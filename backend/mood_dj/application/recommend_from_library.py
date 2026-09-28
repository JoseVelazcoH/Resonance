"""Use case: select a session's cached per-track mood profiles against a prompt's target.

Slice C design: the listener's prompt is analyzed once, on its own, into a
target mood (`PromptProfiler`). Every library track that already has a cached
`TrackMoodProfile` (computed once, ahead of time, during library preparation)
is then checked against that target mood with a precision-first, per-mood
selection policy (`mood_dj.domain.mood_selection_policy`) -- no further model
calls happen per track. A track qualifies when its cached probability for the
target mood clears that mood's threshold (or, for `fear`, when it is the
track's top-1 mood). Tracks without lyrics, instrumental tracks, and tracks
whose lyrics have not been profiled yet (or were profiled under a stale
taxonomy version) are excluded and counted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.mood_selection_policy import MoodSelectionPolicy, keep_probability, load_selection_policy, qualifies
from mood_dj.domain.models import LyricsStatus, PlaylistTrack, Strategy, TrackMoodProfile
from mood_dj.domain.moods import load_moods
from mood_dj.domain.taxonomy import (
    iter_all_situations,
    load_emotion_tree,
    load_situations,
    related_families_and_clusters,
)
from mood_dj.ports.lyrics_repository import LyricsRepository
from mood_dj.ports.mood_profile_repository import MoodProfileRepository
from mood_dj.ports.prompt_profiler import PromptProfiler

logger = logging.getLogger(__name__)

DEFAULT_MAX_TRACKS = 30

LIFT_STAGE_NAMES = ["melancholic", "hopeful", "positive"]
SINGLE_STAGE_NAME = "session"

# Lift keeps its three-stage progression, but each stage now targets a mood
# (rather than a shifted valence point): the listener's own detected mood first,
# then a step toward comfort, then a step toward happiness. No track is ever
# reused across stages.
LIFT_STAGE_MOODS_AFTER_FIRST = ["comfort", "happiness"]

# A situation's related moods only ever add tracks AFTER the primary-mood list,
# and only when the prompt's situation pick is confident enough to trust and the
# primary list did not already fill the cap. Kept deliberately simple: no
# separate cap tuning, no partial-credit scoring, just "still room? still
# qualifies for a related mood? then it's next in line by probability."
SITUATION_CONFIDENCE_GATE = 0.2


class LibraryNotPreparedError(Exception):
    """Raised when a session has no prepared library yet: /library/prepare was never run."""


class RecommendPhase(str, Enum):
    """The stage a recommendation run is currently in, for progress reporting."""

    UNDERSTANDING_MOOD = "understanding your mood"
    RANKING_TRACKS = "ranking tracks"


@dataclass
class RecommendRunProgress:
    """Everything the job manager needs to report about an in-flight recommend run."""

    phase: str
    processed: int = 0
    total: int = 0


RecommendProgressCallback = Callable[[RecommendRunProgress], None]


@dataclass(frozen=True)
class RankedTrack:
    track: PlaylistTrack
    similarity: float


@dataclass(frozen=True)
class LibraryTrackSummary:
    """A lightweight library track, published early (before ranking finishes) so the
    UI can render the whole-library grid while the run is still in progress."""

    id: str
    name: str
    artist: str
    cover_url: str | None


@dataclass(frozen=True)
class RankedLibraryTrack:
    """One library track's ranking outcome, in library order.

    `similarity` is None and `selected` is False for tracks that were excluded
    from ranking (no lyrics, instrumental, or no cached mood profile yet) --
    they are still included in this list so the UI grid always matches the full
    library, they just never qualify for the playlist.
    """

    id: str
    name: str
    artist: str
    cover_url: str | None
    similarity: float | None
    selected: bool


@dataclass(frozen=True)
class PlaylistStage:
    name: str
    tracks: list[RankedTrack] = field(default_factory=list)


@dataclass(frozen=True)
class DetectedEmotion:
    id: str
    label: str
    confidence: float


@dataclass(frozen=True)
class DetectedSituation:
    id: str
    label: str
    confidence: float


@dataclass(frozen=True)
class DetectedTarget:
    valence: float
    arousal: float


@dataclass(frozen=True)
class DetectedMood:
    id: str
    label: str


@dataclass(frozen=True)
class Detected:
    """What Laya understood from the prompt, for transparency in the response."""

    emotion: DetectedEmotion
    family_id: str
    situation: DetectedSituation
    target: DetectedTarget
    mood: DetectedMood = field(default_factory=lambda: DetectedMood(id="", label=""))


@dataclass(frozen=True)
class DecisionsSnapshot:
    """What Laya has decided about the prompt, published as soon as it is known.

    This is available well before track ranking finishes, so the UI can reveal
    it while the (slower) ranking phase is still running.
    """

    strategy: Strategy
    signal_probabilities: dict[str, float]
    detected: Detected


@dataclass(frozen=True)
class PlaylistContribution:
    """How many of the final recommended tracks came from one library playlist."""

    playlist_id: str
    name: str
    image_url: str | None
    track_count: int
    contributed: int


@dataclass(frozen=True)
class PlaylistRecommendation:
    strategy: Strategy
    signal_probabilities: dict[str, float]
    stages: list[PlaylistStage]
    detected: Detected
    excluded_no_lyrics: int
    excluded_instrumental: int
    excluded_no_profile: int
    qualifying_count: int = 0
    # The per-mood threshold used for the primary target mood; `None` for a mood
    # (currently only `fear`) that uses the top-1 fallback instead of a threshold.
    threshold: float | None = None
    playlist_contributions: list[PlaylistContribution] = field(default_factory=list)
    ranked_tracks: list[RankedLibraryTrack] = field(default_factory=list)


class RecommendFromLibraryUseCase:
    """Selects a session's prepared library against a mood prompt using cached profiles."""

    def __init__(
        self,
        library_store: LibraryStore,
        lyrics_repository: LyricsRepository,
        mood_profile_repository: MoodProfileRepository,
        prompt_profiler: PromptProfiler,
        profile_version: str,
        max_tracks: int = DEFAULT_MAX_TRACKS,
        policy: MoodSelectionPolicy | None = None,
    ) -> None:
        self._library_store = library_store
        self._lyrics_repository = lyrics_repository
        self._mood_profile_repository = mood_profile_repository
        self._prompt_profiler = prompt_profiler
        self._profile_version = profile_version
        self._max_tracks = max_tracks
        self._policy = policy if policy is not None else load_selection_policy()
        self._situations = load_situations()
        self._tree = load_emotion_tree()
        self._moods = load_moods()

    def run(
        self,
        session_id: str,
        prompt: str,
        on_progress: RecommendProgressCallback | None = None,
        on_decisions: Callable[[DecisionsSnapshot], None] | None = None,
        on_library_tracks: Callable[[list[LibraryTrackSummary]], None] | None = None,
    ) -> PlaylistRecommendation:
        start = time.perf_counter()
        library = self._library_store.get(session_id)
        if library is None or not library.tracks:
            raise LibraryNotPreparedError(
                f"Library for session {session_id} is not prepared yet; call POST /library/prepare first."
            )

        if on_library_tracks is not None:
            on_library_tracks(
                [
                    LibraryTrackSummary(id=track.id, name=track.name, artist=track.artist, cover_url=track.cover_url)
                    for track in library.tracks
                ]
            )

        def emit(phase: RecommendPhase, processed: int = 0, total: int = 0) -> None:
            if on_progress is not None:
                on_progress(RecommendRunProgress(phase=phase.value, processed=processed, total=total))

        emit(RecommendPhase.UNDERSTANDING_MOOD)
        prompt_profile = self._prompt_profiler.profile(prompt)

        detected = Detected(
            emotion=DetectedEmotion(
                id=prompt_profile.emotion.id,
                label=prompt_profile.emotion.label,
                confidence=prompt_profile.emotion.confidence,
            ),
            family_id=prompt_profile.emotion.family_id,
            situation=DetectedSituation(
                id=prompt_profile.situation.id,
                label=prompt_profile.situation.label,
                confidence=prompt_profile.situation.confidence,
            ),
            target=DetectedTarget(valence=prompt_profile.target_valence, arousal=prompt_profile.target_arousal),
            mood=DetectedMood(id=prompt_profile.mood.id, label=prompt_profile.mood.label),
        )
        if on_decisions is not None:
            on_decisions(
                DecisionsSnapshot(
                    strategy=prompt_profile.strategy,
                    signal_probabilities=prompt_profile.signal_probabilities,
                    detected=detected,
                )
            )

        tracks_by_id, profiles_by_id, excluded_no_lyrics, excluded_instrumental, excluded_no_profile = (
            self._usable_profiles(library.tracks)
        )

        if not profiles_by_id:
            raise LibraryNotPreparedError(
                f"No profiled tracks for session {session_id}; run /library/prepare and retry."
            )

        emit(RecommendPhase.RANKING_TRACKS, total=len(profiles_by_id))

        target_mood_id = prompt_profile.mood.id
        qualifying_count = sum(
            1 for profile in profiles_by_id.values() if qualifies(profile, target_mood_id, self._policy)
        )

        if prompt_profile.strategy is Strategy.LIFT:
            stages = self._lift_stages(tracks_by_id, profiles_by_id, target_mood_id)
        else:
            situation_related_moods = self._situation_related_moods(prompt_profile.situation.id)
            stages = [
                self._single_stage(
                    tracks_by_id, profiles_by_id, target_mood_id, prompt_profile.situation.confidence,
                    situation_related_moods,
                )
            ]

        emit(RecommendPhase.RANKING_TRACKS, len(profiles_by_id), len(profiles_by_id))

        result_track_ids = {ranked.track.id for stage in stages for ranked in stage.tracks}
        keep_probability_by_id = {
            track_id: keep_probability(profile, target_mood_id) for track_id, profile in profiles_by_id.items()
        }
        ranked_tracks = [
            RankedLibraryTrack(
                id=track.id,
                name=track.name,
                artist=track.artist,
                cover_url=track.cover_url,
                similarity=keep_probability_by_id.get(track.id),
                selected=track.id in result_track_ids,
            )
            for track in library.tracks
        ]
        playlist_contributions = [
            PlaylistContribution(
                playlist_id=playlist.id,
                name=playlist.name,
                image_url=playlist.image_url,
                track_count=playlist.track_count,
                contributed=sum(
                    1 for track_id in library.tracks_by_playlist.get(playlist.id, []) if track_id in result_track_ids
                ),
            )
            for playlist in library.playlists
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "RecommendFromLibrary session=%s strategy=%s ranked=%d qualifying=%d elapsed_ms=%.0f",
            session_id,
            prompt_profile.strategy.value,
            len(profiles_by_id),
            qualifying_count,
            elapsed_ms,
        )

        return PlaylistRecommendation(
            strategy=prompt_profile.strategy,
            signal_probabilities=prompt_profile.signal_probabilities,
            stages=stages,
            detected=detected,
            excluded_no_lyrics=excluded_no_lyrics,
            excluded_instrumental=excluded_instrumental,
            excluded_no_profile=excluded_no_profile,
            qualifying_count=qualifying_count,
            threshold=self._policy.threshold_for(target_mood_id),
            playlist_contributions=playlist_contributions,
            ranked_tracks=ranked_tracks,
        )

    def _usable_profiles(
        self, tracks: list[PlaylistTrack]
    ) -> tuple[dict[str, PlaylistTrack], dict[str, TrackMoodProfile], int, int, int]:
        tracks_by_id: dict[str, PlaylistTrack] = {}
        profiles_by_id: dict[str, TrackMoodProfile] = {}
        excluded_no_lyrics = 0
        excluded_instrumental = 0
        excluded_no_profile = 0
        for track in tracks:
            entry = self._lyrics_repository.get(track.id)
            if entry is None or entry.status is LyricsStatus.MISSING:
                excluded_no_lyrics += 1
                continue
            if entry.status is LyricsStatus.INSTRUMENTAL:
                excluded_instrumental += 1
                continue
            profile = self._mood_profile_repository.get(track.id, self._profile_version)
            if profile is None:
                excluded_no_profile += 1
                continue
            tracks_by_id[track.id] = track
            profiles_by_id[track.id] = profile
        return tracks_by_id, profiles_by_id, excluded_no_lyrics, excluded_instrumental, excluded_no_profile

    def _related_emotions(self, situation_id: str) -> tuple[str, ...]:
        for situation in iter_all_situations(self._situations):
            if situation.id == situation_id:
                return situation.related_emotions
        return ()

    def _situation_related_moods(self, situation_id: str) -> frozenset[str]:
        related_families, _related_clusters = related_families_and_clusters(
            self._tree, self._related_emotions(situation_id)
        )
        return frozenset(
            mood_id
            for family_id in related_families
            for mood_id in (self._moods.family_to_mood_id(family_id),)
            if mood_id is not None
        )

    def _tracks_qualifying_for(
        self,
        tracks_by_id: dict[str, PlaylistTrack],
        profiles_by_id: dict[str, TrackMoodProfile],
        mood_id: str,
        exclude_ids: set[str],
        cap: int,
    ) -> list[RankedTrack]:
        """Every not-yet-used track qualifying for `mood_id`, best `p(mood_id)` first, capped."""

        candidates = [
            RankedTrack(track=tracks_by_id[track_id], similarity=keep_probability(profile, mood_id))
            for track_id, profile in profiles_by_id.items()
            if track_id not in exclude_ids and qualifies(profile, mood_id, self._policy)
        ]
        candidates.sort(key=lambda r: r.similarity, reverse=True)
        return candidates[:cap]

    def _single_stage(
        self,
        tracks_by_id: dict[str, PlaylistTrack],
        profiles_by_id: dict[str, TrackMoodProfile],
        target_mood_id: str,
        situation_confidence: float,
        situation_related_moods: frozenset[str],
    ) -> PlaylistStage:
        tracks = self._tracks_qualifying_for(tracks_by_id, profiles_by_id, target_mood_id, set(), self._max_tracks)

        if len(tracks) < self._max_tracks and situation_confidence >= SITUATION_CONFIDENCE_GATE:
            used_ids = {r.track.id for r in tracks}
            bonus: list[RankedTrack] = []
            for related_mood_id in situation_related_moods:
                if related_mood_id == target_mood_id:
                    continue
                for ranked in self._tracks_qualifying_for(
                    tracks_by_id, profiles_by_id, related_mood_id, used_ids, self._max_tracks - len(tracks)
                ):
                    bonus.append(ranked)
                    used_ids.add(ranked.track.id)
            bonus.sort(key=lambda r: r.similarity, reverse=True)
            tracks = tracks + bonus[: self._max_tracks - len(tracks)]

        return PlaylistStage(name=SINGLE_STAGE_NAME, tracks=tracks)

    def _lift_stages(
        self,
        tracks_by_id: dict[str, PlaylistTrack],
        profiles_by_id: dict[str, TrackMoodProfile],
        target_mood_id: str,
    ) -> list[PlaylistStage]:
        stage_moods = [target_mood_id, *LIFT_STAGE_MOODS_AFTER_FIRST]
        per_stage_cap = max(1, self._max_tracks // len(LIFT_STAGE_NAMES))
        used_ids: set[str] = set()
        stages: list[PlaylistStage] = []

        for name, mood_id in zip(LIFT_STAGE_NAMES, stage_moods):
            selected = self._tracks_qualifying_for(tracks_by_id, profiles_by_id, mood_id, used_ids, per_stage_cap)
            for ranked in selected:
                used_ids.add(ranked.track.id)
            stages.append(PlaylistStage(name=name, tracks=selected))

        return stages
