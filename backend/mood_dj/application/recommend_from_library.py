"""Use case: rank a session's cached per-track mood profiles against a prompt's target.

Slice C design: the listener's prompt is analyzed once, on its own, into a
target mood point (`PromptProfiler`). Every library track that already has a
cached `TrackMoodProfile` (computed once, ahead of time, during library
preparation) is then ranked against that target with a pure, fast similarity
function (`mood_dj.domain.track_ranking.similarity`) -- no further model calls
happen per track. Tracks without lyrics, instrumental tracks, and tracks whose
lyrics have not been profiled yet (or were profiled under a stale taxonomy
version) are excluded and counted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from mood_dj.application.library_store import LibraryStore
from mood_dj.domain.models import LyricsStatus, PlaylistTrack, Strategy, TrackMoodProfile
from mood_dj.domain.taxonomy import (
    iter_all_situations,
    load_emotion_tree,
    load_situations,
    related_families_and_clusters,
)
from mood_dj.domain.track_ranking import TargetProfile, similarity
from mood_dj.ports.lyrics_repository import LyricsRepository
from mood_dj.ports.mood_profile_repository import MoodProfileRepository
from mood_dj.ports.prompt_profiler import PromptProfiler

logger = logging.getLogger(__name__)

DEFAULT_MAX_TRACKS = 30

# A track's similarity to the target must clear this to ever appear in a built
# playlist.
MATCH_THRESHOLD = 0.65

LIFT_STAGE_NAMES = ["melancholic", "hopeful", "positive"]
SINGLE_STAGE_NAME = "session"

# Lift shifts the target valence across three stages (sad -> hopeful -> positive)
# so each stage's ranking favors a different tone, without ever repeating a
# track already used by an earlier stage.
LIFT_VALENCE_SHIFTS = [-0.4, 0.0, 0.4]


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
class Detected:
    """What Laya understood from the prompt, for transparency in the response."""

    emotion: DetectedEmotion
    family_id: str
    situation: DetectedSituation
    target: DetectedTarget


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
    threshold: float = MATCH_THRESHOLD


class RecommendFromLibraryUseCase:
    """Ranks a session's prepared library against a mood prompt using cached profiles."""

    def __init__(
        self,
        library_store: LibraryStore,
        lyrics_repository: LyricsRepository,
        mood_profile_repository: MoodProfileRepository,
        prompt_profiler: PromptProfiler,
        profile_version: str,
        max_tracks: int = DEFAULT_MAX_TRACKS,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> None:
        self._library_store = library_store
        self._lyrics_repository = lyrics_repository
        self._mood_profile_repository = mood_profile_repository
        self._prompt_profiler = prompt_profiler
        self._profile_version = profile_version
        self._max_tracks = max_tracks
        self._match_threshold = match_threshold
        self._situations = load_situations()
        self._tree = load_emotion_tree()

    def run(
        self,
        session_id: str,
        prompt: str,
        on_progress: RecommendProgressCallback | None = None,
    ) -> PlaylistRecommendation:
        start = time.perf_counter()
        library = self._library_store.get(session_id)
        if library is None or not library.tracks:
            raise LibraryNotPreparedError(
                f"Library for session {session_id} is not prepared yet; call POST /library/prepare first."
            )

        def emit(phase: RecommendPhase, processed: int = 0, total: int = 0) -> None:
            if on_progress is not None:
                on_progress(RecommendRunProgress(phase=phase.value, processed=processed, total=total))

        emit(RecommendPhase.UNDERSTANDING_MOOD)
        prompt_profile = self._prompt_profiler.profile(prompt)

        tracks_by_id, profiles_by_id, excluded_no_lyrics, excluded_instrumental, excluded_no_profile = (
            self._usable_profiles(library.tracks)
        )

        if not profiles_by_id:
            raise LibraryNotPreparedError(
                f"No profiled tracks for session {session_id}; run /library/prepare and retry."
            )

        emit(RecommendPhase.RANKING_TRACKS, total=len(profiles_by_id))

        related_families, related_clusters = related_families_and_clusters(
            self._tree, self._related_emotions(prompt_profile.situation.id)
        )
        target = TargetProfile(
            valence=prompt_profile.target_valence,
            arousal=prompt_profile.target_arousal,
            polarity_id=prompt_profile.emotion.polarity_id,
            cluster_id=prompt_profile.emotion.cluster_id,
            family_id=prompt_profile.emotion.family_id,
            emotion_id=prompt_profile.emotion.id,
            situation_id=prompt_profile.situation.id,
            situation_related_families=related_families,
            situation_related_clusters=related_clusters,
        )

        base_ranked = [
            RankedTrack(track=tracks_by_id[track_id], similarity=similarity(profile, target))
            for track_id, profile in profiles_by_id.items()
        ]
        qualifying_count = sum(1 for ranked in base_ranked if ranked.similarity >= self._match_threshold)

        if prompt_profile.strategy is Strategy.LIFT:
            stages = self._lift_stages(tracks_by_id, profiles_by_id, target)
        else:
            stages = [self._single_stage(base_ranked)]

        emit(RecommendPhase.RANKING_TRACKS, len(profiles_by_id), len(profiles_by_id))

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
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "RecommendFromLibrary session=%s strategy=%s ranked=%d qualifying=%d elapsed_ms=%.0f",
            session_id,
            prompt_profile.strategy.value,
            len(base_ranked),
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
            threshold=self._match_threshold,
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

    def _single_stage(self, ranked: list[RankedTrack]) -> PlaylistStage:
        qualifying = [r for r in ranked if r.similarity >= self._match_threshold]
        qualifying.sort(key=lambda r: r.similarity, reverse=True)
        deduped: dict[str, RankedTrack] = {}
        for r in qualifying:
            deduped.setdefault(r.track.id, r)
        capped = list(deduped.values())[: self._max_tracks]
        return PlaylistStage(name=SINGLE_STAGE_NAME, tracks=capped)

    def _lift_stages(
        self,
        tracks_by_id: dict[str, PlaylistTrack],
        profiles_by_id: dict[str, TrackMoodProfile],
        base_target: TargetProfile,
    ) -> list[PlaylistStage]:
        used_ids: set[str] = set()
        per_stage_cap = max(1, self._max_tracks // len(LIFT_STAGE_NAMES))
        stages: list[PlaylistStage] = []

        for name, shift in zip(LIFT_STAGE_NAMES, LIFT_VALENCE_SHIFTS):
            shifted_target = TargetProfile(
                valence=max(-1.0, min(1.0, base_target.valence + shift)),
                arousal=base_target.arousal,
                polarity_id=base_target.polarity_id,
                cluster_id=base_target.cluster_id,
                family_id=base_target.family_id,
                emotion_id=base_target.emotion_id,
                situation_id=base_target.situation_id,
                situation_related_families=base_target.situation_related_families,
                situation_related_clusters=base_target.situation_related_clusters,
            )

            ranked = []
            for track_id, profile in profiles_by_id.items():
                if track_id in used_ids:
                    continue
                score = similarity(profile, shifted_target)
                if score >= self._match_threshold:
                    ranked.append(RankedTrack(track=tracks_by_id[track_id], similarity=score))
            ranked.sort(key=lambda r: r.similarity, reverse=True)

            selected = ranked[:per_stage_cap]
            for r in selected:
                used_ids.add(r.track.id)
            stages.append(PlaylistStage(name=name, tracks=selected))

        return stages
