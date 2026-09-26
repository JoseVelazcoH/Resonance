"""Unit tests for RecommendFromLibraryUseCase, using in-memory fakes."""

from __future__ import annotations

import pytest

from mood_dj.application.library_store import LibraryStore
from mood_dj.application.recommend_from_library import (
    LibraryNotPreparedError,
    RecommendFromLibraryUseCase,
    RecommendPhase,
)
from mood_dj.domain.models import LyricsEntry, LyricsStatus, PlaylistTrack, Strategy, TrackMoodProfile
from mood_dj.domain.playlist_strategy import PlaylistSignals
from mood_dj.domain.prompt_profile import EmotionPick, PromptProfile, SituationPick

VERSION = "v1"


def _track(track_id: str) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name=f"Song {track_id}", artist="Artist", album="Album", duration_s=200.0,
        cover_url=None, external_url=None,
    )


def _profile(
    track_id: str,
    valence: float = 0.0,
    arousal: float = 0.0,
    polarity_id: str = "positive",
    cluster_id: str = "cluster-a",
    family_id: str = "family-a",
    emotion_id: str = "emotion-a",
    situation_id: str = "movie_night",
) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        valence=valence,
        arousal=arousal,
        polarity_id=polarity_id,
        cluster_id=cluster_id,
        family_id=family_id,
        emotion_id=emotion_id,
        emotion_confidence=0.9,
        situation_id=situation_id,
        situation_confidence=0.8,
        version=VERSION,
    )


class FakeLyricsRepository:
    def __init__(self, entries: dict[str, LyricsEntry]) -> None:
        self._entries = entries

    def get(self, track_id: str) -> LyricsEntry | None:
        return self._entries.get(track_id)

    def save(self, entry: LyricsEntry) -> None:
        self._entries[entry.track_id] = entry


class FakeMoodProfileRepository:
    def __init__(self, profiles: dict[str, TrackMoodProfile] | None = None) -> None:
        self._profiles = dict(profiles or {})

    def get(self, track_id: str, version: str) -> TrackMoodProfile | None:
        profile = self._profiles.get(track_id)
        if profile is None or profile.version != version:
            return None
        return profile

    def save(self, profile: TrackMoodProfile) -> None:
        self._profiles[profile.track_id] = profile


ACCOMPANY_SIGNALS = PlaylistSignals(feels_bad=True, wants_change=False, wants_energy=False, wants_rest=False)
LIFT_SIGNALS = PlaylistSignals(feels_bad=True, wants_change=True, wants_energy=False, wants_rest=False)

NEUTRAL_EMOTION = EmotionPick(
    id="emotion-a", label="Emotion A", confidence=0.9,
    family_id="family-a", cluster_id="cluster-a", polarity_id="positive",
    valence=0.0, arousal=0.0,
)
NEUTRAL_SITUATION = SituationPick(id="movie_night", label="Movie night", confidence=0.8, valence=0.4, arousal=-0.35)


class FakePromptProfiler:
    def __init__(self, prompt_profile: PromptProfile) -> None:
        self._prompt_profile = prompt_profile
        self.calls: list[str] = []

    def profile(self, prompt: str) -> PromptProfile:
        self.calls.append(prompt)
        return self._prompt_profile


def _accompany_profile(target_valence: float = 0.0, target_arousal: float = 0.0) -> PromptProfile:
    return PromptProfile(
        signals=ACCOMPANY_SIGNALS,
        signal_probabilities={"feels_bad": 0.9, "wants_change": 0.1, "wants_energy": 0.1, "wants_rest": 0.1},
        strategy=Strategy.ACCOMPANY,
        target_valence=target_valence,
        target_arousal=target_arousal,
        emotion=NEUTRAL_EMOTION,
        situation=NEUTRAL_SITUATION,
    )


def _lift_profile() -> PromptProfile:
    return PromptProfile(
        signals=LIFT_SIGNALS,
        signal_probabilities={"feels_bad": 0.9, "wants_change": 0.9, "wants_energy": 0.1, "wants_rest": 0.1},
        strategy=Strategy.LIFT,
        target_valence=-0.5,
        target_arousal=0.0,
        emotion=NEUTRAL_EMOTION,
        situation=NEUTRAL_SITUATION,
    )


def _use_case(library_store, repo, mood_profile_repository, prompt_profiler, max_tracks=30):
    return RecommendFromLibraryUseCase(
        library_store=library_store,
        lyrics_repository=repo,
        mood_profile_repository=mood_profile_repository,
        prompt_profiler=prompt_profiler,
        profile_version=VERSION,
        max_tracks=max_tracks,
    )


def test_run_raises_when_library_not_prepared() -> None:
    use_case = _use_case(
        LibraryStore(), FakeLyricsRepository({}), FakeMoodProfileRepository(), FakePromptProfiler(_accompany_profile())
    )

    with pytest.raises(LibraryNotPreparedError):
        use_case.run("session-1", "I feel sad")


def test_run_excludes_instrumental_missing_and_unprofiled_tracks() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a"), _track("b"), _track("c"), _track("d")])
    repo = FakeLyricsRepository(
        {
            "a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now"),
            "b": LyricsEntry("b", LyricsStatus.INSTRUMENTAL, None, "now"),
            "c": LyricsEntry("c", LyricsStatus.MISSING, None, "now"),
            "d": LyricsEntry("d", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    # "d" has lyrics but never got profiled.
    profiles = FakeMoodProfileRepository({"a": _profile("a", valence=0.0, arousal=0.0)})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile()))

    result = use_case.run("session-1", "I feel sad")

    assert result.excluded_instrumental == 1
    assert result.excluded_no_lyrics == 1
    assert result.excluded_no_profile == 1


def test_run_raises_when_no_track_has_a_profile() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    use_case = _use_case(store, repo, FakeMoodProfileRepository(), FakePromptProfiler(_accompany_profile()))

    with pytest.raises(LibraryNotPreparedError):
        use_case.run("session-1", "I feel sad")


def test_only_tracks_at_or_above_threshold_are_included() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("good"), _track("bad")])
    repo = FakeLyricsRepository(
        {
            "good": LyricsEntry("good", LyricsStatus.LYRICS, "la", "now"),
            "bad": LyricsEntry("bad", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    # "good" matches the target point exactly; "bad" is maximally far away.
    profiles = FakeMoodProfileRepository(
        {
            "good": _profile("good", valence=0.0, arousal=0.0),
            "bad": _profile("bad", valence=1.0, arousal=1.0, emotion_id="other", family_id="other",
                             cluster_id="other", polarity_id="negative", situation_id="sleeping"),
        }
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(0.0, 0.0)))

    result = use_case.run("session-1", "I feel sad")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["good"]
    assert result.qualifying_count == 1
    assert result.threshold == pytest.approx(0.65)


def test_caps_tracks_to_max_tracks() -> None:
    store = LibraryStore()
    tracks = [_track(str(i)) for i in range(50)]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository({t.id: LyricsEntry(t.id, LyricsStatus.LYRICS, "la", "now") for t in tracks})
    profiles = FakeMoodProfileRepository({t.id: _profile(t.id, valence=0.0, arousal=0.0) for t in tracks})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(0.0, 0.0)), max_tracks=10)

    result = use_case.run("session-1", "prompt")

    total_tracks = sum(len(stage.tracks) for stage in result.stages)
    assert total_tracks <= 10


def test_lift_strategy_builds_three_stages_without_repeating_tracks() -> None:
    store = LibraryStore()
    tracks = [_track("sad"), _track("mid"), _track("happy")]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository({t.id: LyricsEntry(t.id, LyricsStatus.LYRICS, "la", "now") for t in tracks})
    profiles = FakeMoodProfileRepository(
        {
            "sad": _profile("sad", valence=-0.9, arousal=0.0),
            "mid": _profile("mid", valence=0.0, arousal=0.0),
            "happy": _profile("happy", valence=0.9, arousal=0.0),
        }
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_lift_profile()))

    result = use_case.run("session-1", "I feel sad but want to feel better")

    assert result.strategy is Strategy.LIFT
    assert [stage.name for stage in result.stages] == ["melancholic", "hopeful", "positive"]
    seen_ids: list[str] = []
    for stage in result.stages:
        for ranked in stage.tracks:
            assert ranked.track.id not in seen_ids
            seen_ids.append(ranked.track.id)


def test_reports_understanding_and_ranking_phases() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a", valence=0.0, arousal=0.0)})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(0.0, 0.0)))
    phases = []

    use_case.run("session-1", "prompt", on_progress=lambda progress: phases.append(progress.phase))

    assert RecommendPhase.UNDERSTANDING_MOOD.value in phases
    assert RecommendPhase.RANKING_TRACKS.value in phases


def test_detected_fields_reflect_the_prompt_profile() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a", valence=0.0, arousal=0.0)})
    prompt_profile = _accompany_profile(0.2, -0.1)
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(prompt_profile))

    result = use_case.run("session-1", "prompt")

    assert result.detected.emotion.id == NEUTRAL_EMOTION.id
    assert result.detected.family_id == NEUTRAL_EMOTION.family_id
    assert result.detected.situation.id == NEUTRAL_SITUATION.id
    assert result.detected.target.valence == pytest.approx(0.2)
    assert result.detected.target.arousal == pytest.approx(-0.1)
