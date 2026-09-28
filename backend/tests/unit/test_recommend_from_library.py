"""Unit tests for RecommendFromLibraryUseCase (precision-first per-mood selection)."""

from __future__ import annotations

import pytest

from mood_dj.application.library_store import LibraryStore
from mood_dj.application.recommend_from_library import (
    LibraryNotPreparedError,
    RecommendFromLibraryUseCase,
    RecommendPhase,
)
from mood_dj.domain.models import LyricsEntry, LyricsStatus, PlaylistSummary, PlaylistTrack, Strategy, TrackMoodProfile
from mood_dj.domain.mood_selection_policy import MoodSelectionPolicy
from mood_dj.domain.playlist_strategy import PlaylistSignals
from mood_dj.domain.prompt_profile import EmotionPick, MoodPick, PromptProfile, SituationPick

VERSION = "v1"

TEST_POLICY = MoodSelectionPolicy(
    meta={},
    thresholds={
        "love": 0.72,
        "happiness": 0.08,
        "comfort": 0.18,
        "sadness": 0.14,
        "loneliness": 0.16,
        "anger": 0.50,
        "fear": None,
    },
)


def _track(track_id: str) -> PlaylistTrack:
    return PlaylistTrack(
        id=track_id, name=f"Song {track_id}", artist="Artist", album="Album", duration_s=200.0,
        cover_url=None, external_url=None,
    )


def _profile(track_id: str, mood_id: str = "happiness", probability: float = 1.0) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        mood_id=mood_id,
        mood_confidence=probability,
        positive_probability=0.5,
        version=VERSION,
        mood_probabilities={mood_id: probability},
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


# Confidence below the situation-bonus gate (0.2), so tests that don't care about
# the situation bonus don't accidentally trigger it via a real taxonomy lookup.
NEUTRAL_SITUATION = SituationPick(id="movie_night", label="Movie night", confidence=0.1, valence=0.4, arousal=-0.35)
# movie_night's related_emotions resolve (through the real taxonomy) to the
# "comfort" mood -- used by the dedicated situation-bonus tests below.
HIGH_CONFIDENCE_SITUATION = SituationPick(
    id="movie_night", label="Movie night", confidence=0.5, valence=0.4, arousal=-0.35
)

ACCOMPANY_SIGNALS = PlaylistSignals(feels_bad=True, wants_change=False, wants_energy=False, wants_rest=False)
LIFT_SIGNALS = PlaylistSignals(feels_bad=True, wants_change=True, wants_energy=False, wants_rest=False)

NEUTRAL_EMOTION = EmotionPick(
    id="emotion-a", label="Emotion A", confidence=0.9,
    family_id="family-a", cluster_id="cluster-a", polarity_id="positive",
    valence=0.0, arousal=0.0,
)
HAPPINESS_MOOD = MoodPick(id="happiness", label="Felicidad")
SADNESS_MOOD = MoodPick(id="sadness", label="Tristeza")
ANGER_MOOD = MoodPick(id="anger", label="Ira")


class FakePromptProfiler:
    def __init__(self, prompt_profile: PromptProfile) -> None:
        self._prompt_profile = prompt_profile
        self.calls: list[str] = []

    def profile(self, prompt: str) -> PromptProfile:
        self.calls.append(prompt)
        return self._prompt_profile


def _accompany_profile(
    mood: MoodPick = HAPPINESS_MOOD,
    situation: SituationPick = NEUTRAL_SITUATION,
    target_valence: float = 0.55,
    target_arousal: float = 0.512,
) -> PromptProfile:
    return PromptProfile(
        signals=ACCOMPANY_SIGNALS,
        signal_probabilities={"feels_bad": 0.9, "wants_change": 0.1, "wants_energy": 0.1, "wants_rest": 0.1},
        strategy=Strategy.ACCOMPANY,
        target_valence=target_valence,
        target_arousal=target_arousal,
        emotion=NEUTRAL_EMOTION,
        situation=situation,
        mood=mood,
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
        mood=SADNESS_MOOD,
    )


def _use_case(library_store, repo, mood_profile_repository, prompt_profiler, max_tracks=30, policy=TEST_POLICY):
    return RecommendFromLibraryUseCase(
        library_store=library_store,
        lyrics_repository=repo,
        mood_profile_repository=mood_profile_repository,
        prompt_profiler=prompt_profiler,
        profile_version=VERSION,
        max_tracks=max_tracks,
        policy=policy,
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
    profiles = FakeMoodProfileRepository({"a": _profile("a")})
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


def test_only_tracks_meeting_the_moods_threshold_are_included() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("good"), _track("bad")])
    repo = FakeLyricsRepository(
        {
            "good": LyricsEntry("good", LyricsStatus.LYRICS, "la", "now"),
            "bad": LyricsEntry("bad", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    profiles = FakeMoodProfileRepository(
        {
            "good": _profile("good", mood_id="happiness", probability=0.9),
            "bad": _profile("bad", mood_id="sadness", probability=1.0),
        }
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD)))

    result = use_case.run("session-1", "I feel great")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["good"]
    assert result.qualifying_count == 1
    assert result.threshold == pytest.approx(0.08)


def test_keep_probability_surfaced_per_track_is_the_targets_mood_probability() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a", mood_id="happiness", probability=0.42)})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD)))

    result = use_case.run("session-1", "prompt")

    ranked = result.stages[0].tracks[0]
    assert ranked.similarity == pytest.approx(0.42)


def test_fear_target_uses_top1_fallback_instead_of_a_probability_threshold() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("top1_fear"), _track("mixed_fear")])
    repo = FakeLyricsRepository(
        {
            "top1_fear": LyricsEntry("top1_fear", LyricsStatus.LYRICS, "la", "now"),
            "mixed_fear": LyricsEntry("mixed_fear", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    profiles = FakeMoodProfileRepository(
        {
            "top1_fear": TrackMoodProfile(
                track_id="top1_fear", mood_id="fear", mood_confidence=0.6, positive_probability=0.2,
                version=VERSION, mood_probabilities={"fear": 0.6, "anger": 0.4},
            ),
            "mixed_fear": TrackMoodProfile(
                track_id="mixed_fear", mood_id="anger", mood_confidence=0.55, positive_probability=0.2,
                version=VERSION, mood_probabilities={"fear": 0.45, "anger": 0.55},
            ),
        }
    )
    fear_mood = MoodPick(id="fear", label="Miedo")
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(mood=fear_mood)))

    result = use_case.run("session-1", "I'm scared")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["top1_fear"]
    assert result.threshold is None


def test_caps_tracks_to_max_tracks() -> None:
    store = LibraryStore()
    tracks = [_track(str(i)) for i in range(50)]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository({t.id: LyricsEntry(t.id, LyricsStatus.LYRICS, "la", "now") for t in tracks})
    profiles = FakeMoodProfileRepository({t.id: _profile(t.id, mood_id="happiness", probability=0.9) for t in tracks})
    use_case = _use_case(
        store, repo, profiles, FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD)), max_tracks=10
    )

    result = use_case.run("session-1", "prompt")

    total_tracks = sum(len(stage.tracks) for stage in result.stages)
    assert total_tracks <= 10


def test_lift_strategy_targets_sadness_then_comfort_then_happiness_without_repeating() -> None:
    store = LibraryStore()
    tracks = [_track("sad"), _track("mid"), _track("happy")]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository({t.id: LyricsEntry(t.id, LyricsStatus.LYRICS, "la", "now") for t in tracks})
    profiles = FakeMoodProfileRepository(
        {
            "sad": _profile("sad", mood_id="sadness", probability=1.0),
            "mid": _profile("mid", mood_id="comfort", probability=1.0),
            "happy": _profile("happy", mood_id="happiness", probability=1.0),
        }
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_lift_profile()))

    result = use_case.run("session-1", "I feel sad but want to feel better")

    assert result.strategy is Strategy.LIFT
    assert [stage.name for stage in result.stages] == ["melancholic", "hopeful", "positive"]
    assert [ranked.track.id for ranked in result.stages[0].tracks] == ["sad"]
    assert [ranked.track.id for ranked in result.stages[1].tracks] == ["mid"]
    assert [ranked.track.id for ranked in result.stages[2].tracks] == ["happy"]
    seen_ids: list[str] = []
    for stage in result.stages:
        for ranked in stage.tracks:
            assert ranked.track.id not in seen_ids
            seen_ids.append(ranked.track.id)


def test_lift_first_stage_targets_the_prompts_detected_mood_not_a_fixed_mood() -> None:
    store = LibraryStore()
    tracks = [_track("angry"), _track("mid"), _track("happy")]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository({t.id: LyricsEntry(t.id, LyricsStatus.LYRICS, "la", "now") for t in tracks})
    profiles = FakeMoodProfileRepository(
        {
            "angry": _profile("angry", mood_id="anger", probability=1.0),
            "mid": _profile("mid", mood_id="comfort", probability=1.0),
            "happy": _profile("happy", mood_id="happiness", probability=1.0),
        }
    )
    lift_profile = _lift_profile()
    angry_profile = PromptProfile(
        signals=lift_profile.signals,
        signal_probabilities=lift_profile.signal_probabilities,
        strategy=Strategy.LIFT,
        target_valence=-0.6,
        target_arousal=0.55,
        emotion=NEUTRAL_EMOTION,
        situation=NEUTRAL_SITUATION,
        mood=ANGER_MOOD,
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(angry_profile))

    result = use_case.run("session-1", "I'm furious but want to calm down")

    assert [ranked.track.id for ranked in result.stages[0].tracks] == ["angry"]


def test_situation_bonus_adds_tracks_after_the_primary_list_when_below_cap() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("primary"), _track("bonus")])
    repo = FakeLyricsRepository(
        {
            "primary": LyricsEntry("primary", LyricsStatus.LYRICS, "la", "now"),
            "bonus": LyricsEntry("bonus", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    profiles = FakeMoodProfileRepository(
        {
            "primary": _profile("primary", mood_id="happiness", probability=0.9),
            # "bonus" doesn't qualify for happiness, but does for "comfort", which
            # movie_night's related_emotions resolve to.
            "bonus": _profile("bonus", mood_id="comfort", probability=0.9),
        }
    )
    use_case = _use_case(
        store, repo, profiles,
        FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD, situation=HIGH_CONFIDENCE_SITUATION)),
        max_tracks=10,
    )

    result = use_case.run("session-1", "prompt")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["primary", "bonus"]


def test_situation_bonus_is_skipped_below_the_confidence_gate() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("primary"), _track("bonus")])
    repo = FakeLyricsRepository(
        {
            "primary": LyricsEntry("primary", LyricsStatus.LYRICS, "la", "now"),
            "bonus": LyricsEntry("bonus", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    profiles = FakeMoodProfileRepository(
        {
            "primary": _profile("primary", mood_id="happiness", probability=0.9),
            "bonus": _profile("bonus", mood_id="comfort", probability=0.9),
        }
    )
    # NEUTRAL_SITUATION's confidence (0.1) is below the 0.2 gate.
    use_case = _use_case(
        store, repo, profiles,
        FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD, situation=NEUTRAL_SITUATION)),
        max_tracks=10,
    )

    result = use_case.run("session-1", "prompt")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["primary"]


def test_situation_bonus_is_skipped_once_the_primary_list_fills_the_cap() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("primary"), _track("bonus")])
    repo = FakeLyricsRepository(
        {
            "primary": LyricsEntry("primary", LyricsStatus.LYRICS, "la", "now"),
            "bonus": LyricsEntry("bonus", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    profiles = FakeMoodProfileRepository(
        {
            "primary": _profile("primary", mood_id="happiness", probability=0.9),
            "bonus": _profile("bonus", mood_id="comfort", probability=0.9),
        }
    )
    use_case = _use_case(
        store, repo, profiles,
        FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD, situation=HIGH_CONFIDENCE_SITUATION)),
        max_tracks=1,
    )

    result = use_case.run("session-1", "prompt")

    track_ids = [ranked.track.id for stage in result.stages for ranked in stage.tracks]
    assert track_ids == ["primary"]


def test_reports_understanding_and_ranking_phases() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a")})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile()))
    phases = []

    use_case.run("session-1", "prompt", on_progress=lambda progress: phases.append(progress.phase))

    assert RecommendPhase.UNDERSTANDING_MOOD.value in phases
    assert RecommendPhase.RANKING_TRACKS.value in phases


def test_detected_fields_reflect_the_prompt_profile() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a")})
    prompt_profile = _accompany_profile(target_valence=0.2, target_arousal=-0.1)
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(prompt_profile))

    result = use_case.run("session-1", "prompt")

    assert result.detected.emotion.id == NEUTRAL_EMOTION.id
    assert result.detected.family_id == NEUTRAL_EMOTION.family_id
    assert result.detected.situation.id == NEUTRAL_SITUATION.id
    assert result.detected.target.valence == pytest.approx(0.2)
    assert result.detected.target.arousal == pytest.approx(-0.1)
    assert result.detected.mood.id == HAPPINESS_MOOD.id
    assert result.detected.mood.label == HAPPINESS_MOOD.label


def test_on_decisions_fires_before_ranking_with_strategy_and_detected() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a")])
    repo = FakeLyricsRepository({"a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now")})
    profiles = FakeMoodProfileRepository({"a": _profile("a")})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile()))
    seen = []

    def on_decisions(decisions):
        seen.append(decisions)

    def on_progress(progress):
        # Decisions must already be available once ranking starts.
        if progress.phase == RecommendPhase.RANKING_TRACKS.value:
            assert len(seen) == 1

    use_case.run("session-1", "prompt", on_progress=on_progress, on_decisions=on_decisions)

    assert len(seen) == 1
    assert seen[0].strategy == Strategy.ACCOMPANY
    assert seen[0].detected.emotion.id == NEUTRAL_EMOTION.id
    assert seen[0].signal_probabilities["feels_bad"] == pytest.approx(0.9)


def test_playlist_contributions_count_result_tracks_per_playlist() -> None:
    store = LibraryStore()
    store.save(
        "session-1",
        {},
        [_track("a"), _track("b"), _track("c")],
        playlist_order=["p1", "p2"],
        tracks_by_playlist={"p1": ["a", "b"], "p2": ["b", "c"]},
        playlists=[
            PlaylistSummary(id="p1", name="Playlist One", image_url=None, track_count=2, snapshot_id="s1"),
            PlaylistSummary(id="p2", name="Playlist Two", image_url="https://img", track_count=2, snapshot_id="s2"),
        ],
    )
    repo = FakeLyricsRepository(
        {
            "a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now"),
            "b": LyricsEntry("b", LyricsStatus.LYRICS, "la", "now"),
            "c": LyricsEntry("c", LyricsStatus.LYRICS, "la", "now"),
        }
    )
    # Only "a" and "b" get a matching profile; "c" is excluded (no profile) so it
    # never counts toward p2's contribution even though it belongs to p2.
    profiles = FakeMoodProfileRepository({"a": _profile("a"), "b": _profile("b")})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile()))

    result = use_case.run("session-1", "prompt")

    contributions = {c.playlist_id: c for c in result.playlist_contributions}
    assert contributions["p1"].contributed == 2
    assert contributions["p1"].name == "Playlist One"
    assert contributions["p1"].track_count == 2
    assert contributions["p2"].contributed == 1
    assert contributions["p2"].image_url == "https://img"


def test_ranked_tracks_cover_the_whole_library_in_library_order() -> None:
    store = LibraryStore()
    tracks = [_track("a"), _track("b"), _track("c")]
    store.save("session-1", {}, tracks)
    repo = FakeLyricsRepository(
        {
            "a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now"),
            "b": LyricsEntry("b", LyricsStatus.LYRICS, "la", "now"),
            "c": LyricsEntry("c", LyricsStatus.MISSING, None, "now"),
        }
    )
    # "a" matches the target and is selected; "b" has a profile but does not
    # qualify; "c" has no lyrics so it never gets a profile at all.
    profiles = FakeMoodProfileRepository(
        {
            "a": _profile("a", mood_id="happiness", probability=0.9),
            "b": _profile("b", mood_id="sadness", probability=1.0),
        }
    )
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile(mood=HAPPINESS_MOOD)))

    result = use_case.run("session-1", "I feel sad")

    assert [ranked.id for ranked in result.ranked_tracks] == ["a", "b", "c"]
    by_id = {ranked.id: ranked for ranked in result.ranked_tracks}
    assert by_id["a"].selected is True
    assert by_id["a"].similarity is not None
    assert by_id["b"].selected is False
    assert by_id["b"].similarity is not None
    assert by_id["c"].selected is False
    assert by_id["c"].similarity is None


def test_on_library_tracks_fires_with_the_whole_library_before_ranking() -> None:
    store = LibraryStore()
    store.save("session-1", {}, [_track("a"), _track("b")])
    repo = FakeLyricsRepository(
        {
            "a": LyricsEntry("a", LyricsStatus.LYRICS, "la", "now"),
            "b": LyricsEntry("b", LyricsStatus.MISSING, None, "now"),
        }
    )
    profiles = FakeMoodProfileRepository({"a": _profile("a")})
    use_case = _use_case(store, repo, profiles, FakePromptProfiler(_accompany_profile()))
    seen = []

    def on_library_tracks(tracks):
        seen.append(tracks)

    def on_progress(progress):
        # The library snapshot must already be available before ranking starts.
        if progress.phase == RecommendPhase.RANKING_TRACKS.value:
            assert len(seen) == 1

    use_case.run("session-1", "prompt", on_progress=on_progress, on_library_tracks=on_library_tracks)

    assert len(seen) == 1
    assert [track.id for track in seen[0]] == ["a", "b"]
