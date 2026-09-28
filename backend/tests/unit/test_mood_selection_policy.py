"""Unit tests for the precision-first per-mood track selection policy."""

from __future__ import annotations

import pytest

from mood_dj.adapters.laya_track_profiler import compute_version
from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.mood_selection_policy import (
    MoodSelectionPolicy,
    keep_probability,
    load_selection_policy,
    parse_selection_policy,
    qualifies,
    validate_selection_policy,
)
from mood_dj.domain.moods import Mood, MoodCatalog, load_moods

VERSION = "v1"

MOODS = MoodCatalog(
    meta={},
    moods=(
        Mood(id="love", label="Amor", criterion="c", valence=0.417, arousal=0.333, polarity="positive", families=()),
        Mood(
            id="happiness", label="Felicidad", criterion="c", valence=0.55, arousal=0.512, polarity="positive",
            families=(),
        ),
        Mood(
            id="comfort", label="Consuelo", criterion="c", valence=0.512, arousal=-0.2, polarity="positive",
            families=(),
        ),
        Mood(
            id="sadness", label="Tristeza", criterion="c", valence=-0.533, arousal=-0.225, polarity="negative",
            families=(),
        ),
        Mood(
            id="loneliness", label="Soledad", criterion="c", valence=-0.362, arousal=-0.063, polarity="negative",
            families=(),
        ),
        Mood(id="anger", label="Ira", criterion="c", valence=-0.6, arousal=0.55, polarity="negative", families=()),
        Mood(id="fear", label="Miedo", criterion="c", valence=-0.333, arousal=0.5, polarity="negative", families=()),
    ),
)


def _track(mood_id: str, mood_probabilities: dict | None = None) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id="t",
        mood_id=mood_id,
        mood_confidence=0.9,
        positive_probability=0.5,
        version=VERSION,
        mood_probabilities=mood_probabilities or {},
    )


# -- loading the real packaged file ------------------------------------------


def test_load_selection_policy_returns_every_mood_id() -> None:
    policy = load_selection_policy()

    mood_ids = {mood.id for mood in load_moods().moods}
    assert set(policy.thresholds.keys()) == mood_ids


def test_load_selection_policy_gives_fear_a_null_threshold() -> None:
    policy = load_selection_policy()

    assert policy.threshold_for("fear") is None


def test_load_selection_policy_thresholds_are_in_unit_interval_or_null() -> None:
    policy = load_selection_policy()

    for mood_id, threshold in policy.thresholds.items():
        assert threshold is None or 0.0 < threshold < 1.0, (mood_id, threshold)


def test_loading_the_real_policy_does_not_change_compute_version() -> None:
    """The policy lives in its own data file precisely so it can evolve without
    forcing a full re-profile of every cached track (see the module docstring)."""

    before = compute_version()
    load_selection_policy()
    after = compute_version()

    assert before == after


# -- parsing / validating a raw dict -----------------------------------------


def _valid_raw() -> dict:
    return {
        "_meta": {"note": "test fixture"},
        "thresholds": {
            "love": 0.72,
            "happiness": 0.08,
            "comfort": 0.18,
            "sadness": 0.14,
            "loneliness": 0.16,
            "anger": 0.5,
            "fear": None,
        },
    }


def test_parse_selection_policy_builds_a_policy_from_a_valid_document() -> None:
    policy = parse_selection_policy(_valid_raw(), MOODS)

    assert policy.threshold_for("love") == pytest.approx(0.72)
    assert policy.threshold_for("fear") is None
    assert policy.meta == {"note": "test fixture"}


def test_validate_selection_policy_flags_a_missing_mood() -> None:
    raw = _valid_raw()["thresholds"]
    del raw["anger"]

    problems = validate_selection_policy(raw, MOODS)

    assert any("anger" in problem for problem in problems)


def test_validate_selection_policy_flags_an_unknown_mood_id() -> None:
    raw = _valid_raw()["thresholds"]
    raw["not-a-mood"] = 0.5

    problems = validate_selection_policy(raw, MOODS)

    assert any("not-a-mood" in problem for problem in problems)


@pytest.mark.parametrize("bad_value", [0.0, 1.0, -0.1, 1.5, "0.5", True])
def test_validate_selection_policy_flags_an_out_of_range_threshold(bad_value: object) -> None:
    raw = _valid_raw()["thresholds"]
    raw["love"] = bad_value

    problems = validate_selection_policy(raw, MOODS)

    assert any("love" in problem for problem in problems)


def test_validate_selection_policy_allows_null_for_top1_fallback() -> None:
    raw = _valid_raw()["thresholds"]
    raw["fear"] = None

    problems = validate_selection_policy(raw, MOODS)

    assert problems == []


def test_parse_selection_policy_raises_on_an_invalid_document() -> None:
    raw = _valid_raw()
    del raw["thresholds"]["anger"]

    with pytest.raises(ValueError):
        parse_selection_policy(raw, MOODS)


# -- keep_probability ----------------------------------------------------------


def test_keep_probability_reads_the_targets_mass_from_the_distribution() -> None:
    track = _track("love", mood_probabilities={"love": 0.3, "sadness": 0.7})

    assert keep_probability(track, "sadness") == pytest.approx(0.7)


def test_keep_probability_falls_back_to_one_hot_on_mood_id_without_a_distribution() -> None:
    track = _track("love")

    assert keep_probability(track, "love") == pytest.approx(1.0)
    assert keep_probability(track, "sadness") == pytest.approx(0.0)


# -- qualifies -----------------------------------------------------------------


def test_qualifies_true_when_probability_meets_the_threshold() -> None:
    policy = MoodSelectionPolicy(meta={}, thresholds={"sadness": 0.14})
    track = _track("sadness", mood_probabilities={"sadness": 0.14})

    assert qualifies(track, "sadness", policy) is True


def test_qualifies_false_when_probability_is_below_the_threshold() -> None:
    policy = MoodSelectionPolicy(meta={}, thresholds={"sadness": 0.14})
    track = _track("sadness", mood_probabilities={"sadness": 0.1})

    assert qualifies(track, "sadness", policy) is False


def test_qualifies_uses_top1_fallback_when_threshold_is_null() -> None:
    policy = MoodSelectionPolicy(meta={}, thresholds={"fear": None})
    top1_fear = _track("fear", mood_probabilities={"fear": 0.4, "anger": 0.6})
    top1_anger = _track("anger", mood_probabilities={"fear": 0.4, "anger": 0.6})

    assert qualifies(top1_fear, "fear", policy) is True
    assert qualifies(top1_anger, "fear", policy) is False
