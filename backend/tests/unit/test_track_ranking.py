"""Unit tests for the pure track-ranking similarity function (flat-mood schema)."""

from __future__ import annotations

import pytest

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.moods import Mood, MoodCatalog
from mood_dj.domain.track_ranking import (
    TargetProfile,
    _situation_bonus,
    expected_position,
    resolve_mood_centroid,
    similarity,
)

MATCH_THRESHOLD = 0.65
VERSION = "v1"

MOODS = MoodCatalog(
    meta={},
    moods=(
        Mood(id="love", label="Amor", criterion="c", valence=0.417, arousal=0.333, polarity="positive",
             families=()),
        Mood(id="happiness", label="Felicidad", criterion="c", valence=0.55, arousal=0.512, polarity="positive",
             families=()),
        Mood(id="comfort", label="Consuelo", criterion="c", valence=0.512, arousal=-0.2, polarity="positive",
             families=()),
        Mood(id="sadness", label="Tristeza", criterion="c", valence=-0.533, arousal=-0.225, polarity="negative",
             families=()),
        Mood(id="loneliness", label="Soledad", criterion="c", valence=-0.362, arousal=-0.063, polarity="negative",
             families=()),
        Mood(id="anger", label="Ira", criterion="c", valence=-0.6, arousal=0.55, polarity="negative",
             families=()),
        Mood(id="fear", label="Miedo", criterion="c", valence=-0.333, arousal=0.5, polarity="negative",
             families=()),
    ),
)


def _track(mood_id: str, positive_probability: float, mood_probabilities: dict | None = None) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id="t",
        mood_id=mood_id,
        mood_confidence=0.9,
        positive_probability=positive_probability,
        version=VERSION,
        mood_probabilities=mood_probabilities or {},
    )


def _target(
    mood_id: str,
    valence: float | None = None,
    arousal: float | None = None,
    situation_id: str = "",
    situation_related_moods: frozenset[str] = frozenset(),
) -> TargetProfile:
    mood = MOODS.by_id(mood_id)
    return TargetProfile(
        valence=mood.valence if valence is None else valence,
        arousal=mood.arousal if arousal is None else arousal,
        mood_id=mood_id,
        situation_id=situation_id,
        situation_related_moods=situation_related_moods,
    )


# -- centroid resolution ------------------------------------------------------


def test_resolve_mood_centroid_returns_the_moods_valence_arousal() -> None:
    assert resolve_mood_centroid(MOODS, "love") == pytest.approx((0.417, 0.333))


def test_resolve_mood_centroid_falls_back_to_origin_for_unknown_mood() -> None:
    assert resolve_mood_centroid(MOODS, "not-a-mood") == (0.0, 0.0)


# -- expected position (distribution-weighted) -------------------------------


def test_expected_position_of_a_one_hot_track_is_its_moods_centroid() -> None:
    track = _track("love", 0.85)

    valence, arousal = expected_position(track, MOODS)

    assert (valence, arousal) == pytest.approx((0.417, 0.333))


def test_expected_position_of_a_mixed_track_blends_component_moods() -> None:
    track = _track("love", 0.6, mood_probabilities={"love": 0.5, "sadness": 0.4, "fear": 0.1})

    valence, arousal = expected_position(track, MOODS)

    expected_valence = 0.5 * 0.417 + 0.4 * -0.533 + 0.1 * -0.333
    expected_arousal = 0.5 * 0.333 + 0.4 * -0.225 + 0.1 * 0.5
    assert valence == pytest.approx(expected_valence)
    assert arousal == pytest.approx(expected_arousal)


# -- similarity: same-mood, opposite-mood, mixed-song scenarios --------------


def test_identical_mood_and_polarity_scores_near_the_maximum() -> None:
    # No situation bonus applies here, so the ceiling is
    # WEIGHT_MOOD_MATCH + WEIGHT_PROXIMITY + WEIGHT_POLARITY = 0.85, not 1.0.
    track = _track("love", positive_probability=0.85)
    target = _target("love")

    assert similarity(track, target, MOODS) > 0.8


def test_opposite_mood_scores_near_zero() -> None:
    track = _track("anger", positive_probability=0.1)
    target = _target("love")

    assert similarity(track, target, MOODS) < 0.3


def test_love_target_admits_love_tracks_and_rejects_anger_and_fear() -> None:
    target = _target("love")
    love_track = _track("love", positive_probability=0.85)
    anger_track = _track("anger", positive_probability=0.1)
    fear_track = _track("fear", positive_probability=0.15)

    assert similarity(love_track, target, MOODS) >= MATCH_THRESHOLD
    assert similarity(anger_track, target, MOODS) < MATCH_THRESHOLD
    assert similarity(fear_track, target, MOODS) < MATCH_THRESHOLD


def test_happiness_target_admits_happiness_tracks_and_rejects_sadness() -> None:
    target = _target("happiness")
    happy_track = _track("happiness", positive_probability=0.85)
    sad_track = _track("sadness", positive_probability=0.15)

    assert similarity(happy_track, target, MOODS) >= MATCH_THRESHOLD
    assert similarity(sad_track, target, MOODS) < MATCH_THRESHOLD


def test_mixed_song_gets_partial_credit_against_both_of_its_component_moods() -> None:
    # A 50% love / 40% sadness / 10% fear song should score higher against a
    # love target than a pure-anger track, and higher against a sadness target
    # than a pure-happiness track -- credit proportional to its distribution.
    mixed = _track("love", positive_probability=0.55, mood_probabilities={"love": 0.5, "sadness": 0.4, "fear": 0.1})
    pure_anger = _track("anger", positive_probability=0.1)
    pure_happiness = _track("happiness", positive_probability=0.85)

    love_target = _target("love")
    sadness_target = _target("sadness")

    assert similarity(mixed, love_target, MOODS) > similarity(pure_anger, love_target, MOODS)
    assert similarity(mixed, sadness_target, MOODS) > similarity(pure_happiness, sadness_target, MOODS)


def test_mixed_song_scores_lower_than_a_pure_match_against_its_dominant_mood() -> None:
    mixed = _track("love", positive_probability=0.6, mood_probabilities={"love": 0.5, "sadness": 0.4, "fear": 0.1})
    pure_love = _track("love", positive_probability=0.85)
    target = _target("love")

    assert similarity(mixed, target, MOODS) < similarity(pure_love, target, MOODS)


def test_track_without_a_distribution_falls_back_to_one_hot_on_mood_id() -> None:
    one_hot_explicit = _track("love", 0.85, mood_probabilities={"love": 1.0})
    no_distribution = _track("love", 0.85)

    target = _target("love")

    assert similarity(one_hot_explicit, target, MOODS) == pytest.approx(similarity(no_distribution, target, MOODS))


# -- polarity agreement --------------------------------------------------------


def test_polarity_disagreement_lowers_the_score_at_equal_mood_and_proximity() -> None:
    target = _target("love")
    agreeing = _track("love", positive_probability=0.9)
    disagreeing = _track("love", positive_probability=0.1)

    assert similarity(agreeing, target, MOODS) > similarity(disagreeing, target, MOODS)


# -- situation relatedness -----------------------------------------------------


def test_situation_related_mood_adds_a_bonus() -> None:
    target_no_situation = _target("love")
    target_with_situation = _target("love", situation_id="date_night", situation_related_moods=frozenset({"comfort"}))
    track = _track("comfort", positive_probability=0.7)

    assert similarity(track, target_with_situation, MOODS) > similarity(track, target_no_situation, MOODS)


def test_situation_bonus_is_proportional_to_probability_mass_on_related_moods() -> None:
    target = _target("love", situation_id="date_night", situation_related_moods=frozenset({"comfort"}))
    high_mass = _track("love", 0.7, mood_probabilities={"love": 0.5, "comfort": 0.5})
    low_mass = _track("love", 0.7, mood_probabilities={"love": 0.9, "comfort": 0.1})

    assert _situation_bonus(high_mass, target) > _situation_bonus(low_mass, target)


def test_no_situation_id_gives_no_situation_bonus_even_with_related_moods_set() -> None:
    target = TargetProfile(
        valence=0.417, arousal=0.333, mood_id="love", situation_id="", situation_related_moods=frozenset({"love"})
    )
    track = _track("love", 0.85)

    with_situation = _target("love", situation_id="date_night", situation_related_moods=frozenset({"love"}))
    assert similarity(track, target, MOODS) < similarity(track, with_situation, MOODS)


# -- bounds ---------------------------------------------------------------


def test_similarity_stays_within_unit_interval() -> None:
    target = _target("love", situation_id="date_night", situation_related_moods=frozenset({"love"}))
    for mood_id in ("love", "happiness", "comfort", "sadness", "loneliness", "anger", "fear"):
        for positive_probability in (0.0, 0.5, 1.0):
            score = similarity(_track(mood_id, positive_probability), target, MOODS)
            assert 0.0 <= score <= 1.0
