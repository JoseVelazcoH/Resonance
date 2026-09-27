"""Unit tests for the pure mood-diagnostics aggregation (flat-mood + distribution)."""

from __future__ import annotations

import math

import pytest

from mood_dj.domain.mood_diagnostics import compute_diagnostics
from mood_dj.domain.models import TrackMoodProfile


def _profile(
    track_id: str,
    mood_id: str,
    mood_confidence: float,
    positive_probability: float,
    mood_probabilities: dict | None = None,
) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id=track_id,
        mood_id=mood_id,
        mood_confidence=mood_confidence,
        positive_probability=positive_probability,
        version="v1",
        mood_probabilities=mood_probabilities or {mood_id: mood_confidence},
    )


def test_empty_profiles_yield_zeroed_report() -> None:
    report = compute_diagnostics([], total_tracks=0)

    assert report.moods == ()
    assert report.profiled_count == 0
    assert report.unprofiled_share == 0.0
    assert report.overall_mean_entropy == 0.0


def test_groups_and_aggregates_by_mood() -> None:
    profiles = [
        _profile("t1", "love", 0.8, 0.9),
        _profile("t2", "love", 0.6, 0.7),
        _profile("t3", "anger", 0.9, 0.1),
    ]

    report = compute_diagnostics(profiles, total_tracks=3)

    moods_by_id = {m.mood_id: m for m in report.moods}
    assert set(moods_by_id) == {"love", "anger"}
    love = moods_by_id["love"]
    assert love.count == 2
    assert love.mean_confidence == pytest.approx(0.7)
    assert love.mean_positive_probability == pytest.approx(0.8)


def test_unprofiled_share_counts_tracks_without_a_profile() -> None:
    profiles = [_profile("t1", "love", 0.8, 0.9)]

    report = compute_diagnostics(profiles, total_tracks=4)

    assert report.profiled_count == 1
    assert report.unprofiled_share == pytest.approx(0.75)


def test_mean_entropy_is_zero_for_a_fully_confident_one_hot_profile() -> None:
    profile = _profile("t1", "love", 1.0, 0.9, mood_probabilities={"love": 1.0})

    report = compute_diagnostics([profile], total_tracks=1)

    assert report.moods[0].mean_entropy == pytest.approx(0.0)
    assert report.overall_mean_entropy == pytest.approx(0.0)


def test_mean_entropy_is_positive_for_a_spread_out_distribution() -> None:
    distribution = {"love": 0.5, "sadness": 0.3, "fear": 0.2}
    profile = _profile("t1", "love", 0.5, 0.6, mood_probabilities=distribution)
    expected_entropy = -sum(p * math.log2(p) for p in distribution.values())

    report = compute_diagnostics([profile], total_tracks=1)

    assert report.moods[0].mean_entropy == pytest.approx(expected_entropy)


def test_mean_probabilities_averages_the_full_distribution_within_a_mood_group() -> None:
    profiles = [
        _profile("t1", "love", 0.6, 0.8, mood_probabilities={"love": 0.6, "sadness": 0.4}),
        _profile("t2", "love", 0.8, 0.8, mood_probabilities={"love": 0.8, "sadness": 0.2}),
    ]

    report = compute_diagnostics(profiles, total_tracks=2)

    love = report.moods[0]
    assert love.mean_probabilities["love"] == pytest.approx(0.7)
    assert love.mean_probabilities["sadness"] == pytest.approx(0.3)


def test_profile_without_a_distribution_falls_back_to_one_hot() -> None:
    profile = TrackMoodProfile(
        track_id="t1", mood_id="love", mood_confidence=0.9, positive_probability=0.8, version="v1"
    )

    report = compute_diagnostics([profile], total_tracks=1)

    assert report.moods[0].mean_entropy == pytest.approx(0.0)
    assert report.moods[0].mean_probabilities == {"love": 1.0}
