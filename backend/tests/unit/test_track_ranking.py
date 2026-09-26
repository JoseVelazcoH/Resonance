"""Unit tests for the pure track-ranking similarity function."""

from __future__ import annotations

import pytest

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.track_ranking import TargetProfile, similarity

VERSION = "v1"


def _track(
    valence: float = 0.0,
    arousal: float = 0.0,
    polarity_id: str = "positive",
    cluster_id: str = "cluster-a",
    family_id: str = "family-a",
    emotion_id: str = "emotion-a",
) -> TrackMoodProfile:
    return TrackMoodProfile(
        track_id="t",
        valence=valence,
        arousal=arousal,
        polarity_id=polarity_id,
        cluster_id=cluster_id,
        family_id=family_id,
        emotion_id=emotion_id,
        emotion_confidence=0.9,
        version=VERSION,
    )


def _target(
    valence: float = 0.0,
    arousal: float = 0.0,
    polarity_id: str = "positive",
    cluster_id: str = "cluster-a",
    family_id: str = "family-a",
    emotion_id: str = "emotion-a",
    situation_id: str = "movie_night",
    situation_related_families: frozenset[str] = frozenset(),
    situation_related_clusters: frozenset[str] = frozenset(),
) -> TargetProfile:
    return TargetProfile(
        valence=valence,
        arousal=arousal,
        polarity_id=polarity_id,
        cluster_id=cluster_id,
        family_id=family_id,
        emotion_id=emotion_id,
        situation_id=situation_id,
        situation_related_families=situation_related_families,
        situation_related_clusters=situation_related_clusters,
    )


def test_identical_profiles_score_the_maximum() -> None:
    track = _track()
    target = _target(situation_related_families=frozenset({"family-a"}))

    assert similarity(track, target) == pytest.approx(1.0)


def test_max_valence_arousal_distance_and_no_tier_match_scores_near_zero() -> None:
    track = _track(valence=1.0, arousal=1.0, polarity_id="negative", cluster_id="c2", family_id="f2",
                    emotion_id="e2")
    target = _target(valence=-1.0, arousal=-1.0)

    assert similarity(track, target) == pytest.approx(0.0, abs=1e-9)


def test_same_emotion_scores_higher_than_same_family_only() -> None:
    target = _target()
    same_emotion = _track()
    same_family_only = _track(emotion_id="different-emotion")

    assert similarity(same_emotion, target) > similarity(same_family_only, target)


def test_tier_ordering_emotion_family_cluster_polarity_none() -> None:
    target = _target()
    same_family = _track(emotion_id="e2")
    same_cluster = _track(emotion_id="e2", family_id="f2")
    same_polarity = _track(emotion_id="e2", family_id="f2", cluster_id="c2")
    none_shared = _track(emotion_id="e2", family_id="f2", cluster_id="c2", polarity_id="negative")

    scores = [similarity(t, target) for t in (same_family, same_cluster, same_polarity, none_shared)]
    assert scores == sorted(scores, reverse=True)


def test_related_family_scores_higher_than_related_cluster_only() -> None:
    # movie_night-related family should outrank a track that only shares the
    # related cluster, at equal valence/arousal proximity and emotion tier.
    target = _target(
        situation_related_families=frozenset({"related-family"}),
        situation_related_clusters=frozenset({"related-cluster"}),
    )
    related_family = _track(family_id="related-family", cluster_id="unrelated-cluster", emotion_id="x")
    related_cluster_only = _track(family_id="other-family", cluster_id="related-cluster", emotion_id="y")
    neither = _track(family_id="other-family", cluster_id="unrelated-cluster", emotion_id="z")

    assert similarity(related_family, target) > similarity(related_cluster_only, target)
    assert similarity(related_cluster_only, target) > similarity(neither, target)


def test_no_situation_id_on_target_gives_no_situation_bonus() -> None:
    target = _target(situation_id="", situation_related_families=frozenset({"family-a"}))
    track = _track()

    with_situation = _target(situation_related_families=frozenset({"family-a"}))
    assert similarity(track, target) < similarity(track, with_situation)


def test_similarity_stays_within_unit_interval() -> None:
    target = _target(valence=-1.0, arousal=1.0, situation_related_families=frozenset({"family-a"}))
    for valence in (-1.0, 0.0, 1.0):
        for arousal in (-1.0, 0.0, 1.0):
            score = similarity(_track(valence=valence, arousal=arousal), target)
            assert 0.0 <= score <= 1.0
