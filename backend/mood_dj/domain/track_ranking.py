"""Pure ranking: how well a cached per-track mood profile matches a prompt's target.

`similarity(track_profile, target)` combines three signals into one score in
[0, 1], as a weighted sum (weights sum to 1.0):

  (a) valence/arousal proximity: 1 - normalized Euclidean distance between the
      track's (valence, arousal) and the target's. Both coordinates live in
      [-1, 1], so the maximum possible distance is 2*sqrt(2).
  (b) emotion-tree tier bonus: same leaf emotion scores highest, then same
      family, then same cluster, then same polarity, then nothing.
  (c) situation relatedness bonus: tracks no longer carry a situation pick (see
      `mood_dj.adapters.laya_track_profiler`, v3) -- situations describe the
      listener's context, not the song. Instead, the target's situation
      `related_emotions` (from `situations.json`) are pre-resolved to the
      families/clusters they belong to (`situation_related_families` /
      `situation_related_clusters`), and a track scores a bonus when its own
      family (stronger) or cluster (weaker) falls in one of those sets. This
      keeps the situation signal in ranking while removing the two most
      expensive per-track questions.

This module has no framework or I/O dependencies, so it is fully unit-testable
with plain dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mood_dj.domain.models import TrackMoodProfile

WEIGHT_VALENCE_AROUSAL = 0.5
WEIGHT_EMOTION_TIER = 0.3
WEIGHT_SITUATION = 0.2

EMOTION_TIER_SAME_EMOTION = 1.0
EMOTION_TIER_SAME_FAMILY = 0.7
EMOTION_TIER_SAME_CLUSTER = 0.4
EMOTION_TIER_SAME_POLARITY = 0.15
EMOTION_TIER_NONE = 0.0

SITUATION_RELATED_FAMILY = 1.0
SITUATION_RELATED_CLUSTER = 0.5
SITUATION_NONE = 0.0

# Both valence and arousal live in [-1, 1], so the widest possible gap on each
# axis is 2.0, making the Euclidean diagonal 2*sqrt(2) the maximum distance.
_MAX_CIRCUMPLEX_DISTANCE = 2.0 * math.sqrt(2.0)


@dataclass(frozen=True)
class TargetProfile:
    """The listener's target mood, in the same coordinate space as `TrackMoodProfile`."""

    valence: float
    arousal: float
    polarity_id: str
    cluster_id: str
    family_id: str
    emotion_id: str
    situation_id: str
    situation_related_families: frozenset[str] = field(default_factory=frozenset)
    situation_related_clusters: frozenset[str] = field(default_factory=frozenset)


def _valence_arousal_proximity(track: TrackMoodProfile, target: TargetProfile) -> float:
    distance = math.sqrt((track.valence - target.valence) ** 2 + (track.arousal - target.arousal) ** 2)
    normalized = min(1.0, distance / _MAX_CIRCUMPLEX_DISTANCE)
    return 1.0 - normalized


def _emotion_tier_bonus(track: TrackMoodProfile, target: TargetProfile) -> float:
    # Tracks profiled since the family-level descent change (see
    # laya_track_profiler.py) have `emotion_id is None`, so family is the finest
    # tier ever reached for them. Older cached profiles that still carry a real
    # `emotion_id` (a different `version`) keep getting the finer same-emotion
    # tier, for backward compatibility.
    if track.emotion_id is not None and track.emotion_id == target.emotion_id:
        return EMOTION_TIER_SAME_EMOTION
    if track.family_id == target.family_id:
        return EMOTION_TIER_SAME_FAMILY
    if track.cluster_id == target.cluster_id:
        return EMOTION_TIER_SAME_CLUSTER
    if track.polarity_id == target.polarity_id:
        return EMOTION_TIER_SAME_POLARITY
    return EMOTION_TIER_NONE


def _situation_bonus(track: TrackMoodProfile, target: TargetProfile) -> float:
    if not target.situation_id:
        return SITUATION_NONE
    if track.family_id in target.situation_related_families:
        return SITUATION_RELATED_FAMILY
    if track.cluster_id in target.situation_related_clusters:
        return SITUATION_RELATED_CLUSTER
    return SITUATION_NONE


def similarity(track: TrackMoodProfile, target: TargetProfile) -> float:
    """Return how well `track` matches `target`, in [0, 1]."""

    proximity = _valence_arousal_proximity(track, target)
    emotion_bonus = _emotion_tier_bonus(track, target)
    situation_bonus = _situation_bonus(track, target)
    return (
        WEIGHT_VALENCE_AROUSAL * proximity
        + WEIGHT_EMOTION_TIER * emotion_bonus
        + WEIGHT_SITUATION * situation_bonus
    )
