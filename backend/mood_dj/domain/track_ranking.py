"""Pure ranking: how well a cached per-track mood profile matches a prompt's target.

`similarity(track_profile, target, moods)` combines four signals into one score
in [0, 1], as a weighted sum (weights sum to 1.0):

  (a) mood match: the probability MASS the track's full mood distribution
      (`TrackMoodProfile.mood_probabilities`, one probability per mood in
      `mood_dj.domain.moods`) assigns to the target's mood -- not just whether
      the track's single argmax mood equals the target's. This is what lets a
      mixed song (e.g. 50% love / 40% sadness) get meaningful partial credit
      against a "sadness" prompt even though "love" is its top pick.
  (b) circumplex proximity: 1 - normalized Euclidean distance between the
      target's (valence, arousal) point and the track's EXPECTED circumplex
      position -- the probability-weighted average of every mood's centroid,
      `sum(p(mood) * mood.centroid)` -- rather than just its top mood's
      centroid, so a mixed song sits between its component moods instead of
      snapping to one.
  (c) polarity agreement: the track's `positive_probability` (the model's own
      binary read of the lyrics) compared against the target's valence sign --
      a track the model called clearly positive/negative agrees or disagrees
      with a target that leans positive/negative.
  (d) situation relatedness: the target's listening situation's
      `related_emotions` (from `situations.json`) are pre-resolved, through the
      emotion tree and then through `moods.json`, to the set of moods they
      imply (`situation_related_moods`); a track scores a bonus proportional to
      the probability mass it places on any of those moods.

This module has no framework or I/O dependencies (`moods` is passed in), so it
is fully unit-testable with plain dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.moods import MoodCatalog

WEIGHT_MOOD_MATCH = 0.45
WEIGHT_PROXIMITY = 0.25
WEIGHT_POLARITY = 0.15
WEIGHT_SITUATION = 0.15

MOOD_MATCH_SAME = 1.0
MOOD_MATCH_NONE = 0.0

SITUATION_RELATED = 1.0
SITUATION_NONE = 0.0

# Both valence and arousal live in [-1, 1], so the widest possible gap on each
# axis is 2.0, making the Euclidean diagonal 2*sqrt(2) the maximum distance.
_MAX_CIRCUMPLEX_DISTANCE = 2.0 * math.sqrt(2.0)


@dataclass(frozen=True)
class TargetProfile:
    """The listener's target mood, in the same coordinate space as `TrackMoodProfile`."""

    valence: float
    arousal: float
    mood_id: str
    situation_id: str = ""
    situation_related_moods: frozenset[str] = field(default_factory=frozenset)


def resolve_mood_centroid(moods: MoodCatalog, mood_id: str) -> tuple[float, float]:
    """Resolve a mood id to its (valence, arousal) circumplex centroid.

    Falls back to (0.0, 0.0) -- the circumplex origin -- for an unrecognized
    mood id (e.g. a profile cached under a stale `moods.json` version), which
    neither favors nor penalizes the track in the proximity term.
    """

    mood = moods.by_id(mood_id)
    if mood is None:
        return 0.0, 0.0
    return mood.valence, mood.arousal


def _track_distribution(track: TrackMoodProfile) -> dict[str, float]:
    """The track's full mood distribution, falling back to a one-hot on `mood_id`.

    The fallback keeps this module working for a profile that has no
    `mood_probabilities` (e.g. hand-built in a test, or a stale cached row from
    before that field existed).
    """

    if track.mood_probabilities:
        return track.mood_probabilities
    return {track.mood_id: 1.0}


def expected_position(track: TrackMoodProfile, moods: MoodCatalog) -> tuple[float, float]:
    """The track's probability-weighted expected (valence, arousal) position.

    `sum(p(mood) * mood.centroid)` over the track's full mood distribution, so a
    mixed song sits between its component moods' centroids instead of snapping
    to only its top pick.
    """

    distribution = _track_distribution(track)
    total_weight = sum(distribution.values()) or 1.0
    valence = 0.0
    arousal = 0.0
    for mood_id, probability in distribution.items():
        mood_valence, mood_arousal = resolve_mood_centroid(moods, mood_id)
        valence += probability * mood_valence
        arousal += probability * mood_arousal
    return valence / total_weight, arousal / total_weight


def _proximity(track: TrackMoodProfile, target: TargetProfile, moods: MoodCatalog) -> float:
    track_valence, track_arousal = expected_position(track, moods)
    distance = math.sqrt((track_valence - target.valence) ** 2 + (track_arousal - target.arousal) ** 2)
    normalized = min(1.0, distance / _MAX_CIRCUMPLEX_DISTANCE)
    return 1.0 - normalized


def _mood_match_bonus(track: TrackMoodProfile, target: TargetProfile) -> float:
    distribution = _track_distribution(track)
    return distribution.get(target.mood_id, MOOD_MATCH_NONE)


def _polarity_agreement(track: TrackMoodProfile, target: TargetProfile) -> float:
    # Map the target's valence ([-1, 1]) onto the same [0, 1] scale as
    # `positive_probability`, then score how close the track's own polarity
    # read is to that target polarity.
    target_positive = (target.valence + 1.0) / 2.0
    return 1.0 - abs(track.positive_probability - target_positive)


def _situation_bonus(track: TrackMoodProfile, target: TargetProfile) -> float:
    if not target.situation_id or not target.situation_related_moods:
        return SITUATION_NONE
    distribution = _track_distribution(track)
    mass = sum(distribution.get(mood_id, 0.0) for mood_id in target.situation_related_moods)
    return SITUATION_RELATED * min(1.0, mass)


def similarity(track: TrackMoodProfile, target: TargetProfile, moods: MoodCatalog) -> float:
    """Return how well `track` matches `target`, in [0, 1]."""

    mood_bonus = _mood_match_bonus(track, target)
    proximity = _proximity(track, target, moods)
    polarity = _polarity_agreement(track, target)
    situation = _situation_bonus(track, target)

    return (
        WEIGHT_MOOD_MATCH * mood_bonus
        + WEIGHT_PROXIMITY * proximity
        + WEIGHT_POLARITY * polarity
        + WEIGHT_SITUATION * situation
    )
