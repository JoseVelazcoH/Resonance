"""Domain: per-mood, precision-first track selection policy.

Replaces the earlier weighted-similarity ranking (mood match + circumplex
proximity + polarity agreement + situation bonus, gated by a single
`MATCH_THRESHOLD`) with a much simpler rule fitted directly against
user-labeled data: a track qualifies as a match for a target mood `m` when the
probability mass its cached distribution (`TrackMoodProfile.mood_probabilities`)
assigns to `m` is at or above a per-mood threshold. There is no proximity,
polarity or blended score anymore -- `p(m)` alone decides qualification, and
`p(m)` itself is the number surfaced to the UI as `keep_probability` (it is the
model's raw probability for the target mood, not a calibrated match score).

Thresholds live in `mood_dj/data/mood_selection_policy.json`, loaded and
validated by this module. That threshold set was fitted on a DEV split of 300
user-labeled tracks with a precision-first rule (see the file's `_meta`):
maximize F0.5 subject to precision clearing the random baseline by at least
0.15 and recall staying at or above 0.20. `fear` never cleared that bar on
either split, so it has no threshold (`null` in the JSON): it instead falls
back to a top-1 rule, qualifying only when `fear` is the track's argmax mood
(`TrackMoodProfile.mood_id`).

This is a SEPARATE data file from `moods.json` on purpose. `moods.json` is
hashed by `mood_dj.adapters.laya_track_profiler.compute_version` to invalidate
every cached `TrackMoodProfile` whenever the taxonomy or question wording
changes; the thresholds here can be re-fit and shipped independently, without
forcing a full re-profile of every track in every user's library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from mood_dj.domain.models import TrackMoodProfile
from mood_dj.domain.moods import MoodCatalog, load_moods


@dataclass(frozen=True)
class MoodSelectionPolicy:
    """One qualification rule per mood.

    `thresholds[mood_id]` is either a probability in (0, 1) that `p(mood_id)`
    must meet or exceed, or `None`, meaning "use the top-1 fallback": the
    track qualifies only when `mood_id` is its argmax mood.
    """

    meta: dict = field(default_factory=dict)
    thresholds: dict[str, float | None] = field(default_factory=dict)

    def threshold_for(self, mood_id: str) -> float | None:
        return self.thresholds.get(mood_id)


def _track_distribution(track: TrackMoodProfile) -> dict[str, float]:
    """The track's full mood distribution, falling back to a one-hot on `mood_id`.

    Keeps this module working for a profile with no `mood_probabilities` (e.g.
    hand-built in a test, or a stale cached row from before that field existed).
    """

    if track.mood_probabilities:
        return track.mood_probabilities
    return {track.mood_id: 1.0}


def keep_probability(track: TrackMoodProfile, mood_id: str) -> float:
    """`p(mood_id)` for `track`: the number surfaced to the UI as `keep_probability`."""

    return _track_distribution(track).get(mood_id, 0.0)


def qualifies(track: TrackMoodProfile, mood_id: str, policy: MoodSelectionPolicy) -> bool:
    """Whether `track` qualifies as a match for `mood_id` under `policy`."""

    threshold = policy.threshold_for(mood_id)
    if threshold is None:
        return track.mood_id == mood_id
    return keep_probability(track, mood_id) >= threshold


def validate_selection_policy(thresholds_raw: dict, moods: MoodCatalog) -> list[str]:
    """Return a list of problems with a raw `thresholds` dict; empty means valid.

    Every mood in `moods` must be present, and every value must be a probability
    strictly between 0 and 1, or `None` (top-1 fallback).
    """

    problems: list[str] = []
    mood_ids = {mood.id for mood in moods.moods}

    for mood_id in mood_ids:
        if mood_id not in thresholds_raw:
            problems.append(f"missing threshold for mood {mood_id!r}")

    for mood_id, value in thresholds_raw.items():
        if mood_id not in mood_ids:
            problems.append(f"unknown mood id {mood_id!r} in thresholds")
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
            problems.append(f"threshold for mood {mood_id!r} must be in (0, 1) or null, got {value!r}")

    return problems


def parse_selection_policy(raw: dict, moods: MoodCatalog) -> MoodSelectionPolicy:
    """Build a `MoodSelectionPolicy` from an already-parsed JSON document.

    Raises `ValueError` when the document is invalid (see `validate_selection_policy`).
    """

    thresholds_raw = raw.get("thresholds", {})
    problems = validate_selection_policy(thresholds_raw, moods)
    if problems:
        raise ValueError(f"invalid mood_selection_policy.json: {'; '.join(problems)}")

    thresholds = {mood_id: (None if value is None else float(value)) for mood_id, value in thresholds_raw.items()}
    return MoodSelectionPolicy(meta=raw.get("_meta", {}), thresholds=thresholds)


def _read_json(filename: str) -> dict:
    package = "mood_dj.data"
    with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_selection_policy() -> MoodSelectionPolicy:
    """Load and parse `mood_selection_policy.json` into a `MoodSelectionPolicy`."""

    raw = _read_json("mood_selection_policy.json")
    return parse_selection_policy(raw, load_moods())
