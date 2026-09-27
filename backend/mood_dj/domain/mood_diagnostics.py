"""Pure aggregation of per-track mood profiles into calibration diagnostics.

Read-only report: per flat mood (see `mood_dj.domain.moods`), the count of
profiled tracks assigned that mood (by argmax), the mean confidence, mean
`positive_probability`, mean Shannon entropy of each track's own mood
distribution (low entropy = the model was decisive; high entropy = it hedged
across several moods), and the mean full probability distribution across all
tracks in the group (which mood the group leans toward besides its own). No
I/O here -- callers gather `TrackMoodProfile` rows and pass them in.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from mood_dj.domain.models import TrackMoodProfile


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0.0)


def _distribution(profile: TrackMoodProfile) -> dict[str, float]:
    if profile.mood_probabilities:
        return profile.mood_probabilities
    return {profile.mood_id: 1.0}


def _mean_distribution(profiles: list[TrackMoodProfile]) -> dict[str, float]:
    if not profiles:
        return {}
    totals: dict[str, float] = {}
    for profile in profiles:
        for mood_id, probability in _distribution(profile).items():
            totals[mood_id] = totals.get(mood_id, 0.0) + probability
    return {mood_id: total / len(profiles) for mood_id, total in totals.items()}


@dataclass(frozen=True)
class MoodDiagnostics:
    """Aggregated stats for every track whose argmax mood is `mood_id`."""

    mood_id: str
    count: int
    mean_confidence: float
    mean_positive_probability: float
    mean_entropy: float
    mean_probabilities: dict[str, float]


@dataclass(frozen=True)
class MoodDiagnosticsReport:
    moods: tuple[MoodDiagnostics, ...]
    profiled_count: int
    total_tracks: int
    unprofiled_share: float
    overall_mean_entropy: float


def compute_diagnostics(profiles: list[TrackMoodProfile], total_tracks: int) -> MoodDiagnosticsReport:
    """Aggregate raw per-track profiles into a `MoodDiagnosticsReport`.

    `total_tracks` is the size of the library the profiles were drawn from
    (including tracks that have no lyrics or no profile yet), used only to
    compute `unprofiled_share`.
    """

    by_mood: dict[str, list[TrackMoodProfile]] = {}
    for profile in profiles:
        by_mood.setdefault(profile.mood_id, []).append(profile)

    moods = tuple(
        MoodDiagnostics(
            mood_id=mood_id,
            count=len(rows),
            mean_confidence=statistics.fmean(r.mood_confidence for r in rows),
            mean_positive_probability=statistics.fmean(r.positive_probability for r in rows),
            mean_entropy=statistics.fmean(_entropy(_distribution(r)) for r in rows),
            mean_probabilities=_mean_distribution(rows),
        )
        for mood_id, rows in sorted(by_mood.items())
    )

    profiled_count = len(profiles)
    unprofiled_share = 0.0 if total_tracks <= 0 else max(0.0, total_tracks - profiled_count) / total_tracks
    overall_mean_entropy = (
        statistics.fmean(_entropy(_distribution(p)) for p in profiles) if profiles else 0.0
    )

    return MoodDiagnosticsReport(
        moods=moods,
        profiled_count=profiled_count,
        total_tracks=total_tracks,
        unprofiled_share=unprofiled_share,
        overall_mean_entropy=overall_mean_entropy,
    )
