"""Integration test against the real multilingual Laya model for TrackProfiler.

Marked `laya` so it is skipped by default (it downloads real model weights and runs
actual inference). Run explicitly with: `uv run pytest -m laya`.

Also measures and prints wall-clock CPU time per track (batched, on this machine)
so the full-library cost can be estimated.
"""

from __future__ import annotations

import time

import pytest

from mood_dj.adapters.laya_track_profiler import LayaTrackProfiler
from mood_dj.ports.track_profiler import TrackForProfiling

pytestmark = pytest.mark.laya

ANGRY_SPANISH_LYRICS = (
    "Estoy harto de tanta mentira, la rabia me quema por dentro, "
    "grito con furia porque ya no aguanto, traicion tras traicion y el odio crece, "
    "quiero romperlo todo, esta ira no se apaga."
)

LOVE_SPANISH_LYRICS = (
    "Te quiero con toda el alma, mi corazon es tuyo desde el primer dia, "
    "cada momento a tu lado es un regalo, sueño con abrazarte para siempre, "
    "eres el amor de mi vida y no imagino el mundo sin ti."
)


def test_profile_orders_angry_and_love_lyrics_sensibly() -> None:
    profiler = LayaTrackProfiler()
    tracks = [
        TrackForProfiling(track_id="angry", artist="Artist", title="Angry Song", lyrics=ANGRY_SPANISH_LYRICS),
        TrackForProfiling(track_id="love", artist="Artist", title="Love Song", lyrics=LOVE_SPANISH_LYRICS),
    ]

    start = time.process_time()
    profiles = profiler.profile(tracks)
    elapsed = time.process_time() - start
    per_track = elapsed / len(tracks)

    print(f"\nLaya track profiling CPU time: {elapsed:.2f}s total, {per_track:.2f}s/track (batched, n={len(tracks)})")

    by_id = {p.track_id: p for p in profiles}
    angry, love = by_id["angry"], by_id["love"]
    print(
        f"angry: mood={angry.mood_id} confidence={angry.mood_confidence:.3f} "
        f"positive_probability={angry.positive_probability:.3f}"
    )
    print(
        f"love:  mood={love.mood_id} confidence={love.mood_confidence:.3f} "
        f"positive_probability={love.positive_probability:.3f}"
    )

    assert angry.mood_id == "anger"
    assert angry.positive_probability < 0.5
    assert love.mood_id == "love"
    assert love.positive_probability > 0.5

    for profile in profiles:
        assert profile.version == profiler.version
        assert set(profile.mood_probabilities.keys()) == {
            "love", "happiness", "comfort", "sadness", "loneliness", "anger", "fear",
        }
        assert profile.mood_probabilities[profile.mood_id] == pytest.approx(profile.mood_confidence)
