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

CALM_SPANISH_LYRICS = (
    "La brisa suave mece las hojas al atardecer, todo esta en paz y en silencio, "
    "respiro tranquilo mientras el sol se despide despacio, "
    "una calma dulce envuelve este momento sereno."
)


def test_profile_orders_angry_and_calm_lyrics_sensibly() -> None:
    profiler = LayaTrackProfiler()
    tracks = [
        TrackForProfiling(track_id="angry", artist="Artist", title="Angry Song", lyrics=ANGRY_SPANISH_LYRICS),
        TrackForProfiling(track_id="calm", artist="Artist", title="Calm Song", lyrics=CALM_SPANISH_LYRICS),
    ]

    start = time.process_time()
    profiles = profiler.profile(tracks)
    elapsed = time.process_time() - start
    per_track = elapsed / len(tracks)

    print(f"\nLaya track profiling CPU time: {elapsed:.2f}s total, {per_track:.2f}s/track (batched, n={len(tracks)})")

    by_id = {p.track_id: p for p in profiles}
    angry, calm = by_id["angry"], by_id["calm"]
    print(f"angry: valence={angry.valence:.3f} arousal={angry.arousal:.3f} family={angry.family_id}")
    print(f"calm:  valence={calm.valence:.3f} arousal={calm.arousal:.3f} family={calm.family_id}")

    # The angry track should read as more negative and more aroused than the calm one.
    assert angry.valence < calm.valence
    assert angry.arousal > calm.arousal
    assert angry.polarity_id == "negative"
    for profile in profiles:
        # Tree descent stops at family for tracks: no per-track emotion-level call.
        assert profile.emotion_id is None
        assert profile.emotion_confidence is None
        # Situations were dropped from track-level questions (v3): tracks no
        # longer carry a situation pick (see mood_dj.domain.track_ranking for
        # the prompt-situation relatedness bonus that replaced it).
        assert profile.situation_id is None
        assert profile.situation_confidence is None
        assert profile.version == profiler.version
