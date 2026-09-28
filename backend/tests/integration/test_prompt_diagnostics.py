"""Prompt-side raw valence/arousal diagnostic against the real Laya model.

Marked `laya` so it is skipped by default (downloads real model weights and runs
actual inference). Run explicitly with: `uv run pytest -m laya`.

Prints each prompt's RAW (unblended) direct valence/arousal score -- the same
miscalibrated question type used per-track (see `mood_dj.domain.mood_selection_policy`)
-- across ~8 varied prompts, so bias on the prompt side can be inspected
alongside the per-track evidence. No strict numeric assertions beyond basic
range sanity: this is a diagnostic, not a regression test of exact model output.
"""

from __future__ import annotations

import pytest

from mood_dj.adapters.laya_prompt_profiler import LayaPromptProfiler

pytestmark = pytest.mark.laya

DIAGNOSTIC_PROMPTS = [
    "Estoy feliz y lleno de energia",
    "Quiero relajarme y estar en calma",
    "Estoy furioso, todo me molesta",
    "Me siento muy triste y solo",
    "Quiero bailar toda la noche",
    "Necesito llorar un rato",
    "Estoy enamorado y todo se siente perfecto",
    "Tengo miedo de lo que viene",
]


def test_prompt_side_raw_valence_arousal_diagnostic() -> None:
    profiler = LayaPromptProfiler()

    print("\nPrompt-side raw direct valence/arousal diagnostic")
    print(f"{'prompt':<45}{'raw valence':>12}{'raw arousal':>12}{'family':>18}")
    for prompt in DIAGNOSTIC_PROMPTS:
        profile = profiler.profile(prompt)
        print(
            f"{prompt:<45}{profile.direct_valence:>12.3f}{profile.direct_arousal:>12.3f}"
            f"{profile.emotion.family_id:>18}"
        )
        assert -1.0 <= profile.direct_valence <= 1.0
        assert -1.0 <= profile.direct_arousal <= 1.0
