"""Unit tests for the pure `blend_target` confidence-weighted blending formula."""

from __future__ import annotations

import pytest

from mood_dj.domain.prompt_profile import EmotionPick, SituationPick, blend_target


def _emotion(confidence: float, valence: float = 0.0, arousal: float = 0.0) -> EmotionPick:
    return EmotionPick(
        id="e", label="E", confidence=confidence, family_id="f", cluster_id="c", polarity_id="p",
        valence=valence, arousal=arousal,
    )


def _situation(confidence: float, valence: float = 0.0, arousal: float = 0.0) -> SituationPick:
    return SituationPick(id="s", label="S", confidence=confidence, valence=valence, arousal=arousal)


def test_zero_confidence_picks_leave_the_direct_score_untouched() -> None:
    valence, arousal = blend_target(0.5, -0.3, _emotion(0.0, 1.0, 1.0), _situation(0.0, -1.0, -1.0))

    assert valence == pytest.approx(0.5)
    assert arousal == pytest.approx(-0.3)


def test_full_confidence_emotion_and_situation_pull_toward_their_coordinates() -> None:
    # With confidence 1.0 each and direct-score weight 1.0, all three terms are
    # equal parts of the average.
    valence, arousal = blend_target(0.0, 0.0, _emotion(1.0, 0.9, 0.6), _situation(1.0, -0.3, 0.3))

    assert valence == pytest.approx((0.0 + 0.9 - 0.3) / 3)
    assert arousal == pytest.approx((0.0 + 0.6 + 0.3) / 3)


def test_higher_confidence_component_dominates_the_blend() -> None:
    low_confidence_emotion = _emotion(0.1, valence=1.0)
    high_confidence_situation = _situation(0.9, valence=-1.0)

    valence, _ = blend_target(0.0, 0.0, low_confidence_emotion, high_confidence_situation)

    assert valence < 0.0
