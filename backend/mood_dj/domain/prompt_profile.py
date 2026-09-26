"""Domain types and pure blending math for prompt-only mood analysis (Slice C).

`PromptProfile` is the output of analyzing ONLY the listener's prompt (no track
lyrics involved): the playlist-strategy signals, the resolved `Strategy`, a
target (valence, arousal) point, and the emotion/situation picks that target
was blended from.
"""

from __future__ import annotations

from dataclasses import dataclass

from mood_dj.domain.models import Strategy
from mood_dj.domain.playlist_strategy import PlaylistSignals

# The direct valence/arousal score question always gets an answer (it is a
# per-prompt scalar, not a confidence-bearing choice), so it is blended in with
# this fixed weight instead of a measured confidence. Emotion and situation
# picks are weighted by their own tree-descent / choice confidence, so a
# genuinely ambiguous pick (low confidence) contributes less to the target than
# a confident one.
DIRECT_SCORE_WEIGHT = 1.0


@dataclass(frozen=True)
class EmotionPick:
    """The emotion the tree descent settled on, with its circumplex coordinates."""

    id: str
    label: str
    confidence: float
    family_id: str
    cluster_id: str
    polarity_id: str
    valence: float
    arousal: float


@dataclass(frozen=True)
class SituationPick:
    """The higher-confidence situation choice, with its circumplex coordinates."""

    id: str
    label: str
    confidence: float
    valence: float
    arousal: float


@dataclass(frozen=True)
class PromptProfile:
    """The full result of analyzing a listener's prompt."""

    signals: PlaylistSignals
    signal_probabilities: dict[str, float]
    strategy: Strategy
    target_valence: float
    target_arousal: float
    emotion: EmotionPick
    situation: SituationPick


def blend_target(
    direct_valence: float,
    direct_arousal: float,
    emotion: EmotionPick,
    situation: SituationPick,
) -> tuple[float, float]:
    """Blend the direct score, emotion pick and situation pick into one target point.

    Weighted average where each component's weight is its own confidence, except
    the direct score, which always carries `DIRECT_SCORE_WEIGHT`::

        target = (direct * W + emotion.coords * emotion.confidence
                   + situation.coords * situation.confidence)
                  / (W + emotion.confidence + situation.confidence)

    A confidence of 0.0 (e.g. a "none of these" situation pick) simply drops that
    term out of the weighted average.
    """

    total_weight = DIRECT_SCORE_WEIGHT + emotion.confidence + situation.confidence
    valence = (
        direct_valence * DIRECT_SCORE_WEIGHT
        + emotion.valence * emotion.confidence
        + situation.valence * situation.confidence
    ) / total_weight
    arousal = (
        direct_arousal * DIRECT_SCORE_WEIGHT
        + emotion.arousal * emotion.confidence
        + situation.arousal * situation.confidence
    ) / total_weight
    return valence, arousal
