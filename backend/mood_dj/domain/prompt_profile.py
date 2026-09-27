"""Domain types and pure blending math for prompt-only mood analysis (Slice C).

`PromptProfile` is the output of analyzing ONLY the listener's prompt (no track
lyrics involved): the playlist-strategy signals, the resolved `Strategy`, a
target (valence, arousal) point, and the emotion/situation picks that target
was blended from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mood_dj.domain.models import Strategy
from mood_dj.domain.playlist_strategy import PlaylistSignals

# The direct valence/arousal score question always gets an answer (it is a
# per-prompt scalar, not a confidence-bearing choice), so it is blended in with
# this fixed weight instead of a measured confidence. Emotion and situation
# picks are weighted by their own tree-descent / choice confidence, so a
# genuinely ambiguous pick (low confidence) contributes less to the target than
# a confident one.
#
# The direct score comes from the same miscalibrated question type used for
# per-track valence/arousal (see `mood_dj.domain.track_ranking`), so its weight
# is reduced below the emotion/situation picks' typical confidence: the emotion
# family and situation coordinates (hand-placed circumplex centroids) are a
# more trustworthy signal than the model's raw direct score. Lowered from 1.0
# (equal weight) to 0.3 so a confident emotion/situation pick dominates the
# blend while the direct score still nudges ambiguous prompts.
DIRECT_SCORE_WEIGHT = 0.3


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
class MoodPick:
    """The flat mood (see `mood_dj.domain.moods`) the prompt's emotion family maps to.

    Bridges the prompt side's deep emotion-tree pick into the same flat
    mood space `TrackMoodProfile` lives in, so ranking (`mood_dj.domain.track_ranking`)
    can compare a prompt's target directly against a track's mood distribution.
    """

    id: str
    label: str


@dataclass(frozen=True)
class PromptProfile:
    """The full result of analyzing a listener's prompt.

    `direct_valence`/`direct_arousal` are the raw, unblended direct-score
    question answers (the same miscalibrated question type used per-track, see
    `mood_dj.domain.track_ranking`), kept here purely for diagnostics/bias
    inspection; ranking and `blend_target` only ever use `target_valence`/
    `target_arousal`.
    """

    signals: PlaylistSignals
    signal_probabilities: dict[str, float]
    strategy: Strategy
    target_valence: float
    target_arousal: float
    emotion: EmotionPick
    situation: SituationPick
    direct_valence: float = 0.0
    direct_arousal: float = 0.0
    mood: MoodPick = field(default_factory=lambda: MoodPick(id="", label=""))


def blend_target(
    direct_valence: float,
    direct_arousal: float,
    emotion: EmotionPick,
    situation: SituationPick,
    direct_weight: float = DIRECT_SCORE_WEIGHT,
) -> tuple[float, float]:
    """Blend the direct score, emotion pick and situation pick into one target point.

    Weighted average where each component's weight is its own confidence, except
    the direct score, which always carries `direct_weight` (defaults to the
    module constant `DIRECT_SCORE_WEIGHT`; the eval harness in
    `scripts/eval_mood.py` overrides it to compare weighting choices, e.g. 0.0
    to drop the direct valence/arousal question entirely)::

        target = (direct * W + emotion.coords * emotion.confidence
                   + situation.coords * situation.confidence)
                  / (W + emotion.confidence + situation.confidence)

    A confidence of 0.0 (e.g. a "none of these" situation pick) simply drops that
    term out of the weighted average.
    """

    total_weight = direct_weight + emotion.confidence + situation.confidence
    valence = (
        direct_valence * direct_weight
        + emotion.valence * emotion.confidence
        + situation.valence * situation.confidence
    ) / total_weight
    arousal = (
        direct_arousal * direct_weight
        + emotion.arousal * emotion.confidence
        + situation.arousal * situation.confidence
    ) / total_weight
    return valence, arousal
