"""PromptProfiler adapter backed by the real Laya Router (multilingual checkpoint).

Analyzes ONLY the listener's prompt (no track lyrics) with as few sequential
`predict` calls as possible:

1. One predict call answers eight questions at once: the four playlist-strategy
   noul signals (same wording as `laya_lyrics_judge.py`), a valence score, an
   arousal score, an emotion polarity choice, and one choice per situation group
   (same wording/options as `laya_track_profiler.py`, so track and prompt
   profiles share the same coordinate space).
2. The emotion tree is then descended with the chosen polarity: cluster choice
   (skipped when the polarity has only one child, same rule as the track
   profiler), family choice, emotion choice -- one predict call per level that
   actually needs one.
3. `target_valence`/`target_arousal` blend the direct scores with the chosen
   emotion's and situation's circumplex coordinates, weighted by confidence
   (see `mood_dj.domain.prompt_profile.blend_target`).

This always returns a profile: an ambiguous or short prompt just yields lower
confidence picks, which `blend_target` naturally down-weights.
"""

from __future__ import annotations

from mood_dj.domain.playlist_strategy import PlaylistSignals, resolve_playlist_strategy
from mood_dj.domain.prompt_profile import EmotionPick, PromptProfile, SituationPick, blend_target
from mood_dj.domain.taxonomy import (
    EmotionTree,
    SituationCatalog,
    iter_all_situations,
    list_children,
    load_emotion_tree,
    load_situations,
)

MODEL_NAME = "multilingual"

VALENCE_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
AROUSAL_LEVELS = ["very calm", "calm", "moderate", "energetic", "very intense"]

NONE_OF_THESE_KEY = "none_of_these"
NONE_OF_THESE_LABEL = "None of these fit what the listener wants."

# Threshold above which a noul (yes/no) probability is treated as "yes". Evaluated
# against the real model (see the situation-question design eval in the change
# history for this module): a lower threshold made pure-emotion prompts trip
# `wants_energy`/`wants_rest` too eagerly, so 0.5 (a plain majority) is kept.
NOUL_SIGNAL_THRESHOLD = 0.5


def _score_to_unit_interval(score: float, level_count: int) -> float:
    max_index = level_count - 1
    normalized = max(0.0, min(1.0, score / max_index))
    return normalized * 2.0 - 1.0


def _signal_questions() -> dict:
    return {
        "feels_bad": {
            "type": "noul",
            "instructions": "Does the listener currently feel bad, sad or low?",
            "criteria": {"false": "The listener does not feel bad.", "true": "The listener feels bad."},
        },
        "wants_change": {
            "type": "noul",
            "instructions": "Does the listener want their mood to change, rather than stay the same?",
            "criteria": {
                "false": "The listener wants to stay in the same mood.",
                "true": "The listener wants their mood to change.",
            },
        },
        "wants_energy": {
            "type": "noul",
            "instructions": "Is the listener asking for energy, movement or something to power through an activity?",
            "criteria": {
                "false": "The listener is not asking for energy or movement.",
                "true": "The listener is asking for energy or movement.",
            },
        },
        "wants_rest": {
            "type": "noul",
            "instructions": "Is the listener asking to relax, wind down or sleep?",
            "criteria": {
                "false": "The listener is not asking to rest or relax.",
                "true": "The listener is asking to rest or relax.",
            },
        },
    }


def _polarity_question(tree: EmotionTree) -> dict:
    return {
        "type": "choice",
        "instructions": "Which emotional polarity best matches what the listener is asking for?",
        "criteria": {
            polarity.id: f"The prompt expresses a mostly {polarity.id} emotional polarity."
            for polarity in tree.polarities
        },
    }


def _situation_question(group_situations) -> dict:
    # `none_of_these` must always be offered, even when a group already holds the
    # domain's max-children-per-node count (10): a group at that cap still needs a
    # way to abstain, otherwise it is forced to pick one of its real situations even
    # for a prompt that matches none of them (e.g. a pure-emotion prompt with no
    # listening context), which produces a confidently wrong forced situation pick.
    criteria = {s.id: s.description for s in group_situations}
    criteria[NONE_OF_THESE_KEY] = NONE_OF_THESE_LABEL
    return {
        "type": "choice",
        "instructions": "Which listening situation best fits what the listener is describing?",
        "criteria": criteria,
    }


def _choice_question(instructions: str, node_children) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {child.id: getattr(child, "description", child.id) for child in node_children},
    }


class LayaPromptProfiler:
    """Adapts the Laya Router (multilingual checkpoint) to the PromptProfiler port."""

    def __init__(
        self,
        router=None,
        tree: EmotionTree | None = None,
        situations: SituationCatalog | None = None,
    ) -> None:
        if router is None:
            from laya import Router

            router = Router()
        self._router = router
        self._tree = tree if tree is not None else load_emotion_tree()
        self._situations = situations if situations is not None else load_situations()

    def profile(self, prompt: str) -> PromptProfile:
        situation_groups = list(self._situations.groups)
        group_a, group_b = situation_groups[0], situation_groups[1]

        questions = {
            **_signal_questions(),
            "valence": {
                "type": "score",
                "instructions": "What is the emotional valence of what the listener wants?",
                "criteria": VALENCE_LEVELS,
            },
            "arousal": {
                "type": "score",
                "instructions": "What is the emotional arousal/energy of what the listener wants?",
                "criteria": AROUSAL_LEVELS,
            },
            "polarity": _polarity_question(self._tree),
            "situation_a": _situation_question(group_a.situations),
            "situation_b": _situation_question(group_b.situations),
        }

        result = self._router.predict(prompt, questions, model=MODEL_NAME)
        answers = result["answers"]

        signal_probabilities = {
            key: float(answers[key]["noul"]) for key in ("feels_bad", "wants_change", "wants_energy", "wants_rest")
        }
        signals = PlaylistSignals(
            feels_bad=signal_probabilities["feels_bad"] >= NOUL_SIGNAL_THRESHOLD,
            wants_change=signal_probabilities["wants_change"] >= NOUL_SIGNAL_THRESHOLD,
            wants_energy=signal_probabilities["wants_energy"] >= NOUL_SIGNAL_THRESHOLD,
            wants_rest=signal_probabilities["wants_rest"] >= NOUL_SIGNAL_THRESHOLD,
        )
        strategy = resolve_playlist_strategy(signals)

        direct_valence = _score_to_unit_interval(float(answers["valence"]["score"]), len(VALENCE_LEVELS))
        direct_arousal = _score_to_unit_interval(float(answers["arousal"]["score"]), len(AROUSAL_LEVELS))

        polarity_id = answers["polarity"]["choice"]
        situation_id, situation_confidence = self._pick_higher_confidence(
            answers["situation_a"], answers["situation_b"]
        )
        situation = self._resolve_situation(situation_id, situation_confidence)

        emotion = self._descend_emotion(prompt, polarity_id)

        target_valence, target_arousal = blend_target(direct_valence, direct_arousal, emotion, situation)

        return PromptProfile(
            signals=signals,
            signal_probabilities=signal_probabilities,
            strategy=strategy,
            target_valence=target_valence,
            target_arousal=target_arousal,
            emotion=emotion,
            situation=situation,
        )

    def _pick_higher_confidence(self, answer_a: dict, answer_b: dict) -> tuple[str, float]:
        choice_a = answer_a["choice"]
        confidence_a = float(answer_a["probabilities"][choice_a])
        choice_b = answer_b["choice"]
        confidence_b = float(answer_b["probabilities"][choice_b])
        if confidence_b > confidence_a:
            return choice_b, confidence_b
        return choice_a, confidence_a

    def _resolve_situation(self, situation_id: str, confidence: float) -> SituationPick:
        if situation_id == NONE_OF_THESE_KEY:
            return SituationPick(id=situation_id, label="None", confidence=0.0, valence=0.0, arousal=0.0)
        for situation in iter_all_situations(self._situations):
            if situation.id == situation_id:
                return SituationPick(
                    id=situation.id,
                    label=situation.label,
                    confidence=confidence,
                    valence=situation.valence,
                    arousal=situation.arousal,
                )
        raise KeyError(f"Unknown situation id: {situation_id!r}")

    def _descend_emotion(self, prompt: str, polarity_id: str) -> EmotionPick:
        polarity = self._find_polarity(polarity_id)

        clusters = list_children(polarity)
        if len(clusters) == 1:
            cluster, cluster_confidence = clusters[0], 1.0
        else:
            questions = {
                "cluster": _choice_question(
                    "Which emotional cluster within this polarity best matches what the listener wants?",
                    clusters,
                )
            }
            result = self._router.predict(prompt, questions, model=MODEL_NAME)
            answer = result["answers"]["cluster"]
            cluster = self._find_cluster(polarity, answer["choice"])
            cluster_confidence = float(answer["probabilities"][answer["choice"]])

        families = list_children(cluster)
        if len(families) == 1:
            family, family_confidence = families[0], 1.0
        else:
            questions = {
                "family": _choice_question(
                    "Which emotion family within this cluster best matches what the listener wants?",
                    families,
                )
            }
            result = self._router.predict(prompt, questions, model=MODEL_NAME)
            answer = result["answers"]["family"]
            family = self._find_family(cluster, answer["choice"])
            family_confidence = float(answer["probabilities"][answer["choice"]])

        emotions = list_children(family)
        if len(emotions) == 1:
            emotion, emotion_confidence = emotions[0], 1.0
        else:
            questions = {
                "emotion": {
                    "type": "choice",
                    "instructions": "Which specific emotion best matches what the listener wants?",
                    "criteria": {e.id: e.label for e in emotions},
                }
            }
            result = self._router.predict(prompt, questions, model=MODEL_NAME)
            answer = result["answers"]["emotion"]
            emotion = self._find_emotion_in_family(family, answer["choice"])
            emotion_confidence = float(answer["probabilities"][answer["choice"]])

        # `cluster_confidence`/`family_confidence` are computed for symmetry with
        # the track profiler's tree descent but are not carried on `EmotionPick`;
        # only the finest (emotion) level confidence is used downstream, since that
        # is the node whose exact coordinates feed `blend_target`.
        del cluster_confidence, family_confidence

        return EmotionPick(
            id=emotion.id,
            label=emotion.label,
            confidence=emotion_confidence,
            family_id=family.id,
            cluster_id=cluster.id,
            polarity_id=polarity.id,
            valence=emotion.valence,
            arousal=emotion.arousal,
        )

    def _find_polarity(self, polarity_id: str):
        for polarity in self._tree.polarities:
            if polarity.id == polarity_id:
                return polarity
        raise KeyError(f"Unknown polarity id: {polarity_id!r}")

    def _find_cluster(self, polarity, cluster_id: str):
        for cluster in polarity.clusters:
            if cluster.id == cluster_id:
                return cluster
        raise KeyError(f"Unknown cluster id: {cluster_id!r}")

    def _find_family(self, cluster, family_id: str):
        for family in cluster.families:
            if family.id == family_id:
                return family
        raise KeyError(f"Unknown family id: {family_id!r}")

    def _find_emotion_in_family(self, family, emotion_id: str):
        for emotion in family.emotions:
            if emotion.id == emotion_id:
                return emotion
        raise KeyError(f"Unknown emotion id: {emotion_id!r}")
