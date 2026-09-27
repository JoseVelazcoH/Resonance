"""Unit tests for `laya_prompt_profiler.py`.

Most tests here are pure, model-free helper tests (choice-question shape
building). The beam and direct-score-weight tests below use a fake Laya
router, in the same style as `test_laya_track_profiler.py`'s `FakeRouter`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mood_dj.adapters.laya_prompt_profiler import NONE_OF_THESE_KEY, LayaPromptProfiler, _situation_question
from mood_dj.domain.taxonomy import load_emotion_tree, load_situations


@dataclass(frozen=True)
class _FakeSituation:
    id: str
    description: str


def _situations(count: int) -> list[_FakeSituation]:
    return [_FakeSituation(id=f"s{i}", description=f"desc {i}") for i in range(count)]


class TestSituationQuestion:
    def test_none_of_these_is_included_when_group_has_fewer_than_ten_situations(self):
        question = _situation_question(_situations(5))
        assert NONE_OF_THESE_KEY in question["criteria"]

    def test_none_of_these_is_also_included_when_group_is_at_the_ten_item_cap(self):
        # A group hitting the MAX_CHILDREN_PER_NODE cap must still be able to abstain,
        # otherwise it is forced to pick a real situation even for prompts that match
        # none of its options (e.g. a pure-emotion prompt with no listening context),
        # which breaks the "no forced high-confidence situation" invariant.
        question = _situation_question(_situations(10))
        assert NONE_OF_THESE_KEY in question["criteria"]
        assert len(question["criteria"]) == 11

    def test_all_real_situation_ids_are_present_as_criteria(self):
        situations = _situations(5)
        question = _situation_question(situations)
        for situation in situations:
            assert question["criteria"][situation.id] == situation.description


# -- LayaPromptProfiler.profile() with a fake router -------------------------


class FakeRouter:
    """Answers every `predict` call deterministically from `overrides`.

    `overrides` maps a question id to either:
    - `{"choice": <key>, "prob": <float>}` for a single dominant choice, or
    - `{"probabilities": {<key>: <float>, ...}}` for an explicit distribution
      (used to pin the polarity margin exactly), or
    - `{"score": <float>}` for a score question.
    Unset question ids default to the first criteria key with a flat
    distribution (or the middle score index).
    """

    def __init__(self, overrides: dict | None = None) -> None:
        self.overrides = overrides or {}
        self.calls: list[dict] = []

    def predict(self, prompt: str, questions: dict, model: str) -> dict:
        self.calls.append(questions)
        answers = {}
        for qid, qdef in questions.items():
            override = self.overrides.get(qid, {})
            if qdef["type"] == "noul":
                answers[qid] = {"noul": override.get("noul", 0.0)}
            elif qdef["type"] == "score":
                levels = qdef["criteria"]
                index = override.get("score", (len(levels) - 1) / 2)
                answers[qid] = {"score": float(index)}
            else:
                keys = list(qdef["criteria"].keys())
                if "probabilities" in override:
                    probabilities = override["probabilities"]
                    choice = max(probabilities, key=probabilities.get)
                else:
                    choice = override.get("choice", keys[0])
                    if choice not in keys:
                        choice = keys[0]
                    prob = override.get("prob", 1.0 / len(keys))
                    probabilities = {k: (prob if k == choice else (1 - prob) / max(1, len(keys) - 1)) for k in keys}
                answers[qid] = {"type": "choice", "choice": choice, "probabilities": probabilities}
        return {"answers": answers}


def _profiler(overrides: dict, beam_margin: float | None = None, direct_score_weight: float | None = None):
    kwargs = {}
    if direct_score_weight is not None:
        kwargs["direct_score_weight"] = direct_score_weight
    return LayaPromptProfiler(
        router=FakeRouter(overrides),
        tree=load_emotion_tree(),
        situations=load_situations(),
        beam_margin=beam_margin,
        **kwargs,
    )


def _base_overrides(polarity_probabilities: dict) -> dict:
    return {
        "polarity": {"probabilities": polarity_probabilities},
        "situation_a": {"choice": "none_of_these", "prob": 1.0},
        "situation_b": {"choice": "none_of_these", "prob": 1.0},
        "family": {"choice": "love_affection", "prob": 0.9},
        "emotion": {"prob": 0.9},
    }


class TestBeamPolaritySelection:
    def test_greedy_by_default_picks_the_top_polarity_even_with_a_tiny_margin(self):
        # "Estoy enamorado"-shaped ambiguity: ambivalent (0.53) barely beats
        # positive (0.43). With no beam_margin configured the profiler must keep
        # the original greedy behavior and pick ambivalent's own family/emotion.
        overrides = _base_overrides({"ambivalent": 0.53, "positive": 0.43, "negative": 0.04})
        overrides["family"] = {"choice": "nostalgia_longing", "prob": 0.97}
        profiler = _profiler(overrides, beam_margin=None)

        profile = profiler.profile("Estoy enamorado")

        assert profile.emotion.polarity_id == "ambivalent"
        assert profile.emotion.family_id == "nostalgia_longing"

    def test_beam_recovers_the_second_polarity_when_its_descent_path_is_more_confident(self):
        # Same near-coin-flip polarity margin (0.53 vs 0.43, margin 0.10 < 0.2),
        # but this time the positive branch's family/emotion picks are far more
        # confident than ambivalent's -- the beam should pick the higher
        # product-of-probabilities path (positive/love_affection), not the
        # greedy top polarity (ambivalent).
        overrides = {
            "polarity": {"probabilities": {"ambivalent": 0.53, "positive": 0.43, "negative": 0.04}},
            "situation_a": {"choice": "none_of_these", "prob": 1.0},
            "situation_b": {"choice": "none_of_these", "prob": 1.0},
        }
        profiler = LayaPromptProfiler(
            router=_BranchAwareRouter(overrides),
            tree=load_emotion_tree(),
            situations=load_situations(),
            beam_margin=0.2,
        )

        profile = profiler.profile("Estoy enamorado")

        assert profile.emotion.polarity_id == "positive"
        assert profile.emotion.family_id == "love_affection"

    def test_beam_does_not_trigger_when_the_margin_is_above_threshold(self):
        overrides = _base_overrides({"positive": 0.9, "ambivalent": 0.05, "negative": 0.05})
        overrides["family"] = {"choice": "love_affection", "prob": 0.9}
        profiler = _profiler(overrides, beam_margin=0.2)

        profile = profiler.profile("Estoy feliz")

        assert profile.emotion.polarity_id == "positive"
        # Only one predict call for "family": the beam never descended a second branch.
        family_calls = [q for q in profiler._router.calls if "family" in q]
        assert len(family_calls) == 1


class _BranchAwareRouter(FakeRouter):
    """Like `FakeRouter`, but the family/emotion answer depends on the branch.

    `positive`'s only family offered is `love_affection` and `ambivalent`'s
    only family offered is `nostalgia_longing` (see emotions.json), so the
    branch is inferred from which criteria keys are on offer, and each branch
    is given a different confidence to exercise the beam's product comparison.
    """

    def predict(self, prompt: str, questions: dict, model: str) -> dict:
        if "family" in questions:
            keys = list(questions["family"]["criteria"].keys())
            if "love_affection" in keys:
                self.overrides = {**self.overrides, "family": {"choice": "love_affection", "prob": 0.95}}
            else:
                self.overrides = {**self.overrides, "family": {"choice": keys[0], "prob": 0.3}}
        if "emotion" in questions:
            self.overrides = {**self.overrides, "emotion": {"prob": 0.9}}
        return super().predict(prompt, questions, model)


class TestDirectScoreWeightOverride:
    def test_direct_score_weight_zero_ignores_the_direct_valence_score(self):
        # A very negative direct score should be fully suppressed when
        # direct_score_weight=0, leaving the target valence driven only by the
        # emotion pick (love_affection, positive valence) and situation.
        overrides = _base_overrides({"positive": 0.9, "ambivalent": 0.05, "negative": 0.05})
        overrides["valence"] = {"score": 0}  # "very negative"
        profiler_default = _profiler(overrides, direct_score_weight=None)
        profiler_zero = _profiler(overrides, direct_score_weight=0.0)

        profile_default = profiler_default.profile("Estoy enamorado")
        profile_zero = profiler_zero.profile("Estoy enamorado")

        assert profile_zero.target_valence > profile_default.target_valence
