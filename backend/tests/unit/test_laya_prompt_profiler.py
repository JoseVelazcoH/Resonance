"""Unit tests for the pure, model-free helpers in `laya_prompt_profiler.py`.

These do not touch the real Laya router; they test the choice-question shape
building, which is plain data transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

from mood_dj.adapters.laya_prompt_profiler import NONE_OF_THESE_KEY, _situation_question


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
