"""Tests for the benchmark runner."""

from any_bench.models import DatasetItem, Difficulty
from any_bench.runner import filter_by_difficulty


def _make_dataset_item(qid: str, difficulty: str) -> DatasetItem:
    return DatasetItem.model_validate(
        {
            "id": qid,
            "domain": "Science",
            "subdomain": "Physics",
            "difficulty": difficulty,
            "question_type": "Conceptual",
            "question": "Test?",
            "canonical_answer": "Answer",
            "expanded_answer": "Expanded",
            "required_facts": ["Fact 1"],
            "negative_responses": [],
            "reasoning_path": {"evidence_summary": "Sum", "logical_steps": ["S1"]},
            "source_references": ["Ref"],
        }
    )


class TestFilterByDifficulty:
    def _all_levels(self) -> list[DatasetItem]:
        return [
            _make_dataset_item("Q_easy", "Easy"),
            _make_dataset_item("Q_medium", "Medium"),
            _make_dataset_item("Q_hard", "Hard"),
            _make_dataset_item("Q_expert", "Expert"),
        ]

    def test_easy_keeps_full_set(self):
        items = self._all_levels()
        result = filter_by_difficulty(items, Difficulty.EASY)
        assert [i.id for i in result] == ["Q_easy", "Q_medium", "Q_hard", "Q_expert"]

    def test_medium_drops_easy(self):
        result = filter_by_difficulty(self._all_levels(), Difficulty.MEDIUM)
        assert [i.id for i in result] == ["Q_medium", "Q_hard", "Q_expert"]

    def test_hard_keeps_hard_and_expert(self):
        result = filter_by_difficulty(self._all_levels(), Difficulty.HARD)
        assert [i.id for i in result] == ["Q_hard", "Q_expert"]

    def test_expert_keeps_only_expert(self):
        result = filter_by_difficulty(self._all_levels(), Difficulty.EXPERT)
        assert [i.id for i in result] == ["Q_expert"]

    def test_no_matching_items_returns_empty(self):
        items = [_make_dataset_item("Q_easy", "Easy")]
        assert filter_by_difficulty(items, Difficulty.EXPERT) == []

    def test_easy_returns_copy_not_original(self):
        items = self._all_levels()
        result = filter_by_difficulty(items, Difficulty.EASY)
        assert result is not items
        assert result == items

    def test_empty_input(self):
        assert filter_by_difficulty([], Difficulty.MEDIUM) == []
