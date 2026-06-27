"""Tests for statistics module."""

import csv
from pathlib import Path

from any_bench.models import DatasetItem, JudgeResult, SectionScores
from any_bench.stats import (
    CSV_COLUMNS,
    _iqr,
    compute_and_write,
    compute_question_stats,
    write_csv,
)


def _make_judge_result(
    qid: str = "Q1",
    composite: int = 4,
    ic: int = 5,
    fa: int = 4,
    rfc: int = 4,
    rq: int = 3,
    rf: int = 5,
    cu: int = 4,
    neg_matches: int = 0,
) -> JudgeResult:
    return JudgeResult(
        id=qid,
        composite_score=composite,
        section_scores=SectionScores(
            instruction_compliance=ic,
            factual_accuracy=fa,
            required_fact_coverage=rfc,
            reasoning_quality=rq,
            relevance_focus=rf,
            clarity_usability=cu,
        ),
        matched_negative_responses=[
            {"response": "bad", "failure_mode": "wrong", "violated_facts": ["f1"]}
            for _ in range(neg_matches)
        ],
        review_summary="Test summary. Good. Accurate. Done.",
    )


def _make_dataset_item(qid: str = "Q1") -> DatasetItem:
    return DatasetItem.model_validate(
        {
            "id": qid,
            "domain": "Science",
            "subdomain": "Physics",
            "difficulty": "Hard",
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


class TestIQR:
    def test_single_value(self):
        assert _iqr([5]) == 0.0

    def test_two_values(self):
        assert _iqr([2, 4]) == 2.0

    def test_even_count(self):
        # [1, 2, 3, 4] -> lower=[1,2] median=1.5, upper=[3,4] median=3.5, IQR=2.0
        assert _iqr([1, 2, 3, 4]) == 2.0

    def test_odd_count(self):
        # [1, 2, 3, 4, 5] -> lower=[1,2] median=1.5, upper=[4,5] median=4.5, IQR=3.0
        assert _iqr([1, 2, 3, 4, 5]) == 3.0

    def test_identical_values(self):
        assert _iqr([3, 3, 3, 3, 3]) == 0.0

    def test_empty_list(self):
        assert _iqr([]) == 0.0


class TestComputeQuestionStats:
    def test_basic_stats(self):
        item = _make_dataset_item()
        results = [_make_judge_result(composite=c) for c in [3, 4, 4, 5, 5]]
        stats = compute_question_stats(item, results, pass_threshold=3)

        assert stats["id"] == "Q1"
        assert stats["domain"] == "Science"
        assert stats["difficulty"] == "Hard"
        assert stats["composite_median"] == 4
        assert stats["composite_min"] == 3
        assert stats["composite_max"] == 5
        assert stats["num_runs"] == 5
        assert stats["pass_threshold"] == 3

    def test_pass_rate(self):
        item = _make_dataset_item()
        # 3 pass (scores 3,4,5), 2 fail (scores 1,2) with threshold=3
        results = [_make_judge_result(composite=c) for c in [1, 2, 3, 4, 5]]
        stats = compute_question_stats(item, results, pass_threshold=3)
        assert stats["pass_rate"] == 0.6

    def test_negative_match_count(self):
        item = _make_dataset_item()
        results = [
            _make_judge_result(neg_matches=1),
            _make_judge_result(neg_matches=2),
            _make_judge_result(neg_matches=0),
        ]
        stats = compute_question_stats(item, results, pass_threshold=3)
        assert stats["negative_match_count"] == 3

    def test_single_run_no_stdev(self):
        item = _make_dataset_item()
        results = [_make_judge_result(composite=4)]
        stats = compute_question_stats(item, results, pass_threshold=3)
        assert stats["composite_stdev"] == 0.0
        assert stats["num_runs"] == 1

    def test_section_stats_present(self):
        item = _make_dataset_item()
        results = [_make_judge_result() for _ in range(3)]
        stats = compute_question_stats(item, results, pass_threshold=3)
        assert "instruction_compliance_median" in stats
        assert "factual_accuracy_iqr" in stats
        assert "clarity_usability_median" in stats

    def test_csv_column_count(self):
        assert len(CSV_COLUMNS) == 27


class TestWriteCSV:
    def test_writes_correct_columns(self, tmp_path: Path):
        item = _make_dataset_item()
        results = [_make_judge_result() for _ in range(3)]
        stats = compute_question_stats(item, results, pass_threshold=3)

        output = tmp_path / "test.csv"
        write_csv([stats], output)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert set(rows[0].keys()) == set(CSV_COLUMNS)

    def test_multiple_questions(self, tmp_path: Path):
        rows = []
        for qid in ["Q1", "Q2", "Q3"]:
            item = _make_dataset_item(qid)
            results = [_make_judge_result(qid=qid) for _ in range(5)]
            rows.append(compute_question_stats(item, results, pass_threshold=3))

        output = tmp_path / "multi.csv"
        write_csv(rows, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        assert len(csv_rows) == 3
        assert [r["id"] for r in csv_rows] == ["Q1", "Q2", "Q3"]


class TestComputeAndWrite:
    def test_end_to_end(self, tmp_path: Path):
        items = [_make_dataset_item("Q1"), _make_dataset_item("Q2")]
        results = {
            "Q1": [_make_judge_result("Q1") for _ in range(5)],
            "Q2": [_make_judge_result("Q2") for _ in range(5)],
        }
        output = tmp_path / "output.csv"
        rows = compute_and_write(items, results, pass_threshold=3, output_path=output)

        assert len(rows) == 2
        assert output.exists()

    def test_skips_items_without_results(self, tmp_path: Path):
        items = [_make_dataset_item("Q1"), _make_dataset_item("Q2")]
        results = {"Q1": [_make_judge_result("Q1") for _ in range(5)]}
        output = tmp_path / "partial.csv"
        rows = compute_and_write(items, results, pass_threshold=3, output_path=output)

        assert len(rows) == 1
        assert rows[0]["id"] == "Q1"
