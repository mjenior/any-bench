"""Tests for checkpoint/resume module."""

from pathlib import Path

import pytest

from any_bench.models import BenchmarkConfig, JudgeResult, SectionScores
from any_bench.resume import (
    Checkpoint,
    CheckpointMismatchError,
    load_checkpoint,
    load_checkpoint_file,
)


def _make_config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    defaults = {
        "dataset_path": tmp_path / "dataset.json",
        "target_model": "gpt-4o",
        "judge_model": "gemini/gemini-2.5-flash",
        "runs_per_question": 3,
        "output_path": tmp_path / "results.csv",
    }
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _make_result(qid: str = "Q1", composite: int = 4) -> JudgeResult:
    return JudgeResult(
        id=qid,
        composite_score=composite,
        section_scores=SectionScores(
            instruction_compliance=5,
            factual_accuracy=4,
            required_fact_coverage=4,
            reasoning_quality=3,
            relevance_focus=5,
            clarity_usability=4,
        ),
        matched_negative_responses=[],
        review_summary="Good. Accurate. Complete. Done.",
    )


class TestCheckpoint:
    def test_new_checkpoint_empty(self, tmp_path: Path):
        config = _make_config(tmp_path)
        cp = Checkpoint(config)
        assert cp.results == {}
        assert cp.completed_runs("Q1") == 0
        assert not cp.is_question_complete("Q1")

    def test_add_result_and_save(self, tmp_path: Path):
        config = _make_config(tmp_path)
        cp = Checkpoint(config)
        result = _make_result("Q1")

        cp.add_result("Q1", result)

        assert cp.completed_runs("Q1") == 1
        assert cp.path.exists()

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        config = _make_config(tmp_path)
        cp = Checkpoint(config)

        for i in range(3):
            cp.add_result("Q1", _make_result("Q1", composite=3 + i))

        # Load into new checkpoint
        cp2 = Checkpoint(config)
        cp2.load()

        assert cp2.completed_runs("Q1") == 3
        assert cp2.results["Q1"][0].composite_score == 3
        assert cp2.results["Q1"][2].composite_score == 5

    def test_is_question_complete(self, tmp_path: Path):
        config = _make_config(tmp_path, runs_per_question=2)
        cp = Checkpoint(config)

        cp.add_result("Q1", _make_result("Q1"))
        assert not cp.is_question_complete("Q1")

        cp.add_result("Q1", _make_result("Q1"))
        assert cp.is_question_complete("Q1")

    def test_multiple_questions(self, tmp_path: Path):
        config = _make_config(tmp_path, runs_per_question=2)
        cp = Checkpoint(config)

        cp.add_result("Q1", _make_result("Q1"))
        cp.add_result("Q2", _make_result("Q2"))
        cp.add_result("Q1", _make_result("Q1"))

        assert cp.is_question_complete("Q1")
        assert not cp.is_question_complete("Q2")

    def test_mismatch_target_model(self, tmp_path: Path):
        config1 = _make_config(tmp_path, target_model="gpt-4o")
        cp = Checkpoint(config1)
        cp.add_result("Q1", _make_result("Q1"))

        config2 = _make_config(tmp_path, target_model="claude-sonnet-4-20250514")
        cp2 = Checkpoint(config2)
        with pytest.raises(CheckpointMismatchError, match="target_model"):
            cp2.load()

    def test_mismatch_judge_model(self, tmp_path: Path):
        config1 = _make_config(tmp_path)
        cp = Checkpoint(config1)
        cp.add_result("Q1", _make_result("Q1"))

        config2 = _make_config(tmp_path, judge_model="gpt-4o")
        cp2 = Checkpoint(config2)
        with pytest.raises(CheckpointMismatchError, match="judge_model"):
            cp2.load()

    def test_mismatch_runs_per_question(self, tmp_path: Path):
        config1 = _make_config(tmp_path, runs_per_question=3)
        cp = Checkpoint(config1)
        cp.add_result("Q1", _make_result("Q1"))

        config2 = _make_config(tmp_path, runs_per_question=5)
        cp2 = Checkpoint(config2)
        with pytest.raises(CheckpointMismatchError, match="runs_per_question"):
            cp2.load()

    def test_no_checkpoint_file_loads_empty(self, tmp_path: Path):
        config = _make_config(tmp_path)
        cp = Checkpoint(config)
        cp.load()  # Should not raise
        assert cp.results == {}


class TestLoadCheckpoint:
    def test_resume_true_loads(self, tmp_path: Path):
        config = _make_config(tmp_path, resume=True)
        # Create checkpoint file first
        cp = Checkpoint(config)
        cp.add_result("Q1", _make_result("Q1"))

        cp2 = load_checkpoint(config)
        assert cp2.completed_runs("Q1") == 1

    def test_resume_false_ignores_existing(self, tmp_path: Path):
        config_save = _make_config(tmp_path, resume=True)
        cp = Checkpoint(config_save)
        cp.add_result("Q1", _make_result("Q1"))

        config_no_resume = _make_config(tmp_path, resume=False)
        cp2 = load_checkpoint(config_no_resume)
        assert cp2.completed_runs("Q1") == 0


class TestLoadCheckpointFile:
    def test_loads_raw_data(self, tmp_path: Path):
        config = _make_config(tmp_path)
        cp = Checkpoint(config)
        cp.add_result("Q1", _make_result("Q1"))

        data = load_checkpoint_file(config.checkpoint_path)
        assert data["target_model"] == "gpt-4o"
        assert "Q1" in data["results"]
        assert len(data["results"]["Q1"]) == 1
