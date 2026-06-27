"""Checkpoint save/load for benchmark resumability."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import BenchmarkConfig, JudgeResult


class CheckpointMismatchError(Exception):
    """Raised when checkpoint config doesn't match current run config."""


class Checkpoint:
    """Manages checkpoint state for a benchmark run.

    Checkpoint format:
    {
        "target_model": str,
        "judge_model": str,
        "runs_per_question": int,
        "results": {question_id: [JudgeResult dict, ...]}
    }
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.path = config.checkpoint_path
        self.results: dict[str, list[JudgeResult]] = {}

    def load(self) -> None:
        """Load checkpoint from disk if it exists. Validates config compatibility."""
        if not self.path.exists():
            return

        data = json.loads(self.path.read_text())

        # Validate config matches
        if data.get("target_model") != self.config.target_model:
            raise CheckpointMismatchError(
                f"Checkpoint target_model '{data.get('target_model')}' "
                f"does not match '{self.config.target_model}'"
            )
        if data.get("judge_model") != self.config.judge_model:
            raise CheckpointMismatchError(
                f"Checkpoint judge_model '{data.get('judge_model')}' "
                f"does not match '{self.config.judge_model}'"
            )
        if data.get("runs_per_question") != self.config.runs_per_question:
            raise CheckpointMismatchError(
                f"Checkpoint runs_per_question '{data.get('runs_per_question')}' "
                f"does not match '{self.config.runs_per_question}'"
            )

        for qid, results_list in data.get("results", {}).items():
            self.results[qid] = [JudgeResult.model_validate(r) for r in results_list]

    def save(self) -> None:
        """Atomically write checkpoint to disk."""
        data = {
            "target_model": self.config.target_model,
            "judge_model": self.config.judge_model,
            "runs_per_question": self.config.runs_per_question,
            "results": {
                qid: [r.model_dump(mode="json") for r in results]
                for qid, results in self.results.items()
            },
        }

        # Atomic write: write to temp file then rename
        dir_path = self.path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self.path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def add_result(self, question_id: str, result: JudgeResult) -> None:
        """Add a judge result for a question and save checkpoint."""
        if question_id not in self.results:
            self.results[question_id] = []
        self.results[question_id].append(result)
        self.save()

    def completed_runs(self, question_id: str) -> int:
        """Return number of completed runs for a question."""
        return len(self.results.get(question_id, []))

    def is_question_complete(self, question_id: str) -> bool:
        """Check if a question has all runs completed."""
        return self.completed_runs(question_id) >= self.config.runs_per_question


def load_checkpoint(config: BenchmarkConfig) -> Checkpoint:
    """Load or create a checkpoint for the given config."""
    checkpoint = Checkpoint(config)
    if config.resume:
        checkpoint.load()
    return checkpoint


def load_checkpoint_file(path: Path) -> dict:
    """Load raw checkpoint data from a file (for stats recompute)."""
    return json.loads(path.read_text())
