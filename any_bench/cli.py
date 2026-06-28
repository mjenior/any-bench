"""Click CLI entry point for any-bench."""

from __future__ import annotations

import asyncio
import enum
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from . import __version__
from .models import BenchmarkConfig, JudgeResult, Difficulty
from .resume import CheckpointMismatchError, load_checkpoint_file
from .stats import compute_and_write, format_summary_report


class EnumChoice(click.Choice):
    """Click choice for a str-valued Enum.

    Accepts the enum's values case-insensitively (e.g. ``easy``, ``EASY`` and
    ``Easy`` all resolve to ``Difficulty.EASY``) and converts the sanitized
    input into the corresponding Enum member.
    """

    def __init__(self, enum_cls: type[enum.Enum]) -> None:
        self.enum_cls = enum_cls
        super().__init__([e.value for e in enum_cls], case_sensitive=False)

    def convert(self, value, param, ctx):  # type: ignore[override]
        if isinstance(value, self.enum_cls):
            return value
        matched = super().convert(value, param, ctx)
        return self.enum_cls(matched)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_system_prompt(value: str | None) -> str | None:
    """Resolve system prompt: if starts with @, read from file."""
    if value is None:
        return None
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            raise click.BadParameter(f"System prompt file not found: {path}")
        return path.read_text()
    return value


load_dotenv()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """any-bench -- corpus-grounded LLM evaluation."""


@cli.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option(
    "-m", 
    "--model", 
    default="anthropic/claude-opus-4.8",
    required=True, 
    help="Target model (litellm format). Default is Opus-4.8")
@click.option(
    "-j",
    "--judge-model",
    default="gemini/gemini-3.5-flash",
    show_default=True,
    help="Judge model identifier (litellm format).",
)
@click.option("-n", "--runs", default=5, show_default=True, help="Runs per question")
@click.option("-c", "--concurrency", default=10, show_default=True, help="Max concurrent API calls")
@click.option(
    "-p",
    "--pass-threshold",
    default=3,
    show_default=True,
    help="Composite score >= this counts as pass",
)
@click.option(
    "-o",
    "--output",
    default="benchmark_results.csv",
    show_default=True,
    type=click.Path(),
    help="Output CSV path",
)
@click.option(
    "-s",
    "--system-prompt",
    default=None,
    help="System prompt for target model (string or @file)",
)
@click.option(
    "-d",
    "--difficulty",
    type=EnumChoice(Difficulty),
    default=Difficulty.EASY.value,
    show_default=True,
    help="Minimum difficulty level of questions to include. Default includes the full test set.",
)
@click.option("--resume/--no-resume", default=True, show_default=True, help="Resume from checkpoint")
@click.option("--target-temperature", default=None, type=float, help="Temperature for target model")
@click.option(
    "-t",
    "--judge-temperature",
    default=0.0,
    show_default=True,
    type=float,
    help="Temperature for judge",
)
@click.option("-v", "--verbose", is_flag=True, help="Per-question progress details")
def run(
    dataset_file: str,
    model: str,
    judge_model: str,
    runs: int,
    concurrency: int,
    pass_threshold: int,
    output: str,
    system_prompt: str | None,
    difficulty: Difficulty,
    resume: bool,
    target_temperature: float | None,
    judge_temperature: float,
    verbose: bool,
) -> None:
    """Run a benchmark against a target model."""
    _setup_logging(verbose)

    system_prompt = _load_system_prompt(system_prompt)

    config = BenchmarkConfig(
        dataset_path=Path(dataset_file),
        target_model=model,
        judge_model=judge_model,
        runs_per_question=runs,
        concurrency=concurrency,
        pass_threshold=pass_threshold,
        output_path=Path(output),
        system_prompt=system_prompt,
        min_difficulty=difficulty,
        resume=resume,
        target_temperature=target_temperature,
        judge_temperature=judge_temperature,
        verbose=verbose,
    )

    try:
        from .runner import run_benchmark

        rows = asyncio.run(run_benchmark(config))
    except CheckpointMismatchError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Use --no-resume to start fresh, or delete the checkpoint file.", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted. Progress saved to checkpoint.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Final performance report
    if rows:
        click.echo("")
        click.echo(format_summary_report(rows, pass_threshold))
        click.echo(f"Results: {output}")
    else:
        click.echo("No results to report.")


@cli.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option(
    "-t",
    "--pass-threshold",
    default=3,
    show_default=True,
    help="Pass threshold",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output CSV path (default: derived from checkpoint filename)",
)
def stats(checkpoint_file: str, pass_threshold: int, output: str | None) -> None:
    """Recompute CSV from checkpoint data."""
    _setup_logging(verbose=False)

    checkpoint_path = Path(checkpoint_file)
    data = load_checkpoint_file(checkpoint_path)

    # Derive output path
    if output is None:
        # checkpoint is foo.checkpoint.json -> foo.csv
        stem = checkpoint_path.stem  # foo.checkpoint
        if stem.endswith(".checkpoint"):
            stem = stem[: -len(".checkpoint")]
        output_path = checkpoint_path.parent / f"{stem}_t{pass_threshold}.csv"
    else:
        output_path = Path(output)

    # We need dataset items for metadata -- checkpoint doesn't store them.
    # Load results and write CSV with available data.
    results: dict[str, list[JudgeResult]] = {}
    for qid, result_dicts in data.get("results", {}).items():
        results[qid] = [JudgeResult.model_validate(r) for r in result_dicts]

    if not results:
        click.echo("No results found in checkpoint.", err=True)
        sys.exit(1)

    # Build minimal DatasetItem-like dicts for stats computation.
    # We only need id, domain, subdomain, difficulty, question_type for the CSV metadata.
    # Since checkpoint doesn't store dataset metadata, we output what we have.
    from .models import DatasetItem, Difficulty, QuestionType

    # Create stub items with the question IDs from checkpoint
    stub_items = []
    for qid in results:
        stub_items.append(
            DatasetItem(
                id=qid,
                domain="",
                subdomain="",
                difficulty=Difficulty.MEDIUM,
                question_type=QuestionType.FACTUAL,
                question="",
                canonical_answer="",
                expanded_answer="",
                required_facts=[],
                negative_responses=[],
                reasoning_path={"evidence_summary": "", "logical_steps": []},
                source_references=[],
            )
        )

    rows = compute_and_write(stub_items, results, pass_threshold, output_path)

    click.echo("")
    click.echo(format_summary_report(rows, pass_threshold))
    click.echo(f"Output: {output_path}")


if __name__ == "__main__":
    cli()
