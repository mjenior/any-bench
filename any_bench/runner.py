"""Async benchmark orchestration pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import random

import litellm
from pydantic import ValidationError

from .judge import JudgeParseError, build_judge_messages, load_judge_prompt, parse_judge_response
from .models import BenchmarkConfig, DatasetItem, Difficulty, JudgeResult
from .resume import load_checkpoint
from .stats import compute_and_write

logger = logging.getLogger("any_bench")

MAX_API_RETRIES = 3
BASE_RETRY_DELAY = 2.0
MAX_JUDGE_PARSE_RETRIES = 3


async def _call_llm(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    retries: int = MAX_API_RETRIES,
) -> str:
    """Call an LLM via litellm with exponential backoff retry."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict = {"model": model, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                delay = BASE_RETRY_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


async def _run_single_evaluation(
    item: DatasetItem,
    config: BenchmarkConfig,
    judge_prompt: str,
    semaphore: asyncio.Semaphore,
) -> JudgeResult | None:
    """Run one target call + judge evaluation for a single question.

    Returns JudgeResult on success, None if all retries exhausted.
    """
    async with semaphore:
        # Call target model
        target_messages = [{"role": "user", "content": item.question}]
        if config.system_prompt:
            target_messages.insert(0, {"role": "system", "content": config.system_prompt})

        try:
            target_response = await _call_llm(
                config.target_model,
                target_messages,
                temperature=config.target_temperature,
            )
        except Exception as e:
            logger.error("Target model call failed for %s after retries: %s", item.id, e)
            return None

        # Call judge model (with parse retries reusing the same target response)
        judge_messages = build_judge_messages(item, target_response, judge_prompt)

        for parse_attempt in range(MAX_JUDGE_PARSE_RETRIES):
            try:
                raw_judge = await _call_llm(
                    config.judge_model,
                    judge_messages,
                    temperature=config.judge_temperature,
                )
                return parse_judge_response(raw_judge)
            except JudgeParseError as e:
                if parse_attempt < MAX_JUDGE_PARSE_RETRIES - 1:
                    logger.warning(
                        "Judge parse error for %s (attempt %d/%d): %s",
                        item.id,
                        parse_attempt + 1,
                        MAX_JUDGE_PARSE_RETRIES,
                        e,
                    )
                else:
                    logger.error(
                        "Judge parse failed for %s after %d attempts: %s",
                        item.id,
                        MAX_JUDGE_PARSE_RETRIES,
                        e,
                    )
            except Exception as e:
                logger.error("Judge model call failed for %s after retries: %s", item.id, e)
                return None

    return None


def load_dataset(path: str | object) -> list[DatasetItem]:
    """Load and validate dataset items from a JSON file.

    Items that fail validation are logged and skipped.
    """
    from pathlib import Path as _Path

    with open(_Path(path)) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "items" in raw:
        raw_items = raw["items"]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError(f"Dataset must be a JSON array or object with 'items' key, got {type(raw)}")

    items = []
    for i, entry in enumerate(raw_items):
        try:
            items.append(DatasetItem.model_validate(entry))
        except ValidationError as e:
            logger.warning("Skipping dataset item %d: %s", i, e)

    if not items:
        raise ValueError("No valid items found in dataset")

    return items


def filter_by_difficulty(
    items: list[DatasetItem], min_difficulty: Difficulty
) -> list[DatasetItem]:
    """Return items at or above ``min_difficulty``.

    ``Difficulty`` is declared in ascending order (Easy -> Expert), so the
    lowest level is a no-op that keeps the full set.
    """
    difficulty_rank = {d: i for i, d in enumerate(Difficulty)}
    min_rank = difficulty_rank[min_difficulty]
    if min_rank == 0:
        return list(items)
    return [item for item in items if difficulty_rank[item.difficulty] >= min_rank]


async def run_benchmark(config: BenchmarkConfig) -> list[dict]:
    """Run the full benchmark pipeline.

    Returns the computed statistics rows.
    """
    # Load dataset
    items = load_dataset(config.dataset_path)
    logger.info("Loaded %d dataset items", len(items))

    # Filter by minimum difficulty.
    before = len(items)
    items = filter_by_difficulty(items, config.min_difficulty)
    if len(items) < before:
        logger.info(
            "Filtered to %d items at difficulty >= %s (dropped %d)",
            len(items),
            config.min_difficulty.value,
            before - len(items),
        )

    # Load judge prompt
    judge_prompt = load_judge_prompt()

    # Load/create checkpoint
    checkpoint = load_checkpoint(config)
    logger.info(
        "Checkpoint: %d questions with results",
        len(checkpoint.results),
    )

    # Set up concurrency control
    semaphore = asyncio.Semaphore(config.concurrency)

    # Process each question
    pending_items = [
        item for item in items if not checkpoint.is_question_complete(item.id)
    ]
    logger.info(
        "%d questions pending (%d already complete)",
        len(pending_items),
        len(items) - len(pending_items),
    )

    async def process_question(item: DatasetItem) -> None:
        """Process all remaining runs for a single question."""
        completed = checkpoint.completed_runs(item.id)
        remaining = config.runs_per_question - completed

        for run_idx in range(remaining):
            run_num = completed + run_idx + 1
            if config.verbose:
                logger.info(
                    "  %s: run %d/%d",
                    item.id,
                    run_num,
                    config.runs_per_question,
                )

            result = await _run_single_evaluation(item, config, judge_prompt, semaphore)
            if result is not None:
                checkpoint.add_result(item.id, result)
            else:
                logger.warning(
                    "  %s: run %d skipped (all retries exhausted)",
                    item.id,
                    run_num,
                )

    # Run questions in parallel (runs within each question are sequential)
    tasks = [asyncio.create_task(process_question(item)) for item in pending_items]

    if tasks:
        await asyncio.gather(*tasks)

    logger.info("Benchmark complete. Computing statistics.")

    # Compute stats and write CSV
    rows = compute_and_write(
        items,
        checkpoint.results,
        config.pass_threshold,
        config.output_path,
    )

    logger.info("Results written to %s", config.output_path)
    return rows
