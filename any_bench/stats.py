"""Statistics computation and CSV output."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .models import DatasetItem, JudgeResult

SECTION_NAMES = [
    "instruction_compliance",
    "factual_accuracy",
    "required_fact_coverage",
    "reasoning_quality",
    "relevance_focus",
    "clarity_usability",
]

CSV_COLUMNS = [
    # Metadata
    "id",
    "domain",
    "subdomain",
    "difficulty",
    "question_type",
    # Composite stats
    "composite_median",
    "composite_iqr",
    "composite_mean",
    "composite_stdev",
    "composite_min",
    "composite_max",
    # Rates
    "pass_rate",
    "pass_threshold",
    "num_runs",
    "negative_match_count",
    # Per-section stats (12 columns: 6 sections x median + iqr)
    *[f"{s}_median" for s in SECTION_NAMES],
    *[f"{s}_iqr" for s in SECTION_NAMES],
]


def _iqr(values: list[int | float]) -> float:
    """Compute interquartile range (Q3 - Q1) using median-of-halves."""
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid:]
    else:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid + 1 :]
    if not lower or not upper:
        return 0.0
    return statistics.median(upper) - statistics.median(lower)


def compute_question_stats(
    item: DatasetItem,
    results: list[JudgeResult],
    pass_threshold: int,
) -> dict:
    """Compute statistics for a single question's judge results."""
    composites = [r.composite_score for r in results]
    num_runs = len(results)

    # Composite stats
    composite_median = statistics.median(composites)
    composite_mean = statistics.mean(composites)
    composite_stdev = statistics.stdev(composites) if num_runs >= 2 else 0.0
    composite_min = min(composites)
    composite_max = max(composites)
    composite_iqr = _iqr(composites)

    # Pass rate
    passes = sum(1 for c in composites if c >= pass_threshold)
    pass_rate = passes / num_runs if num_runs > 0 else 0.0

    # Negative match count
    negative_match_count = sum(len(r.matched_negative_responses) for r in results)

    row = {
        "id": item.id,
        "domain": item.domain,
        "subdomain": item.subdomain,
        "difficulty": item.difficulty.value,
        "question_type": item.question_type.value,
        "composite_median": round(composite_median, 2),
        "composite_iqr": round(composite_iqr, 2),
        "composite_mean": round(composite_mean, 2),
        "composite_stdev": round(composite_stdev, 2),
        "composite_min": composite_min,
        "composite_max": composite_max,
        "pass_rate": round(pass_rate, 4),
        "pass_threshold": pass_threshold,
        "num_runs": num_runs,
        "negative_match_count": negative_match_count,
    }

    # Per-section stats
    for section in SECTION_NAMES:
        scores = [getattr(r.section_scores, section) for r in results]
        row[f"{section}_median"] = round(statistics.median(scores), 2)
        row[f"{section}_iqr"] = round(_iqr(scores), 2)

    return row


def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Write computed statistics rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compute_and_write(
    items: list[DatasetItem],
    results: dict[str, list[JudgeResult]],
    pass_threshold: int,
    output_path: Path,
) -> list[dict]:
    """Compute stats for all questions and write CSV. Returns the rows."""
    rows = []
    for item in items:
        item_results = results.get(item.id, [])
        if not item_results:
            continue
        row = compute_question_stats(item, item_results, pass_threshold)
        rows.append(row)

    write_csv(rows, output_path)
    return rows
