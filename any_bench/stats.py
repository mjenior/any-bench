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


def _fmt_line(label: str, n: int, composite: float, pass_rate: float) -> str:
    return f"  {label:<28} n={n:>4}   composite {composite:>5.2f}   pass {pass_rate:>6.1%}"


def _aggregate(rows: list[dict]) -> tuple[int, float, float]:
    """Return (num_questions, mean composite, mean pass rate) for a set of rows."""
    n = len(rows)
    composite = statistics.mean(r["composite_mean"] for r in rows)
    pass_rate = statistics.mean(r["pass_rate"] for r in rows)
    return n, composite, pass_rate


def format_summary_report(rows: list[dict], pass_threshold: int) -> str:
    """Build a final performance report with per-category and composite stats.

    Summarizes the full test set: an overall composite line, per-section score
    averages, and breakdowns by difficulty and question type.
    """
    if not rows:
        return "No results to report."

    total_runs = sum(r["num_runs"] for r in rows)
    n, composite, pass_rate = _aggregate(rows)
    negatives = sum(r["negative_match_count"] for r in rows)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("FINAL PERFORMANCE REPORT")
    lines.append("=" * 64)
    lines.append(f"Questions evaluated : {n}")
    lines.append(f"Total runs          : {total_runs}")
    lines.append(f"Pass threshold      : composite >= {pass_threshold}")
    lines.append("")
    lines.append("Composite (all questions)")
    lines.append(f"  mean score    : {composite:.2f} / 5")
    lines.append(f"  pass rate     : {pass_rate:.1%}")
    lines.append(f"  median (med.) : {statistics.median(r['composite_median'] for r in rows):.2f}")
    lines.append(f"  range         : {min(r['composite_min'] for r in rows)} - {max(r['composite_max'] for r in rows)}")
    lines.append(f"  negative hits : {negatives}")

    # Per-section (scoring category) averages across questions.
    lines.append("")
    lines.append("Section scores (mean of per-question medians, 0-5)")
    for section in SECTION_NAMES:
        avg = statistics.mean(r[f"{section}_median"] for r in rows)
        label = section.replace("_", " ").title()
        lines.append(f"  {label:<26} : {avg:.2f}")

    # Breakdown helper for a metadata field.
    def _breakdown(title: str, key: str) -> None:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        if len(groups) <= 1 and "" in groups:
            return  # no meaningful metadata (e.g. stats-from-checkpoint stubs)
        lines.append("")
        lines.append(title)
        for label in sorted(groups):
            gn, gc, gp = _aggregate(groups[label])
            lines.append(_fmt_line(label or "(unknown)", gn, gc, gp))

    _breakdown("By difficulty", "difficulty")
    _breakdown("By question type", "question_type")
    _breakdown("By domain", "domain")

    # Best/worst questions: rank by mean composite, breaking ties toward the
    # most consistent runs (lowest stdev) so "consistently" performing
    # questions surface ahead of high-variance ones.
    top_k = min(5, len(rows))
    best = sorted(rows, key=lambda r: (-r["composite_mean"], r["composite_stdev"]))[:top_k]
    worst = sorted(rows, key=lambda r: (r["composite_mean"], r["composite_stdev"]))[:top_k]

    def _ranked(title: str, ranked: list[dict]) -> None:
        lines.append("")
        lines.append(title)
        for r in ranked:
            lines.append(
                f"  {r['id']:<24} composite {r['composite_mean']:>5.2f}"
                f" (sd {r['composite_stdev']:>4.2f})   pass {r['pass_rate']:>6.1%}"
            )

    _ranked(f"Best performing questions (top {top_k}, consistent)", best)
    _ranked(f"Worst performing questions (bottom {top_k}, consistent)", worst)

    lines.append("=" * 64)
    return "\n".join(lines)
