# any-bench

CLI tool for corpus-grounded LLM benchmarking. Sends questions from a benchmark dataset to a target model, judges responses using a separate judge model, and computes statistics.

Designed to be highly flexible to work well for any correctly formatted JSON test dataset like the one shipped in `tests/data/dataset.json`.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

## Usage

### Run a benchmark

```bash
any-bench run dataset.json -m gpt-4o -n 5 -o results.csv
```

### Recompute stats from checkpoint

```bash
any-bench stats results.checkpoint.json -t 4 -o results_strict.csv
```

## Task Runner

This project uses [Task](https://taskfile.dev/) for common operations.

```bash
task test       # run tests
task lint       # run ruff linter
task format     # run ruff formatter
task check      # lint + test
task install    # install dependencies
```

## CLI Reference

### `any-bench run DATASET_FILE [OPTIONS]`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `DATASET_FILE` | positional | required | Path to JSON dataset file |
| `-m, --model` | TEXT | required | `anthropic/claude-opus-4.8` | Target model (litellm format) |
| `-j, --judge-model` | TEXT | `gemini/gemini-3.5-flash` | Judge model identifier |
| `-n, --runs` | INT | `5` | Number of evaluation runs per question |
| `-c, --concurrency` | INT | `10` | Max concurrent API calls |
| `-t, --pass-threshold` | INT | `3` | Composite score >= this counts as pass |
| `-o, --output` | PATH | `benchmark_results.csv` | Output CSV path |
| `-s, --system-prompt` | TEXT | none | System prompt for target model (string or `@file`) |
| `--resume/--no-resume` | flag | `--resume` | Resume from checkpoint if available |
| `--target-temperature` | FLOAT | model default | Temperature for target model |
| `--judge-temperature` | FLOAT | `0.0` | Temperature for judge |
| `-v, --verbose` | flag | off | Per-question progress details |

### `any-bench stats CHECKPOINT_FILE [OPTIONS]`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t, --pass-threshold` | INT | `3` | Pass threshold |
| `-o, --output` | PATH | derived | Output CSV path |

## Dataset Format

Expects JSON files produced by the `create-test-dataset` workflow. Each item includes question, canonical answer, required facts, negative responses, reasoning path, and optional domain context.

## Example Benchmark

A complete worked example ships at [`tests/data/benchmark.json`](tests/data/mljenior_benchmark.json), generated from the source corpus — Matthew Jenior's 2017 University of Michigan PhD dissertation, *Nutrient Niche Space of Clostridium difficile Across Susceptible Microbiomes and the Impact of Infection on Metabolism of the Murine Cecal Microbiota* (159 pp.).

**150 items**, every one answerable only from the dissertation (corpus hard-locked), spanning all four chapters:

The dataset matches the framework's target distribution:

| Difficulty | Count | | Question types (9, balanced 16–18 each) |
|------------|-------|---|------------------------------------------|
| Easy       | 30 (20%) | | Factual, Conceptual, Multi-Hop, Procedural, |
| Medium     | 53 (35%) | | Application, Decision-Making, Troubleshooting, |
| Hard       | 45 (30%) | | Edge_Case, Synthesis |
| Expert     | 22 (15%) | | |

Each item carries at least one **negative response** — a plausible wrong answer a model might give (e.g. swapping a toxin mechanism, inverting a metabolite-score direction) annotated with its `failure_mode` and the `violated_facts` it contradicts, so a judge can recognize known failure modes rather than just matching the gold answer. Items involving specialized terminology or counterintuitive corpus claims also include a `domain_context` block tracing each term/claim back to a section in the source, preventing false-positive hallucination penalties on correct but corpus-specific answers.

Run it like any other dataset:

```bash
any-bench run tests/data/mljenior_benchmark.json -m gpt-4o -n 5 -o mljenior_results.csv
```

## Final Report

After writing the CSV, both `run` and `stats` print a **final performance report** to stdout summarizing the whole test set:

- **Composite (all questions)** — mean score, overall pass rate, median, score range, and total negative-response hits across every question and run.
- **Section scores** — the six scoring categories (instruction compliance, factual accuracy, required-fact coverage, reasoning quality, relevance/focus, clarity/usability), each averaged across questions.
- **Breakdowns** — count, mean composite, and pass rate grouped by difficulty, question type, and domain.
- **Best / worst performing questions** — questions ranked by mean composite, with ties broken toward the lowest standard deviation so *consistently* strong and weak questions surface ahead of high-variance ones. Each line shows the question `id`, mean composite, standard deviation, and pass rate.

```
================================================================
FINAL PERFORMANCE REPORT
================================================================
Questions evaluated : 150
Total runs          : 750
Pass threshold      : composite >= 3

Composite (all questions)
  mean score    : 3.40 / 5
  pass rate     : 62.0%
  ...

Section scores (mean of per-question medians, 0-5)
  Instruction Compliance     : 3.50
  ...

By difficulty
  Easy                         n=  30   composite  4.20   pass  80.0%
  ...

Best performing questions (top 5, consistent)
  q006                     composite  4.47 (sd 0.05)   pass  92.0%
  ...
================================================================
```

> Note: the `stats` command rebuilds from a checkpoint, which does not store dataset metadata, so its difficulty / question-type / domain breakdowns collapse to a single bucket. The composite, section-score, and best/worst sections are fully accurate in both modes.

## How It Works

1. Loads dataset and checkpoint (if resuming)
2. Sends each question to the target model
3. Judges each response against the dataset entry using a separate judge model
4. Saves results to checkpoint after each judgment
5. Computes per-question statistics, writes CSV, and prints the final report
