"""Judge prompt loading, message construction, and response parsing."""

from __future__ import annotations

import importlib.resources
import json
import re

from pydantic import ValidationError

from .models import DatasetItem, JudgeResult


class JudgeParseError(Exception):
    """Raised when the judge response cannot be parsed into a JudgeResult."""


def load_judge_prompt() -> str:
    """Load the bundled judge system prompt."""
    return importlib.resources.files("any_bench").joinpath("prompt.md").read_text()


def build_judge_messages(
    dataset_item: DatasetItem,
    target_response: str,
    judge_prompt: str,
) -> list[dict]:
    """Build the messages list for the judge model call.

    The user message contains all dataset fields plus the target_response as JSON.
    domain_context is included only when present.
    """
    entry = dataset_item.model_dump(mode="json")
    # Remove domain_context if None so judge doesn't see a null field
    if entry.get("domain_context") is None:
        del entry["domain_context"]
    entry["target_response"] = target_response

    return [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": json.dumps(entry, indent=2)},
    ]


def parse_judge_response(raw: str) -> JudgeResult:
    """Parse raw judge model output into a JudgeResult.

    Strips markdown code fences if present, then validates JSON.
    Raises JudgeParseError on failure.
    """
    text = raw.strip()

    # Strip markdown code fences
    fence_pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
    match = fence_pattern.match(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"Invalid JSON from judge: {e}") from e

    try:
        return JudgeResult.model_validate(data)
    except ValidationError as e:
        raise JudgeParseError(f"Judge response failed validation: {e}") from e
