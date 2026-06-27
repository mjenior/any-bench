"""Tests for judge module."""

import json

import pytest

from any_bench.judge import (
    JudgeParseError,
    build_judge_messages,
    load_judge_prompt,
    parse_judge_response,
)
from any_bench.models import DatasetItem


SAMPLE_ITEM = DatasetItem.model_validate(
    {
        "id": "BENCH-TEST-001",
        "domain": "Biology",
        "subdomain": "Ecology",
        "difficulty": "Medium",
        "question_type": "Factual",
        "question": "What is photosynthesis?",
        "canonical_answer": "The process by which plants convert light energy.",
        "expanded_answer": "Photosynthesis is...",
        "required_facts": ["Uses light energy"],
        "negative_responses": [
            {
                "response": "It's respiration",
                "failure_mode": "Confusion",
                "violated_facts": ["Uses light energy"],
            }
        ],
        "reasoning_path": {
            "evidence_summary": "Textbook",
            "logical_steps": ["Step 1"],
        },
        "source_references": ["Ch 3"],
    }
)

SAMPLE_ITEM_WITH_CONTEXT = DatasetItem.model_validate(
    {
        "id": "BENCH-TEST-002",
        "domain": "Biology",
        "subdomain": "Ecology",
        "difficulty": "Hard",
        "question_type": "Multi-Hop",
        "question": "How do chloroplasts work?",
        "canonical_answer": "Chloroplasts contain thylakoids...",
        "expanded_answer": "Detailed explanation...",
        "required_facts": ["Contains thylakoids"],
        "negative_responses": [],
        "reasoning_path": {
            "evidence_summary": "Textbook",
            "logical_steps": ["Step 1"],
        },
        "source_references": ["Ch 4"],
        "domain_context": {
            "specialized_terminology": [
                {
                    "term": "thylakoid",
                    "definition": "membrane-bound compartment",
                    "source_reference": "Section 4.2",
                }
            ],
            "domain_assumptions": [],
            "novel_claims": [],
        },
    }
)

VALID_JUDGE_JSON = json.dumps(
    {
        "id": "BENCH-TEST-001",
        "composite_score": 4,
        "section_scores": {
            "instruction_compliance": 5,
            "factual_accuracy": 4,
            "required_fact_coverage": 4,
            "reasoning_quality": 3,
            "relevance_focus": 5,
            "clarity_usability": 4,
        },
        "matched_negative_responses": [],
        "review_summary": "Good response. Accurate. Covers key facts. Minor gaps.",
    }
)


class TestLoadJudgePrompt:
    def test_loads_successfully(self):
        prompt = load_judge_prompt()
        assert "Judge Agent System Instructions" in prompt
        assert "composite_score" in prompt

    def test_contains_scoring_criteria(self):
        prompt = load_judge_prompt()
        assert "Instruction Compliance" in prompt
        assert "Factual Accuracy" in prompt
        assert "Required Fact Coverage" in prompt


class TestBuildJudgeMessages:
    def test_basic_structure(self):
        msgs = build_judge_messages(SAMPLE_ITEM, "Test response", "System prompt")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System prompt"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_target_response(self):
        msgs = build_judge_messages(SAMPLE_ITEM, "My answer here", "System prompt")
        user_content = json.loads(msgs[1]["content"])
        assert user_content["target_response"] == "My answer here"

    def test_user_message_contains_dataset_fields(self):
        msgs = build_judge_messages(SAMPLE_ITEM, "Answer", "System prompt")
        user_content = json.loads(msgs[1]["content"])
        assert user_content["id"] == "BENCH-TEST-001"
        assert user_content["question"] == "What is photosynthesis?"
        assert "domain_context" not in user_content

    def test_domain_context_included_when_present(self):
        msgs = build_judge_messages(SAMPLE_ITEM_WITH_CONTEXT, "Answer", "System prompt")
        user_content = json.loads(msgs[1]["content"])
        assert "domain_context" in user_content
        assert len(user_content["domain_context"]["specialized_terminology"]) == 1

    def test_domain_context_excluded_when_none(self):
        msgs = build_judge_messages(SAMPLE_ITEM, "Answer", "System prompt")
        user_content = json.loads(msgs[1]["content"])
        assert "domain_context" not in user_content


class TestParseJudgeResponse:
    def test_parse_valid_json(self):
        result = parse_judge_response(VALID_JUDGE_JSON)
        assert result.composite_score == 4
        assert result.section_scores.instruction_compliance == 5

    def test_parse_with_code_fences(self):
        wrapped = f"```json\n{VALID_JUDGE_JSON}\n```"
        result = parse_judge_response(wrapped)
        assert result.composite_score == 4

    def test_parse_with_bare_fences(self):
        wrapped = f"```\n{VALID_JUDGE_JSON}\n```"
        result = parse_judge_response(wrapped)
        assert result.composite_score == 4

    def test_parse_with_whitespace(self):
        padded = f"\n\n  {VALID_JUDGE_JSON}  \n\n"
        result = parse_judge_response(padded)
        assert result.composite_score == 4

    def test_invalid_json_raises(self):
        with pytest.raises(JudgeParseError, match="Invalid JSON"):
            parse_judge_response("not json at all")

    def test_missing_fields_raises(self):
        with pytest.raises(JudgeParseError, match="failed validation"):
            parse_judge_response('{"id": "test"}')

    def test_score_out_of_range_raises(self):
        bad = json.loads(VALID_JUDGE_JSON)
        bad["composite_score"] = 10
        with pytest.raises(JudgeParseError, match="failed validation"):
            parse_judge_response(json.dumps(bad))
