"""Tests for data models."""

import pytest
from pydantic import ValidationError

from any_bench.models import (
    BenchmarkConfig,
    DatasetItem,
    Difficulty,
    JudgeResult,
    QuestionType,
)


SAMPLE_ITEM = {
    "id": "BENCH-TEST-001",
    "domain": "Biology",
    "subdomain": "Ecology",
    "difficulty": "Medium",
    "question_type": "Factual",
    "question": "What is photosynthesis?",
    "canonical_answer": "The process by which plants convert light energy to chemical energy.",
    "expanded_answer": "Photosynthesis is the process used by plants and other organisms...",
    "required_facts": [
        "Uses light energy",
        "Produces glucose",
        "Requires CO2 and water",
    ],
    "negative_responses": [
        {
            "response": "Photosynthesis is cellular respiration.",
            "failure_mode": "Confuses photosynthesis with respiration",
            "violated_facts": ["Uses light energy"],
        }
    ],
    "reasoning_path": {
        "evidence_summary": "Standard biology textbook definition",
        "logical_steps": ["Step 1: Identify the process", "Step 2: State inputs and outputs"],
    },
    "source_references": ["Chapter 3, Section 3.1"],
}


SAMPLE_ITEM_WITH_CONTEXT = {
    **SAMPLE_ITEM,
    "id": "BENCH-TEST-002",
    "domain_context": {
        "specialized_terminology": [
            {
                "term": "chloroplast",
                "definition": "Organelle where photosynthesis occurs",
                "source_reference": "Section 3.2",
            }
        ],
        "domain_assumptions": ["Plants have access to sunlight"],
        "novel_claims": [
            {
                "claim": "Some bacteria perform photosynthesis",
                "evidence": "Cyanobacteria are photosynthetic prokaryotes",
                "source_reference": "Section 4.1",
            }
        ],
        "evaluation_notes": "Accept broad definitions",
    },
}


SAMPLE_JUDGE_RESULT = {
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
    "review_summary": "The response accurately describes photosynthesis. It covers the main inputs and outputs. Minor gaps in detail about the light reactions. Overall a strong answer.",
}


class TestDatasetItem:
    def test_parse_valid_item(self):
        item = DatasetItem.model_validate(SAMPLE_ITEM)
        assert item.id == "BENCH-TEST-001"
        assert item.difficulty == Difficulty.MEDIUM
        assert item.question_type == QuestionType.FACTUAL
        assert len(item.required_facts) == 3
        assert len(item.negative_responses) == 1
        assert item.domain_context is None

    def test_parse_item_with_domain_context(self):
        item = DatasetItem.model_validate(SAMPLE_ITEM_WITH_CONTEXT)
        assert item.domain_context is not None
        assert len(item.domain_context.specialized_terminology) == 1
        assert item.domain_context.specialized_terminology[0].term == "chloroplast"
        assert len(item.domain_context.novel_claims) == 1
        assert item.domain_context.evaluation_notes == "Accept broad definitions"

    def test_all_difficulty_levels(self):
        for level in ["Easy", "Medium", "Hard", "Expert"]:
            item_data = {**SAMPLE_ITEM, "difficulty": level}
            item = DatasetItem.model_validate(item_data)
            assert item.difficulty.value == level

    def test_all_question_types(self):
        types = [
            "Factual",
            "Conceptual",
            "Multi-Hop",
            "Procedural",
            "Application",
            "Decision-Making",
            "Troubleshooting",
            "Edge_Case",
            "Synthesis",
        ]
        for qt in types:
            item_data = {**SAMPLE_ITEM, "question_type": qt}
            item = DatasetItem.model_validate(item_data)
            assert item.question_type.value == qt

    def test_missing_required_field_raises(self):
        bad = {k: v for k, v in SAMPLE_ITEM.items() if k != "question"}
        with pytest.raises(ValidationError):
            DatasetItem.model_validate(bad)

    def test_invalid_difficulty_raises(self):
        bad = {**SAMPLE_ITEM, "difficulty": "Impossible"}
        with pytest.raises(ValidationError):
            DatasetItem.model_validate(bad)


class TestJudgeResult:
    def test_parse_valid_result(self):
        result = JudgeResult.model_validate(SAMPLE_JUDGE_RESULT)
        assert result.composite_score == 4
        assert result.section_scores.instruction_compliance == 5
        assert result.matched_negative_responses == []

    def test_score_out_of_range_raises(self):
        bad = {
            **SAMPLE_JUDGE_RESULT,
            "section_scores": {
                **SAMPLE_JUDGE_RESULT["section_scores"],
                "factual_accuracy": 6,
            },
        }
        with pytest.raises(ValidationError):
            JudgeResult.model_validate(bad)

    def test_negative_score_raises(self):
        bad = {
            **SAMPLE_JUDGE_RESULT,
            "composite_score": -1,
        }
        with pytest.raises(ValidationError):
            JudgeResult.model_validate(bad)

    def test_with_matched_negatives(self):
        data = {
            **SAMPLE_JUDGE_RESULT,
            "matched_negative_responses": [
                {
                    "response": "Wrong answer",
                    "failure_mode": "Confusion",
                    "violated_facts": ["Fact 1"],
                }
            ],
        }
        result = JudgeResult.model_validate(data)
        assert len(result.matched_negative_responses) == 1
        assert result.matched_negative_responses[0].failure_mode == "Confusion"


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig(
            dataset_path="/tmp/test.json",
            target_model="openai/gpt-5.5",
        )
        assert config.judge_model == "gemini/gemini-3.5-flash"
        assert config.runs_per_question == 5
        assert config.concurrency == 10
        assert config.pass_threshold == 3
        assert config.resume is True
        assert config.judge_temperature == 0.0
        assert config.target_temperature is None

    def test_checkpoint_path(self):
        config = BenchmarkConfig(
            dataset_path="/tmp/test.json",
            target_model="openai/gpt-5.5",
            output_path="/tmp/results.csv",
        )
        assert str(config.checkpoint_path) == "/tmp/results.checkpoint.json"

    def test_roundtrip_serialization(self):
        config = BenchmarkConfig(
            dataset_path="/tmp/test.json",
            target_model="anthropic/claude-opus-4.8",
            system_prompt="Be helpful",
        )
        data = config.model_dump(mode="json")
        restored = BenchmarkConfig.model_validate(data)
        assert restored.target_model == "anthropic/claude-opus-4.8"
        assert restored.system_prompt == "Be helpful"
