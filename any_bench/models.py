"""Pydantic data models for dataset items, judge output, and benchmark config."""

from __future__ import annotations

import enum
from pathlib import Path

from pydantic import BaseModel, Field


# --- Enums ---


class Difficulty(str, enum.Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"


class QuestionType(str, enum.Enum):
    FACTUAL = "Factual"
    CONCEPTUAL = "Conceptual"
    MULTI_HOP = "Multi-Hop"
    PROCEDURAL = "Procedural"
    APPLICATION = "Application"
    DECISION_MAKING = "Decision-Making"
    TROUBLESHOOTING = "Troubleshooting"
    EDGE_CASE = "Edge_Case"
    SYNTHESIS = "Synthesis"


# --- Dataset input models ---


class NegativeResponse(BaseModel):
    response: str
    failure_mode: str
    violated_facts: list[str]


class ReasoningPath(BaseModel):
    evidence_summary: str
    logical_steps: list[str]


class SpecializedTerm(BaseModel):
    term: str
    definition: str
    source_reference: str


class NovelClaim(BaseModel):
    claim: str
    evidence: str
    source_reference: str


class DomainContext(BaseModel):
    specialized_terminology: list[SpecializedTerm] = Field(default_factory=list)
    domain_assumptions: list[str] = Field(default_factory=list)
    novel_claims: list[NovelClaim] = Field(default_factory=list)
    evaluation_notes: str | None = None


class DatasetItem(BaseModel):
    id: str
    domain: str
    subdomain: str
    difficulty: Difficulty
    question_type: QuestionType
    question: str
    canonical_answer: str
    expanded_answer: str
    required_facts: list[str]
    negative_responses: list[NegativeResponse]
    reasoning_path: ReasoningPath
    source_references: list[str]
    domain_context: DomainContext | None = None


# --- Judge output models ---


class SectionScores(BaseModel):
    instruction_compliance: int = Field(ge=0, le=5)
    factual_accuracy: int = Field(ge=0, le=5)
    required_fact_coverage: int = Field(ge=0, le=5)
    reasoning_quality: int = Field(ge=0, le=5)
    relevance_focus: int = Field(ge=0, le=5)
    clarity_usability: int = Field(ge=0, le=5)


class MatchedNegativeResponse(BaseModel):
    response: str
    failure_mode: str
    violated_facts: list[str]


class JudgeResult(BaseModel):
    id: str
    composite_score: int = Field(ge=0, le=5)
    section_scores: SectionScores
    matched_negative_responses: list[MatchedNegativeResponse] = Field(default_factory=list)
    review_summary: str


# --- Runtime config ---


class BenchmarkConfig(BaseModel):
    dataset_path: Path
    target_model: str= "anthropic/claude-opus-4.8"
    judge_model: str = "gemini/gemini-3.5-flash"
    runs_per_question: int = 5
    concurrency: int = 10
    pass_threshold: int = 3
    output_path: Path = Path("benchmark_results.csv")
    system_prompt: str | None = None
    min_difficulty: Difficulty = Difficulty.EASY
    resume: bool = True
    target_temperature: float | None = None
    judge_temperature: float = 0.0
    verbose: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.output_path.with_suffix(".checkpoint.json")
