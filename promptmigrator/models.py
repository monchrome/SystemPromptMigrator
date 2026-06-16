"""Pydantic schemas shared by the pipeline and the API layer."""

from pydantic import BaseModel, Field

from .knowledge_base import ModelProfile


class OutputFormatSpec(BaseModel):
    type: str
    spec: str
    enforcement_mechanism: str


class PromptAnalysis(BaseModel):
    intent: str
    role_persona: str
    hard_constraints: list[str]
    output_format: OutputFormatSpec
    reasoning_style: str
    source_model_idioms: list[str]
    migration_risks: list[str]


class CandidateScores(BaseModel):
    intent_fidelity: int = Field(ge=0, le=10)
    constraint_coverage: int = Field(ge=0, le=10)
    target_idiom_fit: int = Field(ge=0, le=10)
    format_enforcement: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)

    @property
    def total(self) -> int:
        return (
            self.intent_fidelity
            + self.constraint_coverage
            + self.target_idiom_fit
            + self.format_enforcement
            + self.clarity
        )


class Candidate(BaseModel):
    prompt: str
    change_log: list[str]
    tip: str
    scores: CandidateScores | None = None
    total_score: int | None = None
    feedback: str | None = None


class MigrationResponse(BaseModel):
    source_model: str
    target_model: str
    migrated_prompt: str
    selected_candidate_index: int
    refined: bool
    analysis: PromptAnalysis
    candidates: list[Candidate]
    source_profile: ModelProfile
    target_profile: ModelProfile
