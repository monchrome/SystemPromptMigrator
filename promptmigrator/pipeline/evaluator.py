"""Stage 3 — LLM-as-judge scoring of candidate rewrites against a fixed rubric."""

import json

from ..knowledge_base import ModelProfile
from ..models import CandidateScores, PromptAnalysis
from ..providers.base import LLMProvider

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_fidelity": {
            "type": "integer",
            "description": "0-10: does the candidate ask for exactly the same task?",
        },
        "constraint_coverage": {
            "type": "integer",
            "description": "0-10: are ALL hard constraints from the analysis preserved?",
        },
        "target_idiom_fit": {
            "type": "integer",
            "description": "0-10: does it follow the target model's conventions?",
        },
        "format_enforcement": {
            "type": "integer",
            "description": (
                "0-10: is the output-format contract correctly re-mapped to the target's "
                "native mechanism (no leftover prefill tricks, correct API recommendation)?"
            ),
        },
        "clarity": {
            "type": "integer",
            "description": "0-10: unambiguous, well-structured, no redundancy.",
        },
        "feedback": {
            "type": "string",
            "description": "Concrete, actionable feedback for a refinement pass.",
        },
    },
    "required": [
        "intent_fidelity",
        "constraint_coverage",
        "target_idiom_fit",
        "format_enforcement",
        "clarity",
        "feedback",
    ],
    "additionalProperties": False,
}

_SYSTEM = """\
You are a strict prompt-migration evaluator. Score a candidate prompt that was
rewritten from {source_model} to run on {target_model}. Judge it ONLY against the
provided analysis of the original prompt and the target model's profile. Score each
rubric dimension as an integer from 0 (fails completely) to 10 (flawless). Be harsh
on any lost constraint or any leftover source-model idiom."""

_USER = """\
{target_profile_block}

<prompt_analysis>
{analysis_json}
</prompt_analysis>

<original_prompt model="{source_model}">
{original_prompt}
</original_prompt>

<candidate_prompt model="{target_model}">
{candidate_prompt}
</candidate_prompt>"""


async def evaluate(
    provider: LLMProvider,
    *,
    target_model: str,
    source_model: str,
    original_prompt: str,
    candidate_prompt: str,
    analysis: PromptAnalysis,
    target_profile: ModelProfile,
    max_tokens: int,
) -> tuple[CandidateScores, str]:
    raw = await provider.complete(
        model=target_model,
        system=_SYSTEM.format(source_model=source_model, target_model=target_model),
        user=_USER.format(
            target_profile_block=target_profile.as_prompt_block("target"),
            analysis_json=analysis.model_dump_json(indent=2),
            source_model=source_model,
            original_prompt=original_prompt,
            target_model=target_model,
            candidate_prompt=candidate_prompt,
        ),
        json_schema=EVALUATION_SCHEMA,
        schema_name="candidate_evaluation",
        max_tokens=max_tokens,
    )
    data = json.loads(raw)
    feedback = data.pop("feedback")
    return CandidateScores.model_validate(data), feedback
