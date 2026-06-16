"""Stage 2 — MIPROv2-style instruction proposal, run on the target model itself.

MIPROv2 diversifies its instruction proposals by varying a meta-level "tip"
(creative / concise / precise / ...) while grounding every proposal in a shared
context (task summary + dataset/program traits). We mirror that: each candidate
gets the same grounded context (the analysis from stage 1 plus the source and
target trait profiles) and a different style tip. Stage 3 then plays the role of
MIPROv2's evaluator by scoring candidates, and `refine` performs one
feedback-driven improvement step on the winner.
"""

from ..knowledge_base import ModelProfile
from ..models import PromptAnalysis
from ..providers.base import LLMProvider

PROPOSAL_TIPS = [
    "Be precise and conservative: preserve the source prompt's structure and wording "
    "wherever it already works; change only what the target model's conventions demand.",
    "Be creative: feel free to restructure the prompt from scratch into the most "
    "idiomatic shape for the target model, as long as intent and constraints survive.",
    "Be concise: compress aggressively. Remove redundancy, boilerplate, and any "
    "instruction the target model follows by default.",
    "Be defensive: make every implicit assumption explicit, spell out edge cases, and "
    "make the output contract impossible to misread.",
    "Be example-driven: where the format or behavior contract matters, demonstrate it "
    "with compact worked examples instead of abstract description.",
]

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "migrated_prompt": {
            "type": "string",
            "description": "The complete rewritten prompt, ready to run on the target model.",
        },
        "change_log": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each meaningful change and why it was made, including any recommended "
                "API-level configuration (structured outputs, thinking/effort settings) "
                "that replaces in-prompt enforcement."
            ),
        },
    },
    "required": ["migrated_prompt", "change_log"],
    "additionalProperties": False,
}

_SYSTEM = """\
You are {target_model}, rewriting a prompt so that it performs optimally when run on
you. The prompt was originally written and tuned for {source_model}.

Follow the cross-model calibration discipline of prompt-optimization research
(MIPROv2's grounded instruction proposal; PromptBridge's cross-model mapping):
ground every decision in (1) the structured analysis of the source prompt,
(2) the source model's conventions, and (3) your own conventions, all provided below.

Requirements:
- Preserve the task intent and EVERY hard constraint from the analysis.
- Re-map output-format enforcement to the target's native mechanism. If the best
  mechanism is an API parameter rather than prompt text (e.g. structured outputs /
  JSON schema mode), keep the prompt focused on field semantics and record the
  recommended API configuration in the change_log.
- Re-map chain-of-thought style: strip or add reasoning elicitation according to the
  target profile (e.g. remove 'think step by step' boilerplate for models that reason
  internally).
- Re-map system-instruction placement, verbosity calibration, and formatting idioms
  (XML tags vs markdown, where examples live) per the target profile.
- Remove source-model idioms that do not transfer; replace them with the target
  equivalent or drop them with a change_log entry.

Style directive for this candidate: {tip}"""

_USER = """\
{source_profile_block}

{target_profile_block}

<prompt_analysis>
{analysis_json}
</prompt_analysis>

<source_prompt model="{source_model}">
{prompt}
</source_prompt>
{notes_block}"""


def _user_content(
    *,
    source_model: str,
    prompt: str,
    analysis: PromptAnalysis,
    source_profile: ModelProfile,
    target_profile: ModelProfile,
    notes: str | None,
) -> str:
    notes_block = f"\n<operator_notes>\n{notes}\n</operator_notes>" if notes else ""
    return _USER.format(
        source_profile_block=source_profile.as_prompt_block("source"),
        target_profile_block=target_profile.as_prompt_block("target"),
        analysis_json=analysis.model_dump_json(indent=2),
        source_model=source_model,
        prompt=prompt,
        notes_block=notes_block,
    )


async def propose(
    provider: LLMProvider,
    *,
    target_model: str,
    source_model: str,
    prompt: str,
    analysis: PromptAnalysis,
    source_profile: ModelProfile,
    target_profile: ModelProfile,
    tip: str,
    notes: str | None,
    max_tokens: int,
) -> str:
    """Returns the raw JSON string matching PROPOSAL_SCHEMA."""
    return await provider.complete(
        model=target_model,
        system=_SYSTEM.format(
            target_model=target_model, source_model=source_model, tip=tip
        ),
        user=_user_content(
            source_model=source_model,
            prompt=prompt,
            analysis=analysis,
            source_profile=source_profile,
            target_profile=target_profile,
            notes=notes,
        ),
        json_schema=PROPOSAL_SCHEMA,
        schema_name="prompt_proposal",
        max_tokens=max_tokens,
    )


_REFINE_SYSTEM = """\
You are {target_model}. Below is the best candidate so far from a prompt-migration
pipeline (original prompt was tuned for {source_model}), together with an evaluator's
feedback. Produce an improved final version that addresses the feedback while keeping
everything the evaluator scored well. Do not regress on intent or hard constraints."""

_REFINE_USER = """\
{target_profile_block}

<prompt_analysis>
{analysis_json}
</prompt_analysis>

<best_candidate>
{candidate}
</best_candidate>

<evaluator_feedback>
{feedback}
</evaluator_feedback>"""


async def refine(
    provider: LLMProvider,
    *,
    target_model: str,
    source_model: str,
    candidate_prompt: str,
    feedback: str,
    analysis: PromptAnalysis,
    target_profile: ModelProfile,
    max_tokens: int,
) -> str:
    """Returns the raw JSON string matching PROPOSAL_SCHEMA."""
    return await provider.complete(
        model=target_model,
        system=_REFINE_SYSTEM.format(target_model=target_model, source_model=source_model),
        user=_REFINE_USER.format(
            target_profile_block=target_profile.as_prompt_block("target"),
            analysis_json=analysis.model_dump_json(indent=2),
            candidate=candidate_prompt,
            feedback=feedback,
        ),
        json_schema=PROPOSAL_SCHEMA,
        schema_name="prompt_proposal",
        max_tokens=max_tokens,
    )
