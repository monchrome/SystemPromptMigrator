"""Stage 1 — extract a structured, model-agnostic spec of the source prompt."""

from ..knowledge_base import ModelProfile
from ..models import PromptAnalysis
from ..providers.base import LLMProvider

# Structured-output schema: every property required, no extras — compatible with
# both Anthropic output_config.format and OpenAI strict json_schema mode.
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "description": "What the prompt is fundamentally asking the model to do.",
        },
        "role_persona": {
            "type": "string",
            "description": "Any role/persona the prompt assigns ('none' if absent).",
        },
        "hard_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every constraint that must survive migration verbatim in meaning.",
        },
        "output_format": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "e.g. json, markdown, plain text, code, table.",
                },
                "spec": {
                    "type": "string",
                    "description": "The exact format contract (schema, fields, layout).",
                },
                "enforcement_mechanism": {
                    "type": "string",
                    "description": (
                        "How the source prompt enforces the format: in-prompt instruction, "
                        "assistant prefill, API structured output, examples, etc."
                    ),
                },
            },
            "required": ["type", "spec", "enforcement_mechanism"],
            "additionalProperties": False,
        },
        "reasoning_style": {
            "type": "string",
            "description": (
                "How the prompt elicits reasoning: explicit chain-of-thought boilerplate, "
                "implicit, scratchpad sections, none."
            ),
        },
        "source_model_idioms": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Source-model-specific tricks that may not transfer: prefill reliance, "
                "magic keywords, special tokens, stop sequences, emphasis hacks."
            ),
        },
        "migration_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things likely to break or behave differently on another model.",
        },
    },
    "required": [
        "intent",
        "role_persona",
        "hard_constraints",
        "output_format",
        "reasoning_style",
        "source_model_idioms",
        "migration_risks",
    ],
    "additionalProperties": False,
}

_SYSTEM = """\
You are a prompt-migration analyst inside an automated prompt-porting pipeline.
You will receive a prompt that was written and tuned for the model "{source_model}".
Produce a faithful, structured specification of what the prompt does so it can be
re-expressed for a different model. Do NOT improve or rewrite the prompt — only
describe it. Be exhaustive about hard constraints and about source-model-specific
idioms (prefill reliance, 'think step by step' boilerplate, magic keywords, special
tokens, emphasis hacks), because those are exactly what breaks during migration."""

_USER = """\
{source_profile_block}

<source_prompt model="{source_model}">
{prompt}
</source_prompt>
{notes_block}"""


async def analyze(
    provider: LLMProvider,
    *,
    executor_model: str,
    source_model: str,
    prompt: str,
    source_profile: ModelProfile,
    notes: str | None,
    max_tokens: int,
) -> PromptAnalysis:
    notes_block = f"\n<operator_notes>\n{notes}\n</operator_notes>" if notes else ""
    raw = await provider.complete(
        model=executor_model,
        system=_SYSTEM.format(source_model=source_model),
        user=_USER.format(
            source_profile_block=source_profile.as_prompt_block("source"),
            source_model=source_model,
            prompt=prompt,
            notes_block=notes_block,
        ),
        json_schema=ANALYSIS_SCHEMA,
        schema_name="prompt_analysis",
        max_tokens=max_tokens,
    )
    return PromptAnalysis.model_validate_json(raw)
