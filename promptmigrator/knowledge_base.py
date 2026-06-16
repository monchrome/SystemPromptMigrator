"""Model-trait knowledge base.

Each profile describes how a model family expects prompts to be written:
system-instruction placement, JSON/format enforcement, chain-of-thought style,
verbosity calibration, and known migration pitfalls. The pipeline injects the
source and target profiles into the proposer prompt so the rewrite is grounded
in real cross-model differences (the PromptBridge-style calibration table)
rather than the rewriting model's guesses.
"""

import re

from pydantic import BaseModel


class ModelProfile(BaseModel):
    family: str
    vendor: str
    example_models: list[str]
    system_prompt_style: str
    json_enforcement: str
    cot_style: str
    verbosity: str
    formatting_idioms: str
    pitfalls: str

    def as_prompt_block(self, label: str) -> str:
        return (
            f"<{label}_model_profile family=\"{self.family}\" vendor=\"{self.vendor}\">\n"
            f"System instructions: {self.system_prompt_style}\n"
            f"JSON / output-format enforcement: {self.json_enforcement}\n"
            f"Chain-of-thought style: {self.cot_style}\n"
            f"Verbosity calibration: {self.verbosity}\n"
            f"Formatting idioms: {self.formatting_idioms}\n"
            f"Migration pitfalls: {self.pitfalls}\n"
            f"</{label}_model_profile>"
        )


_CLAUDE_MODERN = ModelProfile(
    family="claude-4.6+",
    vendor="anthropic",
    example_models=["claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    system_prompt_style=(
        "Dedicated top-level `system` parameter (not a chat message). Keep it stable; "
        "put per-request context in the user turn. XML-style tags (<context>, <rules>, "
        "<examples>) are the preferred way to delimit sections."
    ),
    json_enforcement=(
        "Prefer API-level structured outputs: `output_config.format` with a JSON schema, "
        "or `strict: true` tool schemas — the API then guarantees valid JSON, so the "
        "prompt should describe field semantics, not beg for valid JSON. Assistant-turn "
        "prefill (e.g. seeding the reply with `{`) returns a 400 error on Claude 4.6+ "
        "and must be removed. If no API schema is used, a short instruction plus an "
        "example object inside <output_format> tags works well."
    ),
    cot_style=(
        "Adaptive thinking handles reasoning internally (`thinking: {type: \"adaptive\"}`). "
        "Do NOT add 'think step by step' / 'show your reasoning' boilerplate — it leaks "
        "reasoning into the visible answer. If deep reasoning matters, say *when* it is "
        "warranted, or raise the API `effort` level instead of prompting for it."
    ),
    verbosity=(
        "Follows instructions literally and calibrates length to task complexity. State "
        "the desired length/structure explicitly; positive examples of the desired "
        "concision beat 'do not' lists."
    ),
    formatting_idioms=(
        "XML tags for structure; markdown inside content is fine. Few-shot examples go "
        "in the user turn (or earlier turns), never as a trailing assistant prefill."
    ),
    pitfalls=(
        "Aggressive emphasis ('CRITICAL: you MUST...', 'If in doubt, use X') over-triggers "
        "on 4.6+ — dial it down to plain imperatives. Sampling knobs (temperature/top_p) "
        "are removed on Opus 4.7+/Fable 5; steer style via the prompt instead."
    ),
)

_CLAUDE_LEGACY = ModelProfile(
    family="claude-legacy (<=4.5)",
    vendor="anthropic",
    example_models=["claude-opus-4-5", "claude-sonnet-4-5", "claude-opus-4-1"],
    system_prompt_style=(
        "Dedicated top-level `system` parameter. XML-style tags to delimit sections."
    ),
    json_enforcement=(
        "Assistant prefill (seeding the reply with `{`) and 'Respond only with JSON' "
        "instructions were common; structured outputs are available on 4.5/4.1. "
        "Prefill-based prompts must be rewritten before moving to Claude 4.6+."
    ),
    cot_style=(
        "Extended thinking with an explicit `budget_tokens`; 'think step by step' "
        "boilerplate was sometimes used when thinking was off."
    ),
    verbosity="Tends verbose by default; needed explicit brevity instructions.",
    formatting_idioms="XML tags for structure; few-shot examples in user turns.",
    pitfalls="Prompts often over-emphasize ('MUST', 'CRITICAL') to overcome reluctance.",
)

_GPT = ModelProfile(
    family="gpt",
    vendor="openai",
    example_models=["gpt-5", "gpt-4.1", "gpt-4o"],
    system_prompt_style=(
        "System (or 'developer') message as the first chat message. Markdown headers "
        "and bullet lists are the conventional way to structure long instructions."
    ),
    json_enforcement=(
        "API-level `response_format: {type: \"json_schema\", strict: true}` guarantees "
        "schema-valid output; the word 'JSON' should still appear in the prompt when "
        "using plain json_object mode. Without API enforcement, an explicit 'Respond "
        "with only a JSON object, no prose, no code fences' instruction plus an example "
        "is the idiom."
    ),
    cot_style=(
        "Non-reasoning GPT models benefit from explicit step-by-step elicitation "
        "('First analyze..., then...') and from asking for reasoning before the answer. "
        "Few-shot demonstrations of worked reasoning help."
    ),
    verbosity=(
        "Generally compliant with length instructions; tends to add preamble ('Sure! "
        "Here is...') unless told to answer directly."
    ),
    formatting_idioms=(
        "Markdown structure; delimiters like triple-backticks or ### sections; few-shot "
        "examples as alternating user/assistant messages or inline blocks."
    ),
    pitfalls=(
        "Instructions buried mid-prompt get dropped on long contexts — repeat hard "
        "constraints near the end. 'JSON' keyword required for json_object mode."
    ),
)

_OPENAI_REASONING = ModelProfile(
    family="openai-reasoning (o-series)",
    vendor="openai",
    example_models=["o3", "o4-mini", "o1"],
    system_prompt_style=(
        "'developer' message instead of system. Keep instructions terse and goal-oriented."
    ),
    json_enforcement=(
        "API-level structured outputs (`response_format` json_schema). Avoid long "
        "format lectures in the prompt; a schema plus one example suffices."
    ),
    cot_style=(
        "Reasoning happens internally — do NOT prompt 'think step by step' or ask the "
        "model to show its work; it degrades performance. Zero-shot, clearly-specified "
        "tasks beat heavy few-shot scaffolding."
    ),
    verbosity="Terse prompts work best; state the goal and constraints, skip role-play fluff.",
    formatting_idioms="Markdown or plain text; minimal decoration.",
    pitfalls=(
        "CoT elicitation and large few-shot blocks actively hurt; temperature is not "
        "supported. Strip persona/CoT boilerplate when migrating onto these models."
    ),
)

_GEMINI = ModelProfile(
    family="gemini",
    vendor="google",
    example_models=["gemini-2.5-pro", "gemini-2.0-flash"],
    system_prompt_style=(
        "`system_instruction` config field. Markdown structure; explicit, enumerated "
        "constraints work better than prose."
    ),
    json_enforcement=(
        "API-level `response_mime_type: application/json` plus `response_schema`. In the "
        "prompt itself, show the exact JSON shape and say 'Return JSON matching this "
        "schema exactly'."
    ),
    cot_style=(
        "Thinking models reason internally; for non-thinking variants explicit "
        "step-by-step elicitation helps. Avoid asking thinking variants to expose "
        "reasoning."
    ),
    verbosity="Tends verbose; cap length explicitly.",
    formatting_idioms="Markdown; numbered constraint lists; examples in fenced blocks.",
    pitfalls=(
        "Loosely-specified output formats drift; repeat the format contract at the end "
        "of the prompt."
    ),
)

_OPEN_WEIGHTS = ModelProfile(
    family="open-weights (llama/mistral/qwen)",
    vendor="open",
    example_models=["llama-4", "mistral-large", "qwen-3"],
    system_prompt_style=(
        "System message via the chat template. Keep it short; long system prompts get "
        "diluted."
    ),
    json_enforcement=(
        "Server-dependent (grammar/JSON mode if available). In-prompt: show the exact "
        "JSON shape, demand 'output only the JSON object', and repeat the format "
        "requirement as the final line — smaller models weight the end of the prompt "
        "heavily."
    ),
    cot_style=(
        "Explicit step-by-step elicitation and few-shot worked examples help "
        "substantially."
    ),
    verbosity="Needs explicit length limits and 'no extra commentary' instructions.",
    formatting_idioms="Few-shot examples are the strongest lever; repeat key constraints.",
    pitfalls=(
        "Instruction-following is weaker: make every implicit assumption explicit and "
        "restate the output contract at the end."
    ),
)

_GENERIC = ModelProfile(
    family="generic",
    vendor="unknown",
    example_models=[],
    system_prompt_style="Assume a standard system + user chat structure.",
    json_enforcement=(
        "Show the exact JSON shape, instruct 'output only the JSON object', and provide "
        "one example."
    ),
    cot_style="State when careful reasoning is warranted; avoid demanding visible reasoning.",
    verbosity="State the desired length and structure explicitly.",
    formatting_idioms="Clear delimiters around sections; examples where format matters.",
    pitfalls="Unknown family — keep the prompt self-contained and explicit.",
)

# Ordered: first regex match wins.
_MATCHERS: list[tuple[re.Pattern[str], ModelProfile]] = [
    (re.compile(r"^claude-(fable|opus-4-[678]|sonnet-4-6|haiku-4-5)"), _CLAUDE_MODERN),
    (re.compile(r"^claude"), _CLAUDE_LEGACY),
    (re.compile(r"^(o[134](-|$)|o4-mini)"), _OPENAI_REASONING),
    (re.compile(r"^(gpt|chatgpt)"), _GPT),
    (re.compile(r"^gemini"), _GEMINI),
    (re.compile(r"^(llama|mistral|mixtral|qwen|deepseek)"), _OPEN_WEIGHTS),
]


def get_profile(model_id: str) -> ModelProfile:
    """Return the trait profile for a model ID, falling back to a generic profile."""
    normalized = model_id.strip().lower()
    for pattern, profile in _MATCHERS:
        if pattern.match(normalized):
            return profile
    return _GENERIC


def all_profiles() -> list[ModelProfile]:
    seen: dict[str, ModelProfile] = {}
    for _, profile in _MATCHERS:
        seen.setdefault(profile.family, profile)
    seen.setdefault(_GENERIC.family, _GENERIC)
    return list(seen.values())
