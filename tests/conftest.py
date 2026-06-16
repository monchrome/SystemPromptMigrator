import json
from typing import Any

import pytest

from promptmigrator.pipeline import PromptMigrator
from promptmigrator.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """Returns schema-shaped canned JSON; gives later evaluations higher scores so
    candidate selection is deterministic and observable."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._eval_count = 0
        self._proposal_count = 0

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        max_tokens: int = 16000,
    ) -> str:
        self.calls.append(
            {"model": model, "system": system, "user": user, "schema_name": schema_name}
        )
        if schema_name == "prompt_analysis":
            return json.dumps(
                {
                    "intent": "Summarize support tickets",
                    "role_persona": "support analyst",
                    "hard_constraints": ["Output must be JSON", "Max 3 sentences"],
                    "output_format": {
                        "type": "json",
                        "spec": '{"summary": str, "sentiment": str}',
                        "enforcement_mechanism": "assistant prefill",
                    },
                    "reasoning_style": "think step by step boilerplate",
                    "source_model_idioms": ["prefill with `{`"],
                    "migration_risks": ["prefill 400s on Claude 4.6+"],
                }
            )
        if schema_name == "prompt_proposal":
            self._proposal_count += 1
            return json.dumps(
                {
                    "migrated_prompt": f"migrated prompt v{self._proposal_count}",
                    "change_log": ["removed prefill", "added structured-output note"],
                }
            )
        if schema_name == "candidate_evaluation":
            self._eval_count += 1
            score = min(self._eval_count + 5, 10)
            return json.dumps(
                {
                    "intent_fidelity": score,
                    "constraint_coverage": score,
                    "target_idiom_fit": score,
                    "format_enforcement": score,
                    "clarity": score,
                    "feedback": "tighten the format section",
                }
            )
        raise AssertionError(f"unexpected schema_name {schema_name}")


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def migrator(fake_provider: FakeProvider) -> PromptMigrator:
    return PromptMigrator(provider_for=lambda model: fake_provider)
