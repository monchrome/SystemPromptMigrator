"""Anthropic provider built on the official SDK (streaming + structured outputs)."""

import os
from typing import Any

import anthropic

from .base import LLMProvider, ProviderError

# Models that support adaptive thinking. Older models would 400 on the param.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
)


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
                raise ProviderError(
                    "Anthropic credentials missing: set ANTHROPIC_API_KEY (or log in via "
                    "`ant auth login`).",
                    status_code=503,
                )
            self._client = anthropic.AsyncAnthropic()
        return self._client

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
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
            kwargs["thinking"] = {"type": "adaptive"}
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }

        try:
            async with client.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
        except anthropic.RateLimitError as e:
            raise ProviderError(
                f"Anthropic rate limit hit while calling {model}: {e.message}",
                status_code=429,
                retryable=True,
            ) from e
        except anthropic.AuthenticationError as e:
            raise ProviderError(
                f"Anthropic authentication failed: {e.message}", status_code=503
            ) from e
        except anthropic.NotFoundError as e:
            raise ProviderError(
                f"Unknown Anthropic model '{model}': {e.message}", status_code=422
            ) from e
        except anthropic.BadRequestError as e:
            raise ProviderError(
                f"Anthropic rejected the request for {model}: {e.message}",
                status_code=502,
            ) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(
                f"Anthropic API error ({e.status_code}) calling {model}: {e.message}",
                status_code=502,
                retryable=e.status_code >= 500,
            ) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(
                f"Could not reach the Anthropic API: {e}", status_code=502, retryable=True
            ) from e

        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        if not text.strip():
            raise ProviderError(
                f"Empty completion from {model} (stop_reason={message.stop_reason})."
            )
        return text
