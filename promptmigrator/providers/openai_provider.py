"""OpenAI provider built on the official SDK, for GPT / o-series targets."""

import os
from typing import Any

import openai

from .base import LLMProvider, ProviderError


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ProviderError(
                    "OpenAI credentials missing: set OPENAI_API_KEY.", status_code=503
                )
            self._client = openai.AsyncOpenAI()
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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            }

        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise ProviderError(
                f"OpenAI rate limit hit while calling {model}: {e}",
                status_code=429,
                retryable=True,
            ) from e
        except openai.AuthenticationError as e:
            raise ProviderError(
                f"OpenAI authentication failed: {e}", status_code=503
            ) from e
        except openai.NotFoundError as e:
            raise ProviderError(
                f"Unknown OpenAI model '{model}': {e}", status_code=422
            ) from e
        except openai.APIStatusError as e:
            raise ProviderError(
                f"OpenAI API error ({e.status_code}) calling {model}: {e}",
                status_code=502,
                retryable=e.status_code >= 500,
            ) from e
        except openai.APIConnectionError as e:
            raise ProviderError(
                f"Could not reach the OpenAI API: {e}", status_code=502, retryable=True
            ) from e

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ProviderError(f"Empty completion from {model}.")
        return content
