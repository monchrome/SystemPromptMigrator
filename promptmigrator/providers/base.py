"""Provider abstraction: one async completion method, optionally schema-constrained."""

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(RuntimeError):
    """A provider-level failure, carrying the HTTP status the API should return."""

    def __init__(self, message: str, *, status_code: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMProvider(ABC):
    @abstractmethod
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
        """Run one completion. When json_schema is given, the returned string is
        guaranteed (by the provider's structured-output mechanism) to be valid JSON
        matching the schema."""
