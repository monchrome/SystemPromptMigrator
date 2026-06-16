"""Provider registry: resolve a model ID to an executing provider."""

from .base import LLMProvider, ProviderError

_instances: dict[str, LLMProvider] = {}


def vendor_for_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai"
    raise ProviderError(
        f"No execution provider available for model '{model}'. This service can run "
        "migrations on Anthropic models (claude-*) and OpenAI models (gpt-*, o-series). "
        "Other families are described in the knowledge base but cannot execute the "
        "rewrite, since the target model itself performs it.",
        status_code=422,
    )


def provider_for(model: str) -> LLMProvider:
    vendor = vendor_for_model(model)
    if vendor not in _instances:
        if vendor == "anthropic":
            from .anthropic_provider import AnthropicProvider

            _instances[vendor] = AnthropicProvider()
        elif vendor == "openai":
            from .openai_provider import OpenAIProvider

            _instances[vendor] = OpenAIProvider()
    return _instances[vendor]


__all__ = ["LLMProvider", "ProviderError", "provider_for", "vendor_for_model"]
