import pytest

from promptmigrator.providers import vendor_for_model
from promptmigrator.providers.base import ProviderError


def test_claude_routes_to_anthropic() -> None:
    assert vendor_for_model("claude-opus-4-8") == "anthropic"
    assert vendor_for_model("Claude-Haiku-4-5") == "anthropic"


def test_gpt_and_o_series_route_to_openai() -> None:
    assert vendor_for_model("gpt-5") == "openai"
    assert vendor_for_model("o3") == "openai"


def test_unsupported_target_raises_422() -> None:
    with pytest.raises(ProviderError) as exc_info:
        vendor_for_model("gemini-2.5-pro")
    assert exc_info.value.status_code == 422
