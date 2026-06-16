from promptmigrator.knowledge_base import all_profiles, get_profile


def test_modern_claude_resolves() -> None:
    assert get_profile("claude-opus-4-8").family == "claude-4.6+"
    assert get_profile("claude-fable-5").family == "claude-4.6+"
    assert get_profile("claude-sonnet-4-6").vendor == "anthropic"


def test_legacy_claude_resolves() -> None:
    assert get_profile("claude-sonnet-4-5").family == "claude-legacy (<=4.5)"


def test_openai_families() -> None:
    assert get_profile("gpt-5").family == "gpt"
    assert get_profile("o3").family == "openai-reasoning (o-series)"
    assert get_profile("o4-mini").family == "openai-reasoning (o-series)"


def test_gemini_and_open_weights() -> None:
    assert get_profile("gemini-2.5-pro").vendor == "google"
    assert get_profile("llama-4-70b").vendor == "open"


def test_unknown_falls_back_to_generic() -> None:
    assert get_profile("totally-made-up").family == "generic"


def test_profile_prompt_block_contains_traits() -> None:
    block = get_profile("claude-opus-4-8").as_prompt_block("target")
    assert block.startswith("<target_model_profile")
    assert "JSON / output-format enforcement" in block


def test_all_profiles_unique() -> None:
    families = [p.family for p in all_profiles()]
    assert len(families) == len(set(families))
    assert "generic" in families
