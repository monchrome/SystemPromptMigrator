# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses `uv` for everything (Python 3.12).

```bash
uv sync                                  # install deps (incl. dev group)
uv run pytest                            # full test suite — offline, no API keys needed
uv run pytest tests/test_pipeline.py -k refinement   # single test
uv run promptmigrator                    # start the API server (PM_HOST/PM_PORT, default 127.0.0.1:8000)
uv sync --group ui && uv run streamlit run ui/streamlit_app.py   # web UI (needs the API running; PM_API_URL)
```

Real migrations need `ANTHROPIC_API_KEY` (Claude targets) and/or `OPENAI_API_KEY` (GPT/o-series targets). Tests never touch the network — they inject a `FakeProvider` (see `tests/conftest.py`).

## Architecture

FastAPI microservice that migrates a prompt tuned for one LLM into one optimized for another. The core idea spans several files and is easy to miss reading any one of them:

**Every pipeline stage executes on the *target* model.** `PromptMigrator.migrate()` (pipeline/migrator.py) resolves one provider from `target_model` and uses it for all four stages, so the target LLM rewrites the prompt in its own idiom:

1. `pipeline/analyzer.py` — extracts a structured spec of the source prompt (intent, hard constraints, format-enforcement mechanism, CoT style, source-model idioms).
2. `pipeline/proposer.py` — MIPROv2-style proposal: N candidates generated concurrently, diversified by cycling `PROPOSAL_TIPS` (precise/creative/concise/defensive/example-driven), all grounded in the stage-1 analysis plus both model profiles.
3. `pipeline/evaluator.py` — LLM-as-judge scores each candidate 0–10 on five rubric dimensions; argmax wins.
4. `proposer.refine()` — one feedback-driven pass on the winner; kept only if it re-scores ≥ the unrefined winner.

**Knowledge base ≠ provider support.** `knowledge_base.py` holds trait profiles (system-instruction placement, JSON enforcement, CoT style, pitfalls) for families this service may never execute — Gemini and open-weights models are valid *source* models. Execution support lives separately in `providers/`: `vendor_for_model()` routes `claude-*` → Anthropic, `gpt-*`/`o*` → OpenAI, anything else → `ProviderError(422)`. Adding a new target = new adapter subclassing `LLMProvider` + a branch in `providers/__init__.py`; the knowledge-base profile likely already exists.

**Structured outputs everywhere.** Each stage passes a hand-written JSON schema (`ANALYSIS_SCHEMA`, `PROPOSAL_SCHEMA`, `EVALUATION_SCHEMA`) to `provider.complete()`; providers map it to their native mechanism (Anthropic `output_config.format`, OpenAI strict `response_format`). Schemas must keep every property in `required` and `additionalProperties: false` (OpenAI strict mode demands it), and must avoid numeric min/max constraints (Anthropic doesn't support them — score ranges live in `description` text instead).

**Error flow.** Providers raise `ProviderError(status_code=...)`; the FastAPI exception handler in `api.py` translates it to the HTTP response (429 rate limit, 503 missing creds, 422 unknown/unsupported model, 502 upstream). HTTP-level validation (file type/size/encoding) raises `HTTPException` directly in the endpoint.

**Testability hinges on two injection points:** `PromptMigrator(provider_for=...)` and `create_app(migrator=...)`. The `FakeProvider` dispatches on `schema_name` (`prompt_analysis` / `prompt_proposal` / `candidate_evaluation`) and returns escalating judge scores so selection logic is deterministic. Note `tests/test_api.py` wraps the fake in `_fake_provider_for`, which still calls the real `vendor_for_model()` so unsupported-target 422s stay covered.

## Conventions

- Anthropic calls follow current API rules: adaptive thinking only on models that support it (`_ADAPTIVE_THINKING_PREFIXES` in `anthropic_provider.py`), streaming + `get_final_message()`, never assistant-prefill (400s on Claude 4.6+). The knowledge-base profiles encode these same facts as prompt text — if API behavior changes, update both.
- Provider clients are created lazily (first `complete()` call) so importing/instantiating the app never requires credentials.
- pytest runs with `asyncio_mode = "auto"` — async tests need no decorator.
- The Streamlit UI (`ui/streamlit_app.py`) is a pure HTTP client of the API — it never imports `promptmigrator`. Its deps live in the `ui` dependency group so the API install stays lean; consequently nothing under `tests/` may import streamlit (plain `uv sync` doesn't install it). Validate UI changes with `streamlit.testing.v1.AppTest` via `uv run --group ui`. The UI's target-model dropdown must stay in sync with what `providers/vendor_for_model()` can execute.
