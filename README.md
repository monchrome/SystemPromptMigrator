# PromptMigrator

A Python microservice (FastAPI) that migrates a prompt written for one LLM into an
equivalent prompt optimized for a different LLM. Syntax, formatting idioms, verbosity,
system-instruction placement, JSON-format enforcement, and chain-of-thought style all
vary drastically between model families — so the rewrite is performed **by the target
model itself**, calibrated with techniques borrowed from prompt-optimization research:

- **MIPROv2-style grounded proposal & selection** — the service generates *N*
  diversified candidate rewrites (each with a different meta-level "tip": precise /
  creative / concise / defensive / example-driven), grounded in a structured analysis
  of the source prompt, then scores every candidate with an LLM-as-judge rubric and
  selects the best, followed by one feedback-driven refinement pass.
- **PromptBridge-style cross-model calibration** — a built-in knowledge base of model
  trait profiles (Claude 4.6+/legacy, GPT, OpenAI o-series, Gemini, open-weights)
  describes how each family handles system instructions, JSON enforcement
  (structured outputs vs. prefill vs. `response_format` vs. `response_schema`),
  chain-of-thought, and verbosity. Source and target profiles are injected into every
  pipeline stage so the rewrite maps idioms instead of guessing.

## Pipeline

```
upload .txt ──▶ 1. ANALYZE   structured spec of the source prompt (intent, hard
                             constraints, format contract, CoT style, source idioms)
            ──▶ 2. PROPOSE   N candidate rewrites in parallel, one style tip each
            ──▶ 3. EVALUATE  LLM-judge rubric: intent fidelity, constraint coverage,
                             target-idiom fit, format enforcement, clarity (0–10 each)
            ──▶ 4. REFINE    one feedback pass on the winner; kept only if it scores
                             at least as well
```

All four stages run on the **target** model, so the migrated prompt is written in the
target model's own idiom.

## Supported execution targets

| Target family | Provider | Mechanism |
|---|---|---|
| `claude-*` | Anthropic SDK | streaming, adaptive thinking (4.6+), structured outputs (`output_config.format`) |
| `gpt-*`, `o1/o3/o4-*` | OpenAI SDK | `response_format` strict JSON schema |

Other families (Gemini, Llama, …) are described in the knowledge base — they work as
**source** models, but cannot be targets until a provider adapter is added
(`promptmigrator/providers/`).

## Quick start

```bash
# credentials for whichever target vendors you use
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...      # only if targeting GPT/o-series

uv sync
uv run promptmigrator             # serves on http://127.0.0.1:8000
```

Interactive docs: http://127.0.0.1:8000/docs

### Web UI (Streamlit)

With the API running, start the UI in a second terminal:

```bash
uv sync --group ui
uv run streamlit run ui/streamlit_app.py    # opens on http://127.0.0.1:8501
```

Upload the prompt file, pick source/target models from the dropdowns (or enter a
custom model ID), and run the migration. The UI shows the source-prompt analysis,
a rubric scoreboard comparing all candidates, one tab per option (scores, change
log, judge feedback, full prompt), and lets you pick a final choice and download
it as a `.txt` file. Point it at a remote API with `PM_API_URL`.

## API

### `POST /v1/migrations` (multipart/form-data)

| Field | Type | Notes |
|---|---|---|
| `prompt_file` | file | the prompt as `.txt` / `.md` / `.prompt` (UTF-8, ≤1 MB) |
| `source_model` | str | model the prompt was tuned for, e.g. `gpt-4o` |
| `target_model` | str | model to migrate to, e.g. `claude-opus-4-8` |
| `num_candidates` | int | 1–5, default 3 |
| `refine` | bool | default `true` |
| `notes` | str | optional operator context (task, eval criteria) |

```bash
curl -s http://127.0.0.1:8000/v1/migrations \
  -F "prompt_file=@my_prompt.txt" \
  -F "source_model=gpt-4o" \
  -F "target_model=claude-opus-4-8" \
  -F "num_candidates=3" | jq .
```

Response (abridged):

```json
{
  "migrated_prompt": "...",
  "selected_candidate_index": 1,
  "refined": true,
  "analysis": {
    "intent": "...",
    "hard_constraints": ["..."],
    "output_format": {"type": "json", "spec": "...", "enforcement_mechanism": "assistant prefill"},
    "source_model_idioms": ["..."],
    "migration_risks": ["..."]
  },
  "candidates": [
    {"prompt": "...", "change_log": ["..."], "tip": "...",
     "scores": {"intent_fidelity": 9, "clarity": 8}, "total_score": 43, "feedback": "..."}
  ],
  "source_profile": {"family": "gpt"},
  "target_profile": {"family": "claude-4.6+"}
}
```

`change_log` entries also surface recommended **API-level** configuration when prompt
text alone shouldn't carry the contract (e.g. "enforce JSON via `output_config.format`
json_schema instead of the removed assistant prefill").

### Other endpoints

- `GET /v1/model-profiles` — the cross-model trait knowledge base
- `GET /healthz` — liveness probe

### Errors

| Status | Meaning |
|---|---|
| 400 | bad upload (wrong extension, empty, not UTF-8) or identical source/target |
| 413 | prompt file too large |
| 422 | unsupported target model / unknown model ID |
| 429 | upstream provider rate limit (`retryable: true`) |
| 502/503 | upstream provider error / missing credentials |

## Configuration

| Env var | Default | |
|---|---|---|
| `PM_HOST` / `PM_PORT` | `127.0.0.1` / `8000` | bind address |
| `PM_NUM_CANDIDATES` | `3` | default candidate count |
| `PM_MAX_OUTPUT_TOKENS` | `16000` | per-call `max_tokens` |
| `PM_MAX_PROMPT_BYTES` | `1048576` | upload size limit |

## Development

```bash
uv run pytest          # 22 tests, no network/API keys needed (fake provider)
```

Layout:

```
promptmigrator/
  api.py                  FastAPI app (create_app factory)
  knowledge_base.py       model trait profiles (PromptBridge-style mapping table)
  models.py               request/response schemas
  providers/              Anthropic + OpenAI adapters behind one interface
  pipeline/
    analyzer.py           stage 1 — structured source-prompt spec
    proposer.py           stage 2 — MIPROv2-style diversified proposals + refine
    evaluator.py          stage 3 — LLM-as-judge rubric
    migrator.py           orchestrator
```
