# Contributing to PromptMigrator

Thanks for your interest in improving PromptMigrator! This document explains how to
report issues, propose changes, and submit pull requests.

## Ways to contribute

- **Report a bug** — open an issue describing what you expected, what happened, and the
  steps to reproduce it.
- **Request a feature** — open an issue explaining the use case and the behavior you'd
  like to see.
- **Submit a fix or feature** — open a pull request (see below).

Before starting significant work, please open an issue first so we can discuss the
approach. This avoids duplicated effort and saves you time.

## Development setup

PromptMigrator uses [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/monchrome/SystemPromptMigrator.git
cd SystemPromptMigrator
uv sync                  # install deps, including the dev group
uv run pytest            # run the full test suite (offline, no API keys needed)
```

The test suite never touches the network — it injects a `FakeProvider`, so you do not
need any API keys to develop or run tests.

To run the service or the optional Streamlit UI locally:

```bash
uv run promptmigrator                                          # API server
uv sync --group ui && uv run streamlit run ui/streamlit_app.py # web UI (needs the API running)
```

Real migrations require `ANTHROPIC_API_KEY` (Claude targets) and/or `OPENAI_API_KEY`
(GPT / o-series targets), but contributing code and running tests does not.

## Submitting a pull request

1. **Fork** the repository and create a branch off `main` with a descriptive name
   (e.g. `fix/judge-score-tie` or `feat/gemini-target`).
2. **Make your change.** Keep the diff focused — one logical change per PR. Match the
   style and conventions of the surrounding code (see `CLAUDE.md` for architecture notes
   and project conventions).
3. **Add or update tests.** New behavior should come with tests; bug fixes should come
   with a test that fails before the fix and passes after.
4. **Run the suite locally** and make sure it's green:
   ```bash
   uv run pytest
   ```
5. **Write a clear commit message** and PR description explaining _what_ changed and
   _why_. Link any related issue (e.g. "Closes #123").
6. **Open the PR** against `main`. A maintainer will review it; please be responsive to
   feedback.

### Guidelines

- Keep public behavior and the structured-output schemas backward compatible unless the
  change is intentional and documented. Note that OpenAI strict mode requires every
  schema property in `required` and `additionalProperties: false`, and Anthropic does
  not support numeric min/max constraints — see `CLAUDE.md`.
- Adding a new target model = a new adapter subclassing `LLMProvider` plus a branch in
  `providers/__init__.py`. If you add a target, keep the Streamlit UI's target-model
  dropdown in sync with `providers/vendor_for_model()`.
- Nothing under `tests/` may import `streamlit` (it is not installed by a plain
  `uv sync`). Validate UI changes with `streamlit.testing.v1.AppTest` under
  `uv run --group ui`.
- Be kind in code review, and assume good intent.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. Instead, report them
privately to the maintainer so they can be addressed before disclosure.

## License

By contributing, you agree that your contributions will be licensed under the same
[MIT License](LICENSE) that covers this project.
