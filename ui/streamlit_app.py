"""Streamlit front-end for the PromptMigrator API.

Run with:  uv run streamlit run ui/streamlit_app.py
Configure the API location with PM_API_URL (default http://127.0.0.1:8000).
"""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("PM_API_URL", "http://127.0.0.1:8000")
CUSTOM = "Custom…"

# Source can be any family the knowledge base describes.
SOURCE_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o3",
    "o4-mini",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "llama-4",
    "mistral-large",
    CUSTOM,
]

# Targets are limited to vendors the service can execute (Anthropic + OpenAI).
TARGET_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "o3",
    "o4-mini",
    CUSTOM,
]

RUBRIC_COLUMNS = {
    "intent_fidelity": "Intent fidelity",
    "constraint_coverage": "Constraint coverage",
    "target_idiom_fit": "Target idiom fit",
    "format_enforcement": "Format enforcement",
    "clarity": "Clarity",
}


def model_picker(label: str, options: list[str], key: str) -> str:
    choice = st.selectbox(label, options, key=key)
    if choice == CUSTOM:
        return st.text_input(f"{label} (custom model ID)", key=f"{key}_custom").strip()
    return choice


def call_api(
    api_url: str,
    *,
    file_name: str,
    file_bytes: bytes,
    source_model: str,
    target_model: str,
    num_candidates: int,
    refine: bool,
    notes: str,
) -> dict | None:
    data = {
        "source_model": source_model,
        "target_model": target_model,
        "num_candidates": str(num_candidates),
        "refine": "true" if refine else "false",
    }
    if notes.strip():
        data["notes"] = notes.strip()
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/migrations",
            files={"prompt_file": (file_name, file_bytes, "text/plain")},
            data=data,
            timeout=600,  # a full migration is ~8 model calls
        )
    except requests.ConnectionError:
        st.error(
            f"Could not reach the PromptMigrator API at {api_url}. "
            "Start it with `uv run promptmigrator`."
        )
        return None
    except requests.Timeout:
        st.error("The migration timed out. Try fewer candidates or disable refinement.")
        return None

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Migration failed ({response.status_code}): {detail}")
        return None
    return response.json()


def render_analysis(analysis: dict) -> None:
    with st.expander("Source prompt analysis (stage 1)", expanded=False):
        st.markdown(f"**Intent:** {analysis['intent']}")
        st.markdown(f"**Role / persona:** {analysis['role_persona']}")
        fmt = analysis["output_format"]
        st.markdown(
            f"**Output format:** `{fmt['type']}` — {fmt['spec']}  \n"
            f"**Enforced via:** {fmt['enforcement_mechanism']}"
        )
        st.markdown(f"**Reasoning style:** {analysis['reasoning_style']}")
        for title, key in [
            ("Hard constraints", "hard_constraints"),
            ("Source-model idioms", "source_model_idioms"),
            ("Migration risks", "migration_risks"),
        ]:
            if analysis[key]:
                st.markdown(f"**{title}:**")
                st.markdown("\n".join(f"- {item}" for item in analysis[key]))


def render_scoreboard(result: dict) -> None:
    st.subheader("Rubric scores")
    best_index = result["selected_candidate_index"]
    rows = []
    for i, candidate in enumerate(result["candidates"]):
        scores = candidate["scores"] or {}
        rows.append(
            {
                "Option": f"Candidate {i + 1}" + (" ⭐" if i == best_index else ""),
                **{label: scores.get(key) for key, label in RUBRIC_COLUMNS.items()},
                "Total (/50)": candidate["total_score"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    if result["refined"]:
        st.caption(
            "⭐ = judge's pick. A refinement pass improved it further — the refined "
            "version re-scored at least as well and is offered below as **Final "
            "(refined)**."
        )
    else:
        st.caption("⭐ = judge's pick (refinement pass did not beat it, or was disabled).")


def render_options(result: dict) -> dict[str, str]:
    """Render one tab per reviewable option; return {label: prompt_text}."""
    options: dict[str, str] = {}
    if result["refined"]:
        options["Final (refined)"] = result["migrated_prompt"]
    for i, candidate in enumerate(result["candidates"]):
        label = f"Candidate {i + 1}"
        if i == result["selected_candidate_index"]:
            label += " ⭐"
        options[label] = candidate["prompt"]

    st.subheader("Review the options")
    tabs = st.tabs(list(options.keys()))
    candidates = result["candidates"]
    offset = 1 if result["refined"] else 0
    for tab_index, (tab, label) in enumerate(zip(tabs, options)):
        with tab:
            if result["refined"] and tab_index == 0:
                st.info(
                    "Feedback-driven refinement of the judge's pick "
                    f"(Candidate {result['selected_candidate_index'] + 1})."
                )
            else:
                candidate = candidates[tab_index - offset]
                cols = st.columns(len(RUBRIC_COLUMNS) + 1)
                scores = candidate["scores"] or {}
                for col, (key, name) in zip(cols, RUBRIC_COLUMNS.items()):
                    col.metric(name, scores.get(key))
                cols[-1].metric("Total", f"{candidate['total_score']}/50")
                st.caption(f"Style tip: {candidate['tip']}")
                if candidate["change_log"]:
                    with st.expander("Change log"):
                        st.markdown(
                            "\n".join(f"- {item}" for item in candidate["change_log"])
                        )
                if candidate["feedback"]:
                    with st.expander("Judge feedback"):
                        st.write(candidate["feedback"])
            st.code(options[label], language="text", wrap_lines=True)
    return options


def main() -> None:
    st.set_page_config(page_title="PromptMigrator", page_icon="🔁", layout="wide")
    st.title("🔁 PromptMigrator")
    st.caption(
        "Migrate a prompt from one LLM to another. The target model rewrites the "
        "prompt itself (MIPROv2-style propose → judge → refine), calibrated by a "
        "cross-model trait knowledge base."
    )

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API base URL", DEFAULT_API_URL)
        num_candidates = st.slider("Candidate rewrites", 1, 5, 3)
        refine = st.toggle("Refinement pass", value=True)
        notes = st.text_area(
            "Operator notes (optional)",
            placeholder="Task context, evaluation criteria, things that must not change…",
        )

    col_src, col_tgt = st.columns(2)
    with col_src:
        source_model = model_picker("Source model (prompt was tuned for)", SOURCE_MODELS, "source")
    with col_tgt:
        target_model = model_picker("Target model (migrate to)", TARGET_MODELS, "target")

    uploaded = st.file_uploader("Prompt file", type=["txt", "md", "prompt"])
    if uploaded is not None:
        with st.expander("Preview uploaded prompt"):
            st.code(uploaded.getvalue().decode("utf-8", errors="replace"), language="text")

    ready = uploaded is not None and bool(source_model) and bool(target_model)
    if st.button("Migrate prompt", type="primary", disabled=not ready):
        if source_model.lower() == target_model.lower():
            st.error("Source and target model must differ.")
        else:
            with st.spinner(
                f"Migrating {source_model} → {target_model} "
                f"({num_candidates} candidates{' + refinement' if refine else ''})… "
                "this typically takes 30–90 s."
            ):
                result = call_api(
                    api_url,
                    file_name=uploaded.name,
                    file_bytes=uploaded.getvalue(),
                    source_model=source_model,
                    target_model=target_model,
                    num_candidates=num_candidates,
                    refine=refine,
                    notes=notes,
                )
            if result is not None:
                st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    st.success(
        f"Migrated **{result['source_model']}** → **{result['target_model']}**. "
        f"Judge's pick: Candidate {result['selected_candidate_index'] + 1}"
        + (" (then refined)." if result["refined"] else ".")
    )
    render_analysis(result["analysis"])
    render_scoreboard(result)
    options = render_options(result)

    st.subheader("Pick your final prompt")
    choice = st.radio("Final choice", list(options.keys()), horizontal=True)
    st.download_button(
        "⬇️ Download as .txt",
        data=options[choice],
        file_name=f"migrated_prompt_{result['target_model']}.txt",
        mime="text/plain",
        type="primary",
    )


main()
