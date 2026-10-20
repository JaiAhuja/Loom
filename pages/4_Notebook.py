import hashlib

import streamlit as st

from src.services.notebook import NotebookStore
from src.services.ui.confirm import confirm_destructive
from config.settings import settings
from src.services.llm import get_llm

st.set_page_config(page_title="Evidence Notebook - Loom", page_icon="📝", layout="wide")
st.title("Evidence Notebook")
st.caption("Keep the answers worth remembering, with the evidence that supports them.")

store = NotebookStore()
query = st.text_input("🔍 Search notes", placeholder="Search by title or paper…")
entries = store.search(query)
if not entries:
    st.info(
        "No notebook entries yet. Save a RAG-backed answer from the Chat page "
        "to start building your research memory."
    )
    st.stop()

selected = st.sidebar.radio(
    f"Saved notes ({len(entries)})",
    [entry["filename"] for entry in entries],
    format_func=lambda filename: next(e["title"] for e in entries if e["filename"] == filename)[:60],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.subheader("Synthesize a brief")
brief_selection = st.sidebar.multiselect(
    "Select notes",
    options=[entry["filename"] for entry in entries],
    default=[],
    format_func=lambda filename: next(e["title"] for e in entries if e["filename"] == filename)[:45],
)
brief_instruction = st.sidebar.text_input(
    "Brief focus (optional)", placeholder="Compare findings and open questions"
)
brief_key = hashlib.sha256(
    ("\n".join(sorted(brief_selection)) + "\n" + brief_instruction).encode("utf-8")
).hexdigest()
if st.session_state.get("notebook_brief_key") != brief_key:
    st.session_state.pop("notebook_brief", None)
if st.sidebar.button("✨ Generate brief", disabled=not brief_selection, use_container_width=True):
    selected_notes = [store.load(filename) for filename in brief_selection]
    source_text = "\n\n--- NOTE ---\n\n".join(note.to_markdown() for note in selected_notes)
    prompt = (
        "Create a concise research brief from the notes below. Preserve uncertainty, "
        "separate established findings from open questions, and include an Evidence "
        "section naming the source papers. Do not invent facts. "
        f"Focus: {brief_instruction or 'main findings and connections'}.\n\n{source_text}"
    )
    with st.spinner("Synthesizing selected notes…"):
        try:
            llm = get_llm(model=settings.OLLAMA_MODEL, temperature=0.1)
            brief = llm.invoke(prompt).content
            st.session_state["notebook_brief"] = brief
            st.session_state["notebook_brief_key"] = brief_key
        except Exception as exc:
            st.sidebar.error(f"Could not generate brief: {exc}")

if st.session_state.get("notebook_brief"):
    st.markdown("## Research brief")
    st.markdown(st.session_state["notebook_brief"])
    st.download_button(
        "⬇ Export brief",
        st.session_state["notebook_brief"],
        file_name="loom-research-brief.md",
        mime="text/markdown",
    )
    st.divider()
entry = store.load(selected)

col_title, col_status, col_delete = st.columns([5, 1, 1])
with col_title:
    st.subheader(entry.title)
    st.caption(f"Saved {entry.saved_at}" + (f" · {entry.paper}" if entry.paper else ""))
with col_status:
    st.metric("Status", entry.status)
with col_delete:
    area = st.container()
    if confirm_destructive(
        "🗑 Delete",
        "Delete this notebook entry?",
        key=f"notebook_{selected}",
        trigger_kwargs={"use_container_width": True},
        confirm_container=area,
        on_confirm=lambda: store.delete(selected),
    ):
        st.rerun()

if entry.tags:
    st.caption(" · ".join(f"#{tag}" for tag in entry.tags))
st.markdown("### Question")
st.markdown(entry.question)
st.markdown("### Loom's answer")
st.markdown(entry.answer)
if entry.note:
    st.markdown("### My note")
    st.info(entry.note)

st.markdown("### Evidence")
for index, evidence in enumerate(entry.evidence, 1):
    label = (
        " · ".join(x for x in (evidence.source, evidence.chunk, evidence.pages, evidence.section) if x)
        or "Retrieved excerpt"
    )
    with st.expander(f"{index}. {label}", expanded=index == 1):
        st.markdown(evidence.text)

st.download_button(
    "⬇ Export this note",
    entry.to_markdown(),
    file_name=selected.rsplit(".", 1)[0] + ".md",
    mime="text/markdown",
)
