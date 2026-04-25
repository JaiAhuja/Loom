"""Chat History page - browse, search, resume, download, and delete saved chats.

JSON chats (from :mod:`src.chat.store`) are resumable; legacy ``.md``
files are still listed but read-only.
"""

import streamlit as st

from src.chat import ChatStore
from src.ui.confirm import confirm_destructive

st.set_page_config(
    page_title="Chat History - Loom",
    page_icon="🪡",
    layout="wide",
)

st.title("Chat History")
st.caption("Review, search, and resume previously saved conversations.")

chat_store = ChatStore()

# ----- Search -----
search_q = st.text_input(
    "🔍 Search saved chats (topic or message content)",
    placeholder="e.g. graph theory, attention mechanism, ...",
    key="chat_history_search",
)
entries = chat_store.search(search_q) if search_q else chat_store.list_chats()

if not entries:
    if search_q:
        st.info(f"No chats match '{search_q}'.")
    else:
        st.info(
            "No saved chats yet. Use the **Save Chat to History** button on the "
            "main chat page to save a conversation."
        )
    st.stop()

# ----- Sidebar: list of chats -----
st.sidebar.subheader(f"Saved Chats ({len(entries)})")


def _chat_label(entry: dict) -> str:
    marker = "📝" if entry["type"] == "json" else "📄"
    ts = entry["modified"].strftime("%Y-%m-%d %H:%M")
    return f"{marker} {entry['topic'][:48]} · {ts}"


selected_filename = st.sidebar.radio(
    "Select a chat to view",
    options=[e["filename"] for e in entries],
    format_func=lambda fn: _chat_label(next(e for e in entries if e["filename"] == fn)),
    label_visibility="collapsed",
)

# ----- Main panel: selected chat -----
selected_entry = next((e for e in entries if e["filename"] == selected_filename), None)
if selected_entry is None:
    st.stop()

# Container created before the columns so it renders in full-width context;
# the confirm_destructive widget uses it to avoid squeezing inside col_del.
_del_confirm_area = st.container()

col_title, col_resume, col_dl, col_del = st.columns([4, 1, 1, 1])
with col_title:
    st.subheader(selected_entry["topic"])
    st.caption(
        f"Saved {selected_entry['modified'].strftime('%Y-%m-%d %H:%M')} · "
        f"{'Resumable JSON' if selected_entry['resumable'] else 'Legacy markdown (view-only)'}"
    )

# Resume - only for JSON chats
with col_resume:
    if selected_entry["resumable"]:
        if st.button("▶ Resume", use_container_width=True, type="primary"):
            record = chat_store.load(selected_entry["filename"])
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in record.messages
            ]
            st.session_state["_resumed_from"] = selected_entry["filename"]
            st.success("Conversation loaded - switching to chat...")
            try:
                st.switch_page("app.py")
            except Exception:
                st.info("Open the main chat page to continue.")
    else:
        st.caption("—")

# Download
md_content = chat_store.render_markdown(selected_entry["filename"])
with col_dl:
    st.download_button(
        "⬇ Download",
        data=md_content,
        file_name=selected_entry["filename"].rsplit(".", 1)[0] + ".md",
        mime="text/markdown",
        use_container_width=True,
    )

# Delete (with confirmation)
with col_del:
    if confirm_destructive(
        "🗑 Delete",
        "Delete this chat? This cannot be undone.",
        key=f"chat_{selected_entry['filename']}",
        on_confirm=lambda fn=selected_entry["filename"]: chat_store.delete(fn),
        trigger_kwargs={"use_container_width": True},
        confirm_container=_del_confirm_area,
    ):
        st.rerun()

# Metadata (JSON only)
if selected_entry["type"] == "json" and selected_entry.get("metadata"):
    meta = selected_entry["metadata"]
    with st.expander("Conversation metadata", expanded=False):
        meta_rows = {k: v for k, v in meta.items() if v not in (None, "", False)}
        if meta_rows:
            st.json(meta_rows)
        else:
            st.caption("No metadata recorded.")

st.divider()

# Render the chat content
if selected_entry["type"] == "json":
    record = chat_store.load(selected_entry["filename"])
    for msg in record.messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
else:
    st.markdown(md_content)