"""Run the Loom Streamlit chat interface and coordinate its services."""

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config.settings import configure_langsmith, settings
from src.services.chat import ChatMetadata, ChatStore
from src.services.evaluation import RAGJudge
from src.services.agent import GraphBuilder
from src.services.knowledge_graph import get_neo4j_connection
from src.services.ingestion import IngestionService
from src.services.ingestion.identity import (
    build_identity,
    generate_ingest_id,
    save_upload,
)
from src.services.ui.bootstrap import (
    check_neo4j_status,
    check_ollama_status,
    get_document_processor,
    get_vector_store,
)
from src.services.ui.chrome import (
    EMPTY_RESPONSE_MARKDOWN,
    format_chat_error,
    inject_stylesheet,
    render_brand,
    render_hero,
    render_status_bar,
)
from src.services.ui.confirm import confirm_destructive

st.set_page_config(
    page_title="Loom",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_stylesheet()
configure_langsmith()

graph_builder = GraphBuilder()


def _persist_toggle(label: str, state_key: str, **kwargs):
    """Render a persistent toggle and synchronize its session-state value."""
    value = st.toggle(label, value=st.session_state[state_key], **kwargs)
    st.session_state[state_key] = value
    return value


def _chat_metadata(
    model: str,
    temperature: float,
    use_rag: bool,
    use_graph: bool,
    neo4j_connected: bool,
    collection_name: str | None,
    paper_filter: str | None,
    document_id_filter: str | None,
) -> ChatMetadata:
    """Build chat metadata from the active model and retrieval settings."""
    return ChatMetadata(
        model=model,
        temperature=temperature,
        use_rag=use_rag,
        use_graph=use_graph and neo4j_connected,
        collection_name=collection_name,
        paper_filter=paper_filter,
        document_id=document_id_filter,
    )


def _save_uploaded_pdf(file, ingest_id: str, pdf_dir: str) -> str:
    """Persist an uploaded PDF using its content-based document identity."""
    raw_bytes = bytes(file.getbuffer())
    return save_upload(
        file_bytes=raw_bytes,
        identity=build_identity(raw_bytes, file.name, ingest_id),
        base_dir=pdf_dir,
    )


def _to_lang_message(msg: dict):
    """Convert a stored chat message into its LangChain message type."""
    if not isinstance(msg, dict):
        return None
    content = msg.get("content", "")
    return (
        HumanMessage(content=content)
        if msg.get("role") == "user"
        else (AIMessage(content=content) if msg.get("role") == "assistant" else None)
    )


@st.cache_resource(show_spinner=False, max_entries=10)
def _get_compiled_graph(
    use_rag: bool,
    use_graph: bool,
    collection_name: str | None,
    model: str,
    temperature: float,
    document_id: str | None,
    _neo4j_conn,
    _vector_store,
):
    """Cache-compiled LangGraph for a given feature/config signature.

    Streamlit hashes the positional/keyword args to derive the cache key;
    arguments prefixed with ``_`` are skipped (Streamlit convention) which
    keeps the driver/store singletons out of the hash.
    """
    return graph_builder.build(
        use_rag=use_rag,
        use_graph=use_graph,
        collection_name=collection_name,
        model=model,
        temperature=temperature,
        neo4j_conn=_neo4j_conn,
        vector_store=_vector_store,
        document_id=document_id,
    )


st.session_state.setdefault("messages", [])
st.session_state.setdefault("use_rag_persistent", False)
st.session_state.setdefault("use_graph_persistent", False)
st.session_state.setdefault("use_rag_eval_persistent", False)


with st.sidebar:
    render_brand()

    st.divider()
    is_connected, available_models = check_ollama_status()

    if is_connected:
        st.success(f"✅ Ollama connected ({len(available_models)} models)")
    else:
        st.error("❌ Ollama not running")
        st.caption("Start it with: `ollama serve`")
        st.stop()

    neo4j_connected, neo4j_error = check_neo4j_status()
    if neo4j_connected:
        st.success("✅ Neo4j connected")
    else:
        st.warning("⚠️ Neo4j not connected")
        if neo4j_error:
            st.caption(f"🔍 {neo4j_error}")
        st.caption("Use Neo4j Desktop")

    st.divider()
    st.subheader("⚙️ Model Settings")

    default_model = settings.OLLAMA_MODEL
    if available_models:
        model_options = sorted(set(available_models))
        default_idx = next(
            (i for i, m in enumerate(model_options) if m.startswith(default_model)),
            0,
        )
        model = st.selectbox("Model", options=model_options, index=default_idx)
    else:
        model = st.text_input("Model", value=default_model)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.OLLAMA_TEMPERATURE,
        step=0.1,
        help="Lower = more focused, Higher = more creative",
    )

    st.divider()
    st.subheader("🔧 Features")

    use_rag = _persist_toggle(
        "📄 Enable RAG (Document Q&A)",
        "use_rag_persistent",
        key="use_rag_toggle",
        help="Enable querying your uploaded PDF documents.",
    )

    use_graph = _persist_toggle(
        "🔗 Enable Knowledge Graph",
        "use_graph_persistent",
        key="use_graph_toggle",
        disabled=not neo4j_connected,
        help="Query the Neo4j knowledge graph for cross-paper relationships. Requires Neo4j to be running.",
    )

    if use_graph and neo4j_connected:
        st.info(
            "💡 Knowledge graph is **ON**. The agent can explore "
            "relationships between your papers and concepts."
        )

    use_rag_eval = _persist_toggle(
        "📊 Show RAG Quality Scores",
        "use_rag_eval_persistent",
        key="use_rag_eval_toggle",
        help="After each RAG-assisted reply, run an LLM-as-Judge evaluation "
        "(adds one extra LLM call per response).",
    )

    collection_name = None
    paper_filter = None
    document_id_filter: str | None = None

    if use_rag:
        st.divider()
        st.subheader("📁 Documents")

        store = get_vector_store()
        existing_collections = store.list_collections()

        col_tab1, col_tab2 = st.tabs(["Select", "Create New"])

        with col_tab1:
            if existing_collections:
                collection_name = st.selectbox(
                    "Collection",
                    options=existing_collections,
                    label_visibility="collapsed",
                )
                count = store.get_collection_count(collection_name)
                st.caption(f"📊 {count} chunks stored")
            else:
                st.caption("No collections yet. Create one or upload PDFs.")
                collection_name = "default"

        with col_tab2:
            new_name = st.text_input(
                "New collection name",
                placeholder="e.g., ml-notes",
                label_visibility="collapsed",
            )
            if new_name:
                new_collection = new_name.strip().lower().replace(" ", "-")
                st.caption(f"Will use: `{new_collection}`")
                if st.button("✅ Use this collection", use_container_width=True):
                    st.session_state.selected_collection = new_collection
                    st.rerun()

        if "selected_collection" in st.session_state:
            collection_name = st.session_state.selected_collection

        if collection_name:
            papers_in_collection = store.list_papers(collection_name)
            if papers_in_collection:
                st.markdown("---")

                doc_id_options: list[str | None] = [None] + [p["document_id"] for p in papers_in_collection]

                def _paper_label(doc_id):
                    if doc_id is None:
                        return "All Papers"
                    meta = next(
                        (p for p in papers_in_collection if p["document_id"] == doc_id),
                        None,
                    )
                    if not meta:
                        return doc_id
                    return f"{meta['title']}  [{meta['domain']}]"

                selected_doc_id = st.selectbox(
                    "📄 Filter by Paper",
                    options=doc_id_options,
                    format_func=_paper_label,
                    help="Scope retrieval to a specific paper, or search across all.",
                )
                if selected_doc_id is not None:
                    document_id_filter = selected_doc_id
                    selected_meta = next(
                        p for p in papers_in_collection if p["document_id"] == selected_doc_id
                    )
                    paper_filter = selected_meta["title"]
                    st.caption(
                        f"🔍 RAG scoped to: **{paper_filter}** · "
                        f"`{selected_meta['domain']}` · `{selected_meta['chunk_count']} chunks`"
                    )
                else:
                    domains = {}
                    for p in papers_in_collection:
                        d = p["domain"]
                        domains[d] = domains.get(d, 0) + 1
                    domain_summary = ", ".join(f"{d} ({c})" for d, c in sorted(domains.items()))
                    st.caption(f"🔍 Searching across {len(papers_in_collection)} paper(s) · {domain_summary}")

        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("🔄 Process & Index Documents", use_container_width=True):
            target_collection = collection_name or "default"

            ingest_id = generate_ingest_id()
            pdf_dir = os.path.join(".", "data", "pdfs")
            file_paths = [_save_uploaded_pdf(file, ingest_id, pdf_dir) for file in uploaded_files]

            progress = st.progress(0, text="Initializing...")

            def _on_progress(current: int, total: int, message: str):
                progress.progress(
                    current / total if total else 1.0,
                    text=message,
                )

            processor = get_document_processor()
            svc = IngestionService(
                processor=processor,
                store=store,
                neo4j_conn=get_neo4j_connection() if neo4j_connected else None,
            )
            ing_result = svc.ingest_files(
                file_paths=file_paths,
                collection_name=target_collection,
                model=model,
                on_progress=_on_progress,
            )

            for fr in ing_result.file_results:
                if fr.skipped:
                    kg_label = " + KG" if fr.kg_indexed else ""
                    st.info(
                        f"⏭️ Already in RAG{kg_label}: **{fr.paper_title}** "
                        f"({fr.content_chunks} chunks already stored)"
                    )
                elif fr.success:
                    summary_label = " + summary" if fr.has_summary else ""
                    kg_label = " + KG" if fr.kg_indexed else ""
                    st.success(
                        f"✅ RAG{kg_label}: **{fr.paper_title}** → {fr.content_chunks} chunks{summary_label}"
                    )
                else:
                    st.error(f"❌ RAG: {fr.file_name}: {fr.error}")

            progress.progress(1.0, text="Done!")
            st.rerun()

        if existing_collections and collection_name in existing_collections:
            _del_collection_confirm_area = st.container()

            if confirm_destructive(
                "🗑️ Delete Collection",
                f"Delete collection **{collection_name}** and all its document chunks? "
                "This cannot be undone. (The Knowledge Graph will remain unchanged.)",
                key=f"delete_collection_{collection_name}",
                on_confirm=lambda cn=collection_name: (
                    store.delete_collection(cn),
                    st.session_state.pop(f"_confirm_delete_collection_{cn}", None),
                ),
                trigger_kwargs={"use_container_width": True},
                confirm_container=_del_collection_confirm_area,
            ):
                st.success(f"Deleted collection: `{collection_name}`")
                st.rerun()

    st.divider()
    st.subheader("Session")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "_resumed_from" in st.session_state:
                del st.session_state["_resumed_from"]
            st.rerun()
    with col2:
        if st.session_state.messages:
            from src.services.chat import ChatRecord

            full_export = ChatRecord(
                topic="Conversation",
                messages=st.session_state.messages,
            ).to_markdown()
            st.download_button(
                "Export Chat",
                data=full_export,
                file_name="loom_conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with col3:
        if st.button(
            "Save Chat to History",
            use_container_width=True,
            disabled=not st.session_state.messages,
            key="save_chat_button",
        ):
            chat_store = ChatStore()
            metadata = _chat_metadata(
                model,
                temperature,
                use_rag,
                use_graph,
                neo4j_connected,
                collection_name,
                paper_filter,
                document_id_filter,
            )
            try:
                resumed_filename = st.session_state.get("_resumed_from")
                if resumed_filename:
                    saved_path = chat_store.update(resumed_filename, st.session_state.messages, metadata)
                    st.success(f"Updated chat: {os.path.basename(saved_path)}")
                else:
                    saved_path = chat_store.save(st.session_state.messages, metadata)
                    st.success(f"Saved to chat_history/{os.path.basename(saved_path)}")
            except ValueError as exc:
                st.warning(str(exc))
            except OSError as exc:
                st.error(f"Could not save chat history: {exc}")


render_status_bar(
    model,
    collection_name=collection_name if use_rag else None,
    paper_filter=paper_filter,
    use_graph=use_graph,
)

if not st.session_state.messages:
    render_hero()

for i, msg in enumerate(st.session_state.messages):
    if not isinstance(msg, dict):
        continue
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))

if user_input := st.chat_input("Ask about any concept in DE, DS, or AI..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    lang_messages = [m for msg in st.session_state.messages if (m := _to_lang_message(msg))]

    with st.chat_message("assistant"):
        result = {}
        with st.spinner("Thinking..."):
            try:
                graph = _get_compiled_graph(
                    use_rag=use_rag,
                    use_graph=use_graph and neo4j_connected,
                    collection_name=collection_name,
                    model=model,
                    temperature=temperature,
                    document_id=document_id_filter,
                    _neo4j_conn=get_neo4j_connection() if (use_graph and neo4j_connected) else None,
                    _vector_store=get_vector_store() if use_rag else None,
                )

                result = graph.invoke({"messages": lang_messages}) or {}
                result_messages = result.get("messages") or []
                ai_msg = result_messages[-1] if result_messages else None
                response = getattr(ai_msg, "content", None) or EMPTY_RESPONSE_MARKDOWN

            except Exception as e:
                response = format_chat_error(e, model)

        if use_rag and use_rag_eval:
            retrieved_chunks = [
                msg.content
                for msg in result.get("messages", [])
                if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "query_documents"
            ]
            if retrieved_chunks:
                with st.spinner("Evaluating response quality..."):
                    judge = RAGJudge()
                    eval_result = judge.evaluate(
                        query=user_input,
                        context="\n\n---\n\n".join(retrieved_chunks),
                        answer=response,
                        model=model,
                    )
                if eval_result is not None:
                    response = response + eval_result.as_markdown()

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
