import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import configure_langsmith
from src.graph import GraphBuilder
from src.graph_db import get_neo4j_connection
from src.ingestion import IngestionService
from src.ingestion.identity import build_identity, generate_ingest_id, save_upload
from src.ui.bootstrap import (
    check_neo4j_status,
    check_ollama_status,
    get_document_processor,
    get_vector_store,
)
from src.ui.chrome import (
    EMPTY_RESPONSE_MARKDOWN,
    format_chat_error,
    inject_stylesheet,
    render_brand,
    render_hero,
    render_status_bar,
)
from config.settings import settings

st.set_page_config(
    page_title="Loom",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_stylesheet()
configure_langsmith()

graph_builder = GraphBuilder()


@st.cache_resource(show_spinner=False)
def _get_compiled_graph(
    use_web_search: bool,
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
        use_web_search=use_web_search,
        use_rag=use_rag,
        use_graph=use_graph,
        collection_name=collection_name,
        model=model,
        temperature=temperature,
        neo4j_conn=_neo4j_conn,
        vector_store=_vector_store,
        document_id=document_id,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    render_brand()

    # ----- Ollama Status -----
    st.divider()
    is_connected, available_models = check_ollama_status()

    if is_connected:
        st.success(f"✅ Ollama connected ({len(available_models)} models)")
    else:
        st.error("❌ Ollama not running")
        st.caption("Start it with: `ollama serve`")
        st.stop()

    neo4j_connected = check_neo4j_status()
    if neo4j_connected:
        st.success("✅ Neo4j connected")
    else:
        st.warning("⚠️ Neo4j not connected")
        st.caption("`docker-compose up -d` or use Neo4j Desktop")

    # ----- Model Settings -----
    st.divider()
    st.subheader("⚙️ Model Settings")

    # Model selection from available models
    default_model = settings.OLLAMA_MODEL
    if available_models:
        # Try to find the default model in available models
        model_options = sorted(set(available_models))
        default_idx = 0
        for i, m in enumerate(model_options):
            if m.startswith(default_model):
                default_idx = i
                break
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

    # ----- Feature Toggles -----
    st.divider()
    st.subheader("🔧 Features")

    use_web_search = st.toggle(
        "🌐 Enable Web Search",
        value=False,
        help="Allow the agent to search the internet via DuckDuckGo. "
        "Only enable when you need current information.",
    )

    if use_web_search:
        st.info(
            "💡 Web search is **ON**. The agent can access the internet "
            "when it determines online info would help."
        )

    use_rag = st.toggle(
        "📄 Enable RAG (Document Q&A)",
        value=False,
        help="Enable querying your uploaded PDF documents.",
    )

    use_graph = st.toggle(
        "🔗 Enable Knowledge Graph",
        value=False,
        disabled=not neo4j_connected,
        help="Query the Neo4j knowledge graph for cross-paper relationships. "
        "Requires Neo4j to be running.",
    )

    if use_graph and neo4j_connected:
        st.info(
            "💡 Knowledge graph is **ON**. The agent can explore "
            "relationships between your papers and concepts."
        )

    # ----- Document Management (when RAG is on) -----
    collection_name = None
    paper_filter = None
    document_id_filter: str | None = None

    if use_rag:
        st.divider()
        st.subheader("📁 Documents")

        store = get_vector_store()
        existing_collections = store.list_collections()

        # Collection selection / creation
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

        # Apply persisted collection selection from session state
        if "selected_collection" in st.session_state:
            collection_name = st.session_state.selected_collection

        # ----- Paper Filter Dropdown -----
        if collection_name:
            papers_in_collection = store.list_papers(collection_name)
            if papers_in_collection:
                st.markdown("---")

                # Dropdown is keyed by document_id; label shows title + domain
                doc_id_options: list[str | None] = [None] + [
                    p["document_id"] for p in papers_in_collection
                ]

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
                    paper_filter = selected_meta["title"]  # kept for UI chips
                    st.caption(
                        f"🔍 RAG scoped to: **{paper_filter}** · "
                        f"`{selected_meta['domain']}` · `{selected_meta['chunk_count']} chunks`"
                    )
                else:
                    # Show domain breakdown
                    domains = {}
                    for p in papers_in_collection:
                        d = p["domain"]
                        domains[d] = domains.get(d, 0) + 1
                    domain_summary = ", ".join(
                        f"{d} ({c})" for d, c in sorted(domains.items())
                    )
                    st.caption(
                        f"🔍 Searching across {len(papers_in_collection)} paper(s) · {domain_summary}"
                    )

        # PDF Upload
        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("🔄 Process & Index Documents", use_container_width=True):
            target_collection = collection_name or "default"

            # Save uploaded files to disk under content-addressed paths
            pdf_dir = os.path.join(".", "data", "pdfs")
            ingest_id = generate_ingest_id()
            file_paths = []
            for file in uploaded_files:
                file_bytes = file.getbuffer()
                identity = build_identity(
                    file_bytes=bytes(file_bytes),
                    original_filename=file.name,
                    ingest_id=ingest_id,
                )
                file_path = save_upload(
                    file_bytes=bytes(file_bytes),
                    identity=identity,
                    base_dir=pdf_dir,
                )
                file_paths.append(file_path)

            # Wire Streamlit progress bar into the service callback
            progress = st.progress(0, text="Initializing...")

            def _on_progress(current: int, total: int, message: str):
                progress.progress(
                    current / total if total else 1.0,
                    text=message,
                )

            # Run ingestion through the service
            processor = get_document_processor()
            svc = IngestionService(
                processor=processor,
                store=store,
            )
            ing_result = svc.ingest_files(
                file_paths=file_paths,
                collection_name=target_collection,
                model=model,
                on_progress=_on_progress,
            )

            # Report per-file outcomes
            for fr in ing_result.file_results:
                if fr.success:
                    summary_label = " + summary" if fr.has_summary else ""
                    st.success(
                        f"✅ RAG: **{fr.paper_title}** → "
                        f"{fr.content_chunks} chunks{summary_label}"
                    )
                else:
                    st.error(f"❌ RAG: {fr.file_name}: {fr.error}")

            progress.progress(1.0, text="Done!")
            st.rerun()

        # Delete collection
        if existing_collections and collection_name in existing_collections:
            if st.button("🗑️ Delete Collection", use_container_width=True):
                store.delete_collection(collection_name)
                st.success(f"Deleted `{collection_name}`")
                st.rerun()

    # ----- Session Actions -----
    st.divider()
    st.subheader("Session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            from src.chat import ChatRecord

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

    # Save full chat to disk (explicit action)
    if st.session_state.messages:
        if st.button("Save Chat to History", use_container_width=True):
            from src.chat import ChatMetadata, ChatStore

            chat_store = ChatStore()
            try:
                saved_path = chat_store.save(
                    st.session_state.messages,
                    metadata=ChatMetadata(
                        model=model,
                        temperature=temperature,
                        use_web_search=use_web_search,
                        use_rag=use_rag,
                        use_graph=use_graph and neo4j_connected,
                        collection_name=collection_name,
                        paper_filter=paper_filter,
                        document_id=document_id_filter,
                    ),
                )
                st.success(f"Saved to chat_history/{os.path.basename(saved_path)}")
            except ValueError as exc:
                st.warning(str(exc))
    else:
        st.caption("Start a conversation to enable saving.")


# --- Main Chat Interface ---
render_status_bar(
    model,
    use_web_search=use_web_search,
    collection_name=collection_name if use_rag else None,
    paper_filter=paper_filter,
    use_graph=use_graph,
)

if not st.session_state.messages:
    render_hero()

# ----- Display Chat History -----
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----- Chat Input -----
if user_input := st.chat_input("Ask about any concept in DE, DS, or AI..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build message history for LangChain
    lang_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lang_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lang_messages.append(AIMessage(content=msg["content"]))

    # Build and invoke the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = _get_compiled_graph(
                    use_web_search=use_web_search,
                    use_rag=use_rag,
                    use_graph=use_graph and neo4j_connected,
                    collection_name=collection_name,
                    model=model,
                    temperature=temperature,
                    document_id=document_id_filter,
                    _neo4j_conn=get_neo4j_connection() if (use_graph and neo4j_connected) else None,
                    _vector_store=get_vector_store() if use_rag else None,
                )

                result = graph.invoke({"messages": lang_messages})
                ai_msg = result["messages"][-1]
                response = ai_msg.content or EMPTY_RESPONSE_MARKDOWN

            except Exception as e:
                response = format_chat_error(e, model)

        # Render the response
        st.markdown(response)

    # Save to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
