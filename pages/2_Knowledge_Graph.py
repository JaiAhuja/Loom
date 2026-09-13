import streamlit as st
from pyvis.network import Network

from src.domain import Paper
from src.domain.paper import merge_paper_sources
from src.graph_db import KnowledgeGraphQueries, get_neo4j_connection
from src.rag import VectorStoreManager

# --- Page Configuration ---
st.set_page_config(
    page_title="Knowledge Graph - Loom",
    page_icon="",
    layout="wide",
)

st.title("Knowledge Graph Explorer")
st.caption("Visualize and explore relationships between your research papers")

# --- Display persistent toggle state from main page ---
rag_status = "✅ RAG Enabled" if st.session_state.get("use_rag_persistent", False) else "⚫ RAG Disabled"
kg_status = "✅ KG Enabled" if st.session_state.get("use_graph_persistent", False) else "⚫ KG Disabled"
st.info(f"**Toggle Status:** {rag_status} · {kg_status} (configure on main page)")


# --- Neo4j Connection (shared singleton — same driver used by app.py) ---
try:
    conn = get_neo4j_connection()
    if not conn.is_connected():
        st.error("❌ Neo4j is not connected.")
        st.info(
            "Start Neo4j with:\n"
            "Open Neo4j Desktop and start your database."
        )
        st.stop()
except Exception as e:
    st.error(f"❌ Cannot connect to Neo4j: {e}")
    st.stop()

queries = KnowledgeGraphQueries(conn)

# --- Graph Statistics ---
try:
    stats = queries.get_graph_stats()
except Exception as e:
    error_msg = str(e).lower()
    if "database unavailable" in error_msg or "database `neo4j` is currently unavailable" in error_msg:
        st.error("❌ Neo4j database instance is not available.")
        st.info(
            "**To fix this:**\n"
            "1. Open Neo4j Desktop\n"
            "2. Find your database instance and click **Start** if it's stopped\n"
            "3. Wait for it to show \"Started\" status\n"
            "4. Refresh this page"
        )
        st.stop()
    else:
        st.error(f"❌ Error querying Neo4j: {e}")
        st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📄 Papers", stats.get("papers", 0))
col2.metric("💡 Concepts", stats.get("concepts", 0))
col3.metric("⚙️ Methods", stats.get("methods", 0))
col4.metric("🔬 Findings", stats.get("findings", 0))
col5.metric("🧩 Details", stats.get("details", 0))

if stats.get("papers", 0) == 0:
    st.info(
        "No papers in the knowledge graph yet. "
        "Upload and process PDFs from the main chat page — "
        "papers are automatically indexed into the knowledge graph when Neo4j is running."
    )
    st.stop()


# --- Unified Paper Listing (RAG ∪ KG, keyed by document_id) ---
with st.expander("📚 All papers across stores (RAG ∪ KG)", expanded=False):
    try:
        vstore = VectorStoreManager()
        rag_papers_all: list[dict] = []
        for col_name in vstore.list_collections():
            for p in vstore.list_papers(col_name):
                rag_papers_all.append({
                    "document_id": p["document_id"],
                    "title": p["title"],
                    "domain": p["domain"],
                    "chunk_count": p["chunk_count"],
                })
    except Exception:
        rag_papers_all = []

    _kg_rows = queries.get_all_papers() or []
    kg_papers = [
        {
            "document_id": p.get("document_id"),
            "title": p.get("title"),
            "domain": p.get("domain"),
            "concept_count": p.get("concept_count", 0),
        }
        for p in _kg_rows if p.get("document_id")
    ]

    unified: list[Paper] = merge_paper_sources(rag_papers=rag_papers_all, kg_papers=kg_papers)
    if unified:
        st.dataframe(
            [
                {
                    "Title": p.title,
                    "Domain": p.domain,
                    "Stores": p.status_badge,
                    "RAG chunks": p.chunk_count,
                    "KG concepts": p.concept_count,
                    "document_id": p.document_id,
                }
                for p in sorted(unified, key=lambda x: x.title.lower())
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No papers indexed yet.")

# --- Tabs ---
_tab_labels = [
    "🕸️ Graph View", "📄 Papers", "💡 Concepts", "🔗 Relationships",
]

_tabs = st.tabs(_tab_labels)
tab_graph, tab_papers, tab_concepts, tab_relations = _tabs

# --- Tab 1: Interactive Graph Visualization ---
with tab_graph:
    st.subheader("Interactive Research Graph")
    st.caption("Papers (large nodes) connected to concepts, methods, findings, and typed paper-detail nodes.")

    # ---- Filters (domains / papers / relationship types) ----
    with st.expander("🔧 Filters", expanded=False):
        _all_papers_for_filter = queries.get_all_papers()
        _all_concepts_for_filter = queries.get_all_concepts()
        _all_domains = sorted({
            c.get("domain") for c in _all_concepts_for_filter if c.get("domain")
        })

        f_col1, f_col2, f_col3 = st.columns(3)
        selected_domains = f_col1.multiselect(
            "Domains",
            options=_all_domains,
            default=[],
            help="Limit concepts (and their papers) to these domains. Leave empty to include all.",
        )
        selected_paper_docs = f_col2.multiselect(
            "Papers",
            options=[p["document_id"] for p in _all_papers_for_filter],
            format_func=lambda d: next(
                (p["title"] for p in _all_papers_for_filter if p["document_id"] == d),
                d,
            ),
            default=[],
        )
        selected_rel_types = f_col3.multiselect(
            "Relationship types",
            options=[
                "DISCUSSES", "HAS_DETAIL", "USES_METHOD", "HAS_FINDING",
                "RELATED_TO", "SUPPORTS", "CONTRADICTS", "EXTENDS",
            ],
            default=[],
            help="Leave empty to include all.",
        )

    graph_data = queries.get_graph_for_visualization(
        limit=150,
        domains=selected_domains or None,
        document_ids=selected_paper_docs or None,
        rel_types=selected_rel_types or None,
    )

    if not graph_data["nodes"]:
        st.info("No graph data to display yet.")
    else:
        # Build pyvis network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.01,
        )

        # Color map
        colors = {
            "paper": "#FF6B6B",
            "concept": "#4ECDC4",
            "method": "#FFE66D",
            "finding": "#A8E6CF",
            "detail": "#B388FF",
        }

        for node in graph_data["nodes"]:
            net.add_node(
                node["id"],
                label=node["label"],
                title=node.get("title", node["label"]),
                color=colors.get(node.get("group"), "#888888"),
                size=node.get("size", 15),
            )

        for edge in graph_data["edges"]:
            net.add_edge(
                edge["from"],
                edge["to"],
                label=edge.get("label", ""),
                color=edge.get("color", "#666666"),
                width=edge.get("width", 1),
                dashes=edge.get("dashes", False),
            )

        # Render to HTML and display
        html = net.generate_html()
        st.components.v1.html(html, height=620, scrolling=True)

        # Legend
        st.markdown(
            """
            **Legend:** 🔴 Paper · 🟢 Concept · 🟡 Method ·
            🟣 Detail · Green edge = core topic · Grey edge = mentioned ·
            Blue dashed = related concepts · Red = contradicts · Orange = extends
            """
        )

        # ---- Interactive Node Details ----
        st.divider()
        st.subheader("📊 Node Details Inspector")
        st.caption("Select a paper node from the graph above to view its full details (concepts, methods, findings)")
        
        # Dropdown to select a paper node for detailed view
        paper_nodes = [n for n in graph_data["nodes"] if n.get("group") == "paper"]
        if paper_nodes:
            def _format_paper_option(node):
                return f"{node.get('label', 'Unknown')} ({node.get('domain', 'N/A')})"
            
            selected_node = st.selectbox(
                "View Details",
                options=paper_nodes,
                format_func=_format_paper_option,
                key="node_details_selector",
            )
            
            if selected_node:
                doc_id = selected_node.get("doc_id")
                if doc_id:
                    # Fetch full details for this paper
                    details = queries.get_paper_details(document_id=doc_id)
                    
                    # Display header with metadata
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("💡 Concepts", len(details.get("concepts", [])))
                    col2.metric("⚙️ Methods", len(details.get("methods", [])))
                    col3.metric("🔬 Findings", len(details.get("findings", [])))
                    col4.metric("🧩 Details", len(details.get("details", [])))
                    
                    # Display concepts, methods, findings in expandable sections
                    det_col1, det_col2 = st.columns(2)
                    
                    with det_col1:
                        with st.expander("🧩 Extracted Details", expanded=True):
                            detail_rows = details.get("details", [])
                            if detail_rows:
                                for d in detail_rows:
                                    label = (d.get("edge_label") or d.get("category") or "DETAIL").replace("_", " ")
                                    st.markdown(f"**{label.title()}**")
                                    st.write(d.get("text", ""))
                                    if d.get("evidence"):
                                        st.caption(f"Evidence: {d['evidence']}")
                                    st.divider()
                            else:
                                st.caption("No paper-detail nodes found for this paper")

                        with st.expander("💡 Concepts", expanded=True):
                            concepts = details.get("concepts", [])
                            if concepts:
                                for c in concepts:
                                    depth_badge = "🟢 Core" if c["depth"] == "core" else "⚪ Mentioned"
                                    st.markdown(f"**{c['name']}** ({depth_badge})")
                                    if c.get("description"):
                                        st.caption(f"{c['description']}")
                                    if c.get("domain"):
                                        st.caption(f"Domain: `{c['domain']}`")
                                    st.divider()
                            else:
                                st.caption("No concepts found for this paper")
                        
                        with st.expander("⚙️ Methods", expanded=False):
                            methods = details.get("methods", [])
                            if methods:
                                for m in methods:
                                    st.markdown(f"**{m['name']}**")
                                    if m.get("description"):
                                        st.caption(f"{m['description']}")
                                    st.divider()
                            else:
                                st.caption("No methods found for this paper")
                    
                    with det_col2:
                        with st.expander("🔬 Findings", expanded=False):
                            findings = details.get("findings", [])
                            if findings:
                                for f in findings:
                                    evidence_icon = {
                                        "empirical": "📊",
                                        "theoretical": "📐",
                                        "survey": "📋"
                                    }.get(f.get("evidence_type", ""), "📝")
                                    st.markdown(f"{evidence_icon} {f['claim']}")
                                    if f.get("evidence_type"):
                                        st.caption(f"Type: `{f['evidence_type']}`")
                                    st.divider()
                            else:
                                st.caption("No findings found for this paper")

# --- Tab 2: Paper Explorer ---
with tab_papers:
    st.subheader("Paper Explorer")

    papers = queries.get_all_papers()
    if not papers:
        st.info("No papers indexed yet.")
    else:
        # Selector keyed by document_id (stable), labelled by title
        doc_ids = [p["document_id"] for p in papers]

        def _paper_label(doc_id):
            meta = next((p for p in papers if p["document_id"] == doc_id), None)
            if not meta:
                return doc_id
            return f"{meta.get('title') or '(untitled)'}  [{meta.get('domain', '')}]"

        selected_doc_id = st.selectbox(
            "Select a paper",
            options=doc_ids,
            format_func=_paper_label,
        )

        if selected_doc_id:
            # Paper metadata
            paper_meta = next(
                (p for p in papers if p["document_id"] == selected_doc_id), {},
            )
            cols = st.columns(3)
            cols[0].markdown(f"**Domain:** {paper_meta.get('domain', 'N/A')}")
            cols[1].markdown(f"**Year:** {paper_meta.get('year', 'N/A')}")
            cols[2].markdown(f"**Concepts:** {paper_meta.get('concept_count', 0)}")

            if paper_meta.get("authors"):
                st.markdown(f"**Authors:** {paper_meta['authors']}")
            if paper_meta.get("summary"):
                st.markdown(f"**Summary:** {paper_meta['summary']}")
            st.caption(f"`{selected_doc_id}`")

            st.divider()

            # Paper details (keyed by canonical document_id)
            details = queries.get_paper_details(document_id=selected_doc_id)

            det_col1, det_col2 = st.columns(2)

            with det_col1:
                st.markdown("### 🧩 Extracted Details")
                grouped_details = {}
                for d in details.get("details", []):
                    grouped_details.setdefault(d.get("category") or "detail", []).append(d)
                for category, rows in grouped_details.items():
                    st.markdown(f"**{category.replace('_', ' ').title()}**")
                    for d in rows:
                        st.markdown(f"- {d.get('text', '')}")
                        if d.get("evidence"):
                            st.caption(f"  Evidence: {d['evidence']}")
                if not grouped_details:
                    st.caption("No extracted detail nodes found.")

                st.markdown("### 💡 Concepts")
                for c in details["concepts"]:
                    depth_badge = "🟢 Core" if c["depth"] == "core" else "⚪ Mentions"
                    st.markdown(f"- **{c['name']}** ({depth_badge})")
                    if c.get("description"):
                        st.caption(f"  {c['description']}")

                st.markdown("### ⚙️ Methods")
                for m in details["methods"]:
                    st.markdown(f"- **{m['name']}**")
                    if m.get("description"):
                        st.caption(f"  {m['description']}")

            with det_col2:
                st.markdown("### 🔬 Findings")
                for f in details["findings"]:
                    evidence_icon = {"empirical": "📊", "theoretical": "📐", "survey": "📋"}.get(
                        f.get("evidence_type", ""), "📝"
                    )
                    st.markdown(f"- {evidence_icon} {f['claim']}")

            # ---- Per-paper delete (removes from KG + every Chroma collection) ----
            st.divider()
            del_col1, del_col2 = st.columns([1, 3])
            confirm_key = f"confirm_delete_{selected_doc_id}"
            with del_col1:
                if st.button("🗑️ Delete this paper", key=f"del_btn_{selected_doc_id}"):
                    st.session_state[confirm_key] = True
            if st.session_state.get(confirm_key):
                with del_col2:
                    st.warning(
                        f"Delete **{paper_meta.get('title', selected_doc_id)}** from the KG "
                        "and every RAG collection? This cannot be undone."
                    )
                    confirm_cols = st.columns(2)
                    if confirm_cols[0].button("Confirm delete", key=f"del_confirm_{selected_doc_id}"):
                        kg_stats = queries.delete_paper(selected_doc_id)
                        rag_removed = 0
                        try:
                            vstore_del = VectorStoreManager()
                            for col_name in vstore_del.list_collections():
                                rag_removed += vstore_del.delete_paper(col_name, selected_doc_id)
                        except Exception:
                            pass
                        st.session_state.pop(confirm_key, None)
                        st.success(
                            f"Deleted paper · KG removed "
                            f"{kg_stats.get('findings', 0)} findings, "
                            f"{kg_stats.get('concepts', 0)} orphan concepts, "
                            f"{kg_stats.get('methods', 0)} orphan methods · "
                            f"RAG removed {rag_removed} chunks."
                        )
                        st.rerun()
                    if confirm_cols[1].button("Cancel", key=f"del_cancel_{selected_doc_id}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()

        # Compare two papers
        st.divider()
        st.markdown("### 🔀 Compare Two Papers")
        paper_titles = [p["title"] for p in papers]
        if len(paper_titles) >= 2:
            cmp_col1, cmp_col2 = st.columns(2)
            paper_a = cmp_col1.selectbox("Paper A", paper_titles, key="cmp_a")
            paper_b = cmp_col2.selectbox(
                "Paper B",
                [t for t in paper_titles if t != paper_a],
                key="cmp_b",
            )

            if st.button("🔍 Find Shared Concepts"):
                shared = queries.get_shared_concepts(paper_a, paper_b)
                if shared:
                    st.success(f"Found {len(shared)} shared concepts!")
                    for s in shared:
                        st.markdown(f"- **{s['concept']}** ({s.get('domain', '')})")
                        if s.get("description"):
                            st.caption(f"  {s['description']}")
                else:
                    st.info("No shared concepts found between these papers.")
        else:
            st.caption("Upload at least 2 papers to compare.")

# --- Tab 3: Concept Explorer ---
with tab_concepts:
    st.subheader("Concept Explorer")

    concepts = queries.get_all_concepts()
    if not concepts:
        st.info("No concepts extracted yet.")
    else:
        # Concept table
        st.dataframe(
            [
                {
                    "Concept": c["name"],
                    "Domain": c.get("domain", ""),
                    "Papers": c.get("paper_count", 0),
                    "Description": c.get("description", "")[:100],
                }
                for c in concepts
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Concept deep-dive
        st.divider()
        concept_names = [c["name"] for c in concepts]
        selected_concept = st.selectbox("Explore a concept", concept_names)

        if selected_concept:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📄 Papers discussing this concept")
                papers_for_concept = queries.get_concept_papers(selected_concept)
                for p in papers_for_concept:
                    depth_badge = "🟢 Core" if p.get("depth") == "core" else "⚪ Mentions"
                    st.markdown(f"- **{p['title']}** ({depth_badge})")

            with col2:
                st.markdown("### 🔗 Related concepts")
                related = queries.get_related_concepts(selected_concept)
                for r in related:
                    strength = r.get("strength") or 0
                    bar = "█" * max(1, int(strength * 10))
                    st.markdown(f"- **{r['name']}** {bar} ({strength:.1f})")

            # ---- Evidence across papers for this concept ----
            st.divider()
            st.markdown("### 📚 Evidence across papers for this concept")
            ev = queries.evidence_for_concept(selected_concept)
            ev_total = sum(len(v) for v in ev.values())
            if ev_total == 0:
                st.caption(
                    "No cross-paper finding relationships linked to this concept yet."
                )
            else:
                ev_tabs = st.tabs([
                    f"✅ Supporting ({len(ev['supports'])})",
                    f"⚡ Contradicting ({len(ev['contradicts'])})",
                    f"🔄 Extensions ({len(ev['extends'])})",
                ])
                for ev_tab, bucket_key in zip(ev_tabs, ("supports", "contradicts", "extends")):
                    with ev_tab:
                        items = ev[bucket_key]
                        if not items:
                            st.caption("None found.")
                            continue
                        for item in items:
                            with st.container(border=True):
                                st.markdown(f"**{item['paper_1']}**")
                                st.markdown(f"> {item['finding_1']}")
                                st.markdown(f"**{item['paper_2']}**")
                                st.markdown(f"> {item['finding_2']}")
                                if item.get("reason"):
                                    st.caption(f"Reason: {item['reason']}")

# --- Tab 4: Cross-Paper Relationships ---
with tab_relations:
    st.subheader("Cross-Paper Relationships")
    st.caption("How findings from different papers relate to each other")

    rel_tab1, rel_tab2, rel_tab3 = st.tabs(
        ["🟥 Contradictions", "🟩 Supporting", "🟧 Extensions"]
    )

    _finding_tabs = [
        (rel_tab1, "CONTRADICTS", "⚡", "contradicting"),
        (rel_tab2, "SUPPORTS",    "✅", "supporting"),
        (rel_tab3, "EXTENDS",     "🔄", "extending"),
    ]
    for tab, rel_type, emoji, label in _finding_tabs:
        with tab:
            items = queries.get_cross_paper_findings(rel_type)
            if items:
                for item in items:
                    with st.container(border=True):
                        st.markdown(f"**Paper 1:** {item['paper_1']}")
                        st.markdown(f"> {item['finding_1']}")
                        st.markdown(f"**{emoji} {rel_type} {emoji}**")
                        st.markdown(f"**Paper 2:** {item['paper_2']}")
                        st.markdown(f"> {item['finding_2']}")
                        if item.get("reason"):
                            st.caption(f"Reason: {item['reason']}")
            else:
                st.info(f"No {label} findings detected yet.")

    # ---- F3: Conflict / Agreement matrix ----
    st.divider()
    st.markdown("### 🧮 Paper × Paper Agreement Matrix")
    st.caption(
        "Counts of supporting / contradicting / extending finding edges between each pair of papers."
    )
    matrix_rows = queries.get_conflict_matrix()
    if not matrix_rows:
        st.info("No cross-paper relationships yet.")
    else:
        matrix_data = [
            {
                "Paper A": r["paper_a"],
                "Paper B": r["paper_b"],
                "✅ Supports": r["supports"],
                "⚡ Contradicts": r["contradicts"],
                "🔄 Extends": r["extends"],
            }
            for r in sorted(
                matrix_rows,
                key=lambda x: (x["contradicts"], x["supports"], x["extends"]),
                reverse=True,
            )
        ]
        st.dataframe(matrix_data, use_container_width=True, hide_index=True)
