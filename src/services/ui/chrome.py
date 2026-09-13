"""Reusable Streamlit HTML snippets (brand, hero, feature grid, status chips)."""

from __future__ import annotations

from html import escape
import logging

import streamlit as st

logger = logging.getLogger(__name__)


def inject_stylesheet(path: str = "static/style.css") -> None:
    try:
        with open(path, encoding="utf-8") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
    except OSError:
        logger.debug("Stylesheet not available: %s", path, exc_info=True)


def render_brand() -> None:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="brand-icon">🪡</div>'
        '<div class="brand-name">Loom</div>'
        '<div class="brand-tag">Weaving threads of knowledge together</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_status_bar(
    model: str,
    *,
    collection_name: str | None = None,
    paper_filter: str | None = None,
    use_graph: bool = False,
) -> None:
    chips = [f'<span class="status-chip">🤖 {escape(model)}</span>']
    if collection_name:
        chips.append(f'<span class="status-chip">📄 {escape(collection_name)}</span>')
        if paper_filter:
            chips.append(f'<span class="status-chip">📑 {escape(paper_filter[:30])}</span>')
    if use_graph:
        chips.append('<span class="status-chip">🔗 Graph</span>')
    st.markdown(f'<div class="status-bar">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-icon">🪡</div>'
        '<div class="hero-title">Loom</div>'
        '<div class="hero-subtitle">Weaving threads of knowledge together</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="feature-grid">'
        '  <div class="feature-card"><div class="card-icon">🧠</div>'
        '    <div class="card-title">Deep Explanations</div>'
        '    <div class="card-desc">Concept breakdowns with code examples, analogies & '
        "best practices</div></div>"
        '  <div class="feature-card"><div class="card-icon">📄</div>'
        '    <div class="card-title">RAG over PDFs</div>'
        '    <div class="card-desc">Upload papers & notes — ask questions grounded in your '
        "own material</div></div>"
        '  <div class="feature-card"><div class="card-icon">🔗</div>'
        '    <div class="card-title">Knowledge Graph</div>'
        '    <div class="card-desc">Discover cross-paper relationships, shared concepts & '
        "contradictions</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="quick-start">'
        "<p>💡 Try asking something like:</p>"
        '<div class="pill-row">'
        '  <span class="pill">Explain MapReduce vs Spark</span>'
        '  <span class="pill">What is the attention mechanism?</span>'
        '  <span class="pill">Compare SQL & NoSQL databases</span>'
        '  <span class="pill">How does RAG work?</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def format_chat_error(exc: Exception, model: str) -> str:
    """Turn a chat-invocation exception into a user-facing markdown message."""
    msg = str(exc)
    low = msg.lower()
    if "connection" in low or "refused" in low:
        return (
            "**Connection Error:** Could not reach Ollama.\n\n"
            f"Make sure Ollama is running (`ollama serve`) and the model "
            f"`{model}` is pulled (`ollama pull {model}`)."
        )
    if "not found" in low:
        return f"**Model Error:** Model `{model}` not found.\n\nPull it with: `ollama pull {model}`"
    return (
        f"**Error:** {msg}\n\n"
        f"Check that Ollama is running and the model `{model}` supports "
        "tool calling (if tools are enabled)."
    )


EMPTY_RESPONSE_MARKDOWN = "*(The model returned an empty response. Try rephrasing your question.)*"
