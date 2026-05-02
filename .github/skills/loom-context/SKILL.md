---
name: loom-context
description: 'Loom repository context and change-propagation workflow. Use whenever working in this repo on Loom — the local, open-source knowledge graph + AI tutor built on Streamlit, LangGraph, LangChain, Ollama, ChromaDB, and Neo4j. Use for: making code changes, adding features/tools, refactoring, debugging RAG/KG/agent flow, modifying ingestion or PDF pipeline, editing Streamlit UI, tweaking prompts, updating settings. Loads architecture, module map, conventions, and invariants so background context does not need to be re-supplied. After edits, this skill also evaluates whether SKILL.md, README.md, and/or Architectural Diagram.mmd need to be updated, and updates only the ones that are actually affected.'
---

# Loom Repository Context & Change-Propagation

## When to Use

Trigger this skill on **any** task in the Loom repo, including:

- Adding/modifying agent tools (`src/tools/`)
- Changing the LangGraph agent (`src/graph/`)
- Editing RAG ingestion, chunking, or vector store (`src/rag/`, `src/ingestion/`)
- Touching the Neo4j knowledge graph layer (`src/graph_db/`)
- Modifying LLM provider or paper profiling (`src/llm/`)
- Streamlit UI changes (`app.py`, `pages/`, `src/ui/`)
- Settings / `.env` / config changes (`config/settings.py`)
- Tests (`tests/`), docker-compose, requirements

If the task is *purely* a one-line bug fix or doc typo, you may skip the propagation step but still load the context.

## Repository Snapshot

**Loom** is a local, fully-offline-capable knowledge graph & AI tutor for research papers. No API keys required for the core path. Everything runs on the user's machine.

**Stack**: Streamlit · LangChain · LangGraph · Ollama · ChromaDB · Neo4j · Docling · Pyvis · Pydantic Settings · pytest

**Default models** (overridable via `.env`):
- Chat: `granite4:tiny-h` (Ollama)
- Embeddings: `qwen3-embedding:4b` (Ollama)

**Entry points**:
- `app.py` — Streamlit chat page (main entry: `streamlit run app.py`)
- `pages/2_Knowledge_Graph.py` — KG explorer (Pyvis viz, paper/concept browsers)
- `pages/3_Chat_History.py` — saved chat viewer

## Architecture Map (must-know)

```
Streamlit UI (app.py + pages/)
        │
        ▼
LangGraph ReAct agent (src/graph/{state,nodes,builder}.py)
        │
        ├──► query_documents      (src/tools/rag_tool.py)        ──► ChromaDB (src/rag/store.py)
        └──► query_knowledge_graph (src/tools/safe_graph_tool.py) ──► GraphQueryService (src/graph_db/service.py)
                                                                      └──► KnowledgeGraphQueries (queries.py)
                                                                            └──► Neo4jConnection (connection.py)

Ingestion: PDF ─► IngestionService (src/ingestion/service.py)
                  ├─► DocumentIdentity (ingestion/identity.py)   # filename → document_id, MD5 → storage path
                  ├─► DocumentProcessor (rag/processor.py)       # Docling → Markdown → HybridChunker chunks
                  │     └─► extract_paper_profile (llm/paper_profile.py)  # title/domain/concepts/methods/findings
                  ├─► VectorStoreManager (rag/store.py)          # idempotent upsert by deterministic chunk_id
                  └─► KnowledgeGraphWriter (graph_db/writer.py)  # write profile + link findings + link concepts
                        ├─► write_paper_profile()                # create Paper, Concept, Method, Finding nodes
                        ├─► link_findings()                      # LLM detects SUPPORTS/CONTRADICTS/EXTENDS between findings
                        └─► link_concepts()                      # LLM detects RELATED_TO/SUBTOPIC_OF/EXTENDS between concepts

LLM: src/llm/provider.py — cached get_llm() / get_embeddings() factories
Domain: src/domain/{paper,taxonomy}.py — Paper model + canonical domain taxonomy (shared by RAG + KG)
Eval: src/evaluation/judge.py — LLM-as-Judge (context relevance, faithfulness, answer relevance)
Chat persistence: src/chat/store.py — JSON files under data/chat_history/
UI helpers: src/ui/{bootstrap,chrome,confirm}.py
Utils: src/utils/json_parser.py — robust LLM-JSON extractor (handles code-fenced output)
```

## Critical Invariants (do not break)

1. **No raw Cypher from agent or UI.** All graph access goes through `GraphQueryService` (9 named intents) or `KnowledgeGraphQueries` (parameterised). Adding a new graph capability = add an intent + a parameterised query, never accept Cypher strings.
2. **Identity is filename-derived, storage is content-addressed.** `document_id` = sanitised filename stem (stable across re-uploads). MD5 hash → `data/pdfs/<aa>/<hash>/...` path. `chunk_id` is deterministic so re-ingestion **upserts**, never duplicates.
3. **Concepts are domain-scoped.** `concept_key = normalized_domain:name` to prevent cross-domain collisions in Neo4j.
4. **Single LLM call for paper profiling.** `extract_paper_profile()` in `llm/paper_profile.py` returns a `PaperProfile` (title, domain, summary, concepts, methods, findings) in one shot. Don't fan it out into per-field calls.
5. **Settings come from `config/settings.py` (Pydantic).** Never read env vars directly in modules — go through the settings singleton.
6. **Feature flags gate tool assembly** in `src/graph/builder.py`. RAG/KG toggles in the sidebar must remain wired through the builder so the agent only sees enabled tools.
7. **Neo4j is optional.** Code paths must degrade gracefully when Neo4j is unreachable (UI shows status, agent simply lacks the KG tool).
8. **Granite tokenizer is local.** `granite_tokenizer/tokenizer.json` is used by the HybridChunker — do not require network downloads at runtime.

## Conventions

- **Python 3.10+**, type hints expected, prefer dataclasses/Pydantic for structured data.
- **No new top-level files** unless necessary — extend existing modules.
- **Tests** live in `tests/` and run via `pytest` (config in `pytest.ini`).
- **Imports**: absolute from `src.*` / `config.*`.
- **Streamlit caching**: service instances cached via `src/ui/bootstrap.py` — reuse it, don't re-cache ad hoc.
- **Markdown output style**: chat responses are formatted for Substack/blog paste — preserve heading levels and code-fence languages.
- **Destructive UI actions** (clear chats, drop graph) must use `src/ui/confirm.py`.

## Workflow

### Step 1 — Load context (always)
Read this file. Skim the relevant module(s) for the task. Do **not** ask the user to re-explain what Loom is or how the agent/RAG/KG fit together.

### Step 2 — Implement the change
Follow the invariants and conventions above. Run/extend tests in `tests/` when behaviour changes.

### Step 3 — Decide which docs need updating

After the implementation, evaluate each of the three doc artefacts independently. Update **only** those that are actually affected. Do not update for cosmetic-only edits.

| Artefact | Update when… | Skip when… |
|---|---|---|
| **`.github/skills/loom-context/SKILL.md`** (this file) | A new module/folder is added under `src/`; an invariant changes; a new entry point or tool category appears; default model/stack swaps; a new pipeline stage is introduced. | Internal refactor inside an existing module; bug fixes; renaming local symbols; test-only changes. |
| **`README.md`** | User-visible behaviour changes: new feature, new sidebar toggle, new page, new env var, changed default model, new prerequisite, new docker-compose service, new install step, changed Quick Start, new tool the user can invoke, new export format. | Pure internals with no user impact; refactors; non-public helper changes. |
| **`Architectural Diagram.mmd`** | A new node appears in the architecture (new module, service, external dep) or a connection between existing nodes is added/removed/redirected (e.g. agent gains/loses a tool, RAG gains a new backend, ingestion calls a new component). | Logic changes inside an already-drawn node that don't change its connections or responsibilities summary. |

For each artefact you update:
- Make the **minimum** change required to reflect reality.
- Keep tone, style, emoji usage, and section ordering consistent with the existing file.
- For the Mermaid diagram, validate it renders (use the mermaid validator if available) before saving.

### Step 4 — Report
Briefly state:
1. What changed in code.
2. Which of {SKILL.md, README.md, Architectural Diagram.mmd} were updated and why — and which were intentionally **not** updated and why.

## Quick Commands

```powershell
# Run app
streamlit run app.py

# Run tests
pytest

# Start Neo4j
docker-compose up -d
```

## Anti-patterns to flag

- Reading `os.environ` directly in feature modules → use `config.settings`.
- Building Cypher strings in tools or pages → add an intent in `GraphQueryService`.
- Calling Ollama directly → use `src/llm/provider.py` factories.
- Re-implementing chunk IDs or document IDs → reuse `src/ingestion/identity.py` + `src/rag/processor.py`.
- Bypassing `IngestionService` and writing to ChromaDB ad hoc → loses dedup guarantees.
