# 🪡 Loom

> *Weaving threads of knowledge together*

A **local, fully open-source knowledge graph & AI tutor** for researchers, students, and practitioners reading papers and articles on Data Engineering, Data Science, and Artificial Intelligence. Built with Streamlit, LangChain, LangGraph, and powered by Ollama — no API keys required for core functionality.

> Upload papers and articles you're reading, ask questions, and let Loom surface the hidden connections, shared concepts, and contradictions between them — all running locally on your machine.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Local LLM** | Powered by [Ollama](https://ollama.com/) — 100% free, no API keys, runs on your hardware |
| **LangGraph Orchestration** | ReAct-style agent with dynamic tool routing via [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **RAG Pipeline** | Upload PDFs → [Docling](https://github.com/DS4SD/docling) extraction → [ChromaDB](https://www.trychroma.com/) storage → Smart retrieval |
| **Web Search** | Optional [DuckDuckGo](https://duckduckgo.com/) integration — explicit opt-in, no API key needed |
| **Beautiful Output** | Publication-ready Markdown formatted for [Substack](https://substack.com/) / blogs |
| **Knowledge Graph** | [Neo4j](https://neo4j.com/) knowledge graph — auto-extracted entities & relationships across papers |
| **Graph Explorer** | Interactive [Pyvis](https://pyvis.readthedocs.io/) visualization + paper comparison + safe intent-based queries |
| **LangSmith Tracing** | Optional [LangSmith](https://smith.langchain.com/) observability and debugging |
| **Chat History** | Save conversations to disk and export full chats as Markdown |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │   Sidebar    │  │  Page 1: Chat Interface                  │ │
│  │ • Model cfg  │  │  User ──► Agent ──► Markdown Response    │ │
│  │ • Toggles    │  │                                          │ │
│  │ • PDF Upload │  │  Page 2: Knowledge Graph Explorer        │ │
│  │ • Collections│  │  Papers ◄─► Concepts ◄─► Findings        │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LangGraph     │
                    │   Agent Loop    │◄──── LangSmith Tracing
                    │  (ReAct Style)  │
                    └───┬────┬───┬────┘
                        │    │   │
              ┌─────────▼┐ ┌▼───▼───────┐ ┌──────────────┐
              │Web Search│ │ RAG Query  │ │ Graph Query  │
              │DuckDuckGo│ │ ChromaDB   │ │ Neo4j        │
              │(opt-in)  │ │ + Docling  │ │ (opt-in)     │
              └──────────┘ └────────────┘ └──────────────┘
                        │        │              │
                    ┌───▼────────▼──────────────▼─┐
                    │        Ollama LLM            │
                    │       (Local Model)          │
                    └──────────────────────────────┘
```

### Data Flow

1. **User asks a question** in the Streamlit chat
2. **LangGraph agent** receives the query with conversation history
3. **Agent decides** whether to use tools (web search, RAG) or answer directly
4. If **RAG is enabled** → queries ChromaDB for relevant document chunks
5. If **Web Search is enabled** → searches DuckDuckGo for current info
6. If **Knowledge Graph is enabled** → queries Neo4j via safe, intent-based tool (no raw Cypher)
7. **LLM generates** a comprehensive, Markdown-formatted response
8. **Response is displayed** in Streamlit and can be saved or exported

### Ingestion & Identity

When PDFs are uploaded they pass through a deterministic identity pipeline:

1. **MD5 hash** computed from file bytes → stable `document_id` (`md5:<hex>`)
2. **Unique `ingest_id`** assigned per upload session
3. **Deterministic `chunk_id`** generated per text chunk → idempotent vector writes (upsert)
4. Files stored under a hash-based path in `data/pdfs/`
5. Re-uploading the same file overwrites existing chunks instead of duplicating them

### Knowledge Graph Pipeline

1. **PDF uploaded** via Streamlit sidebar
2. **Identity assigned** — MD5 hash → `document_id` (`md5:<hex>`), unique `ingest_id`
3. **Docling** converts PDF to Markdown (shared with RAG pipeline)
4. **LLM extracts** structured entities: papers, concepts, methods, findings, authors
5. **Entity resolution** — LLM fuzzy-matches against existing graph entities to avoid duplicates
6. **Paper node keyed by `document_id`** — distinct PDFs with the same title stay separate
7. **Concepts scoped by domain** — `concept_key` = `normalized_domain:name` prevents cross-domain collisions
8. **Neo4j stores** nodes and relationships (DISCUSSES, SUPPORTS, CONTRADICTS, EXTENDS, etc.)
9. **Cross-paper detection** — automatic discovery of supporting, contradicting, and extending findings

### Graph Query Safety

The agent and UI never generate or execute raw Cypher queries. Instead:

- **Agent tool**: `query_knowledge_graph` accepts one of 9 pre-defined intents
  (`graph_stats`, `paper_list`, `paper_details`, `shared_concepts`, `concept_papers`,
  `related_concepts`, `contradiction_list`, `support_list`, `extension_list`) plus
  typed parameters. Unknown intents fail safely.
- **Graph Explorer page**: uses `KnowledgeGraphQueries` — a class of pre-built,
  parameterised Cypher queries. No raw Cypher input is exposed to users.

---

## 📋 Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running
- **[Docker](https://www.docker.com/)** (for Neo4j knowledge graph — optional)
- **Git** (for cloning)
- ~8GB RAM minimum (depends on model size)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loom.git
cd loom
```

### 2. Install Ollama & Pull Models

Install Ollama from [ollama.com](https://ollama.com/), then pull the required models:

```bash
# Main chat model (default — change via OLLAMA_MODEL in .env)
ollama pull gemma4:e4b        # Default model

# Embedding model (required for RAG — change via OLLAMA_EMBEDDING_MODEL in .env)
ollama pull qwen3-embedding:4b  # Required only if using RAG
```

### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `docling` package is large (~2GB) due to ML models. If you don't plan to use RAG immediately, you can skip it and install later:
> ```bash
> pip install -r requirements.txt --exclude docling
> # Install later when needed:
> pip install docling
> ```

### 5. Configure Environment

Create a `.env` file in the project root to override defaults:

```bash
# .env (all optional — sensible defaults are built-in)
OLLAMA_MODEL=gemma4:e4b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

See the Configuration section below for the full list of settings.

### 6. Start Neo4j (Optional — for Knowledge Graph)

```bash
docker-compose up -d
```

This starts a Neo4j Community instance on `neo4j://127.0.0.1:7687` with the Neo4j Browser at `http://localhost:7474`. Default credentials: `neo4j` / `Loom-Weave-Threads`.

> ⚠️ **Change the default password before running on anything other than a private dev machine.** The bundled `docker-compose.yml` ships with `Loom-Weave-Threads` hard-coded for convenience — that value is public in this repo and **must not** be used on shared, networked, or cloud hosts. Set `NEO4J_PASSWORD` in a local `.env` file (picked up by both `docker-compose.yml` and the app's `.env`) and re-run `docker-compose up -d`.

> **Alternative:** Use [Neo4j Desktop](https://neo4j.com/download/) and create a local database. Update `.env` with your connection details.

### 7. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
loom/
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Neo4j Docker setup
├── app.py                    # Streamlit application entry point
│
├── pages/
│   ├── 2_Knowledge_Graph.py     # Knowledge Graph explorer page
│   └── 3_Chat_History.py        # Saved chat history viewer
│
├── config/
│   ├── __init__.py
│   └── settings.py           # Pydantic settings (loads .env)
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── identity.py       # Document identity (hash, document_id, ingest_id)
│   │   └── service.py        # Ingestion orchestration (PDF → RAG + KG)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py       # Ollama LLM & embeddings factory
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py          # LangGraph state schema
│   │   ├── nodes.py          # Agent node, routing, system prompts
│   │   └── builder.py        # Graph builder (assembles the agent)
│   │
│   ├── graph_db/
│   │   ├── __init__.py
│   │   ├── connection.py     # Neo4j connection manager
│   │   ├── schema.py         # Graph schema (node/rel types, constraints)
│   │   ├── queries.py        # Predefined graph queries + visualization
│   │   └── service.py        # Typed intent-based graph query service
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── processor.py      # Docling PDF → chunks pipeline
│   │   └── store.py          # ChromaDB vector store manager (upsert)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_search.py     # DuckDuckGo search tool
│   │   ├── rag_tool.py       # Document query tool factory
│   │   └── safe_graph_tool.py # Intent-based graph query tool
│   │
│   └── utils/
│       ├── __init__.py
│       ├── json_parser.py       # Shared LLM JSON response parser
│       └── markdown_handler.py  # Save/load Markdown files
│
├── data/
│   ├── pdfs/                 # Uploaded PDF storage
│   └── chroma_db/            # ChromaDB persistent storage
│
└── outputs/                  # Saved Markdown responses
```

---

## ⚙️ Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:e4b` | Default chat model |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model (for RAG) |
| `OLLAMA_TEMPERATURE` | `0.1` | Default temperature |
| `LANGSMITH_API_KEY` | *(empty)* | LangSmith API key (optional) |
| `LANGCHAIN_PROJECT` | `Loom` | LangSmith project name |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `OUTPUT_DIR` | `./outputs` | Markdown output directory |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `Loom-Weave-Threads` | Neo4j password (set in docker-compose) |
| `NEO4J_DATABASE` | *(unset)* | Named DB (Enterprise only); leave unset for Community |

### LangSmith Setup (Optional)

1. Create a free account at [smith.langchain.com](https://smith.langchain.com/)
2. Generate an API key
3. Set in `.env`:
   ```
   LANGSMITH_API_KEY=lsv2_pt_xxxxx
   LANGSMITH_TRACING=true
   ```

---

## 🔧 Usage Guide

### Basic Chatting

Just type a question in the chat:

> *"Explain the CAP theorem with real-world examples"*
> *"What's the difference between Spark RDD and DataFrame?"*
> *"How does backpropagation work? Include the math."*

The agent will respond with a comprehensive, structured Markdown answer.

### Using Web Search

1. Toggle **"🌐 Enable Web Search"** in the sidebar
2. Ask questions that benefit from current information:

> *"What are the latest features in Apache Spark 4.0?"*
> *"What's new in the 2024 State of Data Engineering survey?"*

The agent decides when to search based on the question. You don't need to explicitly ask it to search.

### Using RAG (Document Q&A)

1. Toggle **"📄 Enable RAG"** in the sidebar
2. Upload PDFs using the file uploader
3. Click **"🔄 Process & Index Documents"** to process them
4. Ask questions about your documents:

> *"Summarize the key findings from the uploaded paper"*
> *"What does the textbook say about gradient descent?"*

### Using the Knowledge Graph

1. Start Neo4j: `docker-compose up -d`
2. Toggle **"🔗 Enable Knowledge Graph"** in the sidebar
3. Upload PDFs — entities are **automatically extracted** into the graph alongside RAG indexing
4. Ask cross-paper questions:

> *"How are my papers on transformers related to the attention mechanism paper?"*
> *"What concepts appear in both papers?"*
> *"Are there any contradicting findings across my papers?"*

5. Visit the **📊 Knowledge Graph** page (sidebar) for:
   - Interactive graph visualization
   - Paper explorer & comparison
   - Concept deep-dives
   - Cross-paper relationship analysis

### Saving & Exporting

- **Export Chat**: Click "Export Chat" in the sidebar to download the full conversation as Markdown
- **Save Chat to History**: Click "Save Chat to History" to persist the conversation to `data/chat_history/`
- **Markdown format**: Exported files are formatted for direct paste into Substack or academic blogs

---

## 🧩 Extending the Agent

The architecture is designed for easy extension:

### Adding a New Tool

1. Create a new file in `src/tools/`:

```python
# src/tools/my_tool.py
from langchain_core.tools import tool

def create_my_tool():
    @tool
    def my_custom_tool(query: str) -> str:
        """Description of what this tool does."""
        # Your tool logic here
        return result
    return my_custom_tool
```

2. Register it in the graph builder (`src/graph/builder.py`):

```python
def _gather_tools(self, ...):
    tools = []
    # ... existing tools ...
    if use_my_tool:
        tools.append(create_my_tool())
    return tools
```

3. Add a toggle in the Streamlit sidebar (`app.py`).

### Adding a New Document Loader

Extend `DocumentProcessor` in `src/rag/processor.py` to support new file types beyond PDFs.

### Custom System Prompts

Edit the prompts in `src/graph/nodes.py` to customize the agent's personality, expertise areas, or output format.

---

## ⚠️ Migration Notes

### Graph identity: title → document_id (v0.6)

Paper nodes in Neo4j are now keyed by `document_id` (a content-addressed
`md5:<hex>` hash) instead of `title`. This prevents distinct PDFs that
share the same title from being collapsed into one node.

**Impact on existing data:** Paper nodes created before this change have no
`document_id` property and will not match new writes. To migrate a running
instance, either:

1. **Clear and re-ingest** — drop the graph (`MATCH (n) DETACH DELETE n` in
   the Neo4j browser) and re-upload your PDFs, or
2. **Back-fill manually** — set `document_id` on existing Paper and Finding
   nodes (e.g. `SET p.document_id = 'file:' + p.title`).

Finding nodes now carry a `paper_document_id` property alongside the
existing `paper_title` for reliable cross-paper linking.

### Concept identity: name-only → domain-scoped (v0.7)

Concept nodes are now uniquely keyed by `concept_key`
(`normalized_domain:lowercased_name`) instead of `name` alone. This prevents
cross-domain collisions (e.g. "attention" in AI vs. psychology).

**Impact on existing data:** Old Concept nodes have no `concept_key` property.
Re-ingestion after clearing the graph is recommended.

### Graph query safety (v0.8–v0.11)

The LLM-to-Cypher query tool has been permanently removed and replaced with
an intent-based service that only executes pre-written, parameterised Cypher.
The Knowledge Graph explorer page no longer exposes a raw Cypher input.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) — Local LLM runtime
- [LangChain](https://www.langchain.com/) — LLM orchestration framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent framework
- [LangSmith](https://smith.langchain.com/) — Observability platform
- [Streamlit](https://streamlit.io/) — UI framework
- [IBM Docling](https://github.com/DS4SD/docling) — Document conversion
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Neo4j](https://neo4j.com/) — Graph database
- [Pyvis](https://pyvis.readthedocs.io/) — Interactive network visualization
- [DuckDuckGo](https://duckduckgo.com/) — Free web search
