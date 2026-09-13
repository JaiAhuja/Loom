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
| **Beautiful Output** | Publication-ready Markdown formatted for [Substack](https://substack.com/) / blogs |
| **Knowledge Graph** | [Neo4j](https://neo4j.com/) knowledge graph — extracted paper profiles, details, concepts, methods, findings, and cross-paper relationships |
| **Graph Explorer** | Interactive [Pyvis](https://pyvis.readthedocs.io/) visualization + paper/detail inspection + paper comparison + safe intent-based queries |
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
│  │ • Collections│  │  Papers ─► Details / Concepts / Findings │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LangGraph     │
                    │   Agent Loop    │◄──── LangSmith Tracing
                    │  (ReAct Style)  │
                    └─┬─────────────┬─┘
                      │             │
              ┌───────▼──┐     ┌────▼────────┐
              │ RAG Query│     │ Graph Query │
              │ ChromaDB │     │ Neo4j       │
              │+ Docling │     │ (opt-in)    │
              └──────────┘     └─────────────┘
                      │             │
                  ┌───▼─────────────▼───┐
                  │     Ollama LLM      │
                  │    (Local Model)    │
                  └─────────────────────┘
```

### Data Flow

1. **User asks a question** in the Streamlit chat
2. **LangGraph agent** receives the query with conversation history
3. **Agent decides** whether to use tools (RAG, knowledge graph) or answer directly
4. If **RAG is enabled** → queries ChromaDB for relevant document chunks
5. If **Knowledge Graph is enabled** → queries Neo4j via safe, intent-based tool (no raw Cypher)
6. **LLM generates** a comprehensive, Markdown-formatted response
7. **Response is displayed** in Streamlit and can be saved or exported

### Ingestion & Identity

When PDFs are uploaded they pass through a deterministic identity pipeline:

1. **MD5 hash** computed from file bytes → used for content-addressed storage under `data/pdfs/`
2. **Filename-derived `document_id`** — the sanitised filename stem (e.g. `Self-Supervised Learning`) is the stable key used across RAG and KG
3. **Unique `ingest_id`** assigned per upload session
4. **Deterministic `chunk_id`** generated per text chunk → idempotent vector writes (upsert)
5. Re-uploading a document with the same filename-derived `document_id` skips existing RAG chunks instead of duplicating them; if Neo4j is available and the KG is missing or lacks newer detail nodes, Loom catches the KG up from the cached profile or Markdown

### Knowledge Graph Pipeline

1. **PDF uploaded** via Streamlit sidebar
2. **Identity assigned** — sanitized filename stem → `document_id` (e.g. `My-Paper.pdf` → `My-Paper`); MD5 hash → content-addressed storage path; unique `ingest_id` per session
3. **Docling** converts PDF to Markdown (shared with RAG pipeline)
4. **LLM extracts** a structured `PaperProfile`: title, authors, year, domain, a 15-20 sentence summary, contributions, what the paper stands for, what it builds on, what it does not support, limitations, concepts, methods, and findings
5. **Profile sidecar saved** — the `PaperProfile` is persisted as `data/txt/<document_id>_profile.json` alongside the Markdown cache. If the KG needs to be rebuilt later, Loom prefers the saved profile; if the sidecar is missing or predates newer detail fields and a model is available, it re-extracts from the cached Markdown.
6. **Deterministic entity resolution** — graph nodes are merged by stable keys: `Paper.document_id`, `Concept.concept_key`, `Method.name`, `Finding.finding_key`, and `PaperDetail.detail_key`
7. **Paper node keyed by `document_id`** — distinct PDFs with the same title stay separate
8. **Concepts scoped by domain** — `concept_key` = `normalized_domain:name` prevents cross-domain collisions
9. **Detail child nodes** — `PaperDetail` nodes store extracted contributions, positions, build-on statements, unsupported claims, and limitations; the Paper node keeps the central summary
10. **Finding nodes have stable keys** — each `Finding` node is stamped with a `finding_key` (MD5 of `paper_title + claim`) and stores `paper_document_id` for visualization and cross-paper queries
11. **Cross-paper linking** — after each paper is written, Loom compares new findings and concepts against existing graph content and writes directed `SUPPORTS`, `CONTRADICTS`, `EXTENDS`, `RELATED_TO`, or `SUBTOPIC_OF` edges where applicable
12. **Neo4j stores** nodes and relationships (`Paper`, `PaperDetail`, `Concept`, `Method`, `Finding`; `HAS_DETAIL`, `DISCUSSES`, `USES_METHOD`, `HAS_FINDING`, `SUPPORTS`, `CONTRADICTS`, `EXTENDS`, etc.)

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
- **[Neo4j Desktop](https://neo4j.com/download/)** (for Neo4j knowledge graph — optional)
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
ollama pull granite4:tiny-h        # Default model

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
OLLAMA_MODEL=granite4:tiny-h
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

See the Configuration section below for the full list of settings.

### 6. Start Neo4j (Optional — for Knowledge Graph)

Install [Neo4j Desktop](https://neo4j.com/download/) and create a local DBMS:
1. Open Neo4j Desktop -> click **"New"** -> **"Create Prject"**
2. Inside the project, click **"Add"** -> **"Local DBMS"**
3. Set a password (e.g.`Loom-Weave-Threads`) and choose Neo4j 5.x
4. Click **"Start"** on the database instance

### 7. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

Loom uses a service-oriented modular monolith. The packages under
`src/services/` group application capabilities by ownership and integration
boundary, while `src/domain/` contains shared domain concepts. These are
in-process services today; separating one into a deployable microservice later
would require defining an API and moving its infrastructure boundary without
changing the domain model.

```
loom/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── app.py                    # Streamlit application entry point
├── granite_tokenizer/        # Local Granite tokenizer files (HybridChunker)
│
├── pages/
│   ├── 2_Knowledge_Graph.py  # Knowledge Graph explorer page
│   └── 3_Chat_History.py     # Saved chat history viewer
│
├── config/
│   ├── __init__.py
│   └── settings.py           # Pydantic settings (loads .env)
│
├── src/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── paper.py          # Paper domain model
│   │   └── taxonomy.py       # Canonical domain taxonomy (shared by RAG + KG)
│   │
│   └── services/
│       ├── __init__.py
│       ├── agent/
│       │   ├── state.py      # LangGraph state schema
│       │   ├── nodes.py      # Agent node, routing, system prompts
│       │   └── builder.py    # Graph builder (assembles the agent)
│       ├── chat/
│       │   └── store.py      # Chat history persistence (save/load/search JSON)
│       ├── common/
│       │   └── json_parser.py # Shared LLM JSON response parser
│       ├── evaluation/
│       │   └── judge.py      # LLM-as-a-Judge RAG quality scorer
│       ├── ingestion/
│       │   ├── identity.py   # Document identity and content-addressed storage
│       │   └── service.py    # PDF ingestion orchestration
│       ├── knowledge_graph/
│       │   ├── connection.py # Neo4j connection manager
│       │   ├── schema.py     # Graph schema, constraints, and identity keys
│       │   ├── queries.py    # Predefined graph queries and visualization data
│       │   ├── service.py    # Typed intent-based graph query service
│       │   └── writer.py     # PaperProfile to Neo4j writer and linker
│       ├── llm/
│       │   ├── paper_profile.py # Structured paper profiling
│       │   └── provider.py   # Ollama LLM and embedding factories
│       ├── retrieval/
│       │   ├── processor.py  # Docling PDF to chunks pipeline
│       │   └── store.py      # ChromaDB vector store manager
│       ├── tools/
│       │   ├── rag_tool.py   # Document query tool factory
│       │   └── safe_graph_tool.py # Intent-based graph query tool
│       └── ui/
│           ├── bootstrap.py  # Cached dependency checks and factories
│           ├── chrome.py     # Reusable Streamlit presentation components
│           └── confirm.py    # Destructive-action confirmation helper
│
├── data/
│   ├── chat_history/         # Saved conversation JSON files
│   ├── pdfs/                 # Uploaded PDF storage (content-addressed)
│   ├── txt/                  # Docling-extracted Markdown text files
│   │                         #   └─ <document_id>_profile.json — profile sidecar (one per paper)
│   └── chroma_db/            # ChromaDB persistent storage
│
├── outputs/                  # Saved Markdown exports
└── tests/                    # pytest test suite
```

---

## ⚙️ Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `granite4:tiny-h` | Default chat model |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model (for RAG) |
| `OLLAMA_TEMPERATURE` | `0.1` | Default temperature |
| `OLLAMA_NUM_CTX` | `32768` | Context-window token limit (reduce for small models) |
| `LANGSMITH_API_KEY` | *(empty)* | LangSmith API key (optional) |
| `LANGCHAIN_PROJECT` | `Loom` | LangSmith project name |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `OUTPUT_DIR` | `./outputs` | Markdown output directory |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Neo4j connection URI (Neo4j Desktop) |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `Loom-Weave-Threads` | Neo4j password (Neo4j Desktop) |
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

### Using RAG (Document Q&A)

1. Toggle **"📄 Enable RAG"** in the sidebar
2. Upload PDFs using the file uploader
3. Click **"🔄 Process & Index Documents"** to process them
4. Ask questions about your documents:

> *"Summarize the key findings from the uploaded paper"*
> *"What does the textbook say about gradient descent?"*

### Using the Knowledge Graph

1. Start your Neo4j Desktop DBMS
2. Toggle **"📄 Enable RAG"** so the upload/indexing controls are visible
3. Toggle **"🔗 Enable Knowledge Graph"** in the sidebar
4. Upload PDFs and click **"🔄 Process & Index Documents"** — graph nodes are extracted into Neo4j alongside RAG indexing
5. Ask cross-paper questions:

> *"How are my papers on transformers related to the attention mechanism paper?"*
> *"What concepts appear in both papers?"*
> *"Are there any contradicting findings across my papers?"*

6. Visit the **📊 Knowledge Graph** page (sidebar) for:
   - Interactive graph visualization
   - Paper summaries, extracted detail nodes, and paper comparison
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

1. Create a new file in `src/services/tools/`:

```python
# src/services/tools/my_tool.py
from langchain_core.tools import tool

def create_my_tool():
    @tool
    def my_custom_tool(query: str) -> str:
        """Description of what this tool does."""
        # Your tool logic here
        return result
    return my_custom_tool
```

2. Register it in the agent builder (`src/services/agent/builder.py`):

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

Extend `DocumentProcessor` in `src/services/retrieval/processor.py` to support new file types beyond PDFs.

### Custom System Prompts

Edit the prompts in `src/services/agent/nodes.py` to customize the agent's personality, expertise areas, or output format.

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
