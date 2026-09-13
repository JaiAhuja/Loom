# Chat service

The chat service persists saved conversations as structured JSON under
`data/chat_history/` and renders them as Markdown when users export or view
them. Legacy Markdown files can still be listed for backward compatibility.

## Public API

- `ChatStore` — save, update, load, list, search, delete, and render chats.
- `ChatRecord` — conversation topic, messages, metadata, and save timestamp.
- `ChatMetadata` — model, temperature, RAG, graph, collection, and paper-scope
  settings captured with a conversation.

## Usage notes

Use the store from the Streamlit pages rather than writing directly to the
history directory. Filenames are sanitised and messages are normalised before
serialization. The chat-history page uses this service for search, resume,
delete, and Markdown export.

Tests live in `tests/integration/test_chat_store.py`.
