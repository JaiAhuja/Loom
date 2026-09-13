"""Structured chat-history persistence.

Chats are saved as JSON under ``data/chat_history/`` with a small metadata
header plus the raw message list. Legacy ``.md`` files that may already exist
in the directory are still listed (read-only) for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

_DEFAULT_DIR = "./data/chat_history"
_FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")

logger = logging.getLogger(__name__)


@dataclass
class ChatMetadata:
    """Settings used for the conversation being saved."""

    model: str = ""
    temperature: float = 0.0
    use_rag: bool = False
    use_graph: bool = False
    collection_name: Optional[str] = None
    paper_filter: Optional[str] = None
    document_id: Optional[str] = None


@dataclass
class ChatRecord:
    """One saved conversation."""

    topic: str
    messages: list[dict]
    metadata: ChatMetadata = field(default_factory=ChatMetadata)
    saved_at: str = ""

    def to_markdown(self) -> str:
        """Render the record as human-readable markdown."""
        lines = [f"# {self.topic}", ""]
        if self.saved_at:
            lines.append(f"*Saved {self.saved_at}*")
            lines.append("")
        for msg in self.messages:
            role = msg.get("role", "assistant")
            heading = "**You:**" if role == "user" else "**Loom:**"
            lines.extend([heading, "", msg.get("content", ""), "", "---", ""])
        return "\n".join(lines)


def _topic_from_messages(messages: Iterable[dict]) -> str:
    """Derive a short topic string from the first user message."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip().splitlines()
            if text:
                first = text[0].strip()
                return first[:80] if first else "Untitled chat"
    return "Untitled chat"


def _slugify(text: str) -> str:
    slug = _FILENAME_SAFE.sub("-", text.strip()).strip("-")
    return slug[:60] or "chat"


def _normalise_messages(messages: Any) -> list[dict]:
    """Return a safe message list from saved JSON or session state."""
    if not isinstance(messages, list):
        return []

    return [_normalise_message(msg) for msg in messages if isinstance(msg, dict)]


def _normalise_message(msg: dict) -> dict:
    role = msg.get("role") if msg.get("role") in {"user", "assistant", "system", "tool"} else "assistant"
    content = msg.get("content")
    return {"role": role, "content": content if isinstance(content, str) else ""}


def _record_payload(record: ChatRecord) -> dict:
    return {
        "topic": record.topic,
        "saved_at": record.saved_at,
        "metadata": asdict(record.metadata),
        "messages": record.messages,
    }


def _entry(name: str, path: str, topic: str, kind: str, metadata: dict | None = None) -> dict:
    return {
        "filename": name,
        "path": path,
        "topic": topic,
        "type": kind,
        "resumable": kind == "json",
        "metadata": metadata or {},
        "modified": datetime.fromtimestamp(os.path.getmtime(path)),
    }


class ChatStore:
    """Persists chats to ``data/chat_history`` as JSON files."""

    def __init__(self, directory: str = _DEFAULT_DIR) -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _path_for(self, filename: str) -> str:
        safe_name = os.path.basename(str(filename or "").replace("\\", "/"))
        return os.path.join(self.directory, safe_name)

    def save(
        self,
        messages: list[dict],
        metadata: Optional[ChatMetadata] = None,
    ) -> str:
        """Write ``messages`` to a new JSON file and return its path."""
        return self._save(messages, metadata)

    def update(
        self,
        filename: str,
        messages: list[dict],
        metadata: Optional[ChatMetadata] = None,
    ) -> str:
        """Update an existing JSON chat file and return its path."""
        return self._save(messages, metadata, filename)

    def _save(
        self,
        messages: list[dict],
        metadata: Optional[ChatMetadata],
        filename: str | None = None,
    ) -> str:
        if not messages:
            raise ValueError("Cannot save an empty conversation.")

        now = datetime.now()
        if filename:
            path = self._path_for(filename)
            try:
                existing = self.load(filename)
                topic, saved_at = existing.topic, existing.saved_at
            except Exception:
                logger.debug(
                    "Failed to load existing chat before update: %s",
                    filename,
                    exc_info=True,
                )
                topic = _topic_from_messages(messages)
                saved_at = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            topic = _topic_from_messages(messages)
            saved_at = now.strftime("%Y-%m-%d %H:%M:%S")
            path = self._path_for(
                f"{now.strftime('%Y%m%d-%H%M%S')}-{_slugify(topic)}.json"
            )
        self._write_record(path, ChatRecord(
            topic=topic,
            messages=_normalise_messages(messages),
            metadata=metadata or ChatMetadata(),
            saved_at=saved_at,
        ))
        return path

    def _write_record(self, path: str, record: ChatRecord) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_record_payload(record), f, indent=2, ensure_ascii=False)

    def load(self, filename: str) -> ChatRecord:
        """Load a JSON chat file into a :class:`ChatRecord`."""
        with open(self._path_for(filename), "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Chat file is not a JSON object: {filename}")

        meta_dict = data.get("metadata") or {}
        if not isinstance(meta_dict, dict):
            meta_dict = {}
        known = set(ChatMetadata.__dataclass_fields__)
        meta = ChatMetadata(**{k: v for k, v in meta_dict.items() if k in known})

        return ChatRecord(
            topic=data.get("topic") or "Untitled chat",
            messages=_normalise_messages(data.get("messages")),
            metadata=meta,
            saved_at=data.get("saved_at") or "",
        )

    def delete(self, filename: str) -> None:
        path = self._path_for(filename)
        if os.path.isfile(path):
            os.remove(path)

    def list_chats(self) -> list[dict]:
        """Return directory entries (newest first) with light metadata."""
        if not os.path.isdir(self.directory):
            return []

        entries: list[dict] = []
        for name in os.listdir(self.directory):
            path = os.path.join(self.directory, name)
            if not os.path.isfile(path):
                continue

            if name.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        continue
                    topic = data.get("topic") or os.path.splitext(name)[0]
                    metadata = data.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    entries.append(_entry(name, path, topic, "json", metadata))
                except (OSError, json.JSONDecodeError):
                    logger.debug("Skipping unreadable chat history file: %s", path, exc_info=True)
                    continue
            elif name.endswith(".md"):
                entries.append(_entry(name, path, os.path.splitext(name)[0].replace("-", " "), "md"))

        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries

    def search(self, query: str) -> list[dict]:
        """Case-insensitive substring search over topic + message content."""
        if not query:
            return self.list_chats()
        q = query.strip().lower()
        hits: list[dict] = []
        for entry in self.list_chats():
            if q in entry["topic"].lower():
                hits.append(entry)
                continue
            if entry["type"] == "json":
                try:
                    record = self.load(entry["filename"])
                except Exception:
                    logger.debug("Skipping unreadable chat during search: %s", entry["filename"], exc_info=True)
                    continue
                if any(q in (m.get("content") or "").lower() for m in record.messages):
                    hits.append(entry)
            else:
                try:
                    with open(entry["path"], "r", encoding="utf-8") as f:
                        if q in f.read().lower():
                            hits.append(entry)
                except OSError:
                    logger.debug("Skipping unreadable markdown chat during search: %s", entry["path"], exc_info=True)
        return hits

    def render_markdown(self, filename: str) -> str:
        """Render a JSON or legacy .md chat as markdown."""
        if filename.endswith(".json"):
            return self.load(filename).to_markdown()
        with open(self._path_for(filename), "r", encoding="utf-8") as f:
            return f.read()
