"""Local persistence for evidence-backed research notes."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

_DEFAULT_DIR = "./data/notebooks"
_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class Evidence:
    """A retrieved excerpt attached to a notebook note."""

    text: str
    source: str = ""
    chunk: str = ""
    pages: str = ""
    section: str = ""


@dataclass
class NotebookEntry:
    """One durable research finding."""

    title: str
    question: str
    answer: str
    note: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "to-review"
    collection: str | None = None
    paper: str | None = None
    document_id: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    saved_at: str = ""

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"**Status:** {self.status}"]
        if self.tags:
            lines.append(f"**Tags:** {', '.join(self.tags)}")
        if self.paper:
            lines.append(f"**Paper:** {self.paper}")
        lines += ["", "## Question", "", self.question, "", "## Loom's answer", "", self.answer]
        if self.note:
            lines += ["", "## My note", "", self.note]
        if self.evidence:
            lines += ["", "## Evidence", ""]
            for item in self.evidence:
                label = " — ".join(x for x in (item.source, item.chunk, item.pages, item.section) if x)
                lines += [f"### {label or 'Retrieved excerpt'}", "", item.text, ""]
        return "\n".join(lines).rstrip() + "\n"


def _slug(text: str) -> str:
    return _SAFE.sub("-", text.strip()).strip("-")[:60] or "note"


class NotebookStore:
    """Persist notebook entries as readable JSON files."""

    def __init__(self, directory: str = _DEFAULT_DIR) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.directory, os.path.basename(str(filename).replace("\\", "/")))

    def save(self, entry: NotebookEntry) -> str:
        if not entry.question.strip() or not entry.answer.strip():
            raise ValueError("A notebook entry needs a question and answer.")
        now = datetime.now()
        entry.saved_at = entry.saved_at or now.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{now.strftime('%Y%m%d-%H%M%S-%f')}-{_slug(entry.title)}.json"
        path = self._path(filename)
        payload = asdict(entry)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def load(self, filename: str) -> NotebookEntry:
        with open(self._path(filename), encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        evidence = [
            Evidence(**{k: v for k, v in item.items() if k in Evidence.__dataclass_fields__})
            for item in data.get("evidence", [])
            if isinstance(item, dict)
        ]
        return NotebookEntry(
            title=data.get("title") or "Untitled note",
            question=data.get("question") or "",
            answer=data.get("answer") or "",
            note=data.get("note") or "",
            tags=list(data.get("tags") or []),
            status=data.get("status") or "to-review",
            collection=data.get("collection"),
            paper=data.get("paper"),
            document_id=data.get("document_id"),
            evidence=evidence,
            saved_at=data.get("saved_at") or "",
        )

    def list_entries(self) -> list[dict]:
        entries = []
        for name in os.listdir(self.directory):
            if not name.endswith(".json"):
                continue
            try:
                item = self.load(name)
                entries.append(
                    {
                        "filename": name,
                        "title": item.title,
                        "status": item.status,
                        "paper": item.paper,
                        "saved_at": item.saved_at,
                        "modified": datetime.fromtimestamp(os.path.getmtime(self._path(name))),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(entries, key=lambda x: x["modified"], reverse=True)

    def search(self, query: str = "") -> list[dict]:
        if not query.strip():
            return self.list_entries()
        q = query.lower().strip()
        return [e for e in self.list_entries() if q in (e["title"] + " " + (e["paper"] or "")).lower()]

    def delete(self, filename: str) -> None:
        path = self._path(filename)
        if os.path.isfile(path):
            os.remove(path)
