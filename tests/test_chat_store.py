"""Tests for src.services.chat.store — ChatStore save/load/search/delete behaviour."""

import json
import os

from src.services.chat.store import ChatMetadata, ChatRecord, ChatStore


def _msgs():
    return [
        {"role": "user", "content": "What is attention?"},
        {"role": "assistant", "content": "Attention is a mechanism..."},
    ]



def test_save_writes_json_with_topic_and_metadata(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    meta = ChatMetadata(model="granite4:tiny-h", temperature=0.2, use_rag=True)

    path = store.save(_msgs(), metadata=meta)

    assert os.path.isfile(path)
    assert path.endswith(".json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["topic"] == "What is attention?"
    assert data["metadata"]["model"] == "granite4:tiny-h"
    assert data["metadata"]["use_rag"] is True
    assert len(data["messages"]) == 2


def test_save_empty_conversation_raises(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    try:
        store.save([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_save_defaults_metadata_when_omitted(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    path = store.save(_msgs())

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["model"] == ""
    assert data["metadata"]["use_rag"] is False



def test_load_roundtrips(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    path = store.save(_msgs(), metadata=ChatMetadata(model="m", temperature=0.3))

    record = store.load(os.path.basename(path))

    assert isinstance(record, ChatRecord)
    assert record.topic == "What is attention?"
    assert record.metadata.model == "m"
    assert record.metadata.temperature == 0.3
    assert [m["role"] for m in record.messages] == ["user", "assistant"]


def test_load_tolerates_unknown_metadata_keys(tmp_path):
    """Old saved chats with fields we no longer understand still load."""
    store = ChatStore(directory=str(tmp_path))
    payload = {
        "topic": "Legacy",
        "saved_at": "2024-01-01 00:00:00",
        "metadata": {"model": "m", "obsolete_flag": True},
        "messages": _msgs(),
    }
    legacy_path = os.path.join(str(tmp_path), "20240101-000000-legacy.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    record = store.load("20240101-000000-legacy.json")
    assert record.metadata.model == "m"



def test_list_chats_returns_newest_first(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    p1 = store.save([{"role": "user", "content": "first"}])
    p2 = store.save([{"role": "user", "content": "second"}])
    os.utime(p1, (1_700_000_000, 1_700_000_000))
    os.utime(p2, (1_700_000_100, 1_700_000_100))

    entries = store.list_chats()
    assert [e["filename"] for e in entries] == [
        os.path.basename(p2),
        os.path.basename(p1),
    ]
    assert all(e["type"] == "json" and e["resumable"] for e in entries)


def test_list_chats_includes_legacy_md_as_readonly(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    (tmp_path / "legacy-chat.md").write_text("# old chat", encoding="utf-8")

    entries = store.list_chats()
    assert len(entries) == 1
    assert entries[0]["type"] == "md"
    assert entries[0]["resumable"] is False


def test_search_matches_topic_and_content(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    store.save([
        {"role": "user", "content": "transformers question"},
        {"role": "assistant", "content": "..."},
    ])
    store.save([
        {"role": "user", "content": "totally unrelated"},
        {"role": "assistant", "content": "rag pipelines are cool"},
    ])

    topic_hits = store.search("transformers")
    content_hits = store.search("rag")
    miss = store.search("nothingburger")

    assert len(topic_hits) == 1
    assert len(content_hits) == 1
    assert miss == []


def test_search_empty_query_returns_all(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    store.save(_msgs())
    assert len(store.search("")) == 1


def test_delete_removes_file(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    path = store.save(_msgs())
    fname = os.path.basename(path)

    store.delete(fname)
    assert not os.path.exists(path)
    store.delete(fname)


def test_render_markdown_json_and_legacy(tmp_path):
    store = ChatStore(directory=str(tmp_path))
    path = store.save(_msgs())
    md = store.render_markdown(os.path.basename(path))
    assert "**You:**" in md and "**Loom:**" in md

    (tmp_path / "legacy.md").write_text("raw markdown", encoding="utf-8")
    assert store.render_markdown("legacy.md") == "raw markdown"
