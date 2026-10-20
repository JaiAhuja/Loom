from src.services.notebook import Evidence, NotebookEntry, NotebookStore


def test_notebook_round_trip_and_search(tmp_path):
    store = NotebookStore(str(tmp_path))
    path = store.save(
        NotebookEntry(
            title="Attention mechanisms",
            question="How does attention work?",
            answer="It weighs relevant tokens.",
            note="Verify this against the methods section.",
            tags=["transformers"],
            evidence=[Evidence("supporting excerpt", "paper.pdf", "chunk_001")],
        )
    )

    loaded = store.load(path.split("/")[-1])
    assert loaded.title == "Attention mechanisms"
    assert loaded.evidence[0].chunk == "chunk_001"
    assert store.search("attention")[0]["title"] == "Attention mechanisms"
    assert store.search("verify this")[0]["title"] == "Attention mechanisms"
    assert store.search("transformers")[0]["title"] == "Attention mechanisms"
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    assert store.list_entries() and all(e["filename"] != "bad.json" for e in store.list_entries())
    assert "supporting excerpt" in loaded.to_markdown()
