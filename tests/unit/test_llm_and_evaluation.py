"""Unit tests for profile normalization and RAG evaluation formatting."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from src.services.evaluation.judge import (
    EvaluationResult,
    RAGJudge,
    _parse_result,
    _score,
)
from src.services.llm import paper_profile

pytestmark = pytest.mark.unit


def test_profile_section_selection_prioritizes_key_sections():
    text = "Front matter\n# Methods\nmethod details\n# Results\nimportant result\n# Appendix\nappendix"
    selected = paper_profile.select_key_sections(text, budget=60)
    assert "Front matter" in selected
    assert "Results" in selected
    assert len(selected) <= 60


def test_profile_builder_normalizes_domain_and_detail_shapes():
    profile = paper_profile._build_profile(
        {
            "title": "Paper",
            "domain": "nlp",
            "contributions": [
                "A contribution",
                {"text": "Another", "evidence": "Table 1"},
            ],
            "concepts": [{"name": "attention", "domain": "ai"}],
            "findings": [{"claim": "It works", "evidence_type": "empirical"}],
        }
    )
    assert profile is not None
    assert profile.domain == "Natural Language Processing"
    assert profile.contributions[0].text == "A contribution"
    assert profile.concepts[0].domain == "Artificial Intelligence"


def test_profile_response_parser_rejects_error_payloads():
    assert paper_profile._parse_profile_response(SimpleNamespace(content='{"error": "bad"}')) is None
    assert paper_profile._parse_profile_response(SimpleNamespace(content='{"title": "ok"}')) == {
        "title": "ok"
    }


def test_evaluation_scores_are_clamped_and_rendered():
    assert _score(-2) == 0
    assert _score(8) == 5
    assert _score("not a score") == 0
    result = EvaluationResult(5, 3, 1, "Grounded with minor omissions")
    markdown = result.as_markdown()
    assert "Context Relevance" in markdown
    assert "⭐⭐⭐⭐⭐" in markdown
    assert "Grounded with minor omissions" in markdown


def test_evaluation_parser_accepts_response_objects_and_rejects_bad_json():
    good = _parse_result(
        SimpleNamespace(content='{"context_relevance": 4, "faithfulness": 5, "answer_relevance": 3}')
    )
    assert good == EvaluationResult(4, 5, 3, "")
    assert _parse_result(SimpleNamespace(content="bad")) is None


def test_rag_judge_degrades_when_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.services.evaluation.judge.get_llm",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    judge = RAGJudge()
    assert judge.evaluate("q", "ctx", "answer") is None
    assert asyncio.run(judge.aevaluate("q", "ctx", "answer")) is None
