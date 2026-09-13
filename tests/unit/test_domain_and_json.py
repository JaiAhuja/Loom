"""Unit tests for shared domain values and LLM response parsing."""

import json

import pytest

from src.domain.paper import Paper, merge_paper_sources
from src.domain.taxonomy import (
    canonicalize_domain,
    domains_as_string,
    normalize_concept_domain,
)
from src.services.common.json_parser import parse_llm_json

pytestmark = pytest.mark.unit


def test_taxonomy_normalizes_canonical_and_alias_values():
    assert canonicalize_domain(" nlp ") == "Natural Language Processing"
    assert canonicalize_domain("machine_learning") == "Machine Learning"
    assert canonicalize_domain("not-a-domain") == "Other"
    assert normalize_concept_domain("Computer Vision") == "ai"
    assert normalize_concept_domain("Databases") == "data_infra"
    assert "Artificial Intelligence" in domains_as_string()


def test_paper_labels_and_statuses():
    paper = Paper("paper-1", "A Paper", "Machine Learning", in_rag=True, in_kg=True)
    assert paper.display_label == "A Paper  [Machine Learning]"
    assert paper.status_badge == "RAG+KG"
    assert Paper("paper-2", "Other").display_label == "Other"
    assert Paper("paper-2", "Other").status_badge == "—"


def test_merge_paper_sources_combines_records_by_document_id():
    papers = merge_paper_sources(
        rag_papers=[
            {
                "document_id": "shared",
                "title": "Filename title",
                "domain": "Data Science",
                "chunk_count": 4,
            }
        ],
        kg_papers=[
            {
                "document_id": "shared",
                "title": "Extracted title",
                "domain": "Data Science",
                "concept_count": 3,
            },
            {
                "document_id": "kg-only",
                "title": "Graph only",
                "concept_count": 2,
            },
            {"title": "missing id"},
        ],
    )
    assert papers == [
        Paper("shared", "Filename title", "Data Science", True, True, 4, 3),
        Paper("kg-only", "Graph only", "", False, True, 0, 2),
    ]


def test_json_parser_handles_direct_fenced_python_and_invalid_responses():
    expected = {"answer": [1, 2]}
    assert parse_llm_json(json.dumps(expected)) == expected
    assert parse_llm_json('```json\n{"answer": [1, 2]}\n```') == expected
    assert parse_llm_json("```{'answer': [1, 2]}```") == expected
    assert parse_llm_json(None)["error"] == "No response text provided"
    assert parse_llm_json("not json")["error"] == "No valid JSON content found in response"
