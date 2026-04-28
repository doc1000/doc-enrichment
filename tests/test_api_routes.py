"""Tests for the FastAPI service routes.

All package-level enrichment functions are mocked; no real LLM calls are made.
Requires the [service] extra: pip install -e ".[dev,service]"
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentResponse,
    DocumentEnrichmentResponse,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
)
from doc_enrichment.services.api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures / sample payloads
# ---------------------------------------------------------------------------

_DOC_REQUEST = {
    "doc_id": "d1",
    "text": "Sample document text for testing.",
    "title": "Test Doc",
    "source": "https://example.com",
}

_KNOWLEDGE_PAYLOAD_DICT = {
    "title": "Test",
    "summary": "A test summary",
    "key_topics": ["topic"],
    "entities": [],
    "facts": [],
    "intent_purpose": [],
    "target_audience": [],
    "document_type": [],
    "industry": [],
    "life_domain": [],
    "provenance": [],
}

_BRANCH_REQUEST = {
    "branch_id": "b1",
    "left_node_id": "l1",
    "right_node_id": "r1",
    "left_payload": _KNOWLEDGE_PAYLOAD_DICT,
    "right_payload": _KNOWLEDGE_PAYLOAD_DICT,
}


def _doc_response(doc_id: str = "d1") -> DocumentEnrichmentResponse:
    return DocumentEnrichmentResponse(
        doc_id=doc_id,
        prompt_version="v1",
        model_name="mock",
        payload=KnowledgePayload(summary="test"),
    )


def _branch_response(branch_id: str = "b1") -> BranchEnrichmentResponse:
    enrichment = BranchEnrichment(
        sibling_a=SiblingContrast(),
        sibling_b=SiblingContrast(),
        parent_payload=KnowledgePayload(summary="parent"),
        parent_refinement_for_a=ParentRefinement(),
        parent_refinement_for_b=ParentRefinement(),
    )
    return BranchEnrichmentResponse(
        branch_id=branch_id,
        left_node_id="l1",
        right_node_id="r1",
        prompt_version="v1",
        model_name="mock",
        enrichment=enrichment,
    )


# ---------------------------------------------------------------------------
# POST /enrich/documents
# ---------------------------------------------------------------------------

def test_enrich_documents_returns_200():
    mock_fn = AsyncMock(return_value=[_doc_response()])

    with patch("doc_enrichment.services.api.enrich_documents", new=mock_fn):
        resp = client.post("/enrich/documents", json=[_DOC_REQUEST])

    assert resp.status_code == 200


def test_enrich_documents_response_shape():
    mock_fn = AsyncMock(return_value=[_doc_response("d1")])

    with patch("doc_enrichment.services.api.enrich_documents", new=mock_fn):
        resp = client.post("/enrich/documents", json=[_DOC_REQUEST])

    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["doc_id"] == "d1"
    assert body[0]["enrichment_type"] == "document_payload"
    assert body[0]["schema_version"] == "document_payload_v1"


def test_enrich_documents_empty_list_returns_200():
    mock_fn = AsyncMock(return_value=[])

    with patch("doc_enrichment.services.api.enrich_documents", new=mock_fn):
        resp = client.post("/enrich/documents", json=[])

    assert resp.status_code == 200
    assert resp.json() == []


def test_enrich_documents_calls_package_function_once():
    mock_fn = AsyncMock(return_value=[_doc_response()])

    with patch("doc_enrichment.services.api.enrich_documents", new=mock_fn):
        client.post("/enrich/documents", json=[_DOC_REQUEST])

    mock_fn.assert_called_once()


def test_enrich_documents_missing_required_field_returns_422():
    resp = client.post("/enrich/documents", json=[{"title": "no doc_id or text"}])
    assert resp.status_code == 422


def test_enrich_documents_batch_preserves_count():
    mock_fn = AsyncMock(return_value=[_doc_response(f"d{i}") for i in range(3)])

    with patch("doc_enrichment.services.api.enrich_documents", new=mock_fn):
        resp = client.post(
            "/enrich/documents",
            json=[{**_DOC_REQUEST, "doc_id": f"d{i}"} for i in range(3)],
        )

    assert resp.status_code == 200
    assert len(resp.json()) == 3


# ---------------------------------------------------------------------------
# POST /enrich/parents
# ---------------------------------------------------------------------------

def test_enrich_parents_returns_200():
    mock_fn = AsyncMock(return_value=[_branch_response()])

    with patch("doc_enrichment.services.api.enrich_parent_nodes", new=mock_fn):
        resp = client.post("/enrich/parents", json=[_BRANCH_REQUEST])

    assert resp.status_code == 200


def test_enrich_parents_response_shape():
    mock_fn = AsyncMock(return_value=[_branch_response("b1")])

    with patch("doc_enrichment.services.api.enrich_parent_nodes", new=mock_fn):
        resp = client.post("/enrich/parents", json=[_BRANCH_REQUEST])

    body = resp.json()
    assert isinstance(body, list)
    assert body[0]["branch_id"] == "b1"
    assert body[0]["enrichment_type"] == "branch_enrichment"
    assert body[0]["schema_version"] == "branch_enrichment_v1"
    assert body[0]["enrichment"]["parent_payload"]["summary"] == "parent"


def test_enrich_parents_empty_list_returns_200():
    mock_fn = AsyncMock(return_value=[])

    with patch("doc_enrichment.services.api.enrich_parent_nodes", new=mock_fn):
        resp = client.post("/enrich/parents", json=[])

    assert resp.status_code == 200
    assert resp.json() == []


def test_enrich_parents_missing_required_field_returns_422():
    resp = client.post("/enrich/parents", json=[{"branch_id": "b1"}])
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /enrich/contrast
# ---------------------------------------------------------------------------

def test_enrich_contrast_returns_200():
    mock_fn = AsyncMock(return_value=[_branch_response()])

    with patch("doc_enrichment.services.api.enrich_contrast", new=mock_fn):
        resp = client.post("/enrich/contrast", json=[_BRANCH_REQUEST])

    assert resp.status_code == 200


def test_enrich_contrast_response_shape():
    enrichment = BranchEnrichment(
        sibling_a=SiblingContrast(contrast=["a differs"], contrast_tag=["tag-a"]),
        sibling_b=SiblingContrast(contrast=["b differs"], contrast_tag=["tag-b"]),
        parent_payload=KnowledgePayload(summary="contrast parent"),
        parent_refinement_for_a=ParentRefinement(parent_refine=["refine-a"]),
        parent_refinement_for_b=ParentRefinement(parent_refine=["refine-b"]),
    )
    full_response = BranchEnrichmentResponse(
        branch_id="b1",
        left_node_id="l1",
        right_node_id="r1",
        prompt_version="v1",
        model_name="mock",
        enrichment=enrichment,
    )
    mock_fn = AsyncMock(return_value=[full_response])

    with patch("doc_enrichment.services.api.enrich_contrast", new=mock_fn):
        resp = client.post("/enrich/contrast", json=[_BRANCH_REQUEST])

    body = resp.json()
    assert body[0]["enrichment"]["sibling_a"]["contrast"] == ["a differs"]
    assert body[0]["enrichment"]["sibling_b"]["contrast_tag"] == ["tag-b"]
    assert body[0]["enrichment"]["parent_refinement_for_a"]["parent_refine"] == ["refine-a"]


def test_enrich_contrast_empty_list_returns_200():
    mock_fn = AsyncMock(return_value=[])

    with patch("doc_enrichment.services.api.enrich_contrast", new=mock_fn):
        resp = client.post("/enrich/contrast", json=[])

    assert resp.status_code == 200
    assert resp.json() == []


def test_enrich_contrast_missing_required_field_returns_422():
    resp = client.post("/enrich/contrast", json=[{"branch_id": "b1"}])
    assert resp.status_code == 422
