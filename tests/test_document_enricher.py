"""Tests for the document enricher.

All LLM calls are mocked; no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_enrichment.config import EnrichmentConfig, ModelConfig
from doc_enrichment.enrichers.document import enrich_documents
from doc_enrichment.schemas import DocumentEnrichmentRequest, KnowledgePayload

_SAMPLE_PAYLOAD = KnowledgePayload(
    summary="Test summary",
    key_topics=["topic_a"],
    entities=["entity_x"],
    facts=["fact_1"],
)


def _make_mock_chain(return_value: KnowledgePayload) -> MagicMock:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


def _make_request(doc_id: str = "doc-1", text: str = "short text") -> DocumentEnrichmentRequest:
    return DocumentEnrichmentRequest(
        doc_id=doc_id,
        text=text,
        title="Test Doc",
        source="https://example.com",
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

async def test_empty_request_list_returns_empty():
    results = await enrich_documents([])
    assert results == []


# ---------------------------------------------------------------------------
# Single-chunk path
# ---------------------------------------------------------------------------

async def test_single_chunk_returns_validated_payload():
    mock_chain = _make_mock_chain(_SAMPLE_PAYLOAD)

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents([_make_request()])

    assert len(results) == 1
    resp = results[0]
    assert resp.doc_id == "doc-1"
    assert resp.payload == _SAMPLE_PAYLOAD
    assert resp.errors == []


async def test_single_chunk_invokes_extract_with_correct_inputs():
    mock_chain = _make_mock_chain(_SAMPLE_PAYLOAD)

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        req = _make_request(doc_id="d1", text="hello world")
        await enrich_documents([req])

    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["document_id"] == "d1"
    assert call_args["document_text"] == "hello world"
    assert call_args["title"] == "Test Doc"
    assert call_args["source"] == "https://example.com"


# ---------------------------------------------------------------------------
# Response metadata
# ---------------------------------------------------------------------------

async def test_response_metadata_fields_are_populated():
    mock_chain = _make_mock_chain(_SAMPLE_PAYLOAD)
    cfg = EnrichmentConfig(model=ModelConfig(model_name="gpt-test"))

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents([_make_request()], config=cfg)

    resp = results[0]
    assert resp.model_name == "gpt-test"
    assert resp.prompt_version == "v1"
    assert resp.schema_version == "document_payload_v1"
    assert resp.enrichment_type == "document_payload"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_llm_exception_returns_error_response_without_raising():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents([_make_request(doc_id="err-doc")])

    assert len(results) == 1
    resp = results[0]
    assert resp.doc_id == "err-doc"
    assert resp.payload is None
    assert len(resp.errors) == 1
    assert "LLM timeout" in resp.errors[0]


async def test_one_failure_does_not_affect_other_responses():
    good_payload = KnowledgePayload(summary="good doc")

    async def selective_ainvoke(inputs, **kwargs):
        if inputs.get("document_id") == "bad":
            raise ValueError("forced error")
        return good_payload

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=selective_ainvoke)

    requests = [
        _make_request(doc_id="good-1"),
        _make_request(doc_id="bad"),
        _make_request(doc_id="good-2"),
    ]

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(requests)

    assert results[0].payload is not None
    assert results[1].payload is None
    assert len(results[1].errors) == 1
    assert results[2].payload is not None


# ---------------------------------------------------------------------------
# Multi-chunk path
# ---------------------------------------------------------------------------

async def test_multi_chunk_triggers_reduce_chain():
    """Text > 180 000 chars is split; the reduce chain is called once."""
    long_text = "word " * 40_000  # ~200 000 chars

    chunk_payload = KnowledgePayload(summary="chunk result")
    reduce_payload = KnowledgePayload(summary="reduced result")

    call_count = {"n": 0}

    async def mock_ainvoke(inputs, **kwargs):
        call_count["n"] += 1
        if "chunk_payloads" in inputs:
            return reduce_payload
        return chunk_payload

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents([_make_request(text=long_text)])

    assert len(results) == 1
    assert results[0].payload == reduce_payload
    assert call_count["n"] > 1  # at least one chunk call + one reduce call


async def test_multi_chunk_all_chunks_fail_returns_error():
    long_text = "word " * 40_000

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("chunk failure"))

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents([_make_request(text=long_text)])

    resp = results[0]
    assert resp.payload is None
    assert resp.errors != []
