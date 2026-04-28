"""Tests for deterministic batch ordering in enrich_documents.

Ordering guarantee: results[i] always corresponds to requests[i],
regardless of which coroutines happen to complete first.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_enrichment.config import EnrichmentConfig, ModelConfig
from doc_enrichment.enrichers.document import enrich_documents
from doc_enrichment.schemas import DocumentEnrichmentRequest, KnowledgePayload


def _requests(n: int) -> list[DocumentEnrichmentRequest]:
    return [
        DocumentEnrichmentRequest(doc_id=str(i), text=f"document text {i}")
        for i in range(n)
    ]


def _mock_chain_returning_id_as_summary() -> MagicMock:
    """Chain whose summary echoes the document_id so ordering can be verified."""
    async def ainvoke(inputs, **kwargs):
        return KnowledgePayload(summary=inputs["document_id"])

    chain = MagicMock()
    chain.ainvoke = AsyncMock(side_effect=ainvoke)
    return chain


# ---------------------------------------------------------------------------
# Result count matches request count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3, 10, 25])
async def test_result_count_matches_request_count(n: int):
    mock_chain = _mock_chain_returning_id_as_summary()

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(_requests(n))

    assert len(results) == n


# ---------------------------------------------------------------------------
# Response ordering matches request ordering
# ---------------------------------------------------------------------------

async def test_response_order_matches_request_order():
    """results[i].doc_id == requests[i].doc_id for every index."""
    reqs = _requests(15)
    mock_chain = _mock_chain_returning_id_as_summary()

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(reqs)

    for i, (req, resp) in enumerate(zip(reqs, results)):
        assert resp.doc_id == req.doc_id, (
            f"Position {i}: expected doc_id={req.doc_id!r}, got {resp.doc_id!r}"
        )


async def test_payload_summary_aligns_with_request_doc_id():
    """The payload produced for each request contains that request's doc_id."""
    reqs = _requests(10)
    mock_chain = _mock_chain_returning_id_as_summary()

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(reqs)

    for req, resp in zip(reqs, results):
        assert resp.payload is not None
        assert resp.payload.summary == req.doc_id


# ---------------------------------------------------------------------------
# Ordering preserved when tasks complete out of submission order
# ---------------------------------------------------------------------------

async def test_ordering_preserved_with_varied_completion_times():
    """Simulate tasks completing in reverse submission order; index alignment holds."""
    n = 8
    reqs = _requests(n)

    async def delayed_ainvoke(inputs, **kwargs):
        doc_id = inputs["document_id"]
        delay = (n - int(doc_id)) * 0.01  # last submitted doc finishes first
        await asyncio.sleep(delay)
        return KnowledgePayload(summary=doc_id)

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=delayed_ainvoke)

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(reqs)

    for req, resp in zip(reqs, results):
        assert resp.doc_id == req.doc_id
        assert resp.payload is not None
        assert resp.payload.summary == req.doc_id


# ---------------------------------------------------------------------------
# Ordering preserved when some requests fail
# ---------------------------------------------------------------------------

async def test_ordering_preserved_with_mixed_success_and_failure():
    """Errors at even indexes do not shift the positions of successful responses."""
    reqs = _requests(6)

    async def mixed_ainvoke(inputs, **kwargs):
        doc_id = inputs["document_id"]
        if int(doc_id) % 2 == 0:
            raise ValueError(f"forced error for doc {doc_id}")
        return KnowledgePayload(summary=doc_id)

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=mixed_ainvoke)

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(reqs)

    assert len(results) == len(reqs)
    for i, (req, resp) in enumerate(zip(reqs, results)):
        assert resp.doc_id == req.doc_id
        if i % 2 == 0:
            assert resp.payload is None
            assert resp.errors != []
        else:
            assert resp.payload is not None
            assert resp.errors == []


# ---------------------------------------------------------------------------
# Concurrency limit is respected (semaphore not exceeded)
# ---------------------------------------------------------------------------

async def test_concurrency_limit_not_exceeded():
    """Never more than max_concurrency tasks hold the semaphore simultaneously."""
    max_concurrency = 3
    n = 12
    reqs = _requests(n)

    active: list[int] = []
    peak: list[int] = [0]

    async def tracked_ainvoke(inputs, **kwargs):
        active.append(1)
        peak[0] = max(peak[0], len(active))
        await asyncio.sleep(0.01)
        active.pop()
        return KnowledgePayload(summary=inputs["document_id"])

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=tracked_ainvoke)

    cfg = EnrichmentConfig(model=ModelConfig(max_concurrency=max_concurrency))

    with patch(
        "doc_enrichment.enrichers.document.build_document_chain",
        return_value=mock_chain,
    ):
        results = await enrich_documents(reqs, config=cfg)

    assert len(results) == n
    assert peak[0] <= max_concurrency
