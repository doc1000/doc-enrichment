"""Tests for the parent enricher.

All LLM calls are mocked; no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_enrichment.config import EnrichmentConfig, ModelConfig
from doc_enrichment.enrichers.parent import enrich_parent_nodes
from doc_enrichment.schemas import (
    BranchEnrichmentRequest,
    KnowledgePayload,
)

_LEFT = KnowledgePayload(summary="left child")
_RIGHT = KnowledgePayload(summary="right child")
_PARENT = KnowledgePayload(summary="merged parent")


def _make_request(
    branch_id: str = "branch-1",
    left_id: str = "left-1",
    right_id: str = "right-1",
    left: KnowledgePayload = _LEFT,
    right: KnowledgePayload = _RIGHT,
) -> BranchEnrichmentRequest:
    return BranchEnrichmentRequest(
        branch_id=branch_id,
        left_node_id=left_id,
        right_node_id=right_id,
        left_payload=left,
        right_payload=right,
    )


def _mock_parent_chain(return_value: KnowledgePayload = _PARENT) -> MagicMock:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

async def test_empty_request_list_returns_empty():
    results = await enrich_parent_nodes([])
    assert results == []


# ---------------------------------------------------------------------------
# Single merge
# ---------------------------------------------------------------------------

async def test_single_merge_returns_parent_payload():
    mock_chain = _mock_parent_chain(_PARENT)

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes([_make_request()])

    assert len(results) == 1
    resp = results[0]
    assert resp.branch_id == "branch-1"
    assert resp.enrichment is not None
    assert resp.enrichment.parent_payload == _PARENT
    assert resp.errors == []


async def test_merge_sets_contrast_and_refinement_to_empty_defaults():
    """Contrast and refinement fields are empty; PR4 fills them."""
    mock_chain = _mock_parent_chain(_PARENT)

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes([_make_request()])

    enrichment = results[0].enrichment
    assert enrichment.sibling_a.contrast == []
    assert enrichment.sibling_b.contrast == []
    assert enrichment.parent_refinement_for_a.parent_refine == []
    assert enrichment.parent_refinement_for_b.parent_refine == []


async def test_chain_receives_correct_branch_inputs():
    mock_chain = _mock_parent_chain(_PARENT)

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        req = _make_request(branch_id="b-99", left_id="l-1", right_id="r-1")
        await enrich_parent_nodes([req])

    call_inputs = mock_chain.ainvoke.call_args[0][0]
    assert call_inputs["branch_id"] == "b-99"
    assert call_inputs["left_node_id"] == "l-1"
    assert call_inputs["right_node_id"] == "r-1"
    assert "left_payload" in call_inputs
    assert "right_payload" in call_inputs


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

async def test_response_metadata_fields():
    mock_chain = _mock_parent_chain(_PARENT)
    cfg = EnrichmentConfig(model=ModelConfig(model_name="gpt-test-parent"))

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes([_make_request()], config=cfg)

    resp = results[0]
    assert resp.model_name == "gpt-test-parent"
    assert resp.prompt_version == "v1"
    assert resp.schema_version == "branch_enrichment_v1"
    assert resp.enrichment_type == "branch_enrichment"
    assert resp.left_node_id == "left-1"
    assert resp.right_node_id == "right-1"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_llm_exception_returns_error_response_without_raising():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes([_make_request(branch_id="err-branch")])

    assert len(results) == 1
    resp = results[0]
    assert resp.enrichment is None
    assert any("timeout" in e for e in resp.errors)


async def test_one_failure_does_not_affect_other_responses():
    async def selective_ainvoke(inputs, **kwargs):
        if inputs["branch_id"] == "bad":
            raise ValueError("forced failure")
        return KnowledgePayload(summary=f"merged {inputs['branch_id']}")

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=selective_ainvoke)

    requests = [
        _make_request(branch_id="good-1"),
        _make_request(branch_id="bad"),
        _make_request(branch_id="good-2"),
    ]

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes(requests)

    assert results[0].enrichment is not None
    assert results[1].enrichment is None
    assert len(results[1].errors) == 1
    assert results[2].enrichment is not None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

async def test_response_order_matches_request_order():
    async def ainvoke(inputs, **kwargs):
        return KnowledgePayload(summary=inputs["branch_id"])

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=ainvoke)

    requests = [_make_request(branch_id=str(i)) for i in range(10)]

    with patch(
        "doc_enrichment.enrichers.parent.build_parent_chain",
        return_value=mock_chain,
    ):
        results = await enrich_parent_nodes(requests)

    for req, resp in zip(requests, results):
        assert resp.branch_id == req.branch_id
