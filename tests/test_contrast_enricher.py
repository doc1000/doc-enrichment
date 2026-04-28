"""Tests for the contrast enricher.

All LLM calls are mocked; no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_enrichment.config import EnrichmentConfig, ModelConfig
from doc_enrichment.enrichers.contrast import enrich_contrast
from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentRequest,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
)

_LEFT = KnowledgePayload(summary="left child")
_RIGHT = KnowledgePayload(summary="right child")

_FULL_ENRICHMENT = BranchEnrichment(
    sibling_a=SiblingContrast(
        contrast=["a differs here", "a is more specific", "a focuses on X"],
        contrast_tag=["tag-a1", "tag-a2", "tag-a3"],
    ),
    sibling_b=SiblingContrast(
        contrast=["b differs here", "b is broader", "b covers Y"],
        contrast_tag=["tag-b1", "tag-b2", "tag-b3"],
    ),
    parent_payload=KnowledgePayload(summary="combined parent"),
    parent_refinement_for_a=ParentRefinement(
        parent_refine=["refine-a1", "refine-a2", "refine-a3"],
        parent_refine_tag=["rtag-a1", "rtag-a2", "rtag-a3"],
    ),
    parent_refinement_for_b=ParentRefinement(
        parent_refine=["refine-b1", "refine-b2", "refine-b3"],
        parent_refine_tag=["rtag-b1", "rtag-b2", "rtag-b3"],
    ),
)


def _make_request(
    branch_id: str = "branch-1",
    left_id: str = "left-1",
    right_id: str = "right-1",
) -> BranchEnrichmentRequest:
    return BranchEnrichmentRequest(
        branch_id=branch_id,
        left_node_id=left_id,
        right_node_id=right_id,
        left_payload=_LEFT,
        right_payload=_RIGHT,
    )


def _mock_contrast_chain(return_value: BranchEnrichment = _FULL_ENRICHMENT) -> MagicMock:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

async def test_empty_request_list_returns_empty():
    results = await enrich_contrast([])
    assert results == []


# ---------------------------------------------------------------------------
# Single request — payload content
# ---------------------------------------------------------------------------

async def test_single_request_returns_full_enrichment():
    mock_chain = _mock_contrast_chain(_FULL_ENRICHMENT)

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast([_make_request()])

    assert len(results) == 1
    resp = results[0]
    assert resp.enrichment == _FULL_ENRICHMENT
    assert resp.errors == []


async def test_enrichment_sibling_fields_populated():
    mock_chain = _mock_contrast_chain(_FULL_ENRICHMENT)

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast([_make_request()])

    e = results[0].enrichment
    assert e.sibling_a.contrast == ["a differs here", "a is more specific", "a focuses on X"]
    assert e.sibling_b.contrast_tag == ["tag-b1", "tag-b2", "tag-b3"]
    assert e.parent_refinement_for_a.parent_refine == ["refine-a1", "refine-a2", "refine-a3"]
    assert e.parent_refinement_for_b.parent_refine_tag == ["rtag-b1", "rtag-b2", "rtag-b3"]
    assert e.parent_payload.summary == "combined parent"


async def test_chain_receives_payload_a_and_payload_b():
    """Contrast chain uses payload_a/payload_b, not left_payload/right_payload."""
    mock_chain = _mock_contrast_chain(_FULL_ENRICHMENT)

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        await enrich_contrast([_make_request()])

    call_inputs = mock_chain.ainvoke.call_args[0][0]
    assert "payload_a" in call_inputs
    assert "payload_b" in call_inputs
    assert "left_payload" not in call_inputs
    assert "right_payload" not in call_inputs


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

async def test_response_metadata_fields():
    mock_chain = _mock_contrast_chain(_FULL_ENRICHMENT)
    cfg = EnrichmentConfig(model=ModelConfig(model_name="gpt-contrast-test"))

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast([_make_request()], config=cfg)

    resp = results[0]
    assert resp.model_name == "gpt-contrast-test"
    assert resp.prompt_version == "v1"
    assert resp.schema_version == "branch_enrichment_v1"
    assert resp.enrichment_type == "branch_enrichment"
    assert resp.branch_id == "branch-1"
    assert resp.left_node_id == "left-1"
    assert resp.right_node_id == "right-1"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_llm_exception_returns_error_response_without_raising():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("contrast timeout"))

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast([_make_request(branch_id="err-branch")])

    assert len(results) == 1
    resp = results[0]
    assert resp.enrichment is None
    assert any("contrast timeout" in e for e in resp.errors)


async def test_one_failure_does_not_affect_other_responses():
    async def selective_ainvoke(inputs, **kwargs):
        if "bad" in inputs.get("payload_a", ""):
            raise ValueError("forced failure")
        return _FULL_ENRICHMENT

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=selective_ainvoke)

    bad_left = KnowledgePayload(summary="bad payload")
    requests = [
        _make_request(branch_id="good-1"),
        BranchEnrichmentRequest(
            branch_id="bad",
            left_node_id="l-bad",
            right_node_id="r-bad",
            left_payload=bad_left,
            right_payload=_RIGHT,
        ),
        _make_request(branch_id="good-2"),
    ]

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast(requests)

    assert results[0].enrichment is not None
    assert results[1].enrichment is None
    assert len(results[1].errors) == 1
    assert results[2].enrichment is not None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

async def test_response_order_matches_request_order():
    async def ainvoke(inputs, **kwargs):
        return BranchEnrichment(
            sibling_a=SiblingContrast(),
            sibling_b=SiblingContrast(),
            parent_payload=KnowledgePayload(summary=inputs["payload_a"][:10]),
            parent_refinement_for_a=ParentRefinement(),
            parent_refinement_for_b=ParentRefinement(),
        )

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=ainvoke)

    requests = [_make_request(branch_id=str(i)) for i in range(10)]

    with patch(
        "doc_enrichment.enrichers.contrast.build_contrast_chain",
        return_value=mock_chain,
    ):
        results = await enrich_contrast(requests)

    for req, resp in zip(requests, results):
        assert resp.branch_id == req.branch_id
