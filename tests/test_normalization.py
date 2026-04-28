"""Tests for normalize_node_enrichments.

Pure function; no mocking needed.
"""
from __future__ import annotations

import pytest

from doc_enrichment.normalization import NodeEnrichmentData, normalize_node_enrichments
from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentResponse,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enrichment(
    contrast_a: list[str] | None = None,
    contrast_tag_a: list[str] | None = None,
    refine_a: list[str] | None = None,
    refine_tag_a: list[str] | None = None,
    contrast_b: list[str] | None = None,
    contrast_tag_b: list[str] | None = None,
    refine_b: list[str] | None = None,
    refine_tag_b: list[str] | None = None,
) -> BranchEnrichment:
    return BranchEnrichment(
        sibling_a=SiblingContrast(
            contrast=contrast_a or [],
            contrast_tag=contrast_tag_a or [],
        ),
        sibling_b=SiblingContrast(
            contrast=contrast_b or [],
            contrast_tag=contrast_tag_b or [],
        ),
        parent_payload=KnowledgePayload(summary="parent"),
        parent_refinement_for_a=ParentRefinement(
            parent_refine=refine_a or [],
            parent_refine_tag=refine_tag_a or [],
        ),
        parent_refinement_for_b=ParentRefinement(
            parent_refine=refine_b or [],
            parent_refine_tag=refine_tag_b or [],
        ),
    )


def _make_response(
    branch_id: str,
    left_id: str,
    right_id: str,
    enrichment: BranchEnrichment | None = None,
) -> BranchEnrichmentResponse:
    return BranchEnrichmentResponse(
        branch_id=branch_id,
        left_node_id=left_id,
        right_node_id=right_id,
        prompt_version="v1",
        model_name="mock",
        enrichment=enrichment,
        errors=[] if enrichment is not None else ["error"],
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_responses_returns_empty_dict():
    assert normalize_node_enrichments([]) == {}


# ---------------------------------------------------------------------------
# Single response — key mapping
# ---------------------------------------------------------------------------

def test_single_response_maps_both_node_ids():
    enrichment = _make_enrichment()
    resp = _make_response("b1", "left-1", "right-1", enrichment)

    result = normalize_node_enrichments([resp])

    assert "left-1" in result
    assert "right-1" in result
    assert len(result) == 2


def test_left_node_gets_sibling_a_and_refinement_for_a():
    enrichment = _make_enrichment(
        contrast_a=["a-diff-1", "a-diff-2"],
        contrast_tag_a=["tag-a"],
        refine_a=["refine-a-1"],
        refine_tag_a=["rtag-a"],
        contrast_b=["b-diff-1"],
        contrast_tag_b=["tag-b"],
        refine_b=["refine-b-1"],
        refine_tag_b=["rtag-b"],
    )
    resp = _make_response("b1", "node-L", "node-R", enrichment)

    result = normalize_node_enrichments([resp])

    left = result["node-L"]
    assert left.contrast == ["a-diff-1", "a-diff-2"]
    assert left.contrast_tag == ["tag-a"]
    assert left.parent_refine == ["refine-a-1"]
    assert left.parent_refine_tag == ["rtag-a"]


def test_right_node_gets_sibling_b_and_refinement_for_b():
    enrichment = _make_enrichment(
        contrast_a=["a-diff"],
        contrast_tag_a=["tag-a"],
        refine_a=["refine-a"],
        refine_tag_a=["rtag-a"],
        contrast_b=["b-diff-1", "b-diff-2", "b-diff-3"],
        contrast_tag_b=["tag-b1", "tag-b2"],
        refine_b=["refine-b-1", "refine-b-2"],
        refine_tag_b=["rtag-b"],
    )
    resp = _make_response("b1", "node-L", "node-R", enrichment)

    result = normalize_node_enrichments([resp])

    right = result["node-R"]
    assert right.contrast == ["b-diff-1", "b-diff-2", "b-diff-3"]
    assert right.contrast_tag == ["tag-b1", "tag-b2"]
    assert right.parent_refine == ["refine-b-1", "refine-b-2"]
    assert right.parent_refine_tag == ["rtag-b"]


# ---------------------------------------------------------------------------
# Error responses are skipped
# ---------------------------------------------------------------------------

def test_response_with_none_enrichment_is_skipped():
    error_resp = _make_response("b-err", "left-err", "right-err", enrichment=None)

    result = normalize_node_enrichments([error_resp])

    assert result == {}


def test_mixed_success_and_error_responses():
    ok_enrichment = _make_enrichment(
        contrast_a=["ca"], contrast_tag_a=["cta"],
        refine_a=["ra"], refine_tag_a=["rta"],
        contrast_b=["cb"], contrast_tag_b=["ctb"],
        refine_b=["rb"], refine_tag_b=["rtb"],
    )
    responses = [
        _make_response("b-ok", "left-ok", "right-ok", ok_enrichment),
        _make_response("b-err", "left-err", "right-err", enrichment=None),
    ]

    result = normalize_node_enrichments(responses)

    assert "left-ok" in result
    assert "right-ok" in result
    assert "left-err" not in result
    assert "right-err" not in result


# ---------------------------------------------------------------------------
# Multiple responses
# ---------------------------------------------------------------------------

def test_multiple_responses_all_nodes_mapped():
    responses = [
        _make_response(f"b{i}", f"left-{i}", f"right-{i}", _make_enrichment())
        for i in range(5)
    ]

    result = normalize_node_enrichments(responses)

    assert len(result) == 10  # 2 nodes per response
    for i in range(5):
        assert f"left-{i}" in result
        assert f"right-{i}" in result


def test_each_node_id_maps_to_node_enrichment_data_instance():
    responses = [
        _make_response("b1", "l1", "r1", _make_enrichment()),
        _make_response("b2", "l2", "r2", _make_enrichment()),
    ]

    result = normalize_node_enrichments(responses)

    for node_data in result.values():
        assert isinstance(node_data, NodeEnrichmentData)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_input_produces_same_output():
    enrichment = _make_enrichment(
        contrast_a=["x"], contrast_tag_a=["xt"],
        refine_a=["y"], refine_tag_a=["yt"],
        contrast_b=["p"], contrast_tag_b=["pt"],
        refine_b=["q"], refine_tag_b=["qt"],
    )
    responses = [_make_response("b1", "l1", "r1", enrichment)]

    result_a = normalize_node_enrichments(responses)
    result_b = normalize_node_enrichments(responses)

    assert result_a == result_b


def test_list_fields_are_independent_copies():
    """Mutating the original enrichment does not affect the normalized output."""
    enrichment = _make_enrichment(contrast_a=["original"])
    resp = _make_response("b1", "l1", "r1", enrichment)

    result = normalize_node_enrichments([resp])
    result["l1"].contrast.append("mutated")

    assert enrichment.sibling_a.contrast == ["original"]


# ---------------------------------------------------------------------------
# Last-response-wins for duplicate node_ids
# ---------------------------------------------------------------------------

def test_duplicate_node_id_last_response_wins():
    enrichment_first = _make_enrichment(contrast_a=["first"])
    enrichment_last = _make_enrichment(contrast_a=["last"])

    responses = [
        _make_response("b1", "shared-node", "r1", enrichment_first),
        _make_response("b2", "shared-node", "r2", enrichment_last),
    ]

    result = normalize_node_enrichments(responses)

    assert result["shared-node"].contrast == ["last"]
