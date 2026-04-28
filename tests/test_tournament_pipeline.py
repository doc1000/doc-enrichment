"""Tests for the tournament pipeline.

enrich_parent_nodes is mocked throughout; these tests verify bracket structure,
play-in logic, and ordering without real LLM calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from doc_enrichment.errors import PipelineError
from doc_enrichment.pipeline import _prev_power_of_two, run_tournament
from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentResponse,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _leaf(node_id: str) -> tuple[str, KnowledgePayload]:
    return (node_id, KnowledgePayload(summary=f"leaf:{node_id}"))


def _success_response(branch_id: str, left_id: str, right_id: str) -> BranchEnrichmentResponse:
    parent = KnowledgePayload(summary=f"merged:{left_id}+{right_id}")
    enrichment = BranchEnrichment(
        sibling_a=SiblingContrast(),
        sibling_b=SiblingContrast(),
        parent_payload=parent,
        parent_refinement_for_a=ParentRefinement(),
        parent_refinement_for_b=ParentRefinement(),
    )
    return BranchEnrichmentResponse(
        branch_id=branch_id,
        left_node_id=left_id,
        right_node_id=right_id,
        prompt_version="v1",
        model_name="mock",
        enrichment=enrichment,
    )


def _failure_response(branch_id: str, left_id: str, right_id: str) -> BranchEnrichmentResponse:
    return BranchEnrichmentResponse(
        branch_id=branch_id,
        left_node_id=left_id,
        right_node_id=right_id,
        prompt_version="v1",
        model_name="mock",
        enrichment=None,
        errors=["forced failure"],
    )


def _mock_enrich_parent_nodes(call_log: list | None = None):
    """Return an async mock that builds plausible merged responses."""
    async def _side_effect(requests, config=None):
        resps = []
        for req in requests:
            r = _success_response(req.branch_id, req.left_node_id, req.right_node_id)
            resps.append(r)
            if call_log is not None:
                call_log.append(req.branch_id)
        return resps

    return AsyncMock(side_effect=_side_effect)


# ---------------------------------------------------------------------------
# _prev_power_of_two
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, 1), (2, 2), (3, 2), (4, 4), (5, 4),
    (6, 4), (7, 4), (8, 8), (9, 8), (16, 16), (17, 16),
])
def test_prev_power_of_two(n: int, expected: int):
    assert _prev_power_of_two(n) == expected


def test_prev_power_of_two_zero_raises():
    with pytest.raises(ValueError):
        _prev_power_of_two(0)


# ---------------------------------------------------------------------------
# Edge cases: empty and single leaf
# ---------------------------------------------------------------------------

async def test_empty_leaves_raises():
    with pytest.raises(ValueError):
        await run_tournament([])


async def test_single_leaf_returns_immediately_without_llm_call():
    leaf = _leaf("n0")
    with patch("doc_enrichment.pipeline.enrich_parent_nodes") as mock:
        result = await run_tournament([leaf])

    mock.assert_not_called()
    assert result.summary == "leaf:n0"


# ---------------------------------------------------------------------------
# Power-of-two inputs (no play-in)
# ---------------------------------------------------------------------------

async def test_two_leaves_one_round():
    leaves = [_leaf("a"), _leaf("b")]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        result = await run_tournament(leaves)

    assert len(call_log) == 1
    assert call_log[0] == "round_1_pair_0"
    assert "a" in result.summary and "b" in result.summary


async def test_four_leaves_two_rounds():
    leaves = [_leaf(str(i)) for i in range(4)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        result = await run_tournament(leaves)

    assert len(call_log) == 3  # 2 in round 1, 1 in round 2
    round1 = [c for c in call_log if c.startswith("round_1")]
    round2 = [c for c in call_log if c.startswith("round_2")]
    assert len(round1) == 2
    assert len(round2) == 1


async def test_eight_leaves_three_rounds():
    leaves = [_leaf(str(i)) for i in range(8)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        await run_tournament(leaves)

    assert len(call_log) == 7  # 4 + 2 + 1
    assert len([c for c in call_log if c.startswith("round_1")]) == 4
    assert len([c for c in call_log if c.startswith("round_2")]) == 2
    assert len([c for c in call_log if c.startswith("round_3")]) == 1


# ---------------------------------------------------------------------------
# Non-power-of-two inputs (play-in)
# ---------------------------------------------------------------------------

async def test_three_leaves_play_in_then_one_round():
    leaves = [_leaf(str(i)) for i in range(3)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        await run_tournament(leaves)

    assert len(call_log) == 2  # 1 play-in + 1 regular
    assert call_log[0] == "playin_0"
    assert call_log[1] == "round_1_pair_0"


async def test_five_leaves_play_in_then_two_rounds():
    # n=5 → target=4, play_in=1 (uses leaves 0,1), remaining=[2,3,4]
    # After play-in: [playin_0, 2, 3, 4] = 4 nodes → 2 rounds
    leaves = [_leaf(str(i)) for i in range(5)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        await run_tournament(leaves)

    assert call_log[0] == "playin_0"
    assert len([c for c in call_log if c.startswith("round_1")]) == 2
    assert len([c for c in call_log if c.startswith("round_2")]) == 1


async def test_six_leaves_play_in_count():
    # n=6 → target=4, play_in=2 (uses leaves 0-3), remaining=[4,5]
    # After play-in: [playin_0, playin_1, 4, 5] = 4 nodes → 2 rounds
    leaves = [_leaf(str(i)) for i in range(6)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        await run_tournament(leaves)

    play_in = [c for c in call_log if c.startswith("playin")]
    assert len(play_in) == 2
    assert "playin_0" in play_in
    assert "playin_1" in play_in


async def test_seven_leaves_play_in_count():
    # n=7 → target=4, play_in=3 (uses leaves 0-5), remaining=[6]
    # After play-in: [playin_0, playin_1, playin_2, 6] = 4 nodes → 2 rounds
    leaves = [_leaf(str(i)) for i in range(7)]
    call_log: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(call_log),
    ):
        await run_tournament(leaves)

    play_in = [c for c in call_log if c.startswith("playin")]
    assert len(play_in) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

async def test_same_inputs_same_bracket_order():
    """The same leaf list produces the same call sequence across two runs."""
    leaves = [_leaf(str(i)) for i in range(6)]

    log_a: list[str] = []
    log_b: list[str] = []

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(log_a),
    ):
        await run_tournament(leaves)

    with patch(
        "doc_enrichment.pipeline.enrich_parent_nodes",
        new=_mock_enrich_parent_nodes(log_b),
    ):
        await run_tournament(leaves)

    assert log_a == log_b


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

async def test_failed_merge_raises_pipeline_error():
    async def failing_enrich(requests, config=None):
        return [_failure_response(req.branch_id, req.left_node_id, req.right_node_id)
                for req in requests]

    leaves = [_leaf("a"), _leaf("b")]

    with patch("doc_enrichment.pipeline.enrich_parent_nodes", new=AsyncMock(side_effect=failing_enrich)):
        with pytest.raises(PipelineError, match="branch"):
            await run_tournament(leaves)


async def test_partial_failure_in_round_raises_pipeline_error():
    """Even one failed merge in a round stops the tournament."""
    call_n = {"n": 0}

    async def partial_fail_enrich(requests, config=None):
        call_n["n"] += 1
        resps = []
        for i, req in enumerate(requests):
            if call_n["n"] == 1 and i == 1:  # fail the second pair in round 1
                resps.append(_failure_response(req.branch_id, req.left_node_id, req.right_node_id))
            else:
                resps.append(_success_response(req.branch_id, req.left_node_id, req.right_node_id))
        return resps

    leaves = [_leaf(str(i)) for i in range(4)]

    with patch("doc_enrichment.pipeline.enrich_parent_nodes", new=AsyncMock(side_effect=partial_fail_enrich)):
        with pytest.raises(PipelineError):
            await run_tournament(leaves)
