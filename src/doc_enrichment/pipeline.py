from __future__ import annotations

from doc_enrichment.config import EnrichmentConfig
from doc_enrichment.enrichers.parent import enrich_parent_nodes
from doc_enrichment.errors import PipelineError
from doc_enrichment.schemas import BranchEnrichmentRequest, KnowledgePayload


def _prev_power_of_two(n: int) -> int:
    """Return the largest power of 2 that is ≤ n."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return 1 << (n.bit_length() - 1)


async def run_tournament(
    leaves: list[tuple[str, KnowledgePayload]],
    config: EnrichmentConfig | None = None,
) -> KnowledgePayload:
    """Reduce a list of (node_id, KnowledgePayload) pairs to one root payload.

    Algorithm:
    1. Play-in round: if len(leaves) is not a power of 2, pair the first
       2 * (N - prev_power_of_two(N)) nodes to reach the nearest lower
       power-of-two count.
    2. Regular rounds: pair adjacent nodes and merge until one root remains.

    Raises:
        ValueError: If ``leaves`` is empty.
        PipelineError: If any merge call fails (enrichment is None in the response).
    """
    if not leaves:
        raise ValueError("run_tournament requires at least one leaf node.")

    if len(leaves) == 1:
        return leaves[0][1]

    if config is None:
        config = EnrichmentConfig()

    items: list[tuple[str, KnowledgePayload]] = list(leaves)

    # ------------------------------------------------------------------
    # Play-in round
    # ------------------------------------------------------------------
    n = len(items)
    target = _prev_power_of_two(n)
    play_in_count = n - target

    if play_in_count > 0:
        play_in_nodes = items[: 2 * play_in_count]
        remaining = items[2 * play_in_count :]

        requests = [
            BranchEnrichmentRequest(
                branch_id=f"playin_{i}",
                left_node_id=play_in_nodes[2 * i][0],
                right_node_id=play_in_nodes[2 * i + 1][0],
                left_payload=play_in_nodes[2 * i][1],
                right_payload=play_in_nodes[2 * i + 1][1],
            )
            for i in range(play_in_count)
        ]

        responses = await enrich_parent_nodes(requests, config)
        items = _extract_merged(responses) + remaining

    # ------------------------------------------------------------------
    # Regular rounds: len(items) is now a power of 2
    # ------------------------------------------------------------------
    round_num = 0
    while len(items) > 1:
        round_num += 1
        requests = [
            BranchEnrichmentRequest(
                branch_id=f"round_{round_num}_pair_{i}",
                left_node_id=items[2 * i][0],
                right_node_id=items[2 * i + 1][0],
                left_payload=items[2 * i][1],
                right_payload=items[2 * i + 1][1],
            )
            for i in range(len(items) // 2)
        ]

        responses = await enrich_parent_nodes(requests, config)
        items = _extract_merged(responses)

    return items[0][1]


def _extract_merged(
    responses: list,
) -> list[tuple[str, KnowledgePayload]]:
    """Pull (branch_id, parent_payload) from responses; raise PipelineError on failure."""
    merged: list[tuple[str, KnowledgePayload]] = []
    for resp in responses:
        if resp.enrichment is None:
            raise PipelineError(
                f"Tournament merge failed for branch '{resp.branch_id}': {resp.errors}"
            )
        merged.append((resp.branch_id, resp.enrichment.parent_payload))
    return merged
