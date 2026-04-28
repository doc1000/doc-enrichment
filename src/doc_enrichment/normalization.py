from __future__ import annotations

from pydantic import BaseModel, Field

from doc_enrichment.schemas import BranchEnrichmentResponse


class NodeEnrichmentData(BaseModel):
    """Combined contrast and parent-refinement data for a single child node.

    Produced by ``normalize_node_enrichments`` from a ``BranchEnrichmentResponse``.
    The contrast fields describe how this node differs from its sibling.
    The parent_refine fields describe how this node differs from its parent.
    """

    contrast: list[str] = Field(default_factory=list)
    contrast_tag: list[str] = Field(default_factory=list)
    parent_refine: list[str] = Field(default_factory=list)
    parent_refine_tag: list[str] = Field(default_factory=list)


def normalize_node_enrichments(
    responses: list[BranchEnrichmentResponse],
) -> dict[str, NodeEnrichmentData]:
    """Map each child node_id to its combined contrast and refinement data.

    For each response with a populated ``enrichment``:
    - ``left_node_id``  → sibling_a contrast  + parent_refinement_for_a
    - ``right_node_id`` → sibling_b contrast  + parent_refinement_for_b

    Responses whose ``enrichment`` is ``None`` (error cases) are skipped.
    If the same node_id appears in multiple responses, the last one wins.
    """
    result: dict[str, NodeEnrichmentData] = {}

    for resp in responses:
        if resp.enrichment is None:
            continue

        e = resp.enrichment

        result[resp.left_node_id] = NodeEnrichmentData(
            contrast=list(e.sibling_a.contrast),
            contrast_tag=list(e.sibling_a.contrast_tag),
            parent_refine=list(e.parent_refinement_for_a.parent_refine),
            parent_refine_tag=list(e.parent_refinement_for_a.parent_refine_tag),
        )

        result[resp.right_node_id] = NodeEnrichmentData(
            contrast=list(e.sibling_b.contrast),
            contrast_tag=list(e.sibling_b.contrast_tag),
            parent_refine=list(e.parent_refinement_for_b.parent_refine),
            parent_refine_tag=list(e.parent_refinement_for_b.parent_refine_tag),
        )

    return result
