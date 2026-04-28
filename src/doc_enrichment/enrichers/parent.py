from __future__ import annotations

import asyncio

from doc_enrichment.chains import build_parent_chain
from doc_enrichment.config import EnrichmentConfig
from doc_enrichment.prompt_loader import load_prompt
from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentRequest,
    BranchEnrichmentResponse,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
)


async def _enrich_one_parent(
    request: BranchEnrichmentRequest,
    parent_chain,
    semaphore: asyncio.Semaphore,
    prompt_version: str,
    model_name: str,
) -> BranchEnrichmentResponse:
    base = BranchEnrichmentResponse(
        branch_id=request.branch_id,
        left_node_id=request.left_node_id,
        right_node_id=request.right_node_id,
        prompt_version=prompt_version,
        model_name=model_name,
    )

    try:
        async with semaphore:
            parent_payload: KnowledgePayload = await parent_chain.ainvoke({
                "branch_id": request.branch_id,
                "left_node_id": request.left_node_id,
                "right_node_id": request.right_node_id,
                "left_payload": request.left_payload.model_dump_json(indent=2),
                "right_payload": request.right_payload.model_dump_json(indent=2),
            })

        enrichment = BranchEnrichment(
            sibling_a=SiblingContrast(),
            sibling_b=SiblingContrast(),
            parent_payload=parent_payload,
            parent_refinement_for_a=ParentRefinement(),
            parent_refinement_for_b=ParentRefinement(),
        )
        return base.model_copy(update={"enrichment": enrichment})

    except Exception as exc:  # noqa: BLE001
        return base.model_copy(update={"errors": [str(exc)]})


async def enrich_parent_nodes(
    requests: list[BranchEnrichmentRequest],
    config: EnrichmentConfig | None = None,
) -> list[BranchEnrichmentResponse]:
    """Merge pairs of child payloads into parent KnowledgePayloads, in input order.

    Each request is processed concurrently up to ``config.model.max_concurrency``.
    The ``enrichment.parent_payload`` field of each response carries the merged
    KnowledgePayload.  Contrast and refinement fields are left empty; PR4 fills
    those via the contrast enricher.

    Response ordering is guaranteed to match request ordering.
    """
    if not requests:
        return []

    if config is None:
        config = EnrichmentConfig()

    prompt_data = load_prompt(
        "parent",
        version=config.prompts.parent_version,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt_version: str = prompt_data.get("version", config.prompts.parent_version)

    parent_chain = build_parent_chain(config)
    semaphore = asyncio.Semaphore(config.model.max_concurrency)

    results = await asyncio.gather(
        *[
            _enrich_one_parent(
                request=req,
                parent_chain=parent_chain,
                semaphore=semaphore,
                prompt_version=prompt_version,
                model_name=config.model.model_name,
            )
            for req in requests
        ]
    )

    return list(results)
