from __future__ import annotations

import asyncio

from doc_enrichment.chains import build_contrast_chain
from doc_enrichment.config import EnrichmentConfig
from doc_enrichment.prompt_loader import load_prompt
from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentRequest,
    BranchEnrichmentResponse,
)


async def _enrich_one_contrast(
    request: BranchEnrichmentRequest,
    contrast_chain,
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
            enrichment: BranchEnrichment = await contrast_chain.ainvoke({
                "payload_a": request.left_payload.model_dump_json(indent=2),
                "payload_b": request.right_payload.model_dump_json(indent=2),
            })
        return base.model_copy(update={"enrichment": enrichment})

    except Exception as exc:  # noqa: BLE001
        return base.model_copy(update={"errors": [str(exc)]})


async def enrich_contrast(
    requests: list[BranchEnrichmentRequest],
    config: EnrichmentConfig | None = None,
) -> list[BranchEnrichmentResponse]:
    """Generate full BranchEnrichment for each sibling pair, in input order.

    The contrast chain produces all five fields: sibling_a, sibling_b,
    parent_payload, parent_refinement_for_a, and parent_refinement_for_b.

    Response ordering is guaranteed to match request ordering.
    """
    if not requests:
        return []

    if config is None:
        config = EnrichmentConfig()

    prompt_data = load_prompt(
        "contrast",
        version=config.prompts.contrast_version,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt_version: str = prompt_data.get("version", config.prompts.contrast_version)

    contrast_chain = build_contrast_chain(config)
    semaphore = asyncio.Semaphore(config.model.max_concurrency)

    results = await asyncio.gather(
        *[
            _enrich_one_contrast(
                request=req,
                contrast_chain=contrast_chain,
                semaphore=semaphore,
                prompt_version=prompt_version,
                model_name=config.model.model_name,
            )
            for req in requests
        ]
    )

    return list(results)
