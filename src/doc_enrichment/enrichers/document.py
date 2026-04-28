from __future__ import annotations

import asyncio
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_enrichment.chains import build_document_chain
from doc_enrichment.config import EnrichmentConfig
from doc_enrichment.prompt_loader import load_prompt
from doc_enrichment.schemas import (
    DocumentEnrichmentRequest,
    DocumentEnrichmentResponse,
    KnowledgePayload,
)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=50_000,
    chunk_overlap=1_000,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_LONG_DOC_THRESHOLD = 180_000


async def _enrich_one(
    request: DocumentEnrichmentRequest,
    extract_chain,
    reduce_chain,
    semaphore: asyncio.Semaphore,
    prompt_version: str,
    model_name: str,
) -> DocumentEnrichmentResponse:
    base = DocumentEnrichmentResponse(
        doc_id=request.doc_id,
        node_id=request.node_id,
        prompt_version=prompt_version,
        model_name=model_name,
    )

    try:
        text = request.text or ""
        chunks = _SPLITTER.split_text(text) if len(text) > _LONG_DOC_THRESHOLD else [text]

        async with semaphore:
            if len(chunks) == 1:
                payload: KnowledgePayload = await extract_chain.ainvoke({
                    "document_id": request.doc_id,
                    "title": request.title,
                    "source": request.source,
                    "document_text": chunks[0],
                })
                return base.model_copy(update={"payload": payload})

            chunk_results = await asyncio.gather(
                *[
                    extract_chain.ainvoke({
                        "document_id": request.doc_id,
                        "source": request.source,
                        "title": request.title,
                        "document_text": chunk,
                    })
                    for chunk in chunks
                ],
                return_exceptions=True,
            )

            good_chunks = [
                r.model_dump()
                for r in chunk_results
                if not isinstance(r, Exception)
            ]

            if not good_chunks:
                return base.model_copy(update={"errors": ["all_chunks_failed"]})

            reduced: KnowledgePayload = await reduce_chain.ainvoke({
                "document_id": request.doc_id,
                "source": request.source,
                "title": request.title,
                "chunk_payloads": json.dumps(good_chunks, indent=2),
            })
            return base.model_copy(update={"payload": reduced})

    except Exception as exc:  # noqa: BLE001
        return base.model_copy(update={"errors": [str(exc)]})


async def enrich_documents(
    requests: list[DocumentEnrichmentRequest],
    config: EnrichmentConfig | None = None,
) -> list[DocumentEnrichmentResponse]:
    """Enrich a list of documents and return responses in input order.

    Concurrency is limited to ``config.model.max_concurrency`` simultaneous
    documents.  The semaphore spans the full per-document workload including
    any multi-chunk extraction, matching the original notebook behaviour.

    Response ordering is guaranteed to match request ordering regardless of
    which requests complete first.
    """
    if not requests:
        return []

    if config is None:
        config = EnrichmentConfig()

    prompt_data = load_prompt(
        "document",
        version=config.prompts.document_version,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt_version: str = prompt_data.get("version", config.prompts.document_version)

    extract_chain = build_document_chain(config, sub_key="extract")
    reduce_chain = build_document_chain(config, sub_key="reduce")
    semaphore = asyncio.Semaphore(config.model.max_concurrency)

    results = await asyncio.gather(
        *[
            _enrich_one(
                request=req,
                extract_chain=extract_chain,
                reduce_chain=reduce_chain,
                semaphore=semaphore,
                prompt_version=prompt_version,
                model_name=config.model.model_name,
            )
            for req in requests
        ]
    )

    return list(results)
