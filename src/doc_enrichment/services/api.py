from fastapi import Body, FastAPI

from doc_enrichment import enrich_contrast, enrich_documents, enrich_parent_nodes
from doc_enrichment.schemas import (
    BranchEnrichmentRequest,
    BranchEnrichmentResponse,
    DocumentEnrichmentRequest,
    DocumentEnrichmentResponse,
)

app = FastAPI(title="doc-enrichment", version="0.1.0")


@app.post("/enrich/documents")
async def post_enrich_documents(
    requests: list[DocumentEnrichmentRequest] = Body(...),
) -> list[DocumentEnrichmentResponse]:
    return await enrich_documents(requests)


@app.post("/enrich/parents")
async def post_enrich_parents(
    requests: list[BranchEnrichmentRequest] = Body(...),
) -> list[BranchEnrichmentResponse]:
    return await enrich_parent_nodes(requests)


@app.post("/enrich/contrast")
async def post_enrich_contrast(
    requests: list[BranchEnrichmentRequest] = Body(...),
) -> list[BranchEnrichmentResponse]:
    return await enrich_contrast(requests)
