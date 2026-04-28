from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core payload models (match starting_code.md structures exactly)
# ---------------------------------------------------------------------------

class KnowledgePayload(BaseModel):
    title: str | None = None
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    intent_purpose: list[str] = Field(default_factory=list)
    target_audience: list[str] = Field(default_factory=list)
    document_type: list[str] = Field(default_factory=list)
    industry: list[str] = Field(default_factory=list)
    life_domain: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class SiblingContrast(BaseModel):
    contrast: list[str] = Field(default_factory=list)
    contrast_tag: list[str] = Field(default_factory=list)


class ParentRefinement(BaseModel):
    parent_refine: list[str] = Field(default_factory=list)
    parent_refine_tag: list[str] = Field(default_factory=list)


class BranchEnrichment(BaseModel):
    """Full output of the contrast/refinement LLM call for a sibling pair."""
    sibling_a: SiblingContrast
    sibling_b: SiblingContrast
    parent_payload: KnowledgePayload
    parent_refinement_for_a: ParentRefinement
    parent_refinement_for_b: ParentRefinement


# ---------------------------------------------------------------------------
# Request contracts
# ---------------------------------------------------------------------------

class DocumentEnrichmentRequest(BaseModel):
    doc_id: str
    node_id: str | None = None
    text: str
    title: str = ""
    source: str = ""
    instructions: dict[str, Any] = Field(default_factory=dict)


class BranchEnrichmentRequest(BaseModel):
    """Input for parent-node or sibling-contrast enrichment.

    left_payload and right_payload are the already-validated KnowledgePayloads
    for the two child nodes. The caller is responsible for providing them.
    """
    branch_id: str
    left_node_id: str
    right_node_id: str
    left_payload: KnowledgePayload
    right_payload: KnowledgePayload
    instructions: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response contracts
# ---------------------------------------------------------------------------

class UsageMetadata(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DocumentEnrichmentResponse(BaseModel):
    request_id: str | None = None
    doc_id: str
    node_id: str | None = None
    enrichment_type: Literal["document_payload"] = "document_payload"
    schema_version: str = "document_payload_v1"
    prompt_version: str
    model_name: str
    payload: KnowledgePayload | None = None
    errors: list[str] = Field(default_factory=list)
    usage: UsageMetadata = Field(default_factory=UsageMetadata)


class BranchEnrichmentResponse(BaseModel):
    """Response for both parent-node enrichment and sibling-contrast enrichment.

    enrichment.parent_payload is the KnowledgePayload for the branch node.
    enrichment.sibling_a/b and enrichment.parent_refinement_for_a/b carry the
    contrast and refinement data that callers attach to the child nodes.
    """
    request_id: str | None = None
    branch_id: str
    left_node_id: str
    right_node_id: str
    enrichment_type: Literal["branch_enrichment"] = "branch_enrichment"
    schema_version: str = "branch_enrichment_v1"
    prompt_version: str
    model_name: str
    enrichment: BranchEnrichment | None = None
    errors: list[str] = Field(default_factory=list)
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
