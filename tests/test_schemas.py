import pytest
from pydantic import ValidationError

from doc_enrichment.schemas import (
    BranchEnrichment,
    BranchEnrichmentRequest,
    BranchEnrichmentResponse,
    DocumentEnrichmentRequest,
    DocumentEnrichmentResponse,
    KnowledgePayload,
    ParentRefinement,
    SiblingContrast,
    UsageMetadata,
)


# ---------------------------------------------------------------------------
# KnowledgePayload
# ---------------------------------------------------------------------------

def test_knowledge_payload_minimal():
    payload = KnowledgePayload(summary="A short summary.")
    assert payload.summary == "A short summary."
    assert payload.title is None
    assert payload.key_topics == []
    assert payload.provenance == []


def test_knowledge_payload_full():
    payload = KnowledgePayload(
        title="My Doc",
        summary="Summary text.",
        key_topics=["topic1"],
        entities=["Entity A"],
        facts=["Fact 1"],
        intent_purpose=["educate"],
        target_audience=["developers"],
        document_type=["article"],
        industry=["technology"],
        life_domain=["career"],
        provenance=["node-123"],
    )
    assert payload.title == "My Doc"
    assert payload.industry == ["technology"]


def test_knowledge_payload_missing_summary_raises():
    with pytest.raises(ValidationError):
        KnowledgePayload()


# ---------------------------------------------------------------------------
# SiblingContrast / ParentRefinement
# ---------------------------------------------------------------------------

def test_sibling_contrast_defaults():
    sc = SiblingContrast()
    assert sc.contrast == []
    assert sc.contrast_tag == []


def test_sibling_contrast_values():
    sc = SiblingContrast(contrast=["label A"], contrast_tag=["tag A"])
    assert sc.contrast == ["label A"]


def test_parent_refinement_defaults():
    pr = ParentRefinement()
    assert pr.parent_refine == []
    assert pr.parent_refine_tag == []


# ---------------------------------------------------------------------------
# BranchEnrichment
# ---------------------------------------------------------------------------

def _make_payload() -> KnowledgePayload:
    return KnowledgePayload(summary="stub")


def test_branch_enrichment_valid():
    be = BranchEnrichment(
        sibling_a=SiblingContrast(contrast=["a1"], contrast_tag=["t1"]),
        sibling_b=SiblingContrast(contrast=["b1"], contrast_tag=["t2"]),
        parent_payload=_make_payload(),
        parent_refinement_for_a=ParentRefinement(parent_refine=["r1"], parent_refine_tag=["rt1"]),
        parent_refinement_for_b=ParentRefinement(parent_refine=["r2"], parent_refine_tag=["rt2"]),
    )
    assert be.parent_payload.summary == "stub"
    assert be.sibling_a.contrast == ["a1"]


def test_branch_enrichment_missing_field_raises():
    with pytest.raises(ValidationError):
        BranchEnrichment(
            sibling_a=SiblingContrast(),
            sibling_b=SiblingContrast(),
            parent_payload=_make_payload(),
            # missing parent_refinement_for_a and parent_refinement_for_b
        )


# ---------------------------------------------------------------------------
# DocumentEnrichmentRequest
# ---------------------------------------------------------------------------

def test_document_request_minimal():
    req = DocumentEnrichmentRequest(doc_id="doc-1", text="Some text.")
    assert req.doc_id == "doc-1"
    assert req.node_id is None
    assert req.title == ""
    assert req.instructions == {}


def test_document_request_full():
    req = DocumentEnrichmentRequest(
        doc_id="doc-2",
        node_id="node-5",
        text="Content.",
        title="Title",
        source="https://example.com",
        instructions={"key": "value"},
    )
    assert req.node_id == "node-5"
    assert req.source == "https://example.com"


def test_document_request_missing_required_raises():
    with pytest.raises(ValidationError):
        DocumentEnrichmentRequest(doc_id="doc-3")  # missing text


# ---------------------------------------------------------------------------
# BranchEnrichmentRequest
# ---------------------------------------------------------------------------

def test_branch_request_valid():
    req = BranchEnrichmentRequest(
        branch_id="branch-1",
        left_node_id="node-A",
        right_node_id="node-B",
        left_payload=_make_payload(),
        right_payload=_make_payload(),
    )
    assert req.branch_id == "branch-1"
    assert req.left_payload.summary == "stub"


def test_branch_request_missing_fields_raises():
    with pytest.raises(ValidationError):
        BranchEnrichmentRequest(branch_id="branch-2")


# ---------------------------------------------------------------------------
# DocumentEnrichmentResponse
# ---------------------------------------------------------------------------

def test_document_response_valid():
    resp = DocumentEnrichmentResponse(
        doc_id="doc-1",
        prompt_version="document_v1",
        model_name="gpt-4.1-mini",
        payload=_make_payload(),
    )
    assert resp.enrichment_type == "document_payload"
    assert resp.schema_version == "document_payload_v1"
    assert resp.errors == []
    assert resp.usage.input_tokens == 0


def test_document_response_error_state():
    resp = DocumentEnrichmentResponse(
        doc_id="doc-1",
        prompt_version="document_v1",
        model_name="gpt-4.1-mini",
        errors=["timeout"],
    )
    assert resp.payload is None
    assert "timeout" in resp.errors


def test_document_response_metadata_fields_present():
    resp = DocumentEnrichmentResponse(
        doc_id="doc-1",
        prompt_version="document_v1",
        model_name="gpt-4.1-mini",
        usage=UsageMetadata(input_tokens=100, output_tokens=50),
    )
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 50


# ---------------------------------------------------------------------------
# BranchEnrichmentResponse
# ---------------------------------------------------------------------------

def test_branch_response_valid():
    enrichment = BranchEnrichment(
        sibling_a=SiblingContrast(),
        sibling_b=SiblingContrast(),
        parent_payload=_make_payload(),
        parent_refinement_for_a=ParentRefinement(),
        parent_refinement_for_b=ParentRefinement(),
    )
    resp = BranchEnrichmentResponse(
        branch_id="branch-1",
        left_node_id="node-A",
        right_node_id="node-B",
        prompt_version="contrast_v1",
        model_name="gpt-4.1-mini",
        enrichment=enrichment,
    )
    assert resp.enrichment_type == "branch_enrichment"
    assert resp.schema_version == "branch_enrichment_v1"
    assert resp.enrichment.parent_payload.summary == "stub"


def test_branch_response_missing_required_raises():
    with pytest.raises(ValidationError):
        # missing branch_id, left_node_id, right_node_id, prompt_version, model_name
        BranchEnrichmentResponse()
