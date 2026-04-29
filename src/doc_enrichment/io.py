"""Helpers to build enrichment request objects from plain dicts."""
from __future__ import annotations

from typing import Any

from doc_enrichment.schemas import (
    BranchEnrichmentRequest,
    DocumentEnrichmentRequest,
    KnowledgePayload,
)


def document_requests_from_dicts(
    records: list[dict[str, Any]],
    *,
    doc_id_key: str = "id",
    text_key: str = "text",
    title_key: str = "title",
    source_key: str = "source",
    node_id_key: str | None = None,
) -> list[DocumentEnrichmentRequest]:
    """Build DocumentEnrichmentRequests from a list of dicts.

    Default key names match the notebook query shape::

        SELECT id, url AS source, title, full_text AS text FROM ...
    """
    return [
        DocumentEnrichmentRequest(
            doc_id=str(r[doc_id_key]),
            text=r.get(text_key) or "",
            title=r.get(title_key) or "",
            source=r.get(source_key) or "",
            node_id=str(r[node_id_key]) if node_id_key and r.get(node_id_key) else None,
        )
        for r in records
    ]


def _branch_requests_from_dicts(
    records: list[dict[str, Any]],
    *,
    branch_id_key: str = "branch_id",
    left_node_id_key: str = "left_node_id",
    right_node_id_key: str = "right_node_id",
    left_payload_key: str = "left_payload",
    right_payload_key: str = "right_payload",
) -> list[BranchEnrichmentRequest]:
    return [
        BranchEnrichmentRequest(
            branch_id=str(r[branch_id_key]),
            left_node_id=str(r[left_node_id_key]),
            right_node_id=str(r[right_node_id_key]),
            left_payload=KnowledgePayload.model_validate(r[left_payload_key]),
            right_payload=KnowledgePayload.model_validate(r[right_payload_key]),
        )
        for r in records
    ]


def parent_requests_from_dicts(
    records: list[dict[str, Any]],
    *,
    branch_id_key: str = "branch_id",
    left_node_id_key: str = "left_node_id",
    right_node_id_key: str = "right_node_id",
    left_payload_key: str = "left_payload",
    right_payload_key: str = "right_payload",
) -> list[BranchEnrichmentRequest]:
    """Build BranchEnrichmentRequests for ``enrich_parent_nodes`` from a list of dicts."""
    return _branch_requests_from_dicts(
        records,
        branch_id_key=branch_id_key,
        left_node_id_key=left_node_id_key,
        right_node_id_key=right_node_id_key,
        left_payload_key=left_payload_key,
        right_payload_key=right_payload_key,
    )


def contrast_requests_from_dicts(
    records: list[dict[str, Any]],
    *,
    branch_id_key: str = "branch_id",
    left_node_id_key: str = "left_node_id",
    right_node_id_key: str = "right_node_id",
    left_payload_key: str = "left_payload",
    right_payload_key: str = "right_payload",
) -> list[BranchEnrichmentRequest]:
    """Build BranchEnrichmentRequests for ``enrich_contrast`` from a list of dicts."""
    return _branch_requests_from_dicts(
        records,
        branch_id_key=branch_id_key,
        left_node_id_key=left_node_id_key,
        right_node_id_key=right_node_id_key,
        left_payload_key=left_payload_key,
        right_payload_key=right_payload_key,
    )
