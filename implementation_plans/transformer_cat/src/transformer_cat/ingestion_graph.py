"""
Ingestion LangGraph workflow: classify a single document across all registered
taxonomies and return a dual-layer nested JSON payload.

Graph topology:
  set_conditional_entry_point(route_chunking_decision)
        ├── chunk_markdown  ──┐
        ├── chunk_list     ──►├── featurize_chunks ──► classify_multi_taxonomy ──► END
        └── chunk_prose    ──┘

The final enriched_payload matches the schema defined in code_base.py (lines 317–367):
  {
    "document_id": ...,
    "source": ...,
    "title": ...,
    "timestamp": ...,
    "full_document_taxonomies": { taxonomy_name: {label: prob, ...}, ... },
    "chunks": [
      {
        "chunk_id": ...,
        "chunk_index": ...,
        "body": ...,
        "chunk_taxonomies": { taxonomy_name: {label: prob, ...}, ... }
      }, ...
    ]
  }
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from transformer_cat.chunking import chunk_document_with_tracking, get_router
from transformer_cat.features import extract_features
from transformer_cat.storage import load_registry, load_student_classifier

logger = logging.getLogger("transformer_cat.ingestion_graph")

_SAMPLE_SIZE = 2_000


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    raw_document: dict[str, Any]          # input dict with body, document_id, …
    chunk_documents: list[Document]        # LangChain Documents with lineage metadata
    chunk_route: str                       # 'markdown' | 'list' | 'prose'
    feature_matrix: Optional[np.ndarray]  # (N_chunks, 768)
    enriched_payload: dict[str, Any]      # final dual-layer JSON output


# ---------------------------------------------------------------------------
# Conditional entry point: inspect doc head → choose chunker
# ---------------------------------------------------------------------------

def _route_chunking_decision(state: PipelineState) -> str:
    sample = state["raw_document"]["body"][:_SAMPLE_SIZE]
    header_count = len(re.findall(r"^#{1,3}\s", sample, re.MULTILINE))
    list_count = len(re.findall(r"^([\*\-\+]\s|\d+\.\s)", sample, re.MULTILINE))

    if header_count >= 3:
        return "chunk_markdown"
    if list_count >= 6:
        return "chunk_list"
    return "chunk_prose"


# ---------------------------------------------------------------------------
# Chunking nodes — each calls the same underlying router helper but enforces
# the routing contract so the route label written to state is canonical.
# ---------------------------------------------------------------------------

def _chunk_markdown_node(state: PipelineState) -> dict:
    chunks, route = chunk_document_with_tracking(state["raw_document"])
    return {"chunk_documents": chunks, "chunk_route": route}


def _chunk_list_node(state: PipelineState) -> dict:
    chunks, route = chunk_document_with_tracking(state["raw_document"])
    return {"chunk_documents": chunks, "chunk_route": route}


def _chunk_prose_node(state: PipelineState) -> dict:
    chunks, route = chunk_document_with_tracking(state["raw_document"])
    return {"chunk_documents": chunks, "chunk_route": route}


# ---------------------------------------------------------------------------
# Featurisation node
# ---------------------------------------------------------------------------

def _featurize_chunks_node(state: PipelineState) -> dict:
    chunks = state["chunk_documents"]
    texts = [c.page_content for c in chunks]
    X = extract_features(texts)
    logger.info("Featurised %d chunks → shape %s", len(texts), X.shape)
    return {"feature_matrix": X}


# ---------------------------------------------------------------------------
# Multi-taxonomy classification node
# ---------------------------------------------------------------------------

def _classify_multi_taxonomy_node(state: PipelineState) -> dict:
    X: np.ndarray = state["feature_matrix"]  # type: ignore[assignment]
    chunks = state["chunk_documents"]
    doc = state["raw_document"]

    registry = load_registry()
    if not registry:
        logger.warning("Taxonomy registry is empty — no classifications applied.")

    # --- Build per-chunk taxonomy dicts ---
    chunk_records: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{i}"),
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "body": chunk.page_content,
                "chunk_taxonomies": {},
            }
        )

    for tax_name in registry:
        clf = load_student_classifier(tax_name)
        probs = clf.predict_proba(X)  # (N_chunks, N_classes)

        for i, record in enumerate(chunk_records):
            record["chunk_taxonomies"][tax_name] = dict(
                zip(clf.classes_.tolist(), probs[i].tolist())
            )

    # --- Aggregate to document level (mean across chunks) ---
    full_doc_taxonomies: dict[str, dict[str, float]] = {}
    for tax_name in registry:
        if chunk_records:
            categories = list(chunk_records[0]["chunk_taxonomies"].get(tax_name, {}).keys())
            mean_probs = np.mean(
                [
                    [rec["chunk_taxonomies"][tax_name][cat] for cat in categories]
                    for rec in chunk_records
                ],
                axis=0,
            )
            full_doc_taxonomies[tax_name] = dict(zip(categories, mean_probs.tolist()))

    payload: dict[str, Any] = {
        "document_id": doc.get("document_id", ""),
        "source": doc.get("source", ""),
        "title": doc.get("title", ""),
        "timestamp": doc.get("timestamp", ""),
        "chunk_route": state.get("chunk_route", ""),
        "full_document_taxonomies": full_doc_taxonomies,
        "chunks": chunk_records,
    }

    logger.info(
        "Classified document '%s' — %d chunks × %d taxonomies",
        payload["document_id"],
        len(chunk_records),
        len(registry),
    )
    return {"enriched_payload": payload}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_ingestion_graph() -> StateGraph:
    flow = StateGraph(PipelineState)

    flow.add_node("chunk_markdown", _chunk_markdown_node)
    flow.add_node("chunk_list", _chunk_list_node)
    flow.add_node("chunk_prose", _chunk_prose_node)
    flow.add_node("featurize", _featurize_chunks_node)
    flow.add_node("classify", _classify_multi_taxonomy_node)

    flow.set_conditional_entry_point(
        _route_chunking_decision,
        {
            "chunk_markdown": "chunk_markdown",
            "chunk_list": "chunk_list",
            "chunk_prose": "chunk_prose",
        },
    )

    flow.add_edge("chunk_markdown", "featurize")
    flow.add_edge("chunk_list", "featurize")
    flow.add_edge("chunk_prose", "featurize")
    flow.add_edge("featurize", "classify")
    flow.add_edge("classify", END)

    return flow


ingestion_app = _build_ingestion_graph().compile()
