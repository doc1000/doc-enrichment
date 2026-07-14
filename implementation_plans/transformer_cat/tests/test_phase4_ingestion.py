"""
Phase 4 TDD assertions:
- A markdown document with >=3 '#' headers routes to the markdown splitter.
- Chunks carry correct parent_doc_id and chunk_index lineage metadata.
- enriched_payload contains full_document_taxonomies + per-chunk chunk_taxonomies
  for all registered categories; probabilities sum to ~1 per taxonomy per row.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer_cat.ingestion_graph import ingestion_app, PipelineState
from transformer_cat.storage import load_registry

MOCK_MARKDOWN = """# Introduction

This section provides an overview of global financial markets and quarterly earnings guidance.

## Business Finance

Corporate revenue, mergers and acquisitions transactions, and stock exchange trading
have driven market volatility this quarter.

## Technology Innovation

Artificial intelligence neural networks and cloud computing deployment architectures
continue to reshape the enterprise software landscape.

### Cybersecurity

Encryption vulnerabilities and firewall patch cycles remain critical infrastructure concerns.

## Science and Environment

Peer-reviewed journal publications on climate change carbon emissions confirm accelerating
biodiversity loss across major ecosystems.
"""

DOC_DATA = {
    "document_id": "test_doc_001",
    "source": "tests/mock_document.md",
    "title": "Mock Integration Test Document",
    "body": MOCK_MARKDOWN,
    "timestamp": "2026-06-04T10:00:00",
}


@pytest.fixture(scope="module")
def ingestion_result():
    initial_state: PipelineState = {
        "raw_document": DOC_DATA,
        "chunk_documents": [],
        "chunk_route": "",
        "feature_matrix": None,
        "enriched_payload": {},
    }
    return ingestion_app.invoke(initial_state)


# ---------------------------------------------------------------------------
# Routing assertion
# ---------------------------------------------------------------------------

def test_routes_to_markdown_splitter(ingestion_result):
    payload = ingestion_result["enriched_payload"]
    assert payload["chunk_route"] == "markdown", (
        f"Expected route='markdown', got '{payload['chunk_route']}'"
    )


# ---------------------------------------------------------------------------
# Chunk lineage assertions
# ---------------------------------------------------------------------------

def test_chunks_produced(ingestion_result):
    payload = ingestion_result["enriched_payload"]
    assert len(payload["chunks"]) >= 1, "Expected at least one chunk"


def test_chunk_lineage_metadata(ingestion_result):
    payload = ingestion_result["enriched_payload"]
    for chunk in payload["chunks"]:
        assert chunk["chunk_id"].startswith("test_doc_001"), (
            f"chunk_id does not carry parent id: {chunk['chunk_id']}"
        )
        assert isinstance(chunk["chunk_index"], int)


# ---------------------------------------------------------------------------
# Taxonomy output assertions
# ---------------------------------------------------------------------------

def test_full_document_taxonomies_present(ingestion_result):
    registry = load_registry()
    payload = ingestion_result["enriched_payload"]
    fdoc = payload["full_document_taxonomies"]

    for tax_name in registry:
        assert tax_name in fdoc, f"Missing taxonomy '{tax_name}' in full_document_taxonomies"
        probs = list(fdoc[tax_name].values())
        assert abs(sum(probs) - 1.0) < 1e-4, (
            f"Full-doc probabilities for '{tax_name}' do not sum to 1: {sum(probs)}"
        )


def test_chunk_taxonomies_present_and_normalised(ingestion_result):
    registry = load_registry()
    payload = ingestion_result["enriched_payload"]

    for chunk in payload["chunks"]:
        ctax = chunk["chunk_taxonomies"]
        for tax_name in registry:
            assert tax_name in ctax, (
                f"Chunk {chunk['chunk_id']} missing taxonomy '{tax_name}'"
            )
            probs = list(ctax[tax_name].values())
            assert abs(sum(probs) - 1.0) < 1e-4, (
                f"Chunk probs for '{tax_name}' do not sum to 1: {sum(probs)}"
            )


def test_all_registered_categories_present(ingestion_result):
    registry = load_registry()
    payload = ingestion_result["enriched_payload"]
    fdoc = payload["full_document_taxonomies"]

    for tax_name, meta in registry.items():
        expected = set(meta["categories"])
        actual = set(fdoc[tax_name].keys())
        assert expected == actual, (
            f"Category mismatch for '{tax_name}': expected {expected}, got {actual}"
        )
