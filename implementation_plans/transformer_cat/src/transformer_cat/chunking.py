"""
Structural heuristic chunking router + lineage-tracking wrapper.

Router decision matrix (from code_base.py reference):
  header_count >= 3  →  MarkdownHeaderTextSplitter  (then RecursiveCharacter fallback)
  list_count   >= 6  →  RecursiveCharacterTextSplitter  (list-oriented)
  else               →  NLTKTextSplitter  (prose / TextTiling)

chunk_document_with_tracking wraps any router output with parent lineage metadata
matching the plan's output schema.
"""
from __future__ import annotations

import datetime
import logging
import re
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger("transformer_cat.chunking")

_MARKDOWN_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3")]
_SAMPLE_SIZE = 2_000


class IntelligentChunkingRouter:
    """
    CPU-only heuristic router: samples the first 2 000 characters and decides
    which LangChain splitter best matches the document's structural signature.
    """

    def __init__(self) -> None:
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS
        )
        self.list_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1_000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " "],
        )
        self.prose_splitter = NLTKTextSplitter(chunk_size=2_000)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_route(sample: str) -> str:
        header_count = len(re.findall(r"^#{1,3}\s", sample, re.MULTILINE))
        list_count = len(re.findall(r"^([\*\-\+]\s|\d+\.\s)", sample, re.MULTILINE))
        if header_count >= 3:
            return "markdown"
        if list_count >= 6:
            return "list"
        return "prose"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_and_split(self, file_content: str) -> tuple[list[Document], str]:
        """
        Split *file_content* using the appropriate strategy.

        Returns
        -------
        (chunks, route_name)
            chunks: list of LangChain Document objects.
            route_name: one of 'markdown', 'list', 'prose'.
        """
        sample = file_content[:_SAMPLE_SIZE]
        route = self._detect_route(sample)

        if route == "markdown":
            logger.info("Routing to: Markdown Header Splitter")
            initial_splits = self.md_splitter.split_text(file_content)
            chunks = self.list_splitter.split_documents(initial_splits)
        elif route == "list":
            logger.info("Routing to: Recursive List Splitter")
            chunks = self.list_splitter.create_documents([file_content])
        else:
            logger.info("Routing to: Lexical TextTiling Prose Splitter")
            chunks = self.prose_splitter.create_documents([file_content])

        return chunks, route


# Singleton router — created once per process.
_default_router: IntelligentChunkingRouter | None = None


def get_router() -> IntelligentChunkingRouter:
    global _default_router
    if _default_router is None:
        _default_router = IntelligentChunkingRouter()
    return _default_router


def chunk_document_with_tracking(
    doc_data: dict[str, Any],
    router: IntelligentChunkingRouter | None = None,
) -> tuple[list[Document], str]:
    """
    Split a document dict and enrich every chunk with parent lineage metadata.

    Parameters
    ----------
    doc_data:
        Must contain at least ``body`` (str). Optional keys: ``document_id``,
        ``source``, ``title``, ``timestamp``.
    router:
        Override the default singleton router (useful in tests).

    Returns
    -------
    (tracked_chunks, route_name)
    """
    if router is None:
        router = get_router()

    raw_body: str = doc_data["body"]
    doc_id: str = doc_data.get("document_id", str(uuid.uuid4()))
    chunks, route = router.route_and_split(raw_body)

    tracked: list[Document] = []
    for idx, chunk in enumerate(chunks):
        meta: dict[str, Any] = {
            "parent_doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{idx}",
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "source": doc_data.get("source", ""),
            "title": doc_data.get("title", ""),
            "timestamp": doc_data.get(
                "timestamp", datetime.datetime.now().isoformat()
            ),
            "route": route,
        }
        tracked.append(Document(page_content=chunk.page_content, metadata=meta))

    logger.info(
        "Document %s → %d chunks via '%s' splitter", doc_id, len(chunks), route
    )
    return tracked, route
