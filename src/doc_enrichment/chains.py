from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from doc_enrichment.config import EnrichmentConfig
from doc_enrichment.prompt_loader import get_template
from doc_enrichment.schemas import BranchEnrichment, KnowledgePayload


def build_document_chain(
    config: EnrichmentConfig,
    sub_key: Literal["extract", "reduce"] = "extract",
) -> Runnable:
    """Return a chain that invokes the document extract or reduce prompt.

    ``document_v1.yaml`` has two named sections; ``sub_key`` selects which
    template is used.  Both sections produce a ``KnowledgePayload``.
    """
    template = get_template(
        "document",
        version=config.prompts.document_version,
        sub_key=sub_key,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        model=config.model.model_name,
        temperature=config.model.temperature,
    )
    return prompt | llm.with_structured_output(KnowledgePayload, method="json_schema")


def build_parent_chain(config: EnrichmentConfig) -> Runnable:
    """Return a chain that produces a parent ``KnowledgePayload`` from two child payloads."""
    template = get_template(
        "parent",
        version=config.prompts.parent_version,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        model=config.model.model_name,
        temperature=config.model.temperature,
    )
    return prompt | llm.with_structured_output(KnowledgePayload, method="json_schema")


def build_contrast_chain(config: EnrichmentConfig) -> Runnable:
    """Return a chain that produces a full ``BranchEnrichment`` from two sibling payloads."""
    template = get_template(
        "contrast",
        version=config.prompts.contrast_version,
        prompts_dir=config.prompts.prompts_dir,
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        model=config.model.model_name,
        temperature=config.model.temperature,
    )
    return prompt | llm.with_structured_output(BranchEnrichment, method="json_schema")
