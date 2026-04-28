__version__ = "0.1.0"

from doc_enrichment.enrichers.document import enrich_documents
from doc_enrichment.schemas import (
    KnowledgePayload,
    SiblingContrast,
    ParentRefinement,
    BranchEnrichment,
    UsageMetadata,
    DocumentEnrichmentRequest,
    BranchEnrichmentRequest,
    DocumentEnrichmentResponse,
    BranchEnrichmentResponse,
)
from doc_enrichment.config import EnrichmentConfig, ModelConfig, PromptConfig
from doc_enrichment.errors import DocEnrichmentError, PromptLoadError, EnrichmentError, PipelineError

__all__ = [
    "__version__",
    # enrichment functions
    "enrich_documents",
    # payload models
    "KnowledgePayload",
    "SiblingContrast",
    "ParentRefinement",
    "BranchEnrichment",
    "UsageMetadata",
    # request / response contracts
    "DocumentEnrichmentRequest",
    "BranchEnrichmentRequest",
    "DocumentEnrichmentResponse",
    "BranchEnrichmentResponse",
    # config
    "EnrichmentConfig",
    "ModelConfig",
    "PromptConfig",
    # errors
    "DocEnrichmentError",
    "PromptLoadError",
    "EnrichmentError",
    "PipelineError",
]
