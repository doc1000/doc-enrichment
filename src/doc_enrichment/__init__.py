__version__ = "0.1.0"

from doc_enrichment.enrichers.contrast import enrich_contrast
from doc_enrichment.enrichers.document import enrich_documents
from doc_enrichment.enrichers.parent import enrich_parent_nodes
from doc_enrichment.normalization import NodeEnrichmentData, normalize_node_enrichments
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
    "enrich_parent_nodes",
    "enrich_contrast",
    # normalization
    "normalize_node_enrichments",
    "NodeEnrichmentData",
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
