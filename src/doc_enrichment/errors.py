class DocEnrichmentError(Exception):
    """Base exception for all doc-enrichment errors."""


class PromptLoadError(DocEnrichmentError):
    """Raised when a prompt file cannot be found or parsed."""


class EnrichmentError(DocEnrichmentError):
    """Raised when an LLM enrichment call fails."""


class PipelineError(DocEnrichmentError):
    """Raised when tournament or batch orchestration fails."""
