from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


class ModelConfig(BaseModel):
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_concurrency: int = 8


class PromptConfig(BaseModel):
    """Prompt file locations and active versions.

    prompts_dir defaults to the package-bundled prompts/ directory.
    Override to point at a custom directory during development or testing.
    """
    prompts_dir: Path = Field(default_factory=lambda: _DEFAULT_PROMPTS_DIR)
    document_version: str = "v1"
    parent_version: str = "v1"
    contrast_version: str = "v1"

    model_config = {"arbitrary_types_allowed": True}


class EnrichmentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)

    model_config = {"arbitrary_types_allowed": True}
