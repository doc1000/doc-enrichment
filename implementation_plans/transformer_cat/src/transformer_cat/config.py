"""
Central settings resolved once at import time.
All paths are absolute so modules can be called from any working directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The package lives at  …/transformer_cat/src/transformer_cat/config.py
# BASE_DIR is the transformer_cat/ root two levels up.
_SRC_DIR = Path(__file__).resolve().parent          # transformer_cat/src/transformer_cat
_MODULE_ROOT = _SRC_DIR.parent                      # transformer_cat/src
BASE_DIR: Path = _MODULE_ROOT.parent                # transformer_cat/


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    config_dir: Path = field(default_factory=lambda: BASE_DIR / "config")
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")

    # Derived paths
    registry_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "taxonomies_registry.json"
    )
    initial_taxonomies_path: Path = field(
        default_factory=lambda: BASE_DIR / "config" / "initial_taxonomies.json"
    )
    prompts_path: Path = field(
        default_factory=lambda: BASE_DIR / "config" / "prompts.json"
    )
    anchor_cache_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "anchor_corpus_cache.json"
    )

    # Model identifiers
    feature_model_name: str = field(
        default_factory=lambda: os.getenv(
            "FEATURE_MODEL", "answerdotai/ModernBERT-base"
        )
    )
    teacher_model_name: str = field(
        default_factory=lambda: os.getenv(
            "TEACHER_MODEL",
            "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        )
    )

    # Ollama config (used when BaseChatModel is ChatOllama)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    )

    # LangSmith toggle derived from env
    langsmith_enabled: bool = field(
        default_factory=lambda: bool(
            os.getenv("LANGCHAIN_TRACING_V2") and os.getenv("LANGCHAIN_API_KEY")
        )
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
