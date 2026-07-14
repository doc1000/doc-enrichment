"""
Phase 1 TDD assertions:
- All required directories exist.
- JSON config files load cleanly with expected top-level keys.
- get_settings() returns absolute, existing paths.
- init_tracing() is idempotent and safe in both env states.
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the src package is importable when tests are run from the transformer_cat/ root.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer_cat.config import BASE_DIR, get_settings, Settings
from transformer_cat.logging_utils import init_tracing


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def test_required_dirs_exist():
    s = get_settings()
    for d in (s.base_dir, s.config_dir, s.data_dir, s.models_dir):
        assert d.is_dir(), f"Expected directory {d} to exist"


# ---------------------------------------------------------------------------
# JSON config files
# ---------------------------------------------------------------------------

def test_initial_taxonomies_loads():
    s = get_settings()
    assert s.initial_taxonomies_path.exists(), "initial_taxonomies.json missing"
    data = json.loads(s.initial_taxonomies_path.read_text(encoding="utf-8"))
    assert "news_topics_v1" in data, "Missing key: news_topics_v1"
    assert "document_purpose_v1" in data, "Missing key: document_purpose_v1"


def test_prompts_json_loads():
    s = get_settings()
    assert s.prompts_path.exists(), "prompts.json missing"
    data = json.loads(s.prompts_path.read_text(encoding="utf-8"))
    assert "contrastive_differentiator" in data, "Missing key: contrastive_differentiator"
    assert "taxonomy_enrichment" in data, "Missing key: taxonomy_enrichment"


# ---------------------------------------------------------------------------
# Settings paths are absolute and point to existing locations
# ---------------------------------------------------------------------------

def test_settings_paths_are_absolute_and_exist():
    s = get_settings()
    path_fields = [s.base_dir, s.config_dir, s.data_dir, s.models_dir]
    for p in path_fields:
        assert p.is_absolute(), f"{p} is not absolute"
        assert p.exists(), f"{p} does not exist"


def test_settings_is_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ---------------------------------------------------------------------------
# init_tracing idempotency in both env states
# ---------------------------------------------------------------------------

def test_init_tracing_without_langsmith(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    # Reset idempotency guard so we can test fresh
    import transformer_cat.logging_utils as lu
    lu._initialised = False
    init_tracing()
    init_tracing()  # second call must not raise


def test_init_tracing_with_langsmith(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key-1234")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "test-project")
    import transformer_cat.logging_utils as lu
    lu._initialised = False
    init_tracing()
    init_tracing()  # idempotent
