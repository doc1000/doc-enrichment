"""
Anchor corpus management.

First call to ensure_anchor_cache() downloads 2 000 AG News records from
Hugging Face and persists them to data/anchor_corpus_cache.json.
Subsequent calls skip the download.

load_anchor_corpus(sample_size=None) returns a plain list[str].
Pass sample_size=30 in tests for fast, offline execution.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from transformer_cat.config import get_settings

logger = logging.getLogger("transformer_cat.anchor")

_FULL_SIZE = 2_000


def ensure_anchor_cache() -> Path:
    """Download AG News once and cache to disk. Returns the cache file path."""
    settings = get_settings()
    cache_path = settings.anchor_cache_path

    if cache_path.exists():
        logger.info("Anchor corpus cache already exists at %s", cache_path)
        return cache_path

    logger.info("Downloading AG News anchor corpus (%d records)…", _FULL_SIZE)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for the anchor corpus download. "
            "Run: pip install datasets"
        ) from exc

    dataset = load_dataset("ag_news", split=f"train[:{_FULL_SIZE}]")
    texts: list[str] = list(dataset["text"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(texts, fh, ensure_ascii=False, indent=2)

    logger.info("Anchor corpus cached to %s", cache_path)
    return cache_path


def load_anchor_corpus(sample_size: Optional[int] = None) -> list[str]:
    """
    Return anchor texts.

    Parameters
    ----------
    sample_size:
        If provided, return a *reproducible* random sample of this size.
        Pass 30 in unit tests for sub-5 s offline execution.
    """
    ensure_anchor_cache()
    settings = get_settings()
    with open(settings.anchor_cache_path, "r", encoding="utf-8") as fh:
        texts: list[str] = json.load(fh)

    if sample_size is not None:
        rng = random.Random(42)
        texts = rng.sample(texts, min(sample_size, len(texts)))

    return texts
