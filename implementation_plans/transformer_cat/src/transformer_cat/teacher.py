"""
Zero-shot teacher scoring.

Builds composite label strings from pre-enriched taxonomy categories and
runs the ModernBERT NLI zero-shot pipeline over a list of texts.

Returns a list of clean top-1 label strings (matching the original `label`
key, not the full composite descriptor string).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("transformer_cat.teacher")

_teacher_pipeline = None


def _get_teacher(model_name: str | None = None):
    global _teacher_pipeline
    if _teacher_pipeline is not None:
        return _teacher_pipeline

    from transformers import pipeline  # type: ignore
    from transformer_cat.config import get_settings

    if model_name is None:
        model_name = get_settings().teacher_model_name

    logger.info("Loading zero-shot teacher: %s", model_name)
    _teacher_pipeline = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,  # force CPU
    )
    logger.info("Teacher pipeline ready.")
    return _teacher_pipeline


def _build_composite_labels(categories: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Build human-readable composite descriptor strings from category dicts.

    Each category dict must have at least a 'label' key; optional keys are
    'synonyms' (list[str]) and 'differentiators' (list[str]).

    Returns
    -------
    raw_labels: list[str]
        The original short label strings (e.g. 'business_finance').
    composite_labels: list[str]
        Enriched strings fed to the zero-shot model
        (e.g. 'business_finance: corporate fiscal revenue profit').
    """
    raw_labels: list[str] = []
    composite_labels: list[str] = []

    for cat in categories:
        label: str = cat["label"]
        synonyms: list[str] = cat.get("synonyms", [])
        differentiators: list[str] = cat.get("differentiators", [])
        descriptor_parts = synonyms + differentiators
        composite = f"{label}: {' '.join(descriptor_parts)}" if descriptor_parts else label
        raw_labels.append(label)
        composite_labels.append(composite)

    return raw_labels, composite_labels


def zero_shot_label(
    texts: list[str],
    categories: list[dict[str, Any]],
    model_name: str | None = None,
    batch_size: int = 16,
) -> list[str]:
    """
    Run zero-shot classification over *texts* and return the top-1 label per text.

    Parameters
    ----------
    texts:
        Raw text strings to classify.
    categories:
        List of category dicts from the taxonomy (keys: label, synonyms, differentiators).
    model_name:
        Override the teacher model from settings.
    batch_size:
        Number of texts sent to the pipeline per call.

    Returns
    -------
    list[str] of length len(texts) — each is the winning raw label string.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")

    teacher = _get_teacher(model_name)
    raw_labels, composite_labels = _build_composite_labels(categories)

    # Map composite strings back to short labels for clean output.
    composite_to_raw = dict(zip(composite_labels, raw_labels))

    all_top_labels: list[str] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        results = teacher(batch, composite_labels, multi_label=False)
        if isinstance(results, dict):
            results = [results]  # single-item passthrough normalisation

        for res in results:
            top_composite: str = res["labels"][0]
            top_raw = composite_to_raw.get(top_composite, top_composite)
            all_top_labels.append(top_raw)

    return all_top_labels


def reset_teacher_cache() -> None:
    """Force re-load on next call (useful in tests)."""
    global _teacher_pipeline
    _teacher_pipeline = None
