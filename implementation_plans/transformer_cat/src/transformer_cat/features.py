"""
ModernBERT feature extraction via manual mean pooling.

The tokenizer and model are lazy-initialised on first call and cached as
module-level singletons so repeated calls within a process pay no reload cost.

extract_features(texts) -> np.ndarray of shape (N, 768), dtype float32.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("transformer_cat.features")

_tokenizer = None
_model = None


def _get_model(model_name: Optional[str] = None):
    """Lazy-load tokenizer and model; return (tokenizer, model)."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModel  # type: ignore

    if model_name is None:
        from transformer_cat.config import get_settings
        model_name = get_settings().feature_model_name

    logger.info("Loading feature model: %s", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name)
    _model.eval()
    logger.info("Feature model loaded.")
    return _tokenizer, _model


def extract_features(
    texts: list[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode *texts* with ModernBERT and return mean-pooled embeddings.

    Parameters
    ----------
    texts:
        List of raw text strings to encode.
    model_name:
        Override the model from settings (mostly for testing).
    batch_size:
        Number of texts processed per forward pass.

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), 768).
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")

    tokenizer, model = _get_model(model_name)

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Manual mean pooling — exclude padding tokens using attention_mask.
        # attention_mask shape: (batch, seq_len)
        # last_hidden_state shape: (batch, seq_len, hidden)
        token_embeddings = outputs.last_hidden_state          # (B, S, H)
        mask = inputs["attention_mask"].unsqueeze(-1).float() # (B, S, 1)
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)  # (B, H)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)           # (B, 1)
        pooled = (sum_embeddings / sum_mask).float()                 # (B, H)

        all_embeddings.append(pooled.numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def reset_model_cache() -> None:
    """Force re-load of model on next call (useful in tests)."""
    global _tokenizer, _model
    _tokenizer = None
    _model = None
