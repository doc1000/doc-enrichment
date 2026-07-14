"""
Phase 2 TDD assertions:
- extract_features(['a', 'b']) returns shape (2, 768) and dtype float32.
- save_feature_matrix / load_feature_matrix round-trips values exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer_cat.features import extract_features
from transformer_cat.storage import save_feature_matrix, load_feature_matrix


SENTENCES = [
    "The quarterly earnings report shows record profits.",
    "Scientists discovered a new species in the Amazon rainforest.",
]


def test_extract_features_shape_and_dtype():
    X = extract_features(SENTENCES)
    assert X.shape == (2, 768), f"Expected (2, 768), got {X.shape}"
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"


def test_extract_features_nonzero():
    X = extract_features(SENTENCES)
    assert np.any(X != 0), "All embeddings are zero — model may not have loaded"


def test_feature_matrix_round_trip(tmp_path):
    X = extract_features(SENTENCES)
    texts = list(SENTENCES)
    out = tmp_path / "test_matrix"
    save_feature_matrix(out, X, texts)

    loaded = load_feature_matrix(out)
    assert "features" in loaded
    assert "texts" in loaded

    X_loaded = np.array(loaded["features"])
    assert X_loaded.shape == X.shape
    assert np.allclose(X, X_loaded, rtol=1e-5, atol=1e-7), (
        "Reloaded features do not match saved features"
    )
    # String arrays must round-trip exactly
    assert loaded["texts"] == texts


def test_feature_matrix_extra_arrays(tmp_path):
    X = extract_features(SENTENCES)
    ids = ["doc_0", "doc_1"]
    out = tmp_path / "extras"
    save_feature_matrix(out, X, list(SENTENCES), chunk_ids=np.array(ids, dtype=object))

    loaded = load_feature_matrix(out)
    assert loaded["chunk_ids"] == ids
