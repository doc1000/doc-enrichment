"""
Persistence layer for the transformer_cat module.

Covers three artefact types:
  1. Feature matrices  — saved / loaded as .npz (numpy compressed archives)
  2. Student models    — saved / loaded as .joblib (weight dict for LogisticRegression)
  3. Taxonomy registry — a single JSON file tracking all registered models
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("transformer_cat.storage")


# ---------------------------------------------------------------------------
# Feature-matrix persistence
# ---------------------------------------------------------------------------

def save_feature_matrix(
    path: str | Path,
    features: np.ndarray,
    texts: list[str],
    **extra_arrays: Any,
) -> Path:
    """
    Save a feature matrix and associated string arrays to a compressed .npz file.

    Parameters
    ----------
    path:
        Destination file path (will be created with parents if missing).
    features:
        Float32 embedding matrix of shape (N, D).
    texts:
        Parallel list of source texts (length N).
    **extra_arrays:
        Any additional arrays to bundle (e.g. chunk_ids, metadata strings).

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "features": features,
        "texts": np.array(texts, dtype=object),
    }
    arrays.update(extra_arrays)

    np.savez_compressed(path, **arrays)
    logger.info("Feature matrix saved → %s  shape=%s", path, features.shape)
    return path


def load_feature_matrix(path: str | Path) -> dict[str, Any]:
    """
    Load a feature matrix archive written by save_feature_matrix.

    Returns a dict with at least keys 'features' and 'texts', plus any
    extra arrays that were bundled at save time.
    """
    path = Path(path)
    if not path.exists():
        # numpy appends .npz automatically; try with suffix
        candidate = path.with_suffix(".npz")
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Feature matrix not found: {path}")

    archive = np.load(path, allow_pickle=True)
    result: dict[str, Any] = {}
    for key in archive.files:
        arr = archive[key]
        # Scalar object arrays (text lists) are stored as object dtype
        result[key] = arr.tolist() if arr.dtype == object else arr
    logger.info("Feature matrix loaded ← %s", path)
    return result


# ---------------------------------------------------------------------------
# Student model (LogisticRegression) persistence + registry
# ---------------------------------------------------------------------------

def _get_registry_path() -> Path:
    from transformer_cat.config import get_settings
    return get_settings().registry_path


def register_taxonomy_student_model(
    taxonomy_name: str,
    trained_model: LogisticRegression,
    registry_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> Path:
    """
    Save LogisticRegression weights and update the central JSON registry.

    The joblib payload stores only the weight tensors (coef_, intercept_,
    classes_) so the artefact stays tiny regardless of training data size.

    Returns the path to the written .joblib file.
    """
    if registry_path is None:
        registry_path = _get_registry_path()
    registry_path = Path(registry_path)

    if models_dir is None:
        from transformer_cat.config import get_settings
        models_dir = get_settings().models_dir
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Persist model weights only — keeps payload small and portable.
    model_file = models_dir / f"{taxonomy_name}_student.joblib"
    payload = {
        "weights": trained_model.coef_,
        "intercept": trained_model.intercept_,
        "classes": trained_model.classes_,
    }
    joblib.dump(payload, model_file)
    logger.info("Student model saved → %s", model_file)

    # Update registry JSON.
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry: dict = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    registry[taxonomy_name] = {
        "model_file": str(model_file),
        "categories": trained_model.classes_.tolist(),
        "features_dim": int(trained_model.coef_.shape[1]),
    }
    registry_path.write_text(
        json.dumps(registry, indent=4), encoding="utf-8"
    )
    logger.info("Registry updated → %s  (taxonomy: %s)", registry_path, taxonomy_name)
    return model_file


def load_student_classifier(
    taxonomy_name: str,
    registry_path: str | Path | None = None,
) -> LogisticRegression:
    """
    Rehydrate a LogisticRegression from its saved weight payload.

    The reconstructed classifier supports predict / predict_proba without
    needing the original training data.
    """
    if registry_path is None:
        registry_path = _get_registry_path()
    registry_path = Path(registry_path)

    registry: dict = json.loads(registry_path.read_text(encoding="utf-8"))
    if taxonomy_name not in registry:
        raise KeyError(
            f"Taxonomy '{taxonomy_name}' not found in registry at {registry_path}. "
            f"Available: {list(registry.keys())}"
        )

    model_file = Path(registry[taxonomy_name]["model_file"])
    payload: dict = joblib.load(model_file)

    clf = LogisticRegression(class_weight="balanced")
    clf.classes_ = np.array(payload["classes"])
    clf.coef_ = np.array(payload["weights"])
    clf.intercept_ = np.array(payload["intercept"])
    logger.info("Student classifier loaded ← %s", model_file)
    return clf


def load_registry(registry_path: str | Path | None = None) -> dict:
    """Return the full registry dict; empty dict if not yet created."""
    if registry_path is None:
        registry_path = _get_registry_path()
    registry_path = Path(registry_path)
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
