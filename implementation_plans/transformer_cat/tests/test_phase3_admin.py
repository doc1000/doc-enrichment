"""
Phase 3 TDD assertions:
- Bootstrap news_topics_v1 with sample_size=30 anchor docs and no LLM enrichment.
- Assert .joblib file exists after run.
- Assert registry entry has 4 categories and features_dim == 768.
- Assert load_student_classifier().predict_proba(X) returns (N, 4) summing to ~1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer_cat.admin_graph import admin_app, AdminState
from transformer_cat.config import get_settings
from transformer_cat.features import extract_features
from transformer_cat.storage import load_student_classifier, load_registry

TAXONOMY_NAME = "news_topics_v1"


@pytest.fixture(scope="module")
def bootstrapped_state():
    """Run the admin graph once for the whole test module."""
    settings = get_settings()
    # Load categories from config
    taxonomies = json.loads(
        settings.initial_taxonomies_path.read_text(encoding="utf-8")
    )
    tax_body = taxonomies[TAXONOMY_NAME]

    initial_state: AdminState = {
        "taxonomy_name": tax_body["taxonomy_name"],
        "categories_input": tax_body["categories"],
        "force_llm_enrichment": False,
        "provided_labelled_corpus": None,
        "provided_unlabelled_corpus": None,
        "validated_taxonomy": {},  # type: ignore[typeddict-item]
        "training_x": None,
        "training_y": None,
        "anchor_sample_size": 30,
    }

    result = admin_app.invoke(initial_state)
    return result


def test_joblib_file_exists(bootstrapped_state):
    settings = get_settings()
    model_file = settings.models_dir / f"{TAXONOMY_NAME}_student.joblib"
    assert model_file.exists(), f"Expected .joblib at {model_file}"


def test_registry_entry_exists_with_correct_shape(bootstrapped_state):
    registry = load_registry()
    assert TAXONOMY_NAME in registry, f"'{TAXONOMY_NAME}' not found in registry"

    entry = registry[TAXONOMY_NAME]
    assert len(entry["categories"]) == 4, (
        f"Expected 4 categories, got {len(entry['categories'])}"
    )
    assert entry["features_dim"] == 768, (
        f"Expected features_dim=768, got {entry['features_dim']}"
    )


def test_student_classifier_predict_proba(bootstrapped_state):
    clf = load_student_classifier(TAXONOMY_NAME)

    test_texts = [
        "The central bank raised interest rates to control inflation.",
        "A new study reveals coral reefs are recovering near Australia.",
    ]
    X = extract_features(test_texts)
    probs = clf.predict_proba(X)

    assert probs.shape == (2, 4), f"Expected (2, 4), got {probs.shape}"
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"Probability rows do not sum to 1: {row_sums}"
    )


def test_student_classes_match_registry(bootstrapped_state):
    clf = load_student_classifier(TAXONOMY_NAME)
    registry = load_registry()
    registry_categories = set(registry[TAXONOMY_NAME]["categories"])
    model_classes = set(clf.classes_.tolist())
    assert model_classes == registry_categories, (
        f"Model classes {model_classes} != registry categories {registry_categories}"
    )
