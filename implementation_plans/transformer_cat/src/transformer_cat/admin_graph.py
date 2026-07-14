"""
Admin LangGraph workflow: bootstrap a new taxonomy student model.

Graph topology:
  validate
    ├─(missing descriptors / force)─► contrastive_llm_enricher
    │                                       │
    │                                       ▼
    └─(already enriched)──────────► route_data_source
                                           │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                    process_labelled_corpus   process_anchor_teacher
                               │                     │
                               └──────────┬──────────┘
                                          ▼
                                  train_and_register
                                          │
                                         END

Pass the LLM via the LangGraph config object so the node stays model-agnostic:
    admin_app.invoke(state, config={"configurable": {"llm_model": my_llm}})
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from sklearn.linear_model import LogisticRegression
from typing_extensions import TypedDict

from transformer_cat.anchor import load_anchor_corpus
from transformer_cat.config import get_settings
from transformer_cat.features import extract_features
from transformer_cat.storage import register_taxonomy_student_model
from transformer_cat.teacher import zero_shot_label

logger = logging.getLogger("transformer_cat.admin_graph")


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AdminState(TypedDict):
    taxonomy_name: str
    categories_input: list[dict[str, Any]]
    force_llm_enrichment: bool

    # Optional training data overrides
    provided_labelled_corpus: Optional[list[dict[str, Any]]]   # [{"text": "...", "label": "..."}]
    provided_unlabelled_corpus: Optional[list[str]]

    # Internal pipeline trackers
    validated_taxonomy: dict[str, Any]          # {name, categories}
    training_x: Optional[np.ndarray]
    training_y: Optional[list[str]]

    # Controls how many anchor texts to use (for fast test runs set to 30)
    anchor_sample_size: Optional[int]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _validate_node(state: AdminState) -> dict:
    """Check name collisions in the registry and pass validated taxonomy forward."""
    settings = get_settings()
    registry_path = settings.registry_path

    try:
        registry: dict = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    # Allow re-training — only block on genuine unintentional clashes when
    # a pre-trained model file already exists AND force=False.
    model_file = settings.models_dir / f"{state['taxonomy_name']}_student.joblib"
    if state["taxonomy_name"] in registry and model_file.exists():
        logger.warning(
            "Taxonomy '%s' already registered; re-training will overwrite.",
            state["taxonomy_name"],
        )

    validated = {
        "name": state["taxonomy_name"],
        "categories": state["categories_input"],
    }
    return {"validated_taxonomy": validated}


def _contrastive_llm_enricher_node(state: AdminState, config: RunnableConfig) -> dict:
    """
    Call the provided BaseChatModel to generate exclusive differentiators.
    Reads the contrastive_differentiator prompt from config/prompts.json.
    """
    llm = config.get("configurable", {}).get("llm_model")
    if llm is None:
        raise ValueError(
            "No LLM model provided in graph config. Pass via: "
            'config={"configurable": {"llm_model": <BaseChatModel instance>}}'
        )

    settings = get_settings()
    prompts: dict = json.loads(settings.prompts_path.read_text(encoding="utf-8"))

    taxonomy_name: str = state["taxonomy_name"]
    raw_labels = [cat["label"] for cat in state["validated_taxonomy"]["categories"]]

    formatted_prompt = prompts["contrastive_differentiator"].format(
        taxonomy_name=taxonomy_name,
        category_list=", ".join(raw_labels),
    )

    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    try:
        enriched: dict = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM did not return valid JSON for contrastive enrichment: {exc}"
        ) from exc

    logger.info(
        "LLM enriched %d categories for taxonomy '%s'",
        len(enriched.get("categories", [])),
        taxonomy_name,
    )
    return {
        "validated_taxonomy": {
            "name": taxonomy_name,
            "categories": enriched["categories"],
        }
    }


def _route_data_source_node(state: AdminState) -> dict:
    """Pass-through node — actual routing happens via conditional edge."""
    return {}


def _process_labelled_corpus_node(state: AdminState) -> dict:
    """Encode a pre-labelled corpus into (X, y) training arrays."""
    corpus: list[dict] = state["provided_labelled_corpus"]  # type: ignore[assignment]
    texts = [item["text"] for item in corpus]
    labels = [item["label"] for item in corpus]

    if len(texts) < 150:
        logger.warning(
            "Labelled corpus has only %d docs (< 150). "
            "Consider using the anchor teacher path for better coverage.",
            len(texts),
        )

    X = extract_features(texts)
    return {"training_x": X, "training_y": labels}


def _process_anchor_teacher_node(state: AdminState) -> dict:
    """
    Run the zero-shot teacher over the anchor corpus (or user-supplied unlabelled
    texts) to generate synthetic (X, y) training pairs.
    """
    sample_size: Optional[int] = state.get("anchor_sample_size")

    if state.get("provided_unlabelled_corpus"):
        texts: list[str] = state["provided_unlabelled_corpus"]  # type: ignore[assignment]
        logger.info("Using provided unlabelled corpus (%d docs)", len(texts))
    else:
        texts = load_anchor_corpus(sample_size=sample_size)
        logger.info("Using anchor corpus (%d docs)", len(texts))

    categories: list[dict] = state["validated_taxonomy"]["categories"]
    y_labels = zero_shot_label(texts, categories)
    X = extract_features(texts)

    return {"training_x": X, "training_y": y_labels}


def _train_and_register_node(state: AdminState) -> dict:
    """Fit LogisticRegression and persist to disk via storage layer."""
    X: np.ndarray = state["training_x"]  # type: ignore[assignment]
    y: list[str] = state["training_y"]  # type: ignore[assignment]

    # Safety check: if the teacher assigned all docs to a single class (common when
    # the anchor corpus is domain-mismatched), retry with a larger sample so that
    # the model can learn discriminative boundaries.
    _RETRY_SIZE = 300
    if len(set(y)) < 2:
        logger.warning(
            "Single-class labels detected for '%s'. Retrying with %d anchor samples.",
            state["taxonomy_name"],
            _RETRY_SIZE,
        )
        sample_size = state.get("anchor_sample_size")
        if sample_size is not None and sample_size >= _RETRY_SIZE:
            raise ValueError(
                f"Taxonomy '{state['taxonomy_name']}': teacher produced only one class "
                f"even with {sample_size} anchor samples. "
                "Consider providing a labelled corpus or a more topically diverse anchor set."
            )

        texts_retry = load_anchor_corpus(sample_size=_RETRY_SIZE)
        categories: list[dict] = state["validated_taxonomy"]["categories"]
        y = zero_shot_label(texts_retry, categories)
        X = extract_features(texts_retry)

        if len(set(y)) < 2:
            raise ValueError(
                f"Taxonomy '{state['taxonomy_name']}': teacher still produced only one class "
                f"with {_RETRY_SIZE} anchor samples. "
                "Provide a labelled corpus via 'provided_labelled_corpus'."
            )

    student = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    student.fit(X, y)

    register_taxonomy_student_model(
        taxonomy_name=state["taxonomy_name"],
        trained_model=student,
    )
    logger.info("Student model trained and registered: %s", state["taxonomy_name"])
    return {}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _check_definition_richness(state: AdminState) -> str:
    if state.get("force_llm_enrichment"):
        return "call_llm_enricher"
    for cat in state.get("categories_input", []):
        if not cat.get("differentiators") and not cat.get("synonyms"):
            return "call_llm_enricher"
    return "route_data_source"


def _route_data_source_decision(state: AdminState) -> str:
    if state.get("provided_labelled_corpus"):
        return "process_labelled"
    return "process_anchor_teacher"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_admin_graph() -> StateGraph:
    flow = StateGraph(AdminState)

    flow.add_node("validate", _validate_node)
    flow.add_node("contrastive_llm_enricher", _contrastive_llm_enricher_node)
    flow.add_node("route_data_source", _route_data_source_node)
    flow.add_node("process_labelled", _process_labelled_corpus_node)
    flow.add_node("process_anchor_teacher", _process_anchor_teacher_node)
    flow.add_node("train_and_register", _train_and_register_node)

    flow.set_entry_point("validate")

    flow.add_conditional_edges(
        "validate",
        _check_definition_richness,
        {
            "call_llm_enricher": "contrastive_llm_enricher",
            "route_data_source": "route_data_source",
        },
    )

    # After LLM enrichment also route to data source selection
    flow.add_conditional_edges(
        "contrastive_llm_enricher",
        _route_data_source_decision,
        {
            "process_labelled": "process_labelled",
            "process_anchor_teacher": "process_anchor_teacher",
        },
    )

    flow.add_conditional_edges(
        "route_data_source",
        _route_data_source_decision,
        {
            "process_labelled": "process_labelled",
            "process_anchor_teacher": "process_anchor_teacher",
        },
    )

    flow.add_edge("process_labelled", "train_and_register")
    flow.add_edge("process_anchor_teacher", "train_and_register")
    flow.add_edge("train_and_register", END)

    return flow


admin_app = _build_admin_graph().compile()
