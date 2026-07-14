# Transformer categorization (`transformer_cat`)

Local, multi-taxonomy document classification built on **mean-pooled ModernBERT embeddings** and **tiny scikit-learn student classifiers**. Heavy transformer work happens once per text (768-dimensional vectors); each business taxonomy is a separate logistic-regression head that outputs a **full categorical score vector** (class probabilities summing to 1). Taxonomies can be added or retrained without re-embedding the anchor corpus for every ontology change—the teacher labels anchors, the student learns weights on top of fixed feature geometry.

Orchestration uses **LangGraph**: an **admin graph** bootstraps and registers taxonomies; an **ingestion graph** chunks incoming markdown, featurizes chunks, and applies every registered taxonomy at chunk and document level.

---

## Core idea: pooled features + distilled categorical scores

1. **Feature extractor (shared)** — `answerdotai/ModernBERT-base` (override with `FEATURE_MODEL`) tokenizes each text (max 512 tokens), runs the transformer, and **mean-pools** token hidden states using the attention mask so padding does not skew the vector. Output shape: `(N, 768)`.

2. **Teacher (admin / training only)** — `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (override with `TEACHER_MODEL`) assigns synthetic top-1 labels to anchor (or provided) texts. Category definitions are passed as **composite label strings** built from `label`, `synonyms`, and `differentiators` so zero-shot NLI can separate similar classes.

3. **Student (production)** — `LogisticRegression` is fit on `(X, y)` where `X` is pooled embeddings and `y` is either teacher labels or human labels. At inference, `predict_proba` yields **categorical scores** for every class in that taxonomy. Weights are stored as a small `.joblib` payload (~tens of KB), not a full transformer checkpoint.

4. **Document-level scores** — During ingestion, each chunk gets its own probability dict per taxonomy; **document-level** scores are the **arithmetic mean** of chunk probabilities per category (equal weight per chunk).

This pattern decouples **document geometry** (ModernBERT pooling) from **ontology definitions** (JSON categories + student weights), so new taxonomies are mostly configuration plus a fast train step—not a new embedding pipeline.

---

## Repository layout

| Path | Role |
|------|------|
| `src/transformer_cat/` | Package: features, teacher, anchor, chunking, storage, `admin_graph`, `ingestion_graph` |
| `config/initial_taxonomies.json` | Example taxonomy definitions (labels + synonyms + differentiators) |
| `config/prompts.json` | LLM prompts when categories need contrastive enrichment |
| `data/taxonomies_registry.json` | Index of registered taxonomies → model path, category list, `features_dim` |
| `data/anchor_corpus_cache.json` | Cached AG News texts (2000 records) for teacher labeling |
| `models/*_student.joblib` | Per-taxonomy logistic-regression weights |
| `tests/` | Phase tests for config, features, admin bootstrap, ingestion |
| `pipeline_validation.ipynb` | End-to-end validation and charts |

Install dependencies from `requirements.txt`. Run tests with `src` on `PYTHONPATH` (see `tests/conftest.py` / individual test modules).

---

## Configuration and environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `FEATURE_MODEL` | `answerdotai/ModernBERT-base` | Pooled embedding model |
| `TEACHER_MODEL` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | Zero-shot teacher |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | localhost / `llama3.1:8b` | Optional LLM for category enrichment |
| `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY` | unset | Optional LangSmith tracing (`logging_utils.init_tracing`) |

Paths resolve from the `transformer_cat/` root via `transformer_cat.config.get_settings()` (registry, models dir, anchor cache, prompts).

---

## Adding and registering a new taxonomy (admin workflow)

Entry point: compiled graph **`admin_app`** in `transformer_cat.admin_graph`.

### Workflow (graph topology)

```
validate
  ├─ (sparse category defs / force_llm_enrichment) → contrastive_llm_enricher
  └─ (rich defs) ────────────────────────────────→ route_data_source
                                                        │
                        ┌───────────────────────────────┴───────────────────────────────┐
                        ▼                                                               ▼
              process_labelled_corpus                                      process_anchor_teacher
                        │                                                               │
                        └───────────────────────────┬───────────────────────────────────┘
                                                    ▼
                                         train_and_register → END
```

**Steps in plain language:**

1. **Validate** — Accept `taxonomy_name` and `categories_input`. If the name already exists in `data/taxonomies_registry.json` and a model file is present, a warning is logged; re-run **overwrites** the student model and registry entry.

2. **Optional LLM enrichment** — If any category lacks `differentiators` and `synonyms`, or `force_llm_enrichment` is `True`, the graph calls **`contrastive_llm_enricher`** using `config/prompts.json` → `contrastive_differentiator`. You must pass a LangChain chat model:

   ```python
   admin_app.invoke(state, config={"configurable": {"llm_model": my_llm}})
   ```

   The LLM must return JSON with `categories`: list of `{ "label", "synonyms", "differentiators" }`.

3. **Choose training data**
   - **`provided_labelled_corpus`**: list of `{"text": "...", "label": "..."}` → encode texts with `extract_features`, train directly. Warning if fewer than 150 documents.
   - **Default (anchor teacher)**: load anchor texts via `load_anchor_corpus` (AG News cache, optional `anchor_sample_size` for tests), `zero_shot_label`, then `extract_features`. If the teacher assigns only one class, the graph retries with 300 anchor samples; if still single-class, it raises—supply a labelled corpus or a more diverse unlabelled set via `provided_unlabelled_corpus`.

4. **Train and register** — Fit `LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)`, then `register_taxonomy_student_model`:
   - Writes `models/{taxonomy_name}_student.joblib` (coef, intercept, classes only).
   - Updates `data/taxonomies_registry.json` with `model_file`, `categories`, `features_dim` (768).

### Admin inputs (`AdminState`)

| Field | Required | Description |
|-------|----------|-------------|
| `taxonomy_name` | yes | Stable key used in registry and filenames (e.g. `news_topics_v1`) |
| `categories_input` | yes | List of category dicts; each needs at least `"label"`. Optional `"synonyms"`, `"differentiators"` (see `config/initial_taxonomies.json`) |
| `force_llm_enrichment` | no | Default `False`; set `True` to always run the LLM enricher |
| `provided_labelled_corpus` | no | If set, skips anchor teacher path |
| `provided_unlabelled_corpus` | no | Custom texts for teacher labeling instead of AG News anchor |
| `anchor_sample_size` | no | Subsample anchor for fast runs (e.g. `30` in tests) |

Initial state also includes nullable pipeline fields (`validated_taxonomy`, `training_x`, `training_y`)—typically left empty; the graph fills them.

### Admin outputs (artifacts, not graph return payload)

| Artifact | Description |
|----------|-------------|
| `models/{taxonomy_name}_student.joblib` | Student classifier weights |
| `data/taxonomies_registry.json` | New or updated entry for `taxonomy_name` |

After registration, **`load_student_classifier(taxonomy_name)`** and **`load_registry()`** are the supported read APIs for inference code.

### Example: bootstrap from `initial_taxonomies.json`

Same pattern as `tests/test_phase3_admin.py`:

```python
import json
from transformer_cat.admin_graph import admin_app, AdminState
from transformer_cat.config import get_settings

settings = get_settings()
tax = json.loads(settings.initial_taxonomies_path.read_text())["news_topics_v1"]

state: AdminState = {
    "taxonomy_name": tax["taxonomy_name"],
    "categories_input": tax["categories"],
    "force_llm_enrichment": False,
    "provided_labelled_corpus": None,
    "provided_unlabelled_corpus": None,
    "validated_taxonomy": {},
    "training_x": None,
    "training_y": None,
    "anchor_sample_size": 30,  # omit or use None for full 2000-anchor training
}
admin_app.invoke(state)
```

---

## Ingestion and enrichment (after taxonomies are registered)

Entry point: compiled graph **`ingestion_app`** in `transformer_cat.ingestion_graph`.

Every registered taxonomy in `taxonomies_registry.json` is applied automatically—no per-request taxonomy list unless you change the registry.

### Workflow

```
route_chunking_decision (first 2000 chars of body)
  ├─ header_count ≥ 3  → chunk_markdown
  ├─ list_count ≥ 6    → chunk_list
  └─ else              → chunk_prose
            │
            ▼
      featurize (extract_features on all chunk texts)
            │
            ▼
      classify_multi_taxonomy (predict_proba per taxonomy; aggregate doc-level means)
            │
            END
```

**Chunking** (`chunking.py`): heuristic router picks Markdown headers (+ recursive fallback), list-oriented recursive splitting, or NLTK TextTiling prose splitting. **`chunk_document_with_tracking`** attaches lineage: `parent_doc_id`, `chunk_id`, `chunk_index`, `total_chunks`, `source`, `title`, `timestamp`, `route`.

**Classification**: For each taxonomy, load the student model, run `predict_proba` on the chunk feature matrix, attach label → probability maps on each chunk. Empty registry → payload still built but taxonomy dicts stay empty (warning logged).

### Ingestion inputs (`PipelineState`)

| Field | Required | Description |
|-------|----------|-------------|
| `raw_document` | yes | Dict with **`body`** (markdown string). Recommended metadata: `document_id`, `source`, `title`, `timestamp` |

Other `PipelineState` fields are outputs of the graph; initialize as empty/`None` like the tests.

Example document dict (`tests/test_phase4_ingestion.py`):

```python
{
    "document_id": "doc-001",
    "source": "path/or/uri",
    "title": "Title",
    "body": "# Section\n\nMarkdown content…",
    "timestamp": "2026-06-04T10:00:00",
}
```

### Ingestion outputs (`enriched_payload`)

Returned in graph state as `enriched_payload`:

```json
{
  "document_id": "...",
  "source": "...",
  "title": "...",
  "timestamp": "...",
  "chunk_route": "markdown | list | prose",
  "full_document_taxonomies": {
    "<taxonomy_name>": {
      "<category_label>": 0.0,
      "...": 0.0
    }
  },
  "chunks": [
    {
      "chunk_id": "...",
      "chunk_index": 0,
      "body": "chunk text",
      "chunk_taxonomies": {
        "<taxonomy_name>": {
          "<category_label>": 0.0
        }
      }
    }
  ]
}
```

- **`chunk_taxonomies`**: full probability distribution per taxonomy for that chunk (sums to ~1 per taxonomy).
- **`full_document_taxonomies`**: per taxonomy, mean of chunk probabilities across all chunks for each category.

Invoke:

```python
from transformer_cat.ingestion_graph import ingestion_app, PipelineState

result = ingestion_app.invoke({
    "raw_document": doc_data,
    "chunk_documents": [],
    "chunk_route": "",
    "feature_matrix": None,
    "enriched_payload": {},
})
payload = result["enriched_payload"]
```

Downstream systems (search filters, RAG routing, analytics) can use chunk-level scores for precision or document-level scores for cataloging.

---

## Supporting modules (quick reference)

| Module | Responsibility |
|--------|----------------|
| `features.extract_features` | Mean-pooled ModernBERT embeddings, batched |
| `teacher.zero_shot_label` | Teacher top-1 labels from enriched category defs |
| `anchor.ensure_anchor_cache` / `load_anchor_corpus` | AG News anchor download and sampling |
| `storage.register_taxonomy_student_model` | Persist student + update registry |
| `storage.save_feature_matrix` / `load_feature_matrix` | Optional `.npz` cache for offline feature work |
| `chunking.chunk_document_with_tracking` | Router + lineage metadata |

---

## Operational notes

- **First anchor use** downloads Hugging Face `ag_news` (requires `datasets` and network) into `data/anchor_corpus_cache.json`.
- **First featurize / teacher call** downloads transformer weights; models are cached in-process as singletons.
- **Re-training** a taxonomy replaces its `.joblib` and registry entry; ingestion picks up changes on the next `load_registry()` / `load_student_classifier()` call within the same process (restart long-running workers after deploy).
- **Validation**: run `pytest` from this directory; phase 3 covers admin bootstrap, phase 4 covers ingestion routing, lineage, and probability shapes.

For design background and chunking rationale, see `transformer_categorization` and `code_base.py` in this folder (reference notes; the implemented API is under `src/transformer_cat/`).
