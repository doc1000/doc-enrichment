# Doc Enrichment Phased PR Plan

## Goal
Move notebook-based enrichment logic into a small, importable `doc_enrichment` package with an optional FastAPI wrapper, using small reviewable PRs and no database coupling.

## Global Rules For Every PR
- Keep existing behavior from `starting_code.md`; do not redesign prompt logic.
- Validate at boundaries with Pydantic once, then pass typed models internally.
- Use LangChain directly (prompt templates, structured output, batch/abatch) with minimal wrapper code.
- Keep functions single-purpose and avoid generic abstraction layers.
- Include tests for each PR scope before merging.

## PR1 - Package Skeleton + Contracts + Prompt Assets
**Scope**
- Create package skeleton and baseline project files:
  - `pyproject.toml`
  - `src/doc_enrichment/__init__.py`
  - `src/doc_enrichment/schemas.py`
  - `src/doc_enrichment/config.py`
  - `src/doc_enrichment/errors.py`
  - `src/doc_enrichment/prompt_loader.py`
  - `src/doc_enrichment/prompts/document_v1.yaml`
  - `src/doc_enrichment/prompts/parent_v1.yaml`
  - `src/doc_enrichment/prompts/contrast_v1.yaml`
- Define request/response contracts and payload schemas.
- Add prompt/version metadata fields in response contracts (`prompt_version`, `schema_version`, `model_name`).

**Out Of Scope**
- No enrichment execution logic.
- No FastAPI service.

**Acceptance Criteria**
- Package imports cleanly.
- Contracts validate expected sample inputs and outputs.
- Prompt loader resolves versioned prompt files.

**Tests**
- `tests/test_schemas.py`
- `tests/test_prompt_loader.py`

---

## PR2 - LangChain Chains + Document Enrichment
**Scope**
- Add `src/doc_enrichment/chains.py`:
  - `build_document_chain()`
  - `build_parent_chain()`
  - `build_contrast_chain()`
- Add document enricher:
  - `src/doc_enrichment/enrichers/document.py`
- Wire public export in `__init__.py` for:
  - `enrich_documents(...)`
- Ensure batch/abatch ordering remains deterministic.

**Out Of Scope**
- Tournament reduction.
- Parent and contrast orchestration beyond chain creation.

**Acceptance Criteria**
- `enrich_documents` returns validated response objects with required metadata.
- Batch calls preserve request-to-response ordering and ID alignment.

**Tests**
- `tests/test_document_enricher.py`
- `tests/test_batch_ordering.py`

---

## PR3 - Parent Enrichment + Tournament Pipeline
**Scope**
- Add parent enricher:
  - `src/doc_enrichment/enrichers/parent.py`
- Add tournament orchestration:
  - `src/doc_enrichment/pipeline.py`
- Expose:
  - `enrich_parent_nodes(...)`
- Keep retry handling near chain invocation boundaries, not in tournament orchestration logic.

**Out Of Scope**
- Service API.

**Acceptance Criteria**
- Parent pair merges produce validated parent payload responses.
- Tournament reduction:
  - handles play-in to nearest lower power-of-two
  - then reduces adjacent pairs until one root result remains
- Deterministic behavior for stable inputs.

**Tests**
- `tests/test_parent_enricher.py`
- `tests/test_tournament_pipeline.py`

---

## PR4 - Contrast + Refinement Integration
**Scope**
- Add:
  - `src/doc_enrichment/normalization.py`
  - `src/doc_enrichment/enrichers/contrast.py`
- Expose:
  - `enrich_contrast(...)`
- Port sibling contrast/refinement logic from `starting_code.md`.
- Ensure contrast/refinement payload output is keyed and attached consistently to existing branch/doc knowledge-base node identifiers at package boundary.

**Out Of Scope**
- FastAPI routes.

**Acceptance Criteria**
- Contrast path returns validated payloads with full metadata.
- Node ID mapping for sibling contrast/refinement is deterministic and test-covered.

**Tests**
- `tests/test_contrast_enricher.py`
- `tests/test_normalization.py`

---

## PR5 - Thin FastAPI Wrapper + Docs
**Scope**
- Add optional service wrapper:
  - `src/doc_enrichment/services/api.py`
- Routes:
  - `POST /enrich/documents`
  - `POST /enrich/parents`
  - `POST /enrich/contrast`
- Keep route handlers thin; delegate to package functions.
- Add integration usage examples in `README.md`.

**Out Of Scope**
- Persistence/database integration.
- Auth and production infra concerns.

**Acceptance Criteria**
- API routes validate request/response contracts.
- Route tests pass with representative payloads.
- README includes package and service usage snippets.

**Tests**
- `tests/test_api_routes.py`

---

## Explicitly Not Building
- Database writes/adapters.
- Prompt database/CMS.
- Plugin registries or factory-heavy frameworks.
- Generic catch-all `utils.py`.
- Over-general multi-provider abstraction layer not required by current scope.

## Suggested Merge Order
PR1 -> PR2 -> PR3 -> PR4 -> PR5

## Notes For Coding Agent
- Keep each PR small and mergeable on its own.
- Do not mix service concerns into package internals.
- Preserve notebook behavior first; optimize later only if tests stay green.
