# doc-enrichment

A Python package for enriching documents and knowledge-tree branches using LangChain and OpenAI structured output.

## Installation

```bash
# Core package
pip install -e .

# With development tools
pip install -e ".[dev]"

# With the optional FastAPI service
pip install -e ".[dev,service]"
```

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Package usage

### Enrich documents

```python
import asyncio
from doc_enrichment import enrich_documents
from doc_enrichment.schemas import DocumentEnrichmentRequest

requests = [
    DocumentEnrichmentRequest(
        doc_id="doc-1",
        text="Full text of the document...",
        title="My Document",
        source="https://example.com/doc-1",
    ),
    DocumentEnrichmentRequest(
        doc_id="doc-2",
        text="Another document...",
    ),
]

responses = asyncio.run(enrich_documents(requests))

for resp in responses:
    if resp.errors:
        print(f"{resp.doc_id}: ERROR — {resp.errors}")
    else:
        print(f"{resp.doc_id}: {resp.payload.summary}")
```

Documents longer than 180 000 characters are automatically split into chunks, extracted in parallel, and reduced into a single `KnowledgePayload`. Concurrency is controlled by `EnrichmentConfig.model.max_concurrency` (default 8).

---

### Enrich parent nodes

```python
import asyncio
from doc_enrichment import enrich_parent_nodes
from doc_enrichment.schemas import BranchEnrichmentRequest

# left_payload and right_payload are KnowledgePayload objects from a prior
# enrich_documents call or a previous tournament round.
requests = [
    BranchEnrichmentRequest(
        branch_id="branch-1",
        left_node_id="doc-1",
        right_node_id="doc-2",
        left_payload=left_payload,
        right_payload=right_payload,
    ),
]

responses = asyncio.run(enrich_parent_nodes(requests))
parent_payload = responses[0].enrichment.parent_payload
```

---

### Tournament reduction

Reduce an ordered list of leaf payloads to a single root `KnowledgePayload` using a bracket-style tournament. Non-power-of-two inputs are handled with a play-in round.

```python
import asyncio
from doc_enrichment.pipeline import run_tournament

# leaves: list of (node_id, KnowledgePayload)
leaves = [(resp.doc_id, resp.payload) for resp in doc_responses if resp.payload]

root_payload = asyncio.run(run_tournament(leaves))
print(root_payload.summary)
```

---

### Enrich sibling contrast

```python
import asyncio
from doc_enrichment import enrich_contrast, normalize_node_enrichments
from doc_enrichment.schemas import BranchEnrichmentRequest

requests = [
    BranchEnrichmentRequest(
        branch_id="branch-1",
        left_node_id="doc-1",
        right_node_id="doc-2",
        left_payload=left_payload,
        right_payload=right_payload,
    ),
]

responses = asyncio.run(enrich_contrast(requests))

# Map each child node_id to its contrast + refinement data
node_enrichments = normalize_node_enrichments(responses)

for node_id, data in node_enrichments.items():
    print(f"{node_id}: contrast={data.contrast}, refinement={data.parent_refine}")
```

---

### Custom configuration

```python
from doc_enrichment import enrich_documents
from doc_enrichment.config import EnrichmentConfig, ModelConfig

config = EnrichmentConfig(
    model=ModelConfig(
        model_name="gpt-4.1",
        temperature=0.0,
        max_concurrency=4,
    )
)

responses = asyncio.run(enrich_documents(requests, config=config))
```

---

## Service usage

The optional FastAPI wrapper exposes the three enrichment functions as HTTP endpoints.

### Start the server

```bash
uvicorn doc_enrichment.services.api:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/enrich/documents` | Enrich a batch of documents |
| `POST` | `/enrich/parents` | Merge child-payload pairs into parent nodes |
| `POST` | `/enrich/contrast` | Generate full sibling contrast and refinement |

### Example requests

**Enrich documents:**

```bash
curl -X POST http://localhost:8000/enrich/documents \
  -H "Content-Type: application/json" \
  -d '[{"doc_id":"d1","text":"Full document text here...","title":"My Doc"}]'
```

**Enrich parent nodes:**

```bash
curl -X POST http://localhost:8000/enrich/parents \
  -H "Content-Type: application/json" \
  -d '[{
    "branch_id": "b1",
    "left_node_id": "d1",
    "right_node_id": "d2",
    "left_payload": {"summary": "Left summary", ...},
    "right_payload": {"summary": "Right summary", ...}
  }]'
```

**Using httpx in Python:**

```python
import httpx, asyncio

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as c:
        resp = await c.post("/enrich/documents", json=[
            {"doc_id": "d1", "text": "Document text..."}
        ])
        print(resp.json())

asyncio.run(main())
```
