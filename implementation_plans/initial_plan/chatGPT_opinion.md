You are a senior Python software architect helping convert working Jupyter notebook code into a small, clean, importable Python module/service.

Goal:
Create an implementation plan only. Do not write the full code yet. The implementation will later be executed by a coding agent. Keep the plan minimal, concrete, and focused on connective tissue, not rewriting logic.

Context:
I have working notebook code for document enrichment and node/parent enrichment using LangChain. The core logic already works. I want to move it into a separate repo/package that can be imported by ingestion, graph, and visualization systems, and optionally run as a lightweight service.

Use the code in starting_code.md as a core base.  This has the functionality I want - now I want an importable package that can be integrated within larger projects.  you do not need to re-write any prompts or change the core functionality.  I want you to set up the contracts and surfaces so that the functionality is available.

High-level architecture:
- Separate repo/package for enrichment only.
- Contract-based boundary.
- No direct database integration.
- Caller provides documents, identifiers, enrichment instructions, model/client config.
- Package returns validated structured JSON payloads plus minimal metadata.
- Other systems decide where enrichment data is stored.
- Use LangChain because I am familiar with it and it ties the LLM calls, prompt templates, structured output, and batching together simply.

Important constraints:
- Keep implementation small and readable.
- Do not over-engineer.
- Reuse existing notebook logic as much as possible.
- Prefer off-the-shelf LangChain functionality over custom infrastructure.
- Validate once at boundaries using Pydantic, then pass validated data internally.
- Do not repeatedly check/re-check types throughout the code.
- Avoid broad abstractions, plugin systems, registries, or unnecessary factories.
- Avoid building custom orchestration infrastructure.
- Avoid prompt database for now.
- Store prompts as versioned YAML or text files in the repo.
- Keep code understandable to data scientists, RAG developers, and software engineers.
- Async is acceptable where it clearly simplifies batching/concurrency.  Use existing structures.
- Keep route/service layer thin if FastAPI is included.
- Prefer a clean Python package interface first; service wrapper second.

Desired package shape:
doc-enrichment/
  pyproject.toml
  README.md
  src/
    doc_enrichment/
      __init__.py
      client.py              # LLM client wrappers
      schemas.py             # Pydantic input/output contracts
      enrichers/
        document.py          # leaf/document enrichment
        parent.py            # node/parent enrichment
        contrast.py          # sibling contrast/refinement
      prompts/
        document_v1.yaml
        parent_v1.yaml
        contrast_v1.yaml
      services/
        api.py               # FastAPI app, optional
      pipeline.py            # orchestration: batching, tournament, retries
      config.py
      errors.py
      utils.py
  tests/

Core functional areas:
1. Document payload enrichment
   - Input: list of document texts plus doc/node IDs
   - Output: one structured payload per document

2. Parent/node enrichment
   - Input: pairs of child payloads/nodes
   - Output: one structured parent payload per pair

3. Tournament reduction
   - Input: ordered list of chunk/document payloads
   - First reduce to nearest lower power of two using play-in pair merges
   - Then iteratively merge adjacent pairs until one parent payload remains
   - Use existing generate_parent-style logic, adapted through LangChain batch/async batch where appropriate

4. Optional sibling contrast/refinement enrichment
   - Input: sibling payloads or node metadata
   - Output: contrast/refinement payloads keyed by node IDs

Contracts:
Use Pydantic models for request/response boundaries. Keep schemas simple.


Use snake_case internally. If prompt outputs need human-readable keys, normalize once at the boundary.

Prompt handling:
- Put prompts in versioned YAML or text files.
- Include prompt_version in every result.
- Do not create a prompt database.
- Do not create a complex prompt management layer.
- A simple loader function is enough.

LangChain usage:
- Use LangChain prompt templates.
- Use structured output where possible.
- Use batch/abatch for batch processing.
- Do not wrap LangChain in excessive custom layers.
- Do not create unnecessary base classes unless clearly needed.

Implementation plan requirements:
Please produce:
1. A minimal file-by-file implementation plan.
2. The responsibility of each file in one or two sentences.
3. The smallest public API this package should expose.
4. The exact order the coding agent should implement files.
5. What existing notebook functions should map into which module.
6. A short list of things explicitly not to build.
7. A small testing strategy focused on contract validation, prompt rendering, batch ordering, and tournament correctness.
8. Any risks or simplifications you recommend.

Style:
Be concise and decisive. Favor boring, maintainable code. 
```



# expected contract shape for defining iputs and outputs
from pydantic import BaseModel
from typing import Any, Literal


class EnrichmentRequest(BaseModel):
    doc_id: str
    node_id: str | None = None
    text: str
    enrichment_type: Literal["document_payload", "node_parent", "sibling_contrast"]
    instructions: dict[str, Any] = {}


class DocumentPayload(BaseModel):
    canonical_label: list[str]
    short_label: list[str]
    discriminative_subtitle: list[str]
    top_phrases: list[str]
    purpose_or_intent: list[str]
    target_audience: list[str]
    document_type: list[str]
    named_entities: list[str]


class EnrichmentResponse(BaseModel):
    doc_id: str
    node_id: str | None = None
    enrichment_type: str
    prompt_version: str
    model_name: str
    payload: dict[str, Any]


####
It should know about:

text in
metadata in
enrichment type
prompt version
model config
JSON out
validation status
optional trace/debug metadata

That boundary keeps it reusable.
###



### 
Interfaces to expose

Expose both:

1. Python package interface
from doc_enrichment import enrich_documents, enrich_parent_nodes

payloads = await enrich_documents(docs, config=config)
2. Service API

Use FastAPI when you want other systems/agents to call it.

POST /enrich/documents
POST /enrich/parents
POST /enrich/contrast

Keep FastAPI thin. Put logic in the package, not in route handlers.
##


##
Versioning recommendation

Version three things separately:

prompt_version: document_payload_v3
schema_version: document_payload_schema_v1
model_name: gpt-4.1-mini

This matters because you will eventually ask:

“Which prompt/model/schema generated this enrichment?”

Every response should include that metadata.

Suggested response shape
{
  "request_id": "abc",
  "doc_id": "doc-123",
  "node_id": "node-456",
  "enrichment_type": "document_payload",
  "schema_version": "document_payload_v1",
  "prompt_version": "document_payload_v3",
  "model_name": "gpt-4.1-mini",
  "payload": {},
  "errors": [],
  "usage": {
    "input_tokens": 1000,
    "output_tokens": 300
  }
}
##

CHATGPT opinionated recommendation

Do this:

separate repo/package
YAML prompt files under Git
Pydantic request/response contracts
async SDK calls directly
optional FastAPI service wrapper
no direct database writes
return validated JSON + metadata
let graph/ingestion systems own persistence

That gives you a clean module now and a scalable service later without adding much infrastructure.

publish your phased plan (to the extent phases] makes sense) to this folder.