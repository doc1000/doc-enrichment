#! pip install -U langchain
#! pip install -U langchain-openai
#! pip install -U langchain_text_splitters

import asyncio
from typing import Any, Optional, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


class KnowledgePayload(BaseModel):
    title: Optional[str] = None
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    intent_purpose: list[str] = Field(default_factory=list)
    target_audience: list[str] = Field(default_factory=list)
    document_type: list[str] = Field(default_factory=list)
    industry: list[str] = Field(default_factory=list)
    life_domain: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

class SiblingContrast(BaseModel):
    contrast: List[str] = Field(default_factory=list)
    contrast_tag: List[str] = Field(default_factory=list)


class ParentRefinement(BaseModel):
    parent_refine: List[str] = Field(default_factory=list)
    parent_refine_tag: List[str] = Field(default_factory=list)


class BranchEnrichment(BaseModel):
    sibling_a: SiblingContrast
    sibling_b: SiblingContrast
    parent_payload: KnowledgePayload
    parent_refinement_for_a: ParentRefinement
    parent_refinement_for_b: ParentRefinement

extract_prompt = ChatPromptTemplate.from_template("""
You are creating a structured knowledge-base payload for a document.

Return a JSON object matching the schema exactly.

Document metadata:
- document_id: {document_id}
- source: {source}
- title: {title}

Instructions:
- Extract the most important knowledge from the document.
- Be concise but complete.
- Normalize duplicate ideas.
- Prefer concrete facts over vague themes.
- "document_type" describes format, not subject
- "intent_purpose" describes author intent, not subject (convince, educate, imperitive, advertise)
- "industry" describes something like a GICS industry group or industry
- "life_domain" describes where a person would apply this: career, relationships, home, healthcare, vacation, finances, worship,education, politics, legal 

Document text:
{document_text}
""")

reduce_prompt = ChatPromptTemplate.from_template("""
You are consolidating chunk-level knowledge-base payloads into one final payload.

Return a JSON object matching the schema exactly.

Document metadata:
- document_id: {document_id}
- source: {source}
- title: {title}

Chunk payloads:
{chunk_payloads}
""")

#extract_chain_prompt = ChatPromptTemplate.from_template(extract_prompt)
#reduce_chain_prompt = ChatPromptTemplate.from_template(reduce_prompt)


def build_model(model_name: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm.with_structured_output(KnowledgePayload, method="json_schema")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=50000,
    chunk_overlap=1000,
    separators=["\n\n", "\n", ". ", " ", ""],
)

async def build_doc_payload(doc: dict[str, Any], model_name="gpt-4.1-mini"):
    model = build_model(model_name)
    extract_chain = extract_prompt | model
    reduce_chain = reduce_prompt | model

    try:
        text = doc["text"] or ""
        chunks = splitter.split_text(text) if len(text) > 180000 else [text]

        if len(chunks) == 1:
            result = await extract_chain.ainvoke({
                "document_id": doc["id"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "document_text": chunks[0],
            })
            return {
                "document_id": doc["id"],
                "status": "ok",
                "payload": result.model_dump(),
            }

        chunk_results = await asyncio.gather(*[
            extract_chain.ainvoke({
                "document_id": doc["id"],
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "document_text": chunk,
            })
            for chunk in chunks
        ], return_exceptions=True)

        good_chunks = [r.model_dump() for r in chunk_results if not isinstance(r, Exception)]

        if not good_chunks:
            return {
                "document_id": doc["id"],
                "status": "error",
                "error": "all_chunks_failed",
            }

        reduced = await reduce_chain.ainvoke({
            "document_id": doc["id"],
            "source": doc.get("source", ""),
            "title": doc.get("title", ""),
            "chunk_payloads": good_chunks,
        })

        return {
            "document_id": doc["id"],
            "status": "ok",
            "payload": reduced.model_dump(),
            "chunk_count": len(chunks),
            "successful_chunks": len(good_chunks),
        }

    except Exception as e:
        return {
            "document_id": doc["id"],
            "status": "error",
            "error": str(e),
        }

async def process_documents(docs, model_name="gpt-4.1-mini", max_concurrency=8):
    sem = asyncio.Semaphore(max_concurrency)

    async def run_one(doc):
        async with sem:
            return await build_doc_payload(doc, model_name=model_name)

    return await asyncio.gather(*[run_one(doc) for doc in docs], return_exceptions=False)




#  ingest some documents, this will be contract based in the future, but this gives an idea of the shape
qry = """select id, url as source, title, full_text as text
from public.documents d
where exists (
	select doc_id from graph.v_tree_map tm
	where tree_id = '75e5abf9-bf3a-421c-872a-e3a5b81d2eec'
	and leaf = true
	and d.id=tm.doc_id)
"""



df = pd.read_sql(qry, engine)
docs = df.to_dict(orient="records")

### generate document payloads
results = await process_documents(docs=docs, model_name="gpt-4.1-mini")

errors = [r for r in results if r["status"] != "ok"]

pay_dict.update({
    r["document_id"]: r["payload"]
    for r in results
    if r["status"] == "ok"
})

df_pay_dict = pd.DataFrame(pay_dict).T.reset_index()
df_pay_dict.rename(columns={'index':'doc_id'},inplace=True)



# this is a separate pathway.  it takes document payloads and a branch_id from a tree structure and generates enrichment data for the branch.  It will generate parent payload data and a metadata payload that differentiates the child nodes from each other and from the parent.

branch_prompt = ChatPromptTemplate.from_template("""
Create a combined knowledge payload for a parent branch from two child payloads.

Return JSON matching the schema exactly.

Parent branch_id: {branch_id}
Left child node_id: {left_node_id}
Right child node_id: {right_node_id}

Left payload:
{left_payload}

Right payload:
{right_payload}

Instructions:
- Merge and synthesize, do not just concatenate.
- Preserve the most important facts, entities, topics, audience, intent, and life-domain signals.
- Add provenance entries referencing both child node IDs.
""")

branch_template = ChatPromptTemplate.from_template(
    """
You are analyzing two sibling nodes in a hierarchical knowledge tree.

Each sibling node has a JSON knowledge payload with:
title, summary, key_topics, entities, facts, intent_purpose, target_audience,
document_type, industry, life_domain, provenance.

Task:

1) Sibling contrast
- For each sibling, generate 3 human-interpretable labels that highlight how it
  differs from its sibling.
- Generate 3 short tags (3–4 words) per sibling that quickly distinguish it
  from its sibling.
- Focus on differences, not similarities.

2) Parent payload
- Create a combined parent KnowledgePayload that summarizes and unifies both
  siblings.
- Preserve the same fields and structure as the child payloads.
- Summary and labels should reflect the shared content and broad themes.
- Keep generated title concise.  Do not include terms like "Comprehensive knowledge of", "Integrated overview of..".  Just include topics.

3) Parent refinements
- For each child, generate 3 refinement labels that differentiate the child
  from the parent (more specific versions of parent-level ideas).
  Example:
    parent: "customer support"
    child:  "refund disputes"
- Generate 3 short refinement tags (3–4 words) per child that capture what is
  unique to the child compared to the parent.

Return a single JSON object that matches this structure exactly:

{{
  "sibling_a": {{
    "contrast": ["...", "...", "..."],
    "contrast_tag": ["...", "...", "..."]
  }},
  "sibling_b": {{
    "contrast": ["...", "...", "..."],
    "contrast_tag": ["...", "...", "..."]
  }},
  "parent_payload": {{
    "title": ...,
    "summary": ...,
    "key_topics": [...],
    "entities": [...],
    "facts": [...],
    "intent_purpose": [...],
    "target_audience": [...],
    "document_type": [...],
    "industry": [...],
    "life_domain": [...],
    "provenance": [...]
  }},
  "parent_refinement_for_a": {{
    "parent_refine": ["...", "...", "..."],
    "parent_refine_tag": ["...", "...", "..."]
  }},
  "parent_refinement_for_b": {{
    "parent_refine": ["...", "...", "..."],
    "parent_refine_tag": ["...", "...", "..."]
  }}
}}

Use double quotes and valid JSON only. No markdown, no comments, no extra text.

Sibling A payload:
{payload_a}

Sibling B payload:
{payload_b}

response:
"""
)



#model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)#.with_structured_output(BranchEnrichment)
#branch_chain = branch_template | model
async def process_ready_branches(branches, pay_dict, model_name="gpt-4.1-mini", max_concurrency=8):
    sem = asyncio.Semaphore(max_concurrency)
    model = ChatOpenAI(model=model_name, temperature=0)
    branch_chain = branch_prompt | model.with_structured_output(BranchEnrichment)

    async def run_branch(branch):
        left_id, right_id = branch["node_ids"]
        
        try:
            async with sem:
                result: BranchEnrichment = await branch_chain.ainvoke({
                    "branch_id": str(branch["branch_id"]),
                    "left_node_id": str(left_id),
                    "right_node_id": str(right_id),
                    "left_payload": pay_dict[left_id],
                    "right_payload": pay_dict[right_id],
                })
        except Exception as e:
            return {
                "branch_id": branch["branch_id"],
                "status": "error",
                "error": str(e),
            }

        # parent payload becomes the payload for this branch_id node
        pay_dict[branch["branch_id"]] = result.parent_payload.model_dump()

        # optionally store contrast/refinement somewhere
        sibling_meta = {
            "branch_id": branch["branch_id"],
            "left_node_id": left_id,
            "right_node_id": right_id,
            "sibling_a": result.sibling_a.model_dump(),
            "sibling_b": result.sibling_b.model_dump(),
            "parent_refinement_for_a": result.parent_refinement_for_a.model_dump(),
            "parent_refinement_for_b": result.parent_refinement_for_b.model_dump(),
        }

        return {
            "branch_id": branch["branch_id"],
            "status": "ok",
            "payload": result.parent_payload.model_dump(),
            "enrichment": sibling_meta,
        }

    return await asyncio.gather(*(run_branch(b) for b in branches))



    # query to get branch data.  ue this as guidance for the shape of the inputs, but you won't be connecting to the database

    qry = """select branch_id, array_agg(node_id) AS node_ids, sum(case when tz.edge_ix = 1 then 1 else 0 end) as payloads
from graph.v_orchard tz 
where tree_id = '75e5abf9-bf3a-421c-872a-e3a5b81d2eec'
group by branch_id
"""

df_branch = pd.read_sql(qry,engine)
branches = df_branch.to_dict(orient="records")



while True:
    ready = [
        b for b in branches
        if b["branch_id"] not in pay_dict
        and all(node_id in pay_dict for node_id in b["node_ids"])
    ]

    if not ready:
        print("No more ready branches.")
        break

    merged_results = await process_ready_branches(ready, pay_dict)

    new_count = 0
    for r in merged_results:
        if r["status"] == "ok":
            pay_dict[r["branch_id"]] = r["payload"]
            branch_meta_dict[r["branch_id"]] = r["enrichment"]
            new_count += 1

    print(f"Built {new_count} branch payloads this pass.")

    if new_count == 0:
        print("No successful merges this pass; stopping.")
        break


df_pay_dict = pd.DataFrame(branch_meta_dict).T.reset_index()
df_pay_dict.rename(columns={'index':'doc_id'},inplace=True)
df_pay_dict.head()
df_pay_dict.to_csv('df_meta_dict_full.csv',index=False)


#  this extracts the sibling and parent differentiation data and attaches it to the corresponding nodes

import ast

full_list = df_pay_dict_full.to_dict(orient='records')
full_dict = {d['doc_id']:d for d in full_list}

meta_list = df_meta_dict.to_dict(orient='records')
meta_dict = {d['doc_id']:d for d in full_list}

node_dict = {
    node_id: {
        **ast.literal_eval(sibling),
        **ast.literal_eval(parent)
    }
    for item in meta_list
    for node_id, sibling, parent in [
        (item["left_node_id"], item["sibling_a"], item["parent_refinement_for_a"]),
        (item["right_node_id"], item["sibling_b"], item["parent_refinement_for_b"]),
    ]
}

merged = {
    k: {**full_dict.get(k, {}), **node_dict.get(k, {})}
    for k in set(full_dict) | set(node_dict)
}


