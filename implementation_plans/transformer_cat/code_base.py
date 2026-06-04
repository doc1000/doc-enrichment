# the idea is to use a zero-shot teacher to map labels to your anchor files
# then use the student model to fit the new ontology
# then export the tiny weight matrix to production

from datasets import load_dataset

# Pulls a highly diverse, clean text corpus instantly into local memory
dataset = load_dataset("ag_news", split="train[:2000]") 
anchor_texts = dataset["text"] # 2,000 clean documents ready to go


# this is the code to use the base model to get the features

import torch
from transformers import AutoTokenizer, AutoModel

# Use the fast base model without the zero-shot head architecture overhead
model_name = "tasksource/ModernBERT-base-nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(documents_list, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = base_model(**inputs)
    # Perform manual mean pooling to exclude padding token skew
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    token_embeddings = outputs.last_hidden_state
    
    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    X_features = (sum_embeddings / sum_mask).numpy() # Shape: (Num_Docs, 768)

 #Save to disk or DB for future ontology runs 
 np.save("data/document_features.npy", X_features)
 with open("data/anchor_corpus.json", "w") as f:
    json.dump(anchor_texts, f)


import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# 1. Load your pre-computed feature matrix (X)
X_features = np.load("data/document_features.npy")
with open("data/anchor_corpus.json", "r") as f:
    anchor_texts = json.load(f)

# 2. Define the *New* Ontology handed down by stakeholders
new_ontology = ["logistics", "cybersecurity", "human_resources", "legal"]

# 3. Spin up the Zero-Shot Teacher ONLY to map labels to your anchor files
teacher = pipeline("zero-shot-classification", model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0", device=-1)
synthetic_results = teacher(anchor_texts, new_ontology)
y_labels = [res['labels'][0] for res in synthetic_results] # Get top class for each doc

# 4. Fit the new Student Model (Takes less than 1 second)
student_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
student_model.fit(X_features, y_labels)

# 5. Export the tiny weight matrix to production
joblib.dump(student_model, "models/new_ontology_lr.joblib")


# this is the code to export the student model to production

import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Train your student model
classifier = student_model

# 2. Extract the full probability distribution matrix
# Shape will be: (Num_Docs, Num_Classes)
probs_matrix = classifier.predict_proba(X_features)

# 3. Capture the class order so you know which column maps to which category
# e.g., ['business', 'education', 'science'...]
class_labels = classifier.classes_

# 4. Save everything into a single compressed binary file
np.savez_compressed(
    "data/ontology_results.npz",
    doc_ids=document_ids_array,
    probabilities=probs_matrix,
    classes=class_labels
)



## chunking code options

# semantic chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings # or a fast local HuggingFace embedding

# Initialize the smart chunker
# It will calculate statistical thresholds based on standard deviation or percentiles
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), 
    breakpoint_threshold_type="percentile" # Alternative: "standard_deviation"
)

# Pass your large text document
docs = text_splitter.create_documents([large_document_text])


# lexical chunking
from langchain_text_splitters import NLTKTextSplitter

# NLTK TextTiling splits purely on lexical cohesion shifts
# It looks for "document shape" and keyword-density transitions
text_splitter = NLTKTextSplitter(chunk_size=4000) 

docs = text_splitter.split_text(large_document_text)


# Structural chunking

from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)



# intelligent chunking router

import re
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    NLTKTextSplitter, 
    RecursiveCharacterTextSplitter
)

class IntelligentChunkingRouter:
    def __init__(self):
        # Pre-configure the fallback and structural splitters
        self.markdown_headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.markdown_headers)
        
        self.list_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100, 
            separators=["\n\n", "\n", " "]
        )
        
        self.prose_splitter = NLTKTextSplitter(chunk_size=2000)

    def route_and_split(self, file_content: str):
        # 1. Take a micro-sample of the document head to determine shape
        sample = file_content[:2000]
        
        # 2. Extract structural syntax features using basic regex
        header_count = len(re.findall(r'^#{1,3}\s', sample, re.MULTILINE))
        list_count = len(re.findall(r'^([\*\-\+]\s|\d+\.\s)', sample, re.MULTILINE))
        
        # 3. Routing Rules (Executed sequentially in microseconds)
        if header_count >= 3:
            print("Routing to: Markdown Header Splitter")
            # If the markdown structure breaks down later, chain a recursive fallback
            initial_splits = self.md_splitter.split_text(file_content)
            return self.list_splitter.split_documents(initial_splits)
            
        elif list_count >= 6:
            print("Routing to: Recursive List Splitter")
            return self.list_splitter.create_documents([file_content])
            
        else:
            print("Routing to: Lexical TextTiling Prose Splitter")
            # Falls back to examining paragraph-level lexical shifts
            return self.prose_splitter.create_documents([file_content])

# --- Execution Example ---
router = IntelligentChunkingRouter()
chunks = router.route_and_split(your_markdown_string_data)



# chunking with tracking

import uuid
import datetime
from typing import Dict, List
from langchain_core.documents import Document

def chunk_document_with_tracking(doc_data: Dict, router_instance) -> List[Document]:
    """
    Input schema: {
        "document_id": "doc-12345", 
        "source": "sharepoint/file.md", 
        "title": "Q3 Financial Plan", 
        "body": "...", 
        "timestamp": "2026-06-04T15:00:00"
    }
    """
    raw_body = doc_data["body"]
    
    # 1. Route and split the raw text into chunks using our heuristic router
    # This returns LangChain Document objects with temporary structural text
    chunks = router_instance.route_and_split(raw_body)
    
    tracked_chunks = []
    
    # 2. Iterate and enrich each chunk with parent tracking attributes
    for idx, chunk in enumerate(chunks):
        enriched_meta = {
            "parent_doc_id": doc_data.get("document_id", str(uuid.uuid4())),
            "chunk_id": f"{doc_data.get('document_id')}_chunk_{idx}",
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "source": doc_data.get("source"),
            "title": doc_data.get("title"),
            "timestamp": doc_data.get("timestamp", datetime.datetime.now().isoformat())
        }
        
        # Instantiate a clean LangChain Document containing the text and metadata payload
        tracked_chunks.append(
            Document(page_content=chunk.page_content, metadata=enriched_meta)
        )
        
    return tracked_chunks



import numpy as np
import json

# Assume you processed a batch of 500 final chunks
# X_features shape: (500, 768)
# probs_matrix shape: (500, 10)  <- from your model.predict_proba()

# 1. Isolate text strings and dictionary payloads into parallel native tracking lists
chunk_ids = [chunk.metadata["chunk_id"] for chunk in final_chunks_list]
chunk_texts = [chunk.page_content for chunk in final_chunks_list]

# Serialize the full nested metadata dicts to JSON strings for binary safety
metadata_strings = [json.dumps(chunk.metadata) for chunk in final_chunks_list]

# 2. Dump everything into one bulletproof archive file
np.savez_compressed(
    "data/production_enrichment_cache.npz",
    features=X_features,                  # Dense float32 embeddings matrix
    probabilities=probs_matrix,           # Calibrated ontology distributions matrix
    chunk_ids=np.array(chunk_ids),        # Quickly indexable primary keys
    texts=np.array(chunk_texts),          # Raw text backup for instant RAG delivery
    metadata=np.array(metadata_strings),   # Full stringified metadata payloads
    classes=classifier.classes_           # The ordered ontology categories array
)



# Strategic AdvantageThis approach avoids the complexity of setting up external relational databases during development. Everything needed for evaluation, classification adjustments, or vector search filtering is consolidated into a single, high-performance matrix archive format.


# shift to registry for model management

import joblib
import json
import os

def register_taxonomy_student_model(taxonomy_name: str, trained_model, registry_path="data/taxonomies_registry.json"):
    # 1. Save the model weights and categories to a single spot
    model_payload = {
        "weights": trained_model.coef_,
        "intercept": trained_model.intercept_,
        "classes": trained_model.classes_ # Explicitly tracks the category order
    }
    model_file_path = f"data/models/{taxonomy_name}_student.joblib"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    joblib.dump(model_payload, model_file_path)
    
    # 2. Update your global JSON index config
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except FileNotFoundError:
        registry = {}
        
    registry[taxonomy_name] = {
        "model_file": model_file_path,
        "categories": trained_model.classes_.tolist(),
        "features_dim": trained_model.coef_.shape[1]
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)




# To load existing taxonomies and reconstruct a classifier instantly without re-training:

def load_student_classifier(taxonomy_name: str):
    # Pull the weights dictionary
    payload = joblib.load(f"data/models/{taxonomy_name}_student.joblib")
    
    # Rehydrate into a clean scikit-learn instance
    clf = LogisticRegression(class_weight='balanced')
    clf.classes_ = payload["classes"]
    clf.coef_ = payload["weights"]
    clf.intercept_ = payload["intercept"]
    return clf



# The Unified Enrichment Output SchemaWhen a document passes through your pipeline, it returns a single, structured Python dictionary (or JSON payload). This schema maps out the parent attributes, the aggregated full-document taxonomy scores, and an array of individual chunk objects containing their own respective metadata and localized probabilities.

{
  "document_id": "doc_98765",
  "source": "sharepoint/legal_brief.md",
  "title": "2026 Q2 Regulatory Strategy",
  "timestamp": "2026-06-04T15:00:00",
  "full_document_taxonomies": {
    "core_topics_v1": {
      "legal": 0.7240,
      "business": 0.2110,
      "technology": 0.0650
    },
    "compliance_risk_v2": {
      "high_risk": 0.8900,
      "low_risk": 0.1100
    }
  },
  "chunks": [
    {
      "chunk_id": "doc_98765_chunk_0",
      "chunk_index": 0,
      "body": "This section covers the compliance guidelines under the 2026 maritime shipping mandates...",
      "chunk_taxonomies": {
        "core_topics_v1": {
          "legal": 0.9100,
          "business": 0.0500,
          "technology": 0.0400
        },
        "compliance_risk_v2": {
          "high_risk": 0.9500,
          "low_risk": 0.0500
        }
      }
    },
    {
      "chunk_id": "doc_98765_chunk_1",
      "chunk_index": 1,
      "body": "Financial allocations for our technology transformation stack total five million dollars...",
      "chunk_taxonomies": {
        "core_topics_v1": {
          "legal": 0.5380,
          "business": 0.3720,
          "technology": 0.0900
        },
        "compliance_risk_v2": {
          "high_risk": 0.8300,
          "low_risk": 0.1700
        }
      }
    }
  ]
}


# Part 4: How to Implement the Multi-Taxonomy Inference ScriptHere is the operational loop that runs across all your registered models to assemble the nested document schema:

import numpy as np

def run_multi_taxonomy_inference(chunk_objects: list, X_chunk_features: np.ndarray, registry_path="data/taxonomies_registry.json"):
    # Load your central registry tracking file
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    # Initialize dictionary arrays to hold results per taxonomy
    active_taxonomies = list(registry.keys())
    
    # 1. Calculate and append probabilities to individual chunks
    for tax_name in active_taxonomies:
        clf = load_student_classifier(tax_name)
        chunk_probs = clf.predict_proba(X_chunk_features) # Shape: (Num_Chunks, Num_Classes)
        
        for i, chunk in enumerate(chunk_objects):
            if "chunk_taxonomies" not in chunk:
                chunk["chunk_taxonomies"] = {}
                
            # Zip categories to their respective values for clean JSON output
            chunk["chunk_taxonomies"][tax_name] = dict(zip(clf.classes_, chunk_probs[i]))

    # 2. Aggregate chunk probabilities to calculate parent document metrics
    parent_doc_taxonomies = {}
    for tax_name in active_taxonomies:
        # Extract the probability distributions for all child chunks
        all_chunk_probs_for_tax = [c["chunk_taxonomies"][tax_name] for c in chunk_objects]
        categories = list(all_chunk_probs_for_tax[0].keys())
        
        # Calculate the mathematical mean across all child chunk objects
        mean_probs = np.mean([[probs[cat] for cat in categories] for probs in all_chunk_probs_for_tax], axis=0)
        parent_doc_taxonomies[tax_name] = dict(zip(categories, mean_probs))
        
    return parent_doc_taxonomies



# Strategic AdvantageBecause you intend to reuse these exact chunks to calculate semantic vector embeddings later, this architecture keeps your data footprint perfectly clean. The raw text and base features remain decoupled from your classifications. You can pass this output payload straight into down-stream document classification engines, clustering algorithms, or filing architectures




# Minimal LangGraph ImplementationHere is how to structure this entire end-to-end framework using langgraph.1. Define the Global State SchemaThe state acts as a single data bucket that gathers, passes, and retains enrichment details as the document moves through the pipeline steps.

from typing import Dict, List, Any
from typing_extensions import TypedDict

class PipelineState(TypedDict):
    # Inputs
    raw_document: Dict[str, Any]  # Contains id, source, title, body, timestamp
    
    # Mid-pipeline objects
    chunk_documents: List[Any]    # Array of LangChain Document objects
    feature_matrix: Any           # NumPy array (Num_Chunks x 768)
    
    # Final Output
    enriched_payload: Dict[str, Any] # Complete dual-layer JSON output structure

# 2. Define the Graph Nodes (Python Code Steps)Each node is a standard Python function that takes the current state, processes data, and returns an updated dictionary payload
import numpy as np
from langgraph.graph import END

# --- NODE 1: Ingest & Route ---
def route_chunking_decision(state: PipelineState) -> str:
    """Conditional Edge: Inspects the document head to route to a chunking node."""
    sample = state["raw_document"]["body"][:2000]
    header_count = sample.count("#")
    
    if header_count >= 3:
        return "chunk_markdown"
    elif sample.count("* ") >= 6:
        return "chunk_list"
    else:
        return "chunk_prose"

# --- CHUNKING NODES ---
def chunk_markdown_node(state: PipelineState) -> Dict:
    # Execute MarkdownHeaderTextSplitter logic here
    # Wrap text into LangChain Document objects with lineage tracking
    return {"chunk_documents": list_of_tracked_markdown_chunks}

def chunk_prose_node(state: PipelineState) -> Dict:
    # Execute NLTKTextSplitter / TextTiling logic here
    return {"chunk_documents": list_of_tracked_prose_chunks}

# --- NODE 2: Base Featurization ---
def featurize_chunks_node(state: PipelineState) -> Dict:
    chunks = state["chunk_documents"]
    texts = [c.page_content for c in chunks]
    
    # Run the local ModernBERT encoder and mean pooling step
    # X_features = extract_modernbert_features(texts)
    
    return {"feature_matrix": X_features}

# --- NODE 3: Supervised Classification ---
def classify_taxonomy_node(state: PipelineState) -> Dict:
    X_features = state["feature_matrix"]
    chunks = state["chunk_documents"]
    doc_info = state["raw_document"]
    
    # 1. Load active taxonomy classifiers from registry
    # 2. Compute chunk probabilities using classifier.predict_proba(X_features)
    # 3. Aggregate chunk probabilities to calculate parent document scores
    # 4. Assemble the nested final JSON schema
    
    return {"enriched_payload": final_nested_json_dictionary}

# 3. Compile the GraphAssemble the nodes and wire the pipeline edges together into a compiled executable framework.
from langgraph.graph import StateGraph

# Initialize the stateful workflow graph
workflow = StateGraph(PipelineState)

# Add processing nodes to the architecture
workflow.add_node("chunk_markdown", chunk_markdown_node)
workflow.add_node("chunk_prose", chunk_prose_node)
workflow.add_node("featurize", featurize_chunks_node)
workflow.add_node("classify", classify_taxonomy_node)

# Set the entry point by evaluation a conditional routing rule
workflow.set_conditional_entry_point(
    route_chunking_decision,
    {
        "chunk_markdown": "chunk_markdown",
        "chunk_prose": "chunk_prose",
        "chunk_list": "chunk_prose" # Falls back safely if needed
    }
)

# Wire the deterministic operational pathways
workflow.add_edge("chunk_markdown", "featurize")
workflow.add_edge("chunk_prose", "featurize")
workflow.add_edge("featurize", "classify")
workflow.add_edge("classify", END)

# Compile into a runnable system
app = workflow.compile()

# 4. Execute the End-to-End PipelineTo trigger LangSmith tracking automatically, initialize your pipeline variables using standard environment configurations before running your loops:

import os

# Activating these variables routes all operational data logs straight to LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "document_enrichment_pipeline"

# Run a sample payload through the completed graph framework
input_data = {
    "raw_document": {
        "document_id": "doc_abc123",
        "source": "internal/sharepoint_doc.md",
        "title": "Corporate Compliance Directives",
        "body": "# Global Compliance \n## Directives\n This is document body content...",
        "timestamp": "2026-06-04T15:20:00"
    }
}

final_output_state = app.invoke(input_data)

# Print out your final nested dual-layer enrichment records
print(final_output_state["enriched_payload"])


# Externalized Prompt Design (prompts.json)To keep your core orchestration code clean, store your prompt templates in an external configuration file. This allows you to tweak the taxonomy generator without rebuilding your graph.

{
  "taxonomy_enrichment": "You are an expert ontologist. Expand this raw category label into a rich lexical definition for a zero-shot classification model.\nRaw Category: {category_name}\nContext/Domain: {taxonomy_name}\n\nReturn a JSON object with this exact schema:\n{{\n  \"category\": \"{category_name}\",\n  \"synonyms\": [\"word1\", \"word2\"],\n  \"descriptors\": [\"phrase describing intent\", \"another phrase\"],\n  \"sub_categories\": [\"sub1\", \"sub2\"]\n}}"
}


# Implementation of the Administration WorkflowHere is how you write this stateful workflow using LangGraph components:

import json
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from sklearn.linear_model import LogisticRegression

# 1. Define the Administrative Configuration State
class AdminState(TypedDict):
    taxonomy_name: str
    categories_input: List[Dict[str, Any]] # e.g., [{"label": "finance", "descriptors": [...]}]
    force_llm_enrichment: bool
    
    # Optional Training Sources passed via execution call
    provided_labelled_corpus: Optional[List[Dict[str, Any]]]   # [{"text": "...", "label": "finance"}]
    provided_unlabelled_corpus: Optional[List[str]]            # ["text1", "text2"]
    
    # Internal Pipeline Trackers
    validated_taxonomy: Dict[str, Any]
    training_x: Optional[np.ndarray]
    training_y: Optional[List[str]]

# --- NODE 1: Validate Taxonomy ---
def validate_taxonomy_node(state: AdminState) -> Dict:
    # Read global registry to check for naming collisions
    with open("data/taxonomies_registry.json", "get", fallback={}) as f:
        registry = json.load(f)
        
    if state["taxonomy_name"] in registry:
        raise ValueError(f"Taxonomy name '{state['taxonomy_name']}' is already in use.")
        
    return {"validated_taxonomy": {"name": state["taxonomy_name"], "categories": state["categories_input"]}}

# --- ROUTER EDGE: Check Richness ---
def check_definition_richness(state: AdminState) -> str:
    if state["force_llm_enrichment"]:
        return "call_llm_enricher"
        
    # Check if descriptors or synonyms are missing in the input fields
    for cat in state["categories_input"]:
        if not cat.get("descriptors") and not cat.get("synonyms"):
            return "call_llm_enricher"
            
    return "route_data_source"

# --- NODE 2: Call LLM Enricher (Gemini / External Contract Call) ---
def call_llm_enricher_node(state: AdminState) -> Dict:
    # 1. Load your external prompt template
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)
    
    enriched_categories = []
    # 2. Iterate and invoke Gemini to expand missing context definitions
    for cat in state["validated_taxonomy"]["categories"]:
        if not cat.get("descriptors"):
            prompt = prompts["taxonomy_enrichment"].format(
                category_name=cat["label"], 
                taxonomy_name=state["taxonomy_name"]
            )
            # llm_output = gemini_client.generate_structured_json(prompt)
            # enriched_categories.append(llm_output)
            pass
        else:
            enriched_categories.append(cat)
            
    return {"validated_taxonomy": {"name": state["taxonomy_name"], "categories": enriched_categories}}

# --- ROUTER EDGE: Data Source Routing ---
def route_data_source_decision(state: AdminState) -> str:
    if state.get("provided_labelled_corpus"):
        return "process_labelled_corpus"
    elif state.get("provided_unlabelled_corpus"):
        return "process_unlabelled_corpus"
    else:
        return "process_anchor_corpus"

# --- NODE 3: Process Labelled Corpus ---
def process_labelled_corpus_node(state: AdminState) -> Dict:
    corpus = state["provided_labelled_corpus"]
    
    # Data Scientist Fallback Safeguard:
    # If the user supplied fewer than 150 total documents, fall back to anchor generation
    if len(corpus) < 150:
        print("⚠️ Warning: Labelled dataset too small. Merging with anchor dataset.")
        # Trigger flag to mix with anchor data
        
    # extract texts and target labels, generate ModernBERT pooling vectors (X, y)
    return {"training_x": X_matrix, "training_y": y_labels}

# --- NODE 4/5: Run Zero-Shot Teacher Over Unlabelled or Anchor Data ---
def process_unlabelled_or_anchor_node(state: AdminState) -> Dict:
    # 1. Select text pool (either user-provided files or your local fixed fallback dataset)
    texts = state["provided_unlabelled_corpus"] if state.get("provided_unlabelled_corpus") else load_local_anchor_corpus()
    
    # 2. Re-construct a composite string descriptor for each category
    # "finance: revenue profit margin corporate strategy capital stock assets"
    composite_labels = []
    for cat in state["validated_taxonomy"]["categories"]:
        combined_string = f"{cat['label']}: {' '.join(cat.get('synonyms', []))} {' '.join(cat.get('descriptors', []))}"
        composite_labels.append(combined_string)
        
    # 3. Execute the local ModernBERT Zero-shot model over the texts pool
    # teacher_predictions = local_zero_shot_pipeline(texts, composite_labels)
    # y_labels = [map_to_clean_label(res) for res in teacher_predictions]
    # X_matrix = extract_modernbert_features(texts)
    
    return {"training_x": X_matrix, "training_y": y_labels}

# --- NODE 6: Train and Register ---
def train_and_register_node(state: AdminState) -> Dict:
    X = state["training_x"]
    y = state["training_y"]
    
    # Standard Ridge Regularization for small dataset boundaries
    student_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
    student_model.fit(X, y)
    
    # Save the model payload and update metadata configuration arrays inside taxonomies_registry.json
    # register_taxonomy_student_model(state["taxonomy_name"], student_model)
    
    return {}


# Compiling Your Admin Blueprint
admin_flow = StateGraph(AdminState)

admin_flow.add_node("validate", validate_taxonomy_node)
admin_flow.add_node("call_llm_enricher", call_llm_enricher_node)
admin_flow.add_node("process_labelled", process_labelled_corpus_node)
admin_flow.add_node("process_unlabelled_teacher", process_unlabelled_or_anchor_node)

admin_flow.set_entry_point("validate")

admin_flow.add_conditional_edges(
    "validate",
    check_definition_richness,
    {"call_llm_enricher": "call_llm_enricher", "route_data_source": "process_unlabelled_teacher"} # Map routes
)

# Connect downstream nodes to classification execution steps
admin_flow.add_conditional_edges(
    "call_llm_enricher",
    route_data_source_decision,
    {
        "process_labelled": "process_labelled",
        "process_unlabelled_corpus": "process_unlabelled_teacher",
        "process_anchor_corpus": "process_unlabelled_teacher"
    }
)
admin_flow.add_edge("process_labelled", END)
admin_flow.add_edge("process_unlabelled_teacher", END)

admin_app = admin_flow.compile()
