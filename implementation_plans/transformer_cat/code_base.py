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

