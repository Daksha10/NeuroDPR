import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import json
import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# ================= CONFIG =================
PASSAGES_FILE = "pubmed/processed/pubmed_passages.json"
INDEX_FILE = "index/faiss.index"
TOP_K = 5
# =========================================

print("🔄 Loading passages...")
with open(PASSAGES_FILE) as f:
    passages = json.load(f)

print("🔄 Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)
faiss.omp_set_num_threads(1)

print("🔄 Loading DPR Question Encoder...")
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
model.eval()

def search(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True)
    with torch.inference_mode():
        q_emb = model(**inputs).pooler_output.numpy()

    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, TOP_K)

    return [(scores[0][i], passages[idx]) for i, idx in enumerate(indices[0])]

# 🔍 Example query
query = "What causes Parkinson's disease?"
results = search(query)

print("\n🔍 Top Results:\n")
for score, res in results:
    print(f"[Score: {score:.4f}] {res['text'][:200]}...\n")
