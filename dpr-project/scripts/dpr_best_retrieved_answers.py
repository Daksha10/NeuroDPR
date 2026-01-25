import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import faiss
import torch
import numpy as np
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from sentence_transformers import CrossEncoder

# ================= CONFIG =================
PASSAGES_FILE = "pubmed/processed/pubmed_passages.json"
INDEX_FILE = "index/faiss.index"

TOP_K_DPR = 30          # DPR retrieval
TOP_K_RERANK = 5        # After reranking
TOP_K_ANSWERS = 3       # Extractive answers (optional)

MAX_ANSWER_LEN = 30
DEVICE = "cpu"
# =========================================


# ---------- LOAD DATA ----------
print("🔄 Loading passages...")
with open(PASSAGES_FILE) as f:
    passages = json.load(f)

print("🔄 Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)
faiss.omp_set_num_threads(1)


# ---------- DPR QUESTION ENCODER ----------
print("🔄 Loading DPR Question Encoder...")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
).to(DEVICE)
q_encoder.eval()


# ---------- CROSS-ENCODER RERANKER ----------
print("🔄 Loading Cross-Encoder Re-ranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------- OPTIONAL EXTRACTIVE READER ----------
print("🔄 Loading BioBERT Reader (optional)...")
reader_tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    use_fast=True
)
reader_model = AutoModelForQuestionAnswering.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
).to(DEVICE)
reader_model.eval()


# ---------- DPR RETRIEVAL ----------
def retrieve_passages(question: str):
    inputs = q_tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        q_emb = q_encoder(**inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, TOP_K_DPR)

    return [passages[i] for i in indices[0]]


# ---------- KEYWORD FILTER ----------
def keyword_filter(passages, question):
    keywords = [w.lower() for w in question.split() if len(w) > 4]
    filtered = []

    for p in passages:
        text = p["text"].lower()
        if any(k in text for k in keywords):
            filtered.append(p)

    return filtered if filtered else passages


# ---------- RERANKING ----------
def rerank_passages(question, passages, top_k=TOP_K_RERANK):
    pairs = [(question, p["text"][:512]) for p in passages]
    scores = reranker.predict(pairs)

    scored = list(zip(passages, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


# ---------- OPTIONAL EXTRACTIVE QA ----------
def extractive_answer(question, context):
    inputs = reader_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    input_ids = inputs["input_ids"][0]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = reader_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    sep_id = reader_tokenizer.sep_token_id
    sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]
    context_start = sep_positions[1] + 1

    best_score = float("-inf")
    best_answer = ""

    for i in range(context_start, len(start_logits)):
        for j in range(i, min(i + MAX_ANSWER_LEN, len(end_logits))):
            if offset_mapping[i] is None or offset_mapping[j] is None:
                continue

            score = start_logits[i] + end_logits[j]
            if score > best_score:
                start_char = offset_mapping[i][0]
                end_char = offset_mapping[j][1]
                answer = context[start_char:end_char].strip()
                if len(answer.split()) >= 2:
                    best_score = score
                    best_answer = answer

    return best_answer, best_score


# ---------- FULL PIPELINE ----------
def answer_question(question: str):
    # 1. DPR retrieval
    retrieved = retrieve_passages(question)

    # 2. Keyword filtering
    filtered = keyword_filter(retrieved, question)

    # 3. Re-ranking
    reranked = rerank_passages(question, filtered)

    print("\n🔍 BEST RETRIEVED EVIDENCE PASSAGES:\n")

    results = []

    for rank, (p, score) in enumerate(reranked, 1):
        print(f"[{rank}] Rerank Score: {score:.4f}")
        print(p["text"][:400], "...\n")

        # Optional extractive QA
        ans, ans_score = extractive_answer(question, p["text"])
        results.append({
            "rank": rank,
            "rerank_score": score,
            "answer": ans,
            "answer_score": ans_score,
            "passage": p["text"]
        })

    return results


# ================= RUN =================
if __name__ == "__main__":
    question = "What effect does alpha-bisabolol have on gastric lesions?"

    print("\n" + "=" * 80)
    print("❓ Question:", question)
    print("=" * 80)

    results = answer_question(question)

    print("\n📌 BEST ANSWER CANDIDATES:\n")
    for r in results:
        print(f"Rank {r['rank']}")
        print("Extracted Answer:", r["answer"] if r["answer"] else "— (no clean span)")
        print("-" * 80)
