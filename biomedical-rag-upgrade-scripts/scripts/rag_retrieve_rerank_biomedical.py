import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import faiss
import numpy as np
import torch
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from sentence_transformers import CrossEncoder

# ================= CONFIG =================
PASSAGES_FILE = "pubmed/processed/pubmed_passages_enhanced.json"
EMB_FILE = "embeddings_biomed/passage_embeddings.npy"
INDEX_FILE = "index/faiss_biomed.index"

QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_DENSE = 20
TOP_K_RERANK = 5
MIN_RERANK_SCORE = -7.5
MIN_KEYWORD_HITS = 2
RARE_TERM_MIN_LEN = 8
DEVICE = "cpu"
# =========================================


print("Loading passages...")
with open(PASSAGES_FILE, encoding="utf-8") as f:
    passages = json.load(f)

print("Loading / building FAISS index...")
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    os.makedirs("index", exist_ok=True)
    embeddings = np.load(EMB_FILE).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

faiss.omp_set_num_threads(1)

print("Loading biomedical question encoder...")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_ENCODER)
q_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_ENCODER).to(DEVICE)
q_encoder.eval()

print("Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANK_MODEL)


def encode_question(question: str) -> np.ndarray:
    inputs = q_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = q_encoder(**inputs)
        q_emb = outputs.pooler_output.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_emb)
    return q_emb


def retrieve(question: str):
    q_emb = encode_question(question)
    scores, indices = index.search(q_emb, TOP_K_DENSE)
    return [(float(scores[0][i]), passages[idx]) for i, idx in enumerate(indices[0])]


def keyword_filter(question: str, candidates):
    keywords = [w.lower() for w in question.split() if len(w) > 4]
    if not keywords:
        return candidates

    # Enforce rare biomedical terms when present to avoid semantically broad drift.
    rare_terms = [k for k in keywords if len(k) >= RARE_TERM_MIN_LEN]
    must_terms = [
        t for t in rare_terms
        if "-" in t or any(ch.isdigit() for ch in t)
    ]
    if not must_terms and rare_terms:
        must_terms = [rare_terms[0]]

    filtered = []
    for dense_score, p in candidates:
        text = p["text"].lower()
        entities = " ".join(p.get("entities", [])).lower()
        keyword_hits = sum(k in text or k in entities for k in keywords)
        if keyword_hits < MIN_KEYWORD_HITS:
            continue

        if must_terms and not any(t in text or t in entities for t in must_terms):
            continue

        if keyword_hits > 0:
            # Small lexical boost to help reranker prioritize topical candidates.
            filtered.append((dense_score + 0.01 * keyword_hits, p))

    # Hard lexical fallback across corpus when must-terms are missing in dense top-k.
    if must_terms:
        lexical = []
        for p in passages:
            text = p["text"].lower()
            entities = " ".join(p.get("entities", [])).lower()
            if not any(t in text or t in entities for t in must_terms):
                continue
            keyword_hits = sum(k in text or k in entities for k in keywords)
            if keyword_hits > 0:
                lexical.append((0.001 * keyword_hits, p))
        if lexical:
            lexical.sort(key=lambda x: x[0], reverse=True)
            return lexical[:TOP_K_DENSE]

    # Fall back progressively if filtering is still too strict.
    if filtered:
        return filtered

    fallback = []
    for dense_score, p in candidates:
        text = p["text"].lower()
        entities = " ".join(p.get("entities", [])).lower()
        keyword_hits = sum(k in text or k in entities for k in keywords)
        if keyword_hits > 0:
            fallback.append((dense_score + 0.005 * keyword_hits, p))
    return fallback if fallback else candidates


def rerank(question: str, candidates):
    pairs = [(question, p["text"][:512]) for _, p in candidates]
    rerank_scores = reranker.predict(pairs)
    merged = []
    for (dense_score, passage), rerank_score in zip(candidates, rerank_scores):
        merged.append((float(rerank_score), float(dense_score), passage))
    merged.sort(key=lambda x: x[0], reverse=True)
    return merged[:TOP_K_RERANK]


if __name__ == "__main__":
    verification_queries = [
        {
            "label": "BEST-CASE",
            "question": "How does etafenone affect regional myocardial blood flow in dogs?",
            "why": (
                "Contains specific, in-corpus biomedical terms "
                "(etafenone + myocardial blood flow + dogs), so multiple passages match."
            ),
        },
        {
            "label": "WORST-CASE (still partially relevant)",
            "question": "What immunological effects are described for antitumor compounds in preclinical rat models?",
            "why": (
                "Broad wording covers many topics; at least one relevant passage exists, "
                "but lexical overlap is weaker and noisy candidates are more likely."
            ),
        },
    ]

    for item in verification_queries:
        question = item["question"]
        print("\n" + "=" * 90)
        print(f"{item['label']} QUERY")
        print("Question:", question)
        print("Why this behaves this way:", item["why"])
        print("=" * 90)

        dense_results = retrieve(question)
        filtered_results = keyword_filter(question, dense_results)
        if filtered_results and all(score < 0.1 for score, _ in filtered_results):
            print("\nUsing lexical fallback candidates (dense top-k lacked must-terms).")
        reranked = rerank(question, filtered_results)

        if reranked and reranked[0][0] < MIN_RERANK_SCORE:
            print(
                f"\nNo confident retrieval (top rerank score {reranked[0][0]:.4f} < "
                f"{MIN_RERANK_SCORE:.1f}). Results may be weak."
            )

        print("\nTop reranked passages:\n")
        for rank, (rr_score, dense_score, passage) in enumerate(reranked, 1):
            print(f"[{rank}] rerank={rr_score:.4f} dense={dense_score:.4f}")
            print(f"passage_id={passage.get('passage_id')} pmid={passage.get('pmid')}")
            print(passage["text"][:400], "...\n")
