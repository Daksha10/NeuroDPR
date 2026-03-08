import json
import string
from typing import List, Dict, Tuple

import faiss
import numpy as np
import torch
from transformers import (
    DPRQuestionEncoderTokenizer,
    DPRQuestionEncoder,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from sentence_transformers import CrossEncoder

# ================= CONFIG =================
PASSAGES_FILE = "pubmed/processed/pubmed_passages_enhanced.json"
EMB_FILE = "embeddings_biomed/passage_embeddings.npy"
INDEX_FILE = "index/faiss_biomed.index"
EVAL_FILE = "eval/biomedical_eval.json"

QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
# QA-finetuned reader for better extractive answer quality.
READER_MODEL = "deepset/roberta-base-squad2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_VALUES = [5, 10, 20]
TOP_K_DENSE = 20
TOP_K_RERANK = 5
MIN_RERANK_SCORE = -7.5
MIN_KEYWORD_HITS = 2
RARE_TERM_MIN_LEN = 8
MAX_ANSWER_LEN = 30
DEVICE = "cpu"
# =========================================


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, references: List[str]) -> float:
    pred = normalize_text(prediction)
    refs = {normalize_text(r) for r in references}
    return 1.0 if pred in refs else 0.0


def token_f1(prediction: str, references: List[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
        if precision + recall > 0:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def read_answer(reader_tokenizer, reader_model, question: str, context: str) -> Tuple[str, float]:
    inputs = reader_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
    )
    # Fast tokenizers provide sequence ids so we can isolate context tokens robustly.
    sequence_ids = inputs.sequence_ids(0)
    offset_mapping = inputs["offset_mapping"][0].tolist()
    inputs.pop("offset_mapping")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = reader_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    best_score = float("-inf")
    best_answer = ""
    context_token_idxs = [
        i for i, sid in enumerate(sequence_ids)
        if sid == 1 and offset_mapping[i] is not None
    ]

    for i in context_token_idxs:
        for j in range(i, min(i + MAX_ANSWER_LEN, len(end_logits))):
            if j >= len(sequence_ids) or sequence_ids[j] != 1:
                continue
            if offset_mapping[i] is None or offset_mapping[j] is None:
                continue
            if offset_mapping[i][0] == offset_mapping[i][1]:
                continue
            if offset_mapping[j][0] == offset_mapping[j][1]:
                continue

            score = float(start_logits[i].item() + end_logits[j].item())
            if score > best_score:
                start_char = int(offset_mapping[i][0])
                end_char = int(offset_mapping[j][1])
                answer = context[start_char:end_char].strip()
                if answer:
                    best_score = score
                    best_answer = answer
    return best_answer, best_score


def keyword_filter(question: str, candidates, all_passages):
    keywords = [w.lower() for w in question.split() if len(w) > 4]
    if not keywords:
        return candidates

    rare_terms = [k for k in keywords if len(k) >= RARE_TERM_MIN_LEN]
    must_terms = [
        t for t in rare_terms
        if "-" in t or any(ch.isdigit() for ch in t)
    ]
    if not must_terms and rare_terms:
        must_terms = [rare_terms[0]]

    filtered = []
    for p in candidates:
        text = p["text"].lower()
        entities = " ".join(p.get("entities", [])).lower()
        keyword_hits = sum(k in text or k in entities for k in keywords)
        if keyword_hits < MIN_KEYWORD_HITS:
            continue
        if must_terms and not any(t in text or t in entities for t in must_terms):
            continue
        filtered.append((keyword_hits, p))

    if filtered:
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in filtered]

    if must_terms:
        lexical = []
        for p in all_passages:
            text = p["text"].lower()
            entities = " ".join(p.get("entities", [])).lower()
            if not any(t in text or t in entities for t in must_terms):
                continue
            keyword_hits = sum(k in text or k in entities for k in keywords)
            if keyword_hits > 0:
                lexical.append((keyword_hits, p))
        if lexical:
            lexical.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in lexical[:TOP_K_DENSE]]

    fallback = []
    for p in candidates:
        text = p["text"].lower()
        entities = " ".join(p.get("entities", [])).lower()
        keyword_hits = sum(k in text or k in entities for k in keywords)
        if keyword_hits > 0:
            fallback.append((keyword_hits, p))
    if fallback:
        fallback.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in fallback]
    return candidates


print("Loading passages...")
with open(PASSAGES_FILE, encoding="utf-8") as f:
    passages = json.load(f)
passage_id_set = {p["passage_id"] for p in passages}

print("Loading evaluation set...")
with open(EVAL_FILE, encoding="utf-8") as f:
    eval_items: List[Dict] = json.load(f)

print("Validating evaluation IDs against corpus...")
total_missing = 0
for ex in eval_items:
    rel_ids = ex.get("relevant_passage_ids", [])
    valid_ids = [pid for pid in rel_ids if pid in passage_id_set]
    missing_ids = [pid for pid in rel_ids if pid not in passage_id_set]
    if missing_ids:
        total_missing += len(missing_ids)
        q_preview = ex.get("question", "")[:60]
        print(
            f"Warning: {len(missing_ids)} missing relevant_passage_ids "
            f"for question: {q_preview}"
        )
    ex["relevant_passage_ids"] = valid_ids

if total_missing > 0:
    print(f"Total missing relevant_passage_ids filtered out: {total_missing}")

print("Loading FAISS index...")
if not faiss.read_index:
    raise RuntimeError("FAISS unavailable")
index = faiss.read_index(INDEX_FILE)
faiss.omp_set_num_threads(1)

print("Loading models...")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_ENCODER)
q_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_ENCODER).to(DEVICE)
q_encoder.eval()

reranker = CrossEncoder(RERANK_MODEL)

reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, use_fast=True)
reader_model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL).to(DEVICE)
reader_model.eval()

recall_hits = {k: 0 for k in TOP_K_VALUES}
mrr_total = 0.0
em_total = 0.0
f1_total = 0.0
rerank_hit_total = 0
no_confident_retrieval = 0

for ex in eval_items:
    question = ex["question"]
    relevant_pids = set(ex.get("relevant_passage_ids", []))
    gold_answers = ex.get("answers", [])

    q_inputs = q_tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    q_inputs = {k: v.to(DEVICE) for k, v in q_inputs.items()}
    with torch.inference_mode():
        q_outputs = q_encoder(**q_inputs)
        q_emb = q_outputs.pooler_output.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_emb)

    _, dense_indices = index.search(q_emb, TOP_K_DENSE)
    dense_candidates = [passages[i] for i in dense_indices[0]]
    dense_candidates = keyword_filter(question, dense_candidates, passages)
    dense_ranked_ids = [p["passage_id"] for p in dense_candidates]

    pairs = [(question, p["text"][:512]) for p in dense_candidates]
    rr_scores = reranker.predict(pairs)
    reranked = [p for p, _ in sorted(zip(dense_candidates, rr_scores), key=lambda x: x[1], reverse=True)]
    reranked = reranked[:TOP_K_RERANK]
    top_rerank_score = float(max(rr_scores)) if len(rr_scores) > 0 else float("-inf")
    if top_rerank_score < MIN_RERANK_SCORE:
        no_confident_retrieval += 1

    for k in TOP_K_VALUES:
        top_ids = dense_ranked_ids[:k]
        if relevant_pids.intersection(top_ids):
            recall_hits[k] += 1

    rr = 0.0
    for idx, pid in enumerate(dense_ranked_ids, start=1):
        if pid in relevant_pids:
            rr = 1.0 / idx
            break
    mrr_total += rr
    reranked_ids = [p["passage_id"] for p in reranked]
    if relevant_pids.intersection(reranked_ids):
        rerank_hit_total += 1

    best_pred = ""
    best_score = float("-inf")
    for p in reranked:
        pred, score = read_answer(reader_tokenizer, reader_model, question, p["text"])
        if score > best_score:
            best_score = score
            best_pred = pred

    em_total += exact_match(best_pred, gold_answers) if gold_answers else 0.0
    f1_total += token_f1(best_pred, gold_answers) if gold_answers else 0.0

n = len(eval_items)
metrics = {
    **{f"Recall@{k}": recall_hits[k] / n if n else 0.0 for k in TOP_K_VALUES},
    "MRR": mrr_total / n if n else 0.0,
    "ExactMatch": em_total / n if n else 0.0,
    "F1": f1_total / n if n else 0.0,
    "RerankHit@5": rerank_hit_total / n if n else 0.0,
    "NoConfidentRetrievalRate": no_confident_retrieval / n if n else 0.0,
    "num_examples": n,
}

print("\nEvaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
