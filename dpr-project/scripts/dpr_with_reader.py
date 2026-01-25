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

# ================= CONFIG =================
PASSAGES_FILE = "pubmed/processed/pubmed_passages.json"
INDEX_FILE = "index/faiss.index"
TOP_K_RETRIEVAL = 20
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
)
q_encoder.to(DEVICE)
q_encoder.eval()


# ---------- BIOBERT READER ----------
print("🔄 Loading BioBERT Reader...")
reader_tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    use_fast=True              # VERY IMPORTANT
)
reader_model = AutoModelForQuestionAnswering.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)
reader_model.to(DEVICE)
reader_model.eval()


# ---------- RETRIEVAL ----------
def retrieve_passages(question: str):
    inputs = q_tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        q_emb = q_encoder(**inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, TOP_K_RETRIEVAL)

    return [passages[i] for i in indices[0]]


# ---------- READER ----------
def read_answer(question: str, context: str):
    inputs = reader_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True
    )

    # Separate offset mapping (DO NOT pass to model)
    offset_mapping = inputs.pop("offset_mapping")[0]
    input_ids = inputs["input_ids"][0]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = reader_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Identify where context tokens start
    sep_token_id = reader_tokenizer.sep_token_id
    sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

    # BERT format: [CLS] question [SEP] context [SEP]
    context_start = sep_indices[1] + 1

    best_score = float("-inf")
    best_answer = ""

    for start_idx in range(context_start, len(start_logits)):
        for end_idx in range(
            start_idx,
            min(start_idx + MAX_ANSWER_LEN, len(end_logits))
        ):
            if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                continue

            score = start_logits[start_idx] + end_logits[end_idx]

            if score > best_score:
                start_char = offset_mapping[start_idx][0]
                end_char = offset_mapping[end_idx][1]
                answer = context[start_char:end_char].strip()

                if answer:
                    best_score = score
                    best_answer = answer

    return best_answer, best_score


# ---------- FULL PIPELINE ----------
def answer_question(question: str):
    retrieved_passages = retrieve_passages(question)

    best_answer = ""
    best_score = float("-inf")
    best_context = ""

    for passage in retrieved_passages:
        answer, score = read_answer(question, passage["text"])

        if score > best_score and answer:
            best_answer = answer
            best_score = score
            best_context = passage["text"]

    return best_answer, best_context


# ================= RUN =================
if __name__ == "__main__":
    question = "What causes Parkinson's disease?"

    answer, context = answer_question(question)

    print("\n❓ Question:")
    print(question)

    print("\n✅ Answer:")
    print(answer if answer else "No confident answer found")

    print("\n📄 Source Passage:")
    print(context[:400], "...")
