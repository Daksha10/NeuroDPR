import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import os
import time

# ================= CONFIG =================
INPUT_FILE = "pubmed/processed/pubmed_passages.json"
EMB_DIR = "embeddings"
PART_DIR = f"{EMB_DIR}/parts"

FINAL_EMB = f"{EMB_DIR}/passage_embeddings.npy"
FINAL_IDS = f"{EMB_DIR}/passage_ids.json"

BATCH_SIZE = 16          # ⬅️ safer for Mac
MAX_LEN = 256
SAVE_EVERY = 50          # batches
# =========================================

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(PART_DIR, exist_ok=True)

print("🔄 Loading DPR Context Encoder...")
tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
model = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
model.eval()
print("✅ Model loaded")

with open(INPUT_FILE) as f:
    passages = json.load(f)

texts = [p["text"] for p in passages]
passage_ids = [p["passage_id"] for p in passages]

# ---------- RESUME LOGIC ----------
existing_parts = sorted([
    f for f in os.listdir(PART_DIR) if f.endswith(".npy")
])

start_batch = len(existing_parts) * SAVE_EVERY
print(f"🔁 Resuming from batch {start_batch}")

# ---------- ENCODING ----------
batch_embeddings = []
start_time = time.time()

with torch.inference_mode():
    for i in tqdm(
        range(start_batch * BATCH_SIZE, len(texts), BATCH_SIZE),
        desc="Encoding"
    ):
        batch_texts = texts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        emb = model(**inputs).pooler_output.cpu().numpy()
        batch_embeddings.append(emb)

        batch_idx = (i // BATCH_SIZE) + 1

        if batch_idx % SAVE_EVERY == 0:
            part_id = batch_idx // SAVE_EVERY
            part_path = f"{PART_DIR}/emb_part_{part_id}.npy"
            np.save(part_path, np.vstack(batch_embeddings))
            batch_embeddings.clear()

            elapsed = (time.time() - start_time) / 60
            print(f"💾 Saved part {part_id} | {elapsed:.1f} min elapsed")

# Save remaining
if batch_embeddings:
    part_id += 1
    np.save(f"{PART_DIR}/emb_part_{part_id}.npy", np.vstack(batch_embeddings))

print("✅ All parts encoded")

# ---------- MERGE ----------
print("🔗 Merging embeddings...")
parts = sorted(os.listdir(PART_DIR))
all_embeddings = np.vstack([
    np.load(f"{PART_DIR}/{p}") for p in parts
])

np.save(FINAL_EMB, all_embeddings)
with open(FINAL_IDS, "w") as f:
    json.dump(passage_ids, f)

print(f"🎉 DONE: embeddings shape = {all_embeddings.shape}")
