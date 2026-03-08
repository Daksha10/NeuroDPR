import json
import numpy as np
import torch
import os
import time
from tqdm import tqdm
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder

# ================= CONFIG =================
INPUT_FILE = "pubmed/processed/pubmed_passages_enhanced.json"
EMB_DIR = "embeddings_biomed"
PART_DIR = f"{EMB_DIR}/parts"

FINAL_EMB = f"{EMB_DIR}/passage_embeddings.npy"
FINAL_IDS = f"{EMB_DIR}/passage_ids.json"

MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
BATCH_SIZE = 16
MAX_LEN = 256
SAVE_EVERY = 50
DEVICE = "cpu"
# =========================================

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(PART_DIR, exist_ok=True)

print(f"Loading biomedical encoder: {MODEL_NAME}")
tokenizer = DPRContextEncoderTokenizer.from_pretrained(MODEL_NAME)
model = DPRContextEncoder.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded")

with open(INPUT_FILE, encoding="utf-8") as f:
    passages = json.load(f)

texts = []
passage_ids = []
for p in passages:
    entity_suffix = ""
    if p.get("entities"):
        entity_suffix = " Entities: " + ", ".join(p["entities"])
    texts.append((p["text"] + entity_suffix).strip())
    passage_ids.append(p["passage_id"])


existing_parts = sorted([f for f in os.listdir(PART_DIR) if f.endswith(".npy")])
start_batch = len(existing_parts) * SAVE_EVERY
print(f"Resuming from batch {start_batch}")

batch_embeddings = []
part_id = len(existing_parts)
start_time = time.time()

with torch.inference_mode():
    for i in tqdm(range(start_batch * BATCH_SIZE, len(texts), BATCH_SIZE), desc="Encoding"):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = model(**inputs)
        emb = outputs.pooler_output.cpu().numpy().astype("float32")
        batch_embeddings.append(emb)

        batch_idx = (i // BATCH_SIZE) + 1
        if batch_idx % SAVE_EVERY == 0:
            part_id += 1
            part_path = f"{PART_DIR}/emb_part_{part_id}.npy"
            np.save(part_path, np.vstack(batch_embeddings))
            batch_embeddings.clear()
            elapsed = (time.time() - start_time) / 60
            print(f"Saved part {part_id} | {elapsed:.1f} min elapsed")

if batch_embeddings:
    part_id += 1
    np.save(f"{PART_DIR}/emb_part_{part_id}.npy", np.vstack(batch_embeddings))

print("All parts encoded")

print("Merging embeddings...")
parts = sorted([p for p in os.listdir(PART_DIR) if p.endswith(".npy")])
all_embeddings = np.vstack([np.load(f"{PART_DIR}/{p}") for p in parts]).astype("float32")

np.save(FINAL_EMB, all_embeddings)
with open(FINAL_IDS, "w", encoding="utf-8") as f:
    json.dump(passage_ids, f)

print(f"DONE: embeddings shape = {all_embeddings.shape}")
