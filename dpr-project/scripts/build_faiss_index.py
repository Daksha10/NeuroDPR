import faiss
import numpy as np
import os

# ================= CONFIG =================
EMB_FILE = "embeddings/passage_embeddings.npy"
INDEX_DIR = "index"
INDEX_FILE = f"{INDEX_DIR}/faiss.index"
# =========================================

os.makedirs(INDEX_DIR, exist_ok=True)

print("🔄 Loading embeddings...")
embeddings = np.load(EMB_FILE).astype("float32")

print("🔄 Normalizing embeddings...")
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
print(f"📐 Embedding dimension: {dim}")

print("🔧 Building FAISS index...")
index = faiss.IndexFlatIP(dim)   # Inner Product (cosine similarity)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

print(f"🎉 FAISS index built with {index.ntotal} vectors")
