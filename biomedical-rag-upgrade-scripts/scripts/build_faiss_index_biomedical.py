import faiss
import numpy as np
import os

# ================= CONFIG =================
EMB_FILE = "embeddings_biomed/passage_embeddings.npy"
INDEX_DIR = "index"
INDEX_FILE = f"{INDEX_DIR}/faiss_biomed.index"
# =========================================

# 1. Ensure the index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# 2. Load the combined embeddings (vectors) created in the previous step
print("Loading embeddings...")
embeddings = np.load(EMB_FILE).astype("float32")

# 3. Normalize the vectors to Unit Length
# This makes Unit Product (IP) search equivalent to Cosine Similarity
print("Normalizing embeddings...")
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
print(f"Embedding dimension: {dim}")

# 4. Create the FAISS index
# IndexFlatIP uses Inner Product (fastest for small/medium datasets)
print("Building FAISS index...")
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# 5. Save the finished index to disk for use in retrieval
faiss.write_index(index, INDEX_FILE)
print(f"DONE: FAISS index built with {index.ntotal} vectors -> {INDEX_FILE}")
