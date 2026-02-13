import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")  # folder where txt files live
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 5000
OVERLAP = 1000
INDEX_FILE = "karnataka_index.faiss"
CHUNKS_FILE = "chunks.pkl"

# === CHUNK FUNCTION ===
def chunk_text(text, chunk_size=5000, overlap=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# === LOAD MODEL ===
print("🔹 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

chunks = []
metas = []

# === READ ALL .TXT FILES ===
print("🔹 Reading text files...")
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        city = os.path.splitext(file)[0].lower()
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        file_chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        chunks.extend(file_chunks)
        metas.extend([{"city": city}] * len(file_chunks))
        print(f"✅ {file} → {len(file_chunks)} chunks")

print(f"Total chunks collected: {len(chunks)}")

# === CREATE EMBEDDINGS ===
print("🔹 Creating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype("float32")

# === BUILD FAISS INDEX ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors")

# === SAVE FILES ===
print("🔹 Saving files...")
faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump({"chunks": chunks, "meta": metas}, f)

print("🎉 Done! Both FAISS and pickle rebuilt successfully.")
