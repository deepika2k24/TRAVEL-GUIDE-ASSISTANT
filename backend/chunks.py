import faiss, pickle

faiss_index = faiss.read_index("karnataka_index.faiss")
with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
metas = data["meta"]

print("FAISS vectors:", faiss_index.ntotal)
print("Chunks:", len(chunks))
print("Metas:", len(metas))
