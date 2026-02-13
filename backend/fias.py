import pickle, faiss, numpy as np

# Load index
index = faiss.read_index("karnataka_index.faiss")
print(index.ntotal)  # should print >0

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(len(chunks))  # should match ntotal

# Test a dummy search
vec = np.random.rand(1, 384).astype("float32")
d, i = index.search(vec, 4)
print(d, i)
