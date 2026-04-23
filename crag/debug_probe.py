# debug_probe.py
import json, faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.json"

index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH) as f:
    docs = json.load(f)

embed_model = SentenceTransformer("all-mpnet-base-v2")

probe = '"evidence" means includes definition'
p_vec = embed_model.encode([probe], convert_to_numpy=True)
faiss.normalize_L2(p_vec)
sims, indices = index.search(p_vec, 20)

print("Probe fetch results:")
for sim, idx in zip(sims[0], indices[0]):
    d = docs[idx]
    print(f"  [{sim:.3f}] {d['citation']} — {d['title']}")