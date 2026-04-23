import json
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "corpus.jsonl"
INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.json"

print("[INFO] Loading corpus...")

docs = []
texts = []

with open(CORPUS_PATH) as f:
    for line in f:
        d = json.loads(line)

        # ✅ Balanced enrichment (no over-weighting)
        enriched_text = (
            f"{d['source']} "
            f"{d['citation']} "
            f"{d['title']} {d['title']} "
            f"{d['text']}"
        )

        docs.append(d)
        texts.append(enriched_text)

print(f"[INFO] Loaded {len(docs)} documents")

print("[INFO] Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("[INFO] Encoding texts...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("[INFO] Normalizing embeddings...")
faiss.normalize_L2(embeddings)

print("[INFO] Building FAISS index (cosine)...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("[INFO] Saving index...")
faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "w") as f:
    json.dump(docs, f)

print("[DONE] Index + docs saved")