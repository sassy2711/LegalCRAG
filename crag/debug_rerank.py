# debug_rerank.py
import json
import faiss
from retriever import (
    normalize_query, is_definition_query,
    build_definition_probe, rerank, extract_concept_terms,
    embed_model, docs, index
)

query = "what is evidence?"
query_norm = normalize_query(query)

# Primary fetch
q_vec = embed_model.encode([query_norm], convert_to_numpy=True)
faiss.normalize_L2(q_vec)
sims, indices = index.search(q_vec, 75)

seen_ids = set(indices[0].tolist())
candidates = [docs[i] for i in indices[0]]

# Check if Section 3 IEA is in primary
s3 = next((d for d in candidates if d['citation'] == 'Section 3 IEA'), None)
print(f"Section 3 IEA in primary 75? {s3 is not None}")
if s3:
    print(f"  title: {s3['title']}")
    print(f"  text[:100]: {s3['text'][:100]}")

# Check concept terms
terms = extract_concept_terms(query_norm)
print(f"\nConcept terms: {terms}")

# Check if 'evidence' appears in Section 3 text
if s3:
    print(f"'evidence' in Section 3 text? {'evidence' in s3['text'].lower()}")

# Rerank
reranked = rerank(query_norm, candidates)
print(f"\nTop-10 after rerank:")
for i, r in enumerate(reranked[:10]):
    print(f"  {i+1}. {r['citation']} — {r['title']}")