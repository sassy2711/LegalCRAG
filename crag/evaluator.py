# evaluator.py

from retriever import embed_model
import faiss
import numpy as np

CONFIDENCE_THRESHOLD = 0.35  # cosine similarity cutoff

NOISE_PHRASES = [
    "state amendments",
    "letter of request",
    "contracting state",
    "mutual legal assistance",
]


def compute_similarity(query, text):
    vecs = embed_model.encode([query, text], convert_to_numpy=True)
    faiss.normalize_L2(vecs)
    return float(np.dot(vecs[0], vecs[1]))


def has_noise(text):
    t = text.lower()
    return any(phrase in t for phrase in NOISE_PHRASES)


def evaluate(query, results):
    if not results:
        return False, -1.0, "no_results"

    top = results[0]
    text = top.get("text", "")

    # Hard reject noisy docs
    if has_noise(text):
        return False, 0.0, "noise_detected"

    # Semantic similarity between query and top result
    score = compute_similarity(query, text)

    if score < CONFIDENCE_THRESHOLD:
        return False, score, "low_confidence"

    return True, score, "ok"