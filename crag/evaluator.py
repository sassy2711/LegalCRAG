# # evaluator.py

# from retriever import embed_model
# import faiss
# import numpy as np

# CONFIDENCE_THRESHOLD = 0.35  # cosine similarity cutoff

# NOISE_PHRASES = [
#     "state amendments",
#     "letter of request",
#     "contracting state",
#     "mutual legal assistance",
# ]


# def compute_similarity(query, text):
#     vecs = embed_model.encode([query, text], convert_to_numpy=True)
#     faiss.normalize_L2(vecs)
#     return float(np.dot(vecs[0], vecs[1]))


# def has_noise(text):
#     t = text.lower()
#     return any(phrase in t for phrase in NOISE_PHRASES)


# def evaluate(query, results):
#     if not results:
#         return False, -1.0, "no_results"

#     top = results[0]
#     text = top.get("text", "")

#     # Hard reject noisy docs
#     if has_noise(text):
#         return False, 0.0, "noise_detected"

#     # Semantic similarity between query and top result
#     score = compute_similarity(query, text)

#     if score < CONFIDENCE_THRESHOLD:
#         return False, score, "low_confidence"

#     return True, score, "ok"

# evaluator.py

from retriever import reranker
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------

CONFIDENCE_THRESHOLD = 0.5   # cross-encoder score threshold
TOP_K_EVAL = 3              # how many top docs to evaluate

NOISE_PHRASES = [
    "state amendments",
    "letter of request",
    "contracting state",
    "mutual legal assistance",
]


# ----------------------------
# NOISE CHECK
# ----------------------------

def has_noise(text):
    t = text.lower()
    return any(phrase in t for phrase in NOISE_PHRASES)


# ----------------------------
# CROSS-ENCODER SCORING
# ----------------------------

def compute_cross_scores(query, results):
    """
    Returns cross-encoder scores for top-k documents
    """
    pairs = []

    for d in results:
        citation = d.get("citation", "")
        title = d.get("title", "")
        text = d.get("text", "")[:300]

        doc_text = f"{citation}. {title}. {text}"
        pairs.append((query, doc_text))

    scores = reranker.predict(pairs)
    return scores


# ----------------------------
# EVALUATE
# ----------------------------

def evaluate(query, results):
    if not results:
        return False, -1.0, "no_results"

    # Limit to top-k
    top_results = results[:TOP_K_EVAL]

    # ----------------------------
    # HARD NOISE FILTER
    # ----------------------------
    for r in top_results:
        if has_noise(r.get("text", "")):
            return False, 0.0, "noise_detected"

    # ----------------------------
    # CROSS-ENCODER SCORING
    # ----------------------------
    scores = compute_cross_scores(query, top_results)

    best_score = float(np.max(scores))
    avg_score = float(np.mean(scores))

    # ----------------------------
    # DECISION LOGIC
    # ----------------------------

    # If best doc is strong → accept
    if best_score >= CONFIDENCE_THRESHOLD:
        return True, best_score, "ok"

    # If average is decent but no strong hit → weak accept (optional)
    if avg_score >= CONFIDENCE_THRESHOLD * 0.8:
        return False, avg_score, "weak_signal"

    return False, best_score, "low_confidence"