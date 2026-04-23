import json
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.json"

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH) as f:
    docs = json.load(f)

embed_model = SentenceTransformer("all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ----------------------------
# NORMALIZE
# ----------------------------
def normalize_query(query):
    return re.sub(r"\s+", " ", query.lower().strip())


# ----------------------------
# EXTRACT NUMBERS
# ----------------------------
def extract_number(query, keyword):
    match = re.search(rf"{keyword}\s+(\d+[a-zA-Z]?)", query.lower())
    return match.group(1) if match else None


# ----------------------------
# SOURCE INFERENCE
# ----------------------------
SOURCE_KEYWORDS = {
    "IPC":  ["ipc", "indian penal code", "offence", "punishment", "cheating",
             "murder", "theft", "assault", "criminal", "culpable", "abetment"],
    "CRPC": ["crpc", "code of criminal procedure", "fir", "first information",
             "arrest", "bail", "trial", "magistrate", "cognizable", "challan"],
    "IEA":  ["iea", "indian evidence act", "evidence", "witness", "document",
             "proof", "admissible", "confession", "presumption"],
    "CPC":  ["cpc", "code of civil procedure", "civil", "suit", "decree",
             "plaintiff", "defendant", "jurisdiction", "execution"],
    "COI":  ["article", "constitution", "fundamental right", "directive",
             "parliament", "president", "amendment", "preamble"],
}

def infer_source(query):
    q = query.lower()
    scores = {src: 0 for src in SOURCE_KEYWORDS}
    for src, keywords in SOURCE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[src] += 1
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else None


# ----------------------------
# DEFINITION CHECKS
# ----------------------------
def is_definition_query(query):
    q = query.lower()
    return any(p in q for p in ["what is", "define", "meaning", "definition of"])


def is_definition_text(text):
    t = text.lower()
    return (
        re.match(r"^['\"]?\w[\w\s,]+['\"]?\s+(means|includes|is said to)", t) is not None
        or "means" in t[:80]
        or "is said to" in t[:80]
        or "includes" in t[:80]
    )


# ----------------------------
# DEFINITION PROBE QUERY
# ----------------------------
def build_definition_probe(query):
    q = query.lower()
    q = re.sub(r"\b(what is|define|meaning of|definition of|under|the|a|an)\b", "", q)
    q = re.sub(r"\b(ipc|crpc|cpc|iea|coi|constitution of india)\b", "", q)
    q = re.sub(r"[?]", "", q)
    q = re.sub(r"\s+", " ", q).strip()

    if not q:
        return None

    return f'"{q}" means includes definition'


# ----------------------------
# EXTRACT QUERY CONCEPT TERMS
# Strips question scaffolding and source names,
# returns core concept words for boost matching
# ----------------------------
def extract_concept_terms(query):
    q = query.lower()
    q = re.sub(
        r"\b(what is|define|meaning of|definition of|under|the|a|an"
        r"|ipc|crpc|cpc|iea|coi|constitution of india)\b",
        "", q
    )
    q = re.sub(r"[?]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return [w for w in q.split() if len(w) > 2]


# ----------------------------
# NOISE PHRASES
# ----------------------------
NOISE_PHRASES = [
    "state amendments",
    "letter of request",
    "contracting state",
    "mutual legal assistance",
    "shall be construed as a reference",
]

# Titles that signal a section is an interpretation/definitions section.
# Cross-encoder systematically underscores these because the title
# has no lexical overlap with the concept being queried.
INTERPRETATION_TITLES = [
    "interpretation clause",
    "interpretation",
    "definitions",
    "definition",
    "general definitions",
    "words and expressions",
]


# ----------------------------
# RERANK
# ----------------------------
def rerank(query, candidates):
    if not candidates:
        return []

    target_section = extract_number(query, "section")
    target_article = extract_number(query, "article")
    definition_query = is_definition_query(query)
    inferred_source = infer_source(query)
    concept_terms = extract_concept_terms(query) if definition_query else []

    pairs = []
    for d in candidates:
        citation = d.get('citation', '')
        title = d.get('title', '')
        text = d.get('text', '')[:300]
        doc_text = f"{citation}. {title}. {text}"
        pairs.append((query, doc_text))

    scores = reranker.predict(pairs)

    scored = []
    for doc, score in zip(candidates, scores):
        citation = doc.get("citation", "").lower()
        text = doc.get("text", "").lower()
        title = doc.get("title", "").lower()
        source = doc.get("source", "")

        # Exact section/article match
        if target_section and f"section {target_section}" in citation:
            score += 2.0

        if target_article and f"article {target_article}" in citation:
            score += 2.0

        # Source match
        if inferred_source and source == inferred_source:
            score += 0.5

        # Definition text boost
        if definition_query and is_definition_text(text):
            score += 1.0

        # ----------------------------
        # INTERPRETATION CLAUSE BOOST
        # Sections like "Interpretation clause" or "Definitions" are
        # the canonical home of legal definitions but their titles have
        # no lexical overlap with the queried concept, so the cross-encoder
        # scores them low. Boost them when the concept term appears in text.
        # ----------------------------
        if definition_query and any(t in title for t in INTERPRETATION_TITLES):
            if any(term in text for term in concept_terms):
                score = 7.0  # override, not additive

        # Noise penalty
        if any(phrase in text for phrase in NOISE_PHRASES):
            score -= 1.5

        # temporary - replace the existing S3 debug print with this
        if doc.get("citation") == "Section 3 IEA":
            print(f"[DEBUG S3] raw_cross_encoder_score={score - 2.5 - 0.5:.4f} | final_score={score:.4f}")

        scored.append((doc, score))

    # also add this at the end to see what rank 1 scored
    # right before ranked = sorted(...)

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    for d, s in ranked[:5]:
        print(f"[DEBUG TOP] {d['citation']} score={s:.4f}")

    return [r[0] for r in ranked]


# ----------------------------
# RETRIEVE
# ----------------------------
def retrieve(query, k=5):
    query_norm = normalize_query(query)
    definition_query = is_definition_query(query_norm)

    # Primary fetch
    q_vec = embed_model.encode([query_norm], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    sims, indices = index.search(q_vec, k * 15)

    seen_ids = set(indices[0].tolist())
    candidates = [docs[i] for i in indices[0]]

    # Definition probe fetch
    if definition_query:
        probe = build_definition_probe(query_norm)
        if probe:
            p_vec = embed_model.encode([probe], convert_to_numpy=True)
            faiss.normalize_L2(p_vec)
            _, probe_indices = index.search(p_vec, k * 5)

            for idx in probe_indices[0]:
                if idx not in seen_ids:
                    candidates.append(docs[idx])
                    seen_ids.add(idx)

    reranked = rerank(query_norm, candidates)
    return reranked[:k]