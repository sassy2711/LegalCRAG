# generator.py

import re

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

SUB_DEFINITION_SIGNALS = [
    "by personation",
    "primary evidence",
    "secondary evidence",
    "circumstantial",
    "aggravated",
    "attempt to",
    "abetment of",
    "enhanced",
    "special",
]

# Titles that signal a section is the canonical home of definitions
# for an entire Act — scored highest in canonicality
INTERPRETATION_TITLES = [
    "interpretation clause",
    "interpretation",
    "definitions",
    "definition",
    "general definitions",
    "words and expressions",
]


def extract_number(query, keyword):
    match = re.search(rf"{keyword}\s+(\d+[a-zA-Z]?)", query.lower())
    return match.group(1) if match else None


def is_definition_text(text):
    t = text.lower()
    return (
        re.match(r"^['\"]?\w[\w\s,]+['\"]?\s+(means|includes|is said to)", t) is not None
        or "means" in t[:80]
        or "is said to" in t[:80]
        or "includes" in t[:80]
        # ----------------------------
        # FIX 1: Catches interpretation sections like Section 3 IEA
        # whose text opens with "In this Act the following words and
        # expressions are used in the following senses..."
        # These are definition sections but don't start with "means"
        # ----------------------------
        or "following words and expressions" in t[:120]
        or "following words" in t[:80]
    )


def is_sub_definition(doc):
    title = doc.get("title", "").lower()
    return any(sig in title for sig in SUB_DEFINITION_SIGNALS)


def is_interpretation_section(doc):
    """Returns True if this doc is an interpretation/definitions section."""
    title = doc.get("title", "").lower()
    return any(t in title for t in INTERPRETATION_TITLES)


def definition_canonicality_score(doc, query_terms):
    score = 0.0
    title = doc.get("title", "").lower()
    text = doc.get("text", "").lower()
    section = doc.get("section")

    # Penalize sub-definitions hard
    if is_sub_definition(doc):
        score -= 3.0

    # ----------------------------
    # FIX 2: Strongly reward interpretation/definition sections
    # when the queried concept appears in their text.
    # These are the canonical source of legal definitions in each Act.
    # e.g. Section 3 IEA defines Evidence, Court, Facts, Document etc.
    # ----------------------------
    if is_interpretation_section(doc):
        for term in query_terms:
            if term in text:
                score += 5.0  # strong reward — this is the canonical definition
                break  # one match is enough

    # Reward if query terms appear in title (title match = canonical)
    for term in query_terms:
        if term in title:
            score += 2.0

    # Reward if text opens with the concept name being defined
    for term in query_terms:
        if text[:50].startswith(term) or f'"{term}"' in text[:60]:
            score += 2.0

    # Slight reward for lower section numbers
    if section:
        try:
            sec_num = int(re.sub(r"[a-zA-Z]", "", str(section)))
            score -= sec_num * 0.001
        except ValueError:
            pass

    return score


def infer_source(query):
    q = query.lower()
    scores = {src: 0 for src in SOURCE_KEYWORDS}
    for src, keywords in SOURCE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[src] += 1
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else None


def extract_query_terms(query):
    q = query.lower()
    q = re.sub(r"\b(what is|define|meaning of|definition of|under|the|a|an)\b", "", q)
    q = re.sub(r"\b(ipc|crpc|cpc|iea|coi|constitution of india)\b", "", q)
    q = re.sub(r"\?", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return [t for t in q.split() if len(t) > 2]


def select_best_doc(query, results):
    q = query.lower()

    target_section = extract_number(q, "section")
    target_article = extract_number(q, "article")
    inferred_source = infer_source(q)
    is_def_query = any(p in q for p in ["what is", "define", "meaning of", "definition"])
    query_terms = extract_query_terms(query)

    # Step 1: Hard structural constraint
    if target_section:
        for r in results:
            if f"section {target_section}" in r.get("citation", "").lower():
                return r

    if target_article:
        for r in results:
            if f"article {target_article}" in r.get("citation", "").lower():
                return r

    # Step 2: Canonical definition selection
    if is_def_query:
        candidates = []

        for r in results:
            if r.get("source") == inferred_source and is_definition_text(r["text"]):
                candidates.append(r)

        if not candidates:
            for r in results:
                if is_definition_text(r["text"]):
                    candidates.append(r)

        if candidates:
            ranked = sorted(
                candidates,
                key=lambda d: definition_canonicality_score(d, query_terms),
                reverse=True
            )
            best = ranked[0]

            if definition_canonicality_score(best, query_terms) > 0:
                return best

    # Step 3: Source-filtered fallback
    if inferred_source:
        for r in results:
            if r.get("source") == inferred_source:
                return r

    # Step 4: Trust reranker
    return results[0]


def generate_answer(query, results):
    if not results:
        return "No relevant legal provision found."

    best = select_best_doc(query, results)

    seen = set()
    supporting = []
    for r in results:
        c = r["citation"]
        if c not in seen and c != best["citation"]:
            supporting.append(c)
            seen.add(c)
        if len(supporting) == 3:
            break

    answer = f"""Answer:

{best['citation']} — {best.get('title', '')}
{best['text']}

---
Supporting Sections:"""

    for s in supporting:
        answer += f"\n- {s}"

    return answer