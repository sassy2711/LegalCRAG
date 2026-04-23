# corrector.py

import re

ABBREVIATIONS = {
    "ipc": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure",
    "iea": "Indian Evidence Act",
    "fir": "first information report",
    "pil": "public interest litigation",
    "hc":  "high court",
    "sc":  "supreme court",
}

# Intent-based rewrite templates
# If query matches pattern → append context hint
INTENT_HINTS = [
    # FIR / police complaint
    (r"\bfir\b|first information report", "filing complaint police cognizable offence section 154"),
    # Bail
    (r"\bbail\b", "bail application bailable non-bailable offence"),
    # Evidence definition
    (r"\bevidence\b", "definition evidence means facts documents witnesses"),
    # Fundamental rights
    (r"\bfundamental rights?\b", "fundamental rights constitution of india part III"),
    # Anticipatory bail
    (r"\banticipatory bail\b", "anticipatory bail section 438 CRPC"),
    # Culpable homicide
    (r"\bculpable homicide\b", "culpable homicide murder IPC section 299 300"),
]


def expand_abbreviations(query):
    words = query.split()
    expanded = []
    for w in words:
        expanded.append(ABBREVIATIONS.get(w.lower(), w))
    return " ".join(expanded)


def normalize_query(query):
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def apply_intent_hints(query):
    q_lower = query.lower()
    for pattern, hint in INTENT_HINTS:
        if re.search(pattern, q_lower):
            # Only append if hint words not already present
            new_words = [w for w in hint.split() if w.lower() not in q_lower]
            if new_words:
                return query + " " + " ".join(new_words)
    return query


def rewrite_query(query, reason):
    q = normalize_query(query)
    q = expand_abbreviations(q)

    # Always add constitution context for article queries
    if re.search(r"\barticle\s+\d+\b", q.lower()) and "constitution" not in q.lower():
        q = q + " constitution of india"

    # For low_confidence / noise, apply intent hints
    if reason in ("low_confidence", "noise_detected", "refine"):
        q = apply_intent_hints(q)

    return q


def correct_query(query, reason):
    return rewrite_query(query, reason)