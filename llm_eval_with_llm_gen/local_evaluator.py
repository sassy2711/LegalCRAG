import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def judge(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def score_groundedness(query, docs, answer):
    context = "\n\n".join([d["text"][:500] for d in docs])

    prompt = f"""
Query: {query}

Context:
{context}

Answer:
{answer}

Score groundedness from 0 to 1.

Only output a number.
"""

    try:
        return float(judge(prompt).strip())
    except:
        return 0.0


def score_relevance(query, answer):
    prompt = f"""
Query: {query}

Answer:
{answer}

Score relevance from 0 to 1.

Only output a number.
"""
    try:
        return float(judge(prompt).strip())
    except:
        return 0.0