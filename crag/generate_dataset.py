import json
import random

BASE = [
    ("cheating", ["Section 415 IPC"]),
    ("theft", ["Section 378 IPC"]),
    ("murder", ["Section 300 IPC"]),
    ("culpable homicide", ["Section 299 IPC"]),
    ("fir", ["Section 154 CRPC"]),
    ("arrest", ["Section 41 CRPC"]),
    ("bail", ["Section 436 CRPC", "Section 437 CRPC"]),
    ("anticipatory bail", ["Section 438 CRPC"]),
    ("evidence", ["Section 3 IEA"]),
    ("primary evidence", ["Section 62 IEA"]),
    ("secondary evidence", ["Section 63 IEA"]),
    ("confession", ["Section 24 IEA"]),
    ("admission", ["Section 17 IEA"]),
    ("article 21", ["Article 21 COI"]),
    ("article 14", ["Article 14 COI"]),
    ("article 19", ["Article 19 COI"]),
    ("robbery", ["Section 390 IPC"]),
    ("dacoity", ["Section 391 IPC"]),
    ("forgery", ["Section 463 IPC"]),
    ("defamation", ["Section 499 IPC"]),
    ("abetment", ["Section 107 IPC"])
]

TEMPLATES = [
    "What is {x}?",
    "Define {x}",
    "{x} meaning",
    "{x} law india",
    "explain {x}",
    "{x} ipc meaning",
    "{x} crpc meaning",
    "{x} iea meaning",
    "what is {x} in law",
    "definition of {x}",
    "{x} indian law explain",
    "{x} simple meaning",
    "legal meaning of {x}",
    "india law {x}",
    "what is {x} concept",
    "{x} meaning in indian law",
    "{x} definition india",
    "explain {x} in simple terms",
    "what is {x} legal definition",
    "{x} under indian law"
]

def generate_dataset(n=200):
    dataset = []

    while len(dataset) < n:
        concept, gold = random.choice(BASE)
        template = random.choice(TEMPLATES)

        query = template.format(x=concept)

        dataset.append({
            "query": query,
            "gold_citations": gold
        })

    return dataset


if __name__ == "__main__":
    data = generate_dataset(400)

    with open("eval_dataset.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data)} queries")