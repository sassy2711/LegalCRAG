# debug_s3.py
import json

with open("docs.json") as f:
    docs = json.load(f)

# Find Section 3 IEA
for d in docs:
    if d.get("source") == "IEA" and d.get("section") in ["3", "2", "1", "4"]:
        print(f"\n--- {d['citation']} ---")
        print(f"Title: {d['title']}")
        print(f"Text[:200]: {d['text'][:200]}")
        print(f"doc_type: {d['doc_type']}")