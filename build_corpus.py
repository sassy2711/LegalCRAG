import json
import os

OUTPUT_PATH = "corpus.jsonl"

# ----------------------------
# STATUTE FILES (EDITABLE)
# ----------------------------
STATUTE_FILES = {
    "IPC": "Indian-Law-Penal-Code-Json/ipc.json",
    "CRPC": "Indian-Law-Penal-Code-Json/crpc.json",
    "IEA": "Indian-Law-Penal-Code-Json/iea.json",
    "CPC": "Indian-Law-Penal-Code-Json/cpc.json",  
}

COI_PATH = "The_Constitution_Of_India/COI.json"


# ----------------------------
# GENERIC STATUTE PROCESSOR
# ----------------------------
def process_statute(file_path, source_name):
    docs = []

    with open(file_path) as f:
        data = json.load(f)

    for item in data:
        # ----------------------------
        # HANDLE KEY VARIATIONS
        # ----------------------------
        section = (
            item.get("Section") or
            item.get("section")
        )

        title = (
            item.get("section_title") or
            item.get("title") or
            ""
        )

        text = (
            item.get("section_desc") or
            item.get("description") or
            ""
        )

        # ----------------------------
        # CLEAN CPC NOISE
        # ----------------------------
        if source_name == "CPC" and "STATE AMENDMENTS" in text:
            text = text.split("STATE AMENDMENTS")[0]

        # Optional: remove repealed junk
        if text.strip().startswith("Rep."):
            continue

        # Skip empty
        if not text or not section:
            continue

        docs.append({
            "doc_type": "statute",
            "source": source_name,
            "section": str(section),
            "article": None,
            "clause": None,
            "sub_clause": None,
            "title": title,
            "text": text,
            "citation": f"Section {section} {source_name}"
        })

    print(f"[INFO] {source_name}: {len(docs)} sections extracted")

    return docs


# ----------------------------
# CONSTITUTION PROCESSING
# ----------------------------
def process_constitution(coi_data):
    docs = []
    articles = coi_data[0]

    for art in articles:
        art_no = art.get("ArtNo")
        name = art.get("Name", "")

        # Skip omitted
        if "Status" in art and art["Status"] == "Omitted":
            continue

        # ------------------------
        # CASE 1: No clauses → use ArtDesc
        # ------------------------
        if "Clauses" not in art and "ArtDesc" in art:
            text = art["ArtDesc"]

            if "Omitted" in text:
                continue

            citation = "Preamble" if art_no == "0" else f"Article {art_no}"

            docs.append({
                "doc_type": "constitution",
                "source": "COI",
                "section": None,
                "article": art_no,
                "clause": None,
                "sub_clause": None,
                "title": name,
                "text": text,
                "citation": citation
            })

        # ------------------------
        # CASE 2: Clauses
        # ------------------------
        if "Clauses" in art:
            for clause in art["Clauses"]:
                clause_no = clause.get("ClauseNo")

                # If subclauses exist → skip clause-level
                if "SubClauses" in clause:
                    for sub in clause["SubClauses"]:
                        sub_no = sub.get("SubClauseNo")
                        sub_text = sub.get("SubClauseDesc", "")

                        if sub_text:
                            docs.append({
                                "doc_type": "constitution",
                                "source": "COI",
                                "section": None,
                                "article": art_no,
                                "clause": clause_no,
                                "sub_clause": sub_no,
                                "title": name,
                                "text": sub_text,
                                "citation": f"Article {art_no}({clause_no})({sub_no})"
                            })

                else:
                    clause_text = clause.get("ClauseDesc", "")

                    if clause_text:
                        docs.append({
                            "doc_type": "constitution",
                            "source": "COI",
                            "section": None,
                            "article": art_no,
                            "clause": clause_no,
                            "sub_clause": None,
                            "title": name,
                            "text": clause_text,
                            "citation": f"Article {art_no}({clause_no})"
                        })

    return docs


# ----------------------------
# MAIN
# ----------------------------
def main():
    all_docs = []

    print("[INFO] Processing statutes...")

    for source, path in STATUTE_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        print(f"  → {source}")
        docs = process_statute(path, source)
        all_docs.extend(docs)

    print("[INFO] Processing Constitution...")

    with open(COI_PATH) as f:
        coi_data = json.load(f)

    coi_docs = process_constitution(coi_data)
    all_docs.extend(coi_docs)

    print(f"[INFO] Total documents: {len(all_docs)}")

    print("[INFO] Writing corpus.jsonl...")

    with open(OUTPUT_PATH, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("[DONE] corpus.jsonl created")


if __name__ == "__main__":
    main()