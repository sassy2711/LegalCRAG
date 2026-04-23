# evaluate_with_generator.py

import json
import re

from retriever import retrieve
from pipeline import crag_pipeline
from generator import generate_answer


# -----------------------------
# Extract citation from answer
# -----------------------------
def extract_answer_citation(answer):
    """
    Extracts citation like:
    - Section 415 IPC
    - Article 21 COI
    """
    match = re.search(r"(Section\s+\d+[A-Za-z]*\s+\w+|Article\s+\d+)", answer)
    return match.group(1) if match else None


# -----------------------------
# Compute accuracy (answer-level)
# -----------------------------
def compute_accuracy(pred, gold):
    if pred is None:
        return 0
    return 1 if pred in gold else 0


# -----------------------------
# Main evaluation
# -----------------------------
def evaluate(dataset,
             k=5,
             log_file="results_with_generator.jsonl",
             summary_file="metrics_with_generator.json"):

    rag_correct = 0
    crag_correct = 0

    n = len(dataset)

    with open(log_file, "w") as fout:

        for item in dataset:
            query = item["query"]
            gold = item["gold_citations"]

            # -------------------
            # RAG (with generator)
            # -------------------
            rag_results = retrieve(query, k=k)
            rag_answer = generate_answer(query, rag_results)
            rag_pred = extract_answer_citation(rag_answer)

            rag_acc = compute_accuracy(rag_pred, gold)
            rag_correct += rag_acc

            # -------------------
            # CRAG (with generator)
            # -------------------
            crag_answer = crag_pipeline(query)
            crag_pred = extract_answer_citation(crag_answer)

            crag_acc = compute_accuracy(crag_pred, gold)
            crag_correct += crag_acc

            # -------------------
            # LOG
            # -------------------
            fout.write(json.dumps({
                "query": query,
                "gold": gold,

                "rag_pred": rag_pred,
                "crag_pred": crag_pred,

                "rag_correct": rag_acc,
                "crag_correct": crag_acc,

                "comparison": {
                    "gain": crag_acc - rag_acc  # +1, 0, -1
                }
            }) + "\n")

            fout.flush()

    # -------------------
    # FINAL METRICS
    # -------------------
    results = {
        "RAG": {
            "Accuracy": rag_correct / n
        },
        "CRAG": {
            "Accuracy": crag_correct / n
        },
        "Improvement": {
            "Accuracy_gain": (crag_correct - rag_correct) / n
        }
    }

    # -------------------
    # PRINT
    # -------------------
    print("\n===== END-TO-END RESULTS =====\n")

    print("RAG (with generator):")
    print(f"  Accuracy: {results['RAG']['Accuracy']:.3f}")

    print("\nCRAG (with generator):")
    print(f"  Accuracy: {results['CRAG']['Accuracy']:.3f}")

    print("\nImprovement:")
    print(f"  Accuracy gain: {results['Improvement']['Accuracy_gain']:.3f}")

    # -------------------
    # SAVE
    # -------------------
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    evaluate(dataset, k=5)