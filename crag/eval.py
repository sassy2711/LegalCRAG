# evaluate.py

import json
from retriever import retrieve
from pipeline import crag_pipeline


def extract_citations(results):
    return [r["citation"] for r in results]


def compute_metrics(ranked, gold):
    hit = 0
    rr = 0.0

    for i, c in enumerate(ranked):
        if c in gold:
            hit = 1
            rr = 1.0 / (i + 1)
            break

    acc = 1 if ranked and ranked[0] in gold else 0

    return acc, hit, rr


def evaluate(dataset, k=5,
             log_file="results.jsonl",
             summary_file="metrics_summary.json"):

    rag_acc = rag_hit = rag_mrr = 0
    crag_acc = crag_hit = crag_mrr = 0

    n = len(dataset)

    with open(log_file, "w") as fout:

        for item in dataset:
            query = item["query"]
            gold = item["gold_citations"]

            # -------------------
            # RAG
            # -------------------
            rag_results = retrieve(query, k=k)
            rag_ranked = extract_citations(rag_results)

            rag_a, rag_h, rag_rr = compute_metrics(rag_ranked, gold)

            rag_acc += rag_a
            rag_hit += rag_h
            rag_mrr += rag_rr

            # -------------------
            # CRAG
            # -------------------
            _, crag_results = crag_pipeline(query, return_results=True, k=k)
            crag_ranked = extract_citations(crag_results)

            crag_a, crag_h, crag_rr = compute_metrics(crag_ranked, gold)

            crag_acc += crag_a
            crag_hit += crag_h
            crag_mrr += crag_rr

            # -------------------
            # PER-QUERY LOG
            # -------------------
            fout.write(json.dumps({
                "query": query,
                "gold": gold,

                "rag_ranked": rag_ranked,
                "crag_ranked": crag_ranked,

                "rag_metrics": {
                    "accuracy": rag_a,
                    "hit": rag_h,
                    "mrr": rag_rr
                },
                "crag_metrics": {
                    "accuracy": crag_a,
                    "hit": crag_h,
                    "mrr": crag_rr
                },

                "comparison": {
                    "acc_diff": crag_a - rag_a,
                    "hit_diff": crag_h - rag_h,
                    "mrr_diff": crag_rr - rag_rr
                }

            }) + "\n")

            fout.flush()  # important for live logging

    # -------------------
    # FINAL METRICS
    # -------------------
    results = {
        "RAG": {
            "Accuracy": rag_acc / n,
            f"Hit@{k}": rag_hit / n,
            "MRR": rag_mrr / n
        },
        "CRAG": {
            "Accuracy": crag_acc / n,
            f"Hit@{k}": crag_hit / n,
            "MRR": crag_mrr / n
        },
        "Improvement": {
            "Accuracy_gain": (crag_acc - rag_acc) / n,
            f"Hit@{k}_gain": (crag_hit - rag_hit) / n,
            "MRR_gain": (crag_mrr - rag_mrr) / n
        }
    }

    # -------------------
    # PRINT
    # -------------------
    print("\n===== RESULTS =====\n")

    print("RAG:")
    print(f"  Accuracy: {results['RAG']['Accuracy']:.3f}")
    print(f"  Hit@{k}:  {results['RAG'][f'Hit@{k}']:.3f}")
    print(f"  MRR:      {results['RAG']['MRR']:.3f}")

    print("\nCRAG:")
    print(f"  Accuracy: {results['CRAG']['Accuracy']:.3f}")
    print(f"  Hit@{k}:  {results['CRAG'][f'Hit@{k}']:.3f}")
    print(f"  MRR:      {results['CRAG']['MRR']:.3f}")

    print("\nImprovement:")
    print(f"  Accuracy gain: {results['Improvement']['Accuracy_gain']:.3f}")
    print(f"  Hit@{k} gain:  {results['Improvement'][f'Hit@{k}_gain']:.3f}")
    print(f"  MRR gain:      {results['Improvement']['MRR_gain']:.3f}")

    # -------------------
    # SAVE SUMMARY
    # -------------------
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    evaluate(dataset, k=5)