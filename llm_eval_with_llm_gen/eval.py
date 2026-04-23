# import json
# import re

# from retriever import retrieve
# from pipeline import crag_pipeline
# from generator import generate_answer_llm
# from local_evaluator import score_groundedness, score_relevance


# def extract_citation(answer):
#     match = re.search(r"(Section\s+\d+[A-Za-z]*\s+\w+|Article\s+\d+)", answer)
#     return match.group(1) if match else None


# def compute_accuracy(pred, gold):
#     if pred is None:
#         return 0
#     return 1 if pred in gold else 0


# def evaluate(dataset, k=5):
#     rag_acc = crag_acc = 0
#     rag_ground = crag_ground = 0
#     rag_rel = crag_rel = 0

#     for item in dataset:
#         query = item["query"]
#         gold = item["gold_citations"]

#         # -------------------
#         # RAG
#         # -------------------
#         rag_docs = retrieve(query, k=k)
#         rag_answer = generate_answer_llm(query, rag_docs)

#         rag_pred = extract_citation(rag_answer)
#         rag_acc += compute_accuracy(rag_pred, gold)
#         rag_ground += score_groundedness(query, rag_docs, rag_answer)
#         rag_rel += score_relevance(query, rag_answer)

#         # -------------------
#         # CRAG
#         # -------------------
#         crag_out = crag_pipeline(query, use_llm=True, return_context=True)

#         crag_answer = crag_out["answer"]
#         crag_docs = crag_out["docs"]

#         crag_pred = extract_citation(crag_answer)
#         crag_acc += compute_accuracy(crag_pred, gold)
#         crag_ground += score_groundedness(query, crag_docs, crag_answer)
#         crag_rel += score_relevance(query, crag_answer)

#         print("\n----------------------")
#         print("Query:", query)

#         print("\n[RAG Answer]")
#         print(rag_answer)

#         print("\n[CRAG Answer]")
#         print(crag_answer)

#         print("\nScores:")
#         print("RAG Acc:", rag_acc)
#         print("CRAG Acc:", crag_acc)

#     n = len(dataset)

#     print("\n===== FINAL RESULTS =====\n")

#     print("RAG:")
#     print("Accuracy:", rag_acc/n)
#     print("Groundedness:", rag_ground/n)
#     print("Relevance:", rag_rel/n)

#     print("\nCRAG:")
#     print("Accuracy:", crag_acc/n)
#     print("Groundedness:", crag_ground/n)
#     print("Relevance:", crag_rel/n)


# if __name__ == "__main__":
#     with open("eval_dataset.json") as f:
#         dataset = json.load(f)

#     evaluate(dataset)

import json
import re

from retriever import retrieve
from pipeline import crag_pipeline
from generator import generate_answer_llm
from local_evaluator import score_groundedness, score_relevance


def extract_citation(answer):
    match = re.search(r"(Section\s+\d+[A-Za-z]*\s+\w+|Article\s+\d+)", answer)
    return match.group(1) if match else None


def compute_accuracy(pred, gold):
    if pred is None:
        return 0
    return 1 if pred in gold else 0


def evaluate(dataset, k=5,
             log_file="llm_eval_log.jsonl",
             summary_file="llm_eval_running.json"):

    rag_acc = crag_acc = 0
    rag_ground = crag_ground = 0
    rag_rel = crag_rel = 0

    n = len(dataset)

    with open(log_file, "w") as fout:

        for i, item in enumerate(dataset):
            query = item["query"]
            gold = item["gold_citations"]

            # -------------------
            # RAG
            # -------------------
            rag_docs = retrieve(query, k=k)
            rag_answer = generate_answer_llm(query, rag_docs)

            rag_pred = extract_citation(rag_answer)
            rag_a = compute_accuracy(rag_pred, gold)
            rag_g = score_groundedness(query, rag_docs, rag_answer)
            rag_r = score_relevance(query, rag_answer)

            rag_acc += rag_a
            rag_ground += rag_g
            rag_rel += rag_r

            # -------------------
            # CRAG
            # -------------------
            crag_out = crag_pipeline(query, use_llm=True, return_context=True)

            crag_answer = crag_out["answer"]
            crag_docs = crag_out["docs"]

            crag_pred = extract_citation(crag_answer)
            crag_a = compute_accuracy(crag_pred, gold)
            crag_g = score_groundedness(query, crag_docs, crag_answer)
            crag_r = score_relevance(query, crag_answer)

            crag_acc += crag_a
            crag_ground += crag_g
            crag_rel += crag_r

            # -------------------
            # PRINT (debug)
            # -------------------
            print("\n----------------------")
            print(f"[{i+1}/{n}] Query:", query)

            print("\n[RAG Answer]")
            print(rag_answer)

            print("\n[CRAG Answer]")
            print(crag_answer)

            print("\nRunning Accuracy:")
            print("RAG:", rag_acc/(i+1))
            print("CRAG:", crag_acc/(i+1))

            # -------------------
            # WRITE PER-QUERY LOG
            # -------------------
            fout.write(json.dumps({
                "query": query,
                "gold": gold,

                "rag": {
                    "answer": rag_answer,
                    "pred": rag_pred,
                    "accuracy": rag_a,
                    "groundedness": rag_g,
                    "relevance": rag_r
                },
                "crag": {
                    "answer": crag_answer,
                    "pred": crag_pred,
                    "accuracy": crag_a,
                    "groundedness": crag_g,
                    "relevance": crag_r
                }
            }) + "\n")

            fout.flush()  # ✅ ensures live logging

            # -------------------
            # WRITE RUNNING SUMMARY
            # -------------------
            running_summary = {
                "processed": i + 1,

                "RAG": {
                    "Accuracy": rag_acc / (i + 1),
                    "Groundedness": rag_ground / (i + 1),
                    "Relevance": rag_rel / (i + 1)
                },
                "CRAG": {
                    "Accuracy": crag_acc / (i + 1),
                    "Groundedness": crag_ground / (i + 1),
                    "Relevance": crag_rel / (i + 1)
                },
                "Improvement": {
                    "Accuracy_gain": (crag_acc - rag_acc) / (i + 1),
                    "Groundedness_gain": (crag_ground - rag_ground) / (i + 1),
                    "Relevance_gain": (crag_rel - rag_rel) / (i + 1)
                }
            }

            with open(summary_file, "w") as f:
                json.dump(running_summary, f, indent=2)

    # -------------------
    # FINAL PRINT
    # -------------------
    print("\n===== FINAL RESULTS =====\n")

    print("RAG:")
    print("Accuracy:", rag_acc/n)
    print("Groundedness:", rag_ground/n)
    print("Relevance:", rag_rel/n)

    print("\nCRAG:")
    print("Accuracy:", crag_acc/n)
    print("Groundedness:", crag_ground/n)
    print("Relevance:", crag_rel/n)


if __name__ == "__main__":
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    evaluate(dataset)