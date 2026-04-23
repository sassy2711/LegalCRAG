# pipeline.py

from retriever import retrieve
from corrector import correct_query
from evaluator import evaluate
from generator import generate_answer, generate_answer_llm


def crag_pipeline(query, max_hops=2, return_results=False, k=5, use_llm=False, return_context=False):
    current_query = query
    best_results = None
    best_score = -1.0
    best_query = query   # <-- added (safe init)

    seen_queries = set()

    for step in range(max_hops):
        print(f"\n[STEP {step+1}] Query: {current_query}")

        if current_query in seen_queries:
            print("[INFO] Query unchanged, stopping.")
            break
        seen_queries.add(current_query)

        # 👇 only change: pass k
        results = retrieve(current_query, k=k)

        # # DEBUG
        # print("[DEBUG] Retrieved docs:")
        # for r in results:
        #     print(f"  {r['citation']} — {r.get('title','')}")

        # Evaluate quality of this retrieval
        ok, score, reason = evaluate(current_query, results)

        print(f"[EVAL] ok={ok}, score={score:.3f}, reason={reason}")

        # Update best results if this hop is better
        if score > best_score:
            best_score = score
            best_results = results
            best_query = current_query

        # If good enough, stop early
        if ok:
            print("[INFO] Retrieval confident, stopping early.")
            break

        # Otherwise rewrite and retry
        new_query = correct_query(current_query, reason)

        if new_query == current_query:
            print("[INFO] Corrector produced no change, stopping.")
            break

        current_query = new_query

    # 👇 NEW: allow evaluation script to access retrieval
    if return_results:
        return best_query, best_results
    
    # =====================
    # NEW PART
    # =====================
    if use_llm:
        answer = generate_answer_llm(best_query, best_results)
    else:
        answer = generate_answer(best_query, best_results)

    if return_context:
        return {
            "query": best_query,
            "docs": best_results,
            "answer": answer
        }

    return answer


INTERACTIVE_MODE = False

TEST_QUERIES = [
    "What is cheating under IPC?",
    "What is FIR procedure?",
    "What is Article 21?",
    "What is evidence?",
]

if __name__ == "__main__":
    if INTERACTIVE_MODE:
        while True:
            q = input("\nQuery: ")
            answer = crag_pipeline(q)
            print("\n", answer)
    else:
        print("\n[INFO] Running predefined test queries...\n")
        for q in TEST_QUERIES:
            print(f"\nQuery: {q}")
            answer = crag_pipeline(q)
            print("\n", answer)
            print("\n" + "=" * 60)