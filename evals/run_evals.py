"""Retrieval eval: does every eval question surface its supporting corpus notes?

For each question in eval_questions.json, run the keyword retriever and check
whether at least one gold-standard corpus note lands in the top-3. This is the
v1 gate for the RAG layer: it proves the corpus covers the eval set and the
retriever finds it, before any agent is built on top.

Usage:
    python evals/run_evals.py

Exit code is 0 only when every question retrieves a gold note in the top-3.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import load_corpus, retrieve

ROOT = Path(__file__).parent


def main():
    notes = load_corpus(ROOT.parent / "rag" / "corpus.json")
    with open(ROOT / "eval_questions.json") as f:
        questions = json.load(f)["questions"]

    hits, misses = 0, []
    top1_hits = 0
    by_category = Counter()
    by_category_hits = Counter()

    for q in questions:
        ranked = retrieve(q["question"], notes, k=3)
        ranked_ids = [n["id"] for n, _ in ranked]
        gold = set(q["corpus_notes"])
        hit = bool(gold & set(ranked_ids))
        by_category[q["category"]] += 1
        if hit:
            hits += 1
            by_category_hits[q["category"]] += 1
        else:
            misses.append((q["id"], q["question"], sorted(gold), ranked_ids))
        if ranked_ids and ranked_ids[0] in gold:
            top1_hits += 1

    total = len(questions)
    print(f"Retrieval eval: {hits}/{total} questions retrieve a gold note in top-3")
    print(f"Top-1 accuracy:  {top1_hits}/{total}")
    for cat, n in by_category.items():
        print(f"  {cat}: {by_category_hits[cat]}/{n}")

    if misses:
        print("\nMISSES (gold note not in top-3):")
        for qid, question, gold, ranked in misses:
            print(f"  {qid}: {question[:70]}")
            print(f"    gold={gold} retrieved={ranked}")
        return 1
    print("\nAll green: the corpus covers every eval question and the retriever finds it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
