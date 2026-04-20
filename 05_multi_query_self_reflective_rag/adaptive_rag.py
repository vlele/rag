"""
Demo 05: multi-query retrieval + self-reflective critique.

This project turns chapters 8, 9, and 11 into one transparent classroom demo:
- RAG-Fusion style multi-query retrieval
- planning via decomposition
- a simple support/coverage critique loop
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_docs() -> list[dict]:
    data_path = Path(__file__).parent / "data" / "atlas_failure_docs.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def retrieve(query: str, docs: list[dict], top_k: int = 3) -> list[tuple[dict, float]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([doc["title"] + ". " + doc["text"] for doc in docs])
    q = vectorizer.transform([query])
    scores = (matrix @ q.T).toarray().ravel()
    ranked = sorted(
        [(docs[i], float(scores[i])) for i in range(len(docs))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


def reciprocal_rank_fusion(rankings: list[list[tuple[dict, float]]], c: int = 60) -> list[tuple[dict, float]]:
    scores: dict[str, float] = defaultdict(float)
    doc_lookup: dict[str, dict] = {}

    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            scores[doc["id"]] += 1.0 / (c + rank)
            doc_lookup[doc["id"]] = doc

    fused = [(doc_lookup[doc_id], score) for doc_id, score in scores.items()]
    return sorted(fused, key=lambda x: x[1], reverse=True)


def decompose_question(question: str) -> list[str]:
    if "why" in question.lower() and "fail" in question.lower():
        return [
            "Atlas migration project failure timeline",
            "Atlas migration ownership accountability",
            "Atlas migration technical obstacles and deployment failures",
            "Atlas migration stakeholder scope changes",
        ]
    return [question]


def draft_answer(docs: list[dict]) -> str:
    aspect_order = ["timeline", "ownership", "technical", "stakeholder", "budget"]
    ordered = sorted(docs, key=lambda d: aspect_order.index(d["aspect"]) if d["aspect"] in aspect_order else 999)

    bullets = []
    for doc in ordered:
        bullets.append(f"- {doc['text']}")
    return "Draft answer:\n" + "\n".join(bullets)


def critique_answer(selected_docs: list[dict]) -> tuple[list[str], str]:
    required_aspects = ["timeline", "ownership", "technical", "stakeholder"]
    present = {doc["aspect"] for doc in selected_docs}
    missing = [aspect for aspect in required_aspects if aspect not in present]

    if not missing:
        return missing, "Critique: all major aspects are covered."
    return missing, f"Critique: missing support for {', '.join(missing)}."


def reflective_retry(question: str, docs: list[dict], selected_docs: list[dict], missing_aspects: list[str]) -> list[dict]:
    """
    A simple retry strategy: issue targeted aspect queries and add the top result for each missing aspect.
    """
    selected_ids = {doc["id"] for doc in selected_docs}
    additions: list[dict] = []

    aspect_queries = {
        "timeline": f"{question} timeline delay",
        "ownership": f"{question} ownership accountability",
        "technical": f"{question} technical obstacles deployment failures",
        "stakeholder": f"{question} stakeholder scope changes",
    }

    for aspect in missing_aspects:
        ranking = retrieve(aspect_queries[aspect], docs, top_k=3)
        for doc, _score in ranking:
            if doc["aspect"] == aspect and doc["id"] not in selected_ids:
                additions.append(doc)
                selected_ids.add(doc["id"])
                break

    return selected_docs + additions


def unique_docs(ranked: list[tuple[dict, float]], top_k: int = 4) -> list[dict]:
    out = []
    seen = set()
    for doc, _score in ranked:
        if doc["id"] not in seen:
            out.append(doc)
            seen.add(doc["id"])
        if len(out) == top_k:
            break
    return out


def print_ranking(title: str, ranking: list[tuple[dict, float]]) -> None:
    print(title)
    for doc, score in ranking:
        print(f"  {doc['id']:>18} | {score:.4f} | {doc['title']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="Why did the Atlas migration project fail?",
        help="Question to ask.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "fusion", "reflective"],
        default="reflective",
        help="Which stage of the demo to run.",
    )
    args = parser.parse_args()

    docs = load_docs()
    question = args.question

    print(f"\nQuestion: {question}\n")

    if args.mode == "single":
        single = retrieve(question, docs, top_k=4)
        print_ranking("--- Single-query ranking ---", single)
        selected = unique_docs(single, top_k=3)
        print(draft_answer(selected))
        return

    subqueries = decompose_question(question)
    rankings = [retrieve(q, docs, top_k=4) for q in subqueries]

    print("--- Planned subqueries ---")
    for q in subqueries:
        print(f"  - {q}")
    print()

    if args.mode in {"fusion", "reflective"}:
        for q, ranking in zip(subqueries, rankings):
            print_ranking(f"Ranking for: {q!r}", ranking)

        fused = reciprocal_rank_fusion(rankings)
        print_ranking("--- Fused ranking (RAG-Fusion style) ---", fused)
        selected = unique_docs(fused, top_k=4)

        print(draft_answer(selected))

        if args.mode == "fusion":
            return

        missing, critique = critique_answer(selected)
        print("\n" + critique)

        if missing:
            repaired = reflective_retry(question, docs, selected, missing)
            print("\n--- Revised answer after reflective retry ---")
            print(draft_answer(repaired))
            missing2, critique2 = critique_answer(repaired)
            print("\n" + critique2)
        else:
            print("No retry needed.")


if __name__ == "__main__":
    main()
