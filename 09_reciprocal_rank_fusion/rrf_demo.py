"""
Demo 09: Reciprocal Rank Fusion (RRF).

This classroom demo turns the RAG-Fusion ranking idea into a visible, runnable example.
It intentionally uses TF-IDF rather than a hosted embedding model so students can run it
locally and focus on the ranking logic.

Key teaching point:
- Multi-query retrieval improves coverage.
- RRF combines ranked lists without trusting raw scores from any single query.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from collections import Counter
from math import log, sqrt
from typing import Iterable


Doc = dict[str, str]


def load_docs() -> list[Doc]:
    data_path = Path(__file__).parent / "data" / "underwriting_docs.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def doc_text(doc: Doc) -> str:
    return f"{doc['title']}. {doc['text']}"


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for", "from",
    "how", "if", "in", "is", "it", "of", "on", "or", "should", "the", "to",
    "under", "was", "we", "what", "when", "whose", "with"
}


def tokenize(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def build_tfidf_vectors(texts: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    tokenized = [tokenize(t) for t in texts]
    n_docs = len(texts)
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))

    idf = {term: log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    vectors = []
    for toks in tokenized:
        counts = Counter(toks)
        vec = {term: count * idf[term] for term, count in counts.items()}
        norm = sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({term: value / norm for term, value in vec.items()})
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(query))
    vec = {term: count * idf.get(term, 1.0) for term, count in counts.items()}
    norm = sqrt(sum(v * v for v in vec.values())) or 1.0
    return {term: value / norm for term, value in vec.items()}


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def retrieve(query: str, docs: list[Doc], top_k: int = 5) -> list[tuple[Doc, float]]:
    """Return a ranked list using a transparent TF-IDF baseline.

    TF-IDF is not RAG-Fusion itself; it is just the retriever used to produce
    ranked lists that RRF will fuse. The implementation is pure Python so this
    demo can run in a classroom without extra packages.
    """
    doc_vectors, idf = build_tfidf_vectors([doc_text(doc) for doc in docs])
    q_vec = vectorize_query(query, idf)
    scored = [(doc, cosine_sparse(q_vec, doc_vectors[i])) for i, doc in enumerate(docs)]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(
    rankings: Iterable[list[tuple[Doc, float]]],
    rrf_k: int = 60,
) -> list[tuple[Doc, float, list[int]]]:
    """Fuse multiple rankings using Reciprocal Rank Fusion.

    The constant rrf_k dampens the contribution of any single list. A common
    default is 60, but lower values are useful for classroom demonstration.
    """
    scores: dict[str, float] = defaultdict(float)
    ranks_seen: dict[str, list[int]] = defaultdict(list)
    doc_lookup: dict[str, Doc] = {}

    for ranking in rankings:
        for rank, (doc, _raw_score) in enumerate(ranking, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (rrf_k + rank)
            ranks_seen[doc_id].append(rank)
            doc_lookup[doc_id] = doc

    fused = [
        (doc_lookup[doc_id], score, ranks_seen[doc_id])
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item[1], reverse=True)


def make_query_variants(question: str) -> list[str]:
    """A tiny stand-in for LLM-generated query variants.

    In a production RAG-Fusion system, these might be produced by a language
    model. Here we keep them deterministic so the live demo is reliable.
    """
    return [
        question,
        "six month fast track credit score improved 650 720 underwriting memo",
        "denied applicant reopen file after six months 60 point score improvement",
        "credit score improved at least 60 points debt to income below 43 percent",
        "exception memo updated eligibility pathway fast track borrower credit score",
    ]


def print_ranking(title: str, ranking: list[tuple[Doc, float]], max_rows: int = 5) -> None:
    print(title)
    for rank, (doc, score) in enumerate(ranking[:max_rows], start=1):
        print(f"  {rank:>2}. {doc['id']} | score={score:.4f} | {doc['title']}")
    print()


def print_fused(fused: list[tuple[Doc, float, list[int]]], max_rows: int = 7) -> None:
    print("--- RRF fused ranking ---")
    for rank, (doc, score, ranks_seen) in enumerate(fused[:max_rows], start=1):
        ranks = ", ".join(str(r) for r in ranks_seen)
        print(f"  {rank:>2}. {doc['id']} | RRF={score:.5f} | ranks seen=[{ranks}] | {doc['title']}")
    print()


def synthesize_short_answer(top_doc: Doc) -> str:
    return (
        "Suggested answer based on the top fused evidence:\n"
        f"Use {top_doc['title']}. {top_doc['text']}\n"
        "Teaching note: RRF did not prove the answer by itself; it selected the evidence "
        "that should be passed to the generator or to a later reranker."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default=(
            "How should we advise a client denied six months ago whose credit score "
            "rose from 650 to 720 under the new underwriting rules?"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    docs = load_docs()
    variants = make_query_variants(args.question)

    print(f"\nUser question:\n  {args.question}\n")

    baseline = retrieve(args.question, docs, top_k=args.top_k)
    print_ranking("--- Single-query baseline ---", baseline)

    rankings = []
    print("--- Query variants ---")
    for i, variant in enumerate(variants, start=1):
        print(f"  Q{i}: {variant}")
    print()

    for i, variant in enumerate(variants, start=1):
        ranking = retrieve(variant, docs, top_k=args.top_k)
        rankings.append(ranking)
        print_ranking(f"Ranking for Q{i}", ranking)

    fused = reciprocal_rank_fusion(rankings, rrf_k=args.rrf_k)
    print_fused(fused)

    print(synthesize_short_answer(fused[0][0]))


if __name__ == "__main__":
    main()
