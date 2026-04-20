"""
Demo 10: Cross-encoder-style reranking.

This demo is intentionally lightweight and fully local. It uses:
1. A fast bi-encoder-like first stage implemented with TF-IDF.
2. A transparent toy cross-encoder scorer that reads each (query, document) pair jointly.

The toy cross encoder is not a neural model. It is a teaching device that makes the
cross-encoder idea visible: the score depends on how the candidate document answers
this specific query, including conditions and negation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path


Doc = dict[str, str]


@dataclass(frozen=True)
class CrossEncoderScore:
    score: float
    matched_requirements: list[str]
    penalties: list[str]


def load_docs() -> list[Doc]:
    data_path = Path(__file__).parent / "data" / "student_policy_docs.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can", "for",
    "from", "how", "if", "in", "is", "it", "of", "on", "or", "the", "to", "what"
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


def first_stage_retrieve(query: str, docs: list[Doc], top_k: int = 6) -> list[tuple[Doc, float]]:
    """Fast candidate retrieval.

    In production, this is usually BM25, dense-vector search, or hybrid search.
    Here we use a pure-Python TF-IDF ranker so the demo runs anywhere.
    """
    doc_vectors, idf = build_tfidf_vectors([f"{d['title']}. {d['text']}" for d in docs])
    q_vec = vectorize_query(query, idf)
    scored = [(doc, cosine_sparse(q_vec, doc_vectors[i])) for i, doc in enumerate(docs)]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def toy_cross_encoder_score(query: str, doc: Doc) -> CrossEncoderScore:
    """Score the query and candidate together.

    A real cross-encoder would pass this pair through a Transformer as one input:

        [CLS] query [SEP] candidate_document [SEP]

    This toy version makes that joint judgment interpretable with explicit requirements.
    """
    q = normalize(query)
    text = normalize(f"{doc['title']}. {doc['text']}")
    combined = q + " || " + text

    requirements = {
        "missed exam": ["misses an exam", "missed exam", "missed an exam"],
        "illness condition": ["illness", "sick", "medical"],
        "medical documentation": ["medical documentation", "documentation"],
        "deadline after event": ["within 48 hours", "48 hours", "after verification", "after the missed exam"],
        "make-up eligibility": ["qualify for a make-up exam", "make-up exam", "offer a make-up exam"],
    }

    matched: list[str] = []
    score = 0.0
    for label, phrases in requirements.items():
        if any(phrase in text for phrase in phrases):
            matched.append(label)
            score += 1.0

    # Joint interaction bonus: this is what a cross encoder is good at. The document
    # gets extra credit only when multiple conditions needed by THIS query appear together.
    if {"missed exam", "illness condition", "medical documentation", "make-up eligibility"}.issubset(set(matched)):
        score += 2.0
        matched.append("joint condition match bonus")

    if "within 48 hours" in text and "medical documentation" in text:
        score += 0.75
        matched.append("deadline tied to documentation")

    penalties: list[str] = []
    negation_patterns = [
        "does not create eligibility",
        "do not qualify",
        "does not change a grade",
        "not required",
        "separate from the exam make-up process",
    ]
    for pattern in negation_patterns:
        if pattern in text:
            score -= 1.25
            penalties.append(f"negation/limitation: {pattern}")

    # A mild title-query interaction: exact topic titles are often strong signals.
    if "make-up exam" in normalize(doc["title"]):
        score += 0.5
        matched.append("title matches task")

    return CrossEncoderScore(score=score, matched_requirements=matched, penalties=penalties)


def rerank_with_cross_encoder(query: str, candidates: list[tuple[Doc, float]]) -> list[tuple[Doc, float, CrossEncoderScore]]:
    reranked = []
    for doc, retrieval_score in candidates:
        ce = toy_cross_encoder_score(query, doc)
        reranked.append((doc, retrieval_score, ce))
    return sorted(reranked, key=lambda item: item[2].score, reverse=True)


def print_first_stage(candidates: list[tuple[Doc, float]]) -> None:
    print("--- Stage 1: fast candidate retrieval ---")
    for rank, (doc, score) in enumerate(candidates, start=1):
        print(f"  {rank:>2}. {doc['id']} | retrieval={score:.4f} | {doc['title']} [{doc['label']}]")
    print()


def print_reranked(reranked: list[tuple[Doc, float, CrossEncoderScore]]) -> None:
    print("--- Stage 2: cross-encoder-style reranking ---")
    for rank, (doc, retrieval_score, ce) in enumerate(reranked, start=1):
        print(
            f"  {rank:>2}. {doc['id']} | retrieval={retrieval_score:.4f} "
            f"| cross_encoder={ce.score:.2f} | {doc['title']} [{doc['label']}]"
        )
        print(f"      matched: {', '.join(ce.matched_requirements) if ce.matched_requirements else 'none'}")
        print(f"      penalties: {', '.join(ce.penalties) if ce.penalties else 'none'}")
    print()


def explain_answer(top_doc: Doc) -> str:
    return (
        "Reranked answer evidence:\n"
        f"Top document: {top_doc['id']} — {top_doc['title']}\n"
        f"{top_doc['text']}\n\n"
        "Teaching note: the reranker did not search the full corpus. It only rescored the "
        "small candidate pool from stage 1. That is the usual production pattern: broad "
        "retrieval first, expensive precision pass second."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What conditions allow medical documentation after a missed exam to qualify for a make-up exam?",
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    docs = load_docs()
    print(f"\nQuery:\n  {args.query}\n")

    candidates = first_stage_retrieve(args.query, docs, top_k=args.top_k)
    print_first_stage(candidates)

    reranked = rerank_with_cross_encoder(args.query, candidates)
    print_reranked(reranked)

    print(f"Cross-encoder cost model: scored {len(candidates)} query-document pairs.")
    print("In a real neural cross encoder, each pair is a separate model pass, so rerank only a small candidate pool.\n")
    print(explain_answer(reranked[0][0]))


if __name__ == "__main__":
    main()
