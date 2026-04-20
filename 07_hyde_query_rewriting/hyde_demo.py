"""
Demo 07: HyDE query rewriting.

HyDE = Hypothetical Document Embeddings.

The production version usually asks an LLM to generate a short hypothetical answer,
embeds that hypothetical answer, and retrieves documents using the richer text.
This classroom version stays fully local and deterministic by using a small rule-based
hypothetical answer generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


def load_corpus() -> list[dict]:
    return json.loads((Path(__file__).parent / "corpus.json").read_text(encoding="utf-8"))


def retrieve(search_text: str, corpus: list[dict], top_k: int = 3) -> list[tuple[dict, float]]:
    """Simple lexical retrieval so the vocabulary-mismatch effect is visible."""
    documents = [doc["title"] + ". " + doc["text"] for doc in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([search_text])
    scores = (matrix @ query_vec.T).toarray().ravel()
    ranked = sorted(
        [(corpus[i], float(scores[i])) for i in range(len(corpus))],
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


def generate_hypothetical_document(query: str) -> str:
    """
    Stand-in for the LLM call in real HyDE.

    The point is not that this rule is smart. The point is that a plausible answer often
    contains the terminology used by the source documents, while the user's raw query may not.
    """
    q = query.lower()
    if any(term in q for term in ["kicked out", "logged out", "signed out", "session", "login"]):
        return (
            "A good answer would explain session persistence during rolling deployments: "
            "enable load balancer session affinity, configure sticky sessions when appropriate, "
            "store session data in Redis or another shared session store, and avoid losing login "
            "state when application instances restart."
        )
    if any(term in q for term in ["database", "schema", "migration"]):
        return (
            "A good answer would mention reversible schema migrations, database snapshots, "
            "rollback plans, and backward compatibility during releases."
        )
    return (
        "A good answer would restate the likely technical terms, operational cause, "
        "mitigation steps, and source evidence needed to answer the question."
    )


def print_ranking(label: str, ranking: list[tuple[dict, float]]) -> None:
    print(label)
    for rank, (doc, score) in enumerate(ranking, start=1):
        print(f"  {rank}. {doc['id']:<22} score={score:.3f} | {doc['title']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="How can we stop customers from being kicked out during upgrades?",
        help="User query to test.",
    )
    args = parser.parse_args()

    corpus = load_corpus()
    query = args.query

    print(f"User query:\n  {query}\n")

    baseline = retrieve(query, corpus, top_k=3)
    print_ranking("--- Baseline retrieval using the raw query ---", baseline)

    hypothetical = generate_hypothetical_document(query)
    print("HyDE hypothetical document:")
    print("  " + hypothetical + "\n")

    hyde_only = retrieve(hypothetical, corpus, top_k=3)
    print_ranking("--- Retrieval using the hypothetical document ---", hyde_only)

    combined = retrieve(query + " " + hypothetical, corpus, top_k=3)
    print_ranking("--- Retrieval using query + hypothetical document ---", combined)

    print("Teaching takeaway:")
    print("  HyDE does not retrieve from magic. It changes the retrieval text so it contains")
    print("  the concepts likely to appear in the answer, making the retriever less brittle.")


if __name__ == "__main__":
    main()
