"""
Demo 08: RAPTOR-style hierarchical retrieval.

RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval.

This is a compact classroom version:
- leaf nodes are short document chunks
- related leaves are clustered
- each cluster gets a summary node
- retrieval searches summaries first, then descends to supporting leaves

The real paper uses embedding clustering and LLM-generated abstractive summaries.
Here we use TF-IDF, k-means, and extractive summaries so the demo runs locally.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Node:
    node_id: str
    level: int
    text: str
    children: list["Node"] = field(default_factory=list)


def corpus_chunks() -> list[str]:
    return [
        "Apple revenue growth was driven by strong iPhone demand in emerging markets, especially India and Southeast Asia.",
        "Apple Services expanded by 20 percent year over year, increasing recurring revenue from subscriptions, payments, and cloud storage.",
        "Supply-chain resilience helped Apple keep popular devices available while competitors faced logistics delays.",
        "Wearables growth slowed, but accessories still contributed to the installed-base ecosystem and customer retention.",
        "The Atlas migration slipped after a vendor API certification arrived six weeks late and compressed the test schedule.",
        "Atlas ownership was split across infrastructure and application teams, leaving no single lead to resolve blockers quickly.",
        "A memory leak in the authentication service and a schema mismatch caused repeated Atlas deployment failures.",
        "Late compliance requests expanded Atlas scope after development had already started.",
        "Security operations required quarterly access reviews, privileged account monitoring, and evidence collection for auditors.",
        "A phishing simulation showed that finance staff needed stronger training on invoice fraud and malicious attachments.",
        "Compliance teams prioritized source-of-record traceability so policy answers could be tied back to approved documents.",
        "Incident response playbooks required severity classification, owner assignment, containment, and post-incident review.",
    ]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def top_terms(texts: list[str], max_terms: int = 5) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
    return [term for term, _score in ranked[:max_terms]]


def summarize_cluster(cluster_id: int, chunks: list[str]) -> str:
    """Small extractive summary used as the higher-level RAPTOR node."""
    terms = ", ".join(top_terms(chunks, max_terms=6))
    representative = " ".join(split_sentences(" ".join(chunks))[:2])
    return f"Cluster {cluster_id} summary. Key terms: {terms}. Representative evidence: {representative}"


def build_raptor_tree(chunks: list[str], n_clusters: int = 3) -> tuple[Node, list[Node]]:
    """Build a two-level RAPTOR-like tree: root -> summary clusters -> leaf chunks."""
    leaf_nodes = [Node(node_id=f"leaf_{i:02d}", level=0, text=chunk) for i, chunk in enumerate(chunks)]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init=10)
    labels = kmeans.fit_predict(matrix)

    cluster_nodes: list[Node] = []
    for cluster_id in sorted(set(labels)):
        children = [leaf_nodes[i] for i, label in enumerate(labels) if label == cluster_id]
        summary = summarize_cluster(cluster_id, [child.text for child in children])
        cluster_nodes.append(
            Node(node_id=f"summary_{cluster_id}", level=1, text=summary, children=children)
        )

    root_summary = "Root summary. The corpus contains these major branches: " + " | ".join(
        node.text.split(" Representative evidence:")[0] for node in cluster_nodes
    )
    root = Node(node_id="root", level=2, text=root_summary, children=cluster_nodes)
    return root, leaf_nodes


def rank_texts(query: str, items: list[Node], top_k: int) -> list[tuple[Node, float]]:
    texts = [item.text for item in items]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    scores = (matrix @ q.T).toarray().ravel()
    ranked = sorted(zip(items, scores), key=lambda item: float(item[1]), reverse=True)
    return [(node, float(score)) for node, score in ranked[:top_k]]


def flat_retrieval(query: str, leaves: list[Node]) -> list[tuple[Node, float]]:
    return rank_texts(query, leaves, top_k=3)


def raptor_retrieval(query: str, root: Node) -> tuple[list[tuple[Node, float]], list[tuple[Node, float]]]:
    """Retrieve a relevant summary branch, then retrieve leaf evidence inside that branch."""
    summary_rank = rank_texts(query, root.children, top_k=1)
    best_summary = summary_rank[0][0]
    leaf_rank = rank_texts(query, best_summary.children, top_k=min(4, len(best_summary.children)))
    return summary_rank, leaf_rank


def print_tree(root: Node) -> None:
    print("--- RAPTOR tree built at indexing time ---")
    print(f"{root.node_id}: {root.text[:140]}...")
    for summary in root.children:
        print(f"  {summary.node_id}: {summary.text[:150]}...")
        for child in summary.children:
            print(f"    {child.node_id}: {child.text[:90]}...")
    print()


def print_rankings(label: str, ranking: list[tuple[Node, float]]) -> None:
    print(label)
    for rank, (node, score) in enumerate(ranking, start=1):
        preview = node.text.replace("\n", " ")[:120]
        print(f"  {rank}. {node.node_id:<12} score={score:.3f} | {preview}...")
    print()


def synthesize_from_raptor(summary_rank: list[tuple[Node, float]], leaf_rank: list[tuple[Node, float]]) -> str:
    summary = summary_rank[0][0].text
    evidence = [node.text for node, _score in leaf_rank]
    bullets = "\n".join(f"- {line}" for line in evidence)
    return f"High-level orientation:\n{summary}\n\nSupporting leaf evidence:\n{bullets}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What were the main drivers of Apple revenue growth?",
        help="Query to run against the flat and RAPTOR indexes.",
    )
    parser.add_argument("--hide-tree", action="store_true", help="Suppress the printed tree.")
    args = parser.parse_args()

    chunks = corpus_chunks()
    root, leaves = build_raptor_tree(chunks)

    print(f"Query:\n  {args.query}\n")
    if not args.hide_tree:
        print_tree(root)

    flat = flat_retrieval(args.query, leaves)
    print_rankings("--- Flat retrieval over leaf chunks only ---", flat)

    summary_rank, leaf_rank = raptor_retrieval(args.query, root)
    print_rankings("--- RAPTOR step 1: retrieve the best summary branch ---", summary_rank)
    print_rankings("--- RAPTOR step 2: descend into that branch for leaf evidence ---", leaf_rank)

    print("--- RAPTOR-style answer context ---")
    print(synthesize_from_raptor(summary_rank, leaf_rank))
    print("\nTeaching takeaway:")
    print("  RAPTOR makes summaries searchable. The system can retrieve a broad theme first")
    print("  and then recover supporting details, instead of treating every chunk as isolated.")


if __name__ == "__main__":
    main()
