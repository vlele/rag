"""
Demo 06: GraphRAG intuition with a small outage graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ALIASES = {
    "vendor patch": "vendor_patch",
    "patch": "vendor_patch",
    "authentication service": "auth_service",
    "auth service": "auth_service",
    "mfa": "mfa_rule",
    "portal": "student_portal",
    "student portal": "student_portal",
    "lockout": "student_portal",
    "locked out": "student_portal",
    "help desk": "helpdesk_spike",
}


def load_graph() -> tuple[nx.DiGraph, dict]:
    data_path = Path(__file__).parent / "data" / "incident_graph.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))

    graph = nx.DiGraph()
    label_lookup = {}
    for node in payload["nodes"]:
        graph.add_node(node["id"], label=node["label"])
        label_lookup[node["id"]] = node["label"]

    for edge in payload["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"],
            evidence=edge["evidence"],
        )
    return graph, label_lookup


def flat_retrieval(question: str, graph: nx.DiGraph, top_k: int = 3) -> list[tuple[str, float]]:
    evidence_texts = [graph.edges[e]["evidence"] for e in graph.edges]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(evidence_texts)
    q = vectorizer.transform([question])
    scores = (matrix @ q.T).toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(evidence_texts[i], float(scores[i])) for i in top_idx]


def detect_nodes(question: str) -> tuple[str | None, str | None]:
    q = question.lower()
    found = []
    for alias, node_id in ALIASES.items():
        if alias in q:
            found.append(node_id)

    # Keep first distinct nodes in order of discovery.
    distinct = []
    for node_id in found:
        if node_id not in distinct:
            distinct.append(node_id)

    if len(distinct) >= 2:
        return distinct[0], distinct[1]
    if len(distinct) == 1:
        return distinct[0], "student_portal"
    return "vendor_patch", "student_portal"


def graph_answer(question: str, graph: nx.DiGraph, label_lookup: dict[str, str]) -> tuple[list[str], str]:
    start, end = detect_nodes(question)
    path = nx.shortest_path(graph, source=start, target=end)

    edge_evidence = []
    for src, dst in zip(path, path[1:]):
        edge_evidence.append(graph.edges[src, dst]["evidence"])

    path_labels = [label_lookup[node] for node in path]
    answer = (
        "GraphRAG answer: "
        f"The {path_labels[0].lower()} affected the {path_labels[1].lower()}, which propagated through "
        f"{', '.join(label.lower() for label in path_labels[2:-1])} and ultimately blocked access to the "
        f"{path_labels[-1].lower()}."
    )
    return edge_evidence, answer


def main() -> None:
    question = "How did the vendor patch lead to students being locked out of the portal?"
    graph, labels = load_graph()

    print(f"\nQuestion: {question}\n")

    print("--- Flat retrieval ---")
    for text, score in flat_retrieval(question, graph):
        print(f"  score={score:.3f} | {text}")

    evidence, answer = graph_answer(question, graph, labels)

    print("\n--- Graph path evidence ---")
    for step, text in enumerate(evidence, start=1):
        print(f"  Step {step}: {text}")

    print("\n" + answer)
    print("\nKey teaching point:")
    print("Flat retrieval returns pieces. Graph retrieval returns a path of support.")


if __name__ == "__main__":
    main()
