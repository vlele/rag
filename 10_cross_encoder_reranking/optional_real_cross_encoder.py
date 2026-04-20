"""
Optional real cross-encoder reranking demo.

This script requires the sentence-transformers package and a downloadable Hugging Face
model. It is optional because live classrooms often lack internet access or cached models.

Run:
    pip install -r requirements_optional_real_model.txt
    python optional_real_cross_encoder.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from cross_encoder_rerank_demo import first_stage_retrieve


QUERY = "What conditions allow medical documentation after a missed exam to qualify for a make-up exam?"


def load_docs() -> list[dict[str, str]]:
    data_path = Path(__file__).parent / "data" / "student_policy_docs.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def main() -> None:
    docs = load_docs()
    candidates = first_stage_retrieve(QUERY, docs, top_k=6)

    # Small MS MARCO cross encoder. You can replace this with a stronger model if available.
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(QUERY, f"{doc['title']}. {doc['text']}") for doc, _ in candidates]
    scores = model.predict(pairs)

    reranked = sorted(
        [(doc, retrieval_score, float(score)) for (doc, retrieval_score), score in zip(candidates, scores)],
        key=lambda item: item[2],
        reverse=True,
    )

    print(f"\nQuery:\n  {QUERY}\n")
    print("--- Real CrossEncoder reranking ---")
    for rank, (doc, retrieval_score, ce_score) in enumerate(reranked, start=1):
        print(
            f"{rank:>2}. {doc['id']} | retrieval={retrieval_score:.4f} "
            f"| cross_encoder={ce_score:.4f} | {doc['title']} [{doc['label']}]"
        )


if __name__ == "__main__":
    main()
