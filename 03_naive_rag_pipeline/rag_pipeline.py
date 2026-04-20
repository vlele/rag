"""
Demo 03: a tiny naive RAG pipeline.

This project is intentionally framework-light so it can run anywhere.
It mirrors the logic of:
retrieve -> stuff context -> answer with sources
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    source: str
    text: str


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8").strip()))
    return docs


def chunk_text(text: str, source: str) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = [Chunk(source=source, text=p) for p in paragraphs if len(p.split()) > 3]
    return chunks


def sentence_candidates(chunks: Iterable[Chunk]) -> list[Chunk]:
    candidates: list[Chunk] = []
    for chunk in chunks:
        for sentence in SENTENCE_SPLIT_RE.split(chunk.text):
            sentence = sentence.strip()
            if len(sentence.split()) >= 5:
                candidates.append(Chunk(source=chunk.source, text=sentence))
    return candidates


def retrieve(query: str, chunks: list[Chunk], k: int = 3) -> list[tuple[Chunk, float]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [chunk.text for chunk in chunks]
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    scores = (matrix @ query_vec.T).toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def synthesize_answer(query: str, retrieved: list[tuple[Chunk, float]]) -> tuple[str, list[Chunk]]:
    # Re-rank at the sentence level for a lightweight, source-grounded synthesis.
    top_chunks = [chunk for chunk, _ in retrieved]
    candidates = sentence_candidates(top_chunks)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c.text for c in candidates])
    query_vec = vectorizer.transform([query])
    scores = (matrix @ query_vec.T).toarray().ravel()

    chosen: list[Chunk] = []
    used_sources = set()
    for idx in np.argsort(scores)[::-1]:
        candidate = candidates[int(idx)]
        # Favor source diversity so the answer feels synthesized, not copied from one chunk.
        if candidate.source not in used_sources or len(chosen) == 0:
            chosen.append(candidate)
            used_sources.add(candidate.source)
        if len(chosen) == 3:
            break

    summary_parts = []
    for i, c in enumerate(chosen, start=1):
        summary_parts.append(f"[S{i}] {c.text}")

    answer = (
        "Based on the retrieved evidence, the recommended response is:\n\n"
        + "\n".join(summary_parts)
    )
    return answer, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What should a student do after repeated dorm WiFi outages?",
        help="Question to ask the tiny RAG system.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved chunks before the final answer.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    documents = load_documents(data_dir)

    chunks: list[Chunk] = []
    for source, text in documents:
        chunks.extend(chunk_text(text, source))

    retrieved = retrieve(args.query, chunks, k=3)

    print(f"\nQuestion: {args.query}\n")

    if args.show_context:
        print("--- Retrieved chunks (naive RAG context) ---")
        for i, (chunk, score) in enumerate(retrieved, start=1):
            print(f"Chunk {i} | score={score:.3f} | source={chunk.source}")
            print(chunk.text)
            print()

    answer, supporting = synthesize_answer(args.query, retrieved)

    print("--- Final answer ---")
    print(answer)
    print("\n--- Sources used in answer ---")
    for i, chunk in enumerate(supporting, start=1):
        print(f"[S{i}] {chunk.source}")


if __name__ == "__main__":
    main()
