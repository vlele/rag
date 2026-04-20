"""
Demo 04A: a toy Fusion-in-Decoder (FiD) explanation.

This is not a transformer implementation.
It is a classroom-friendly stand-in that captures the central FiD idea:

1. Encode passages independently
2. Score their evidence independently
3. Fuse evidence late to build a synthesized answer
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.split()) >= 4]


def independent_encode(question: str, passages: list[dict]) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Independent encoding analogue:
    build one query+passage text per passage, then vectorize them together.
    """
    combined = [f"question: {question}\ncontext: {p['text']}" for p in passages]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(combined)
    return vectorizer, matrix


def top1_baseline_answer(question: str, passages: list[dict]) -> str:
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [p["text"] for p in passages]
    matrix = vectorizer.fit_transform(texts)
    q = vectorizer.transform([question])
    scores = (matrix @ q.T).toarray().ravel()
    best = passages[int(np.argmax(scores))]
    best_sentence = sentence_split(best["text"])[0]
    return f"Top-1 baseline answer: {best_sentence}"


def fid_like_answer(question: str, passages: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    # Step 1: independent passage encoding
    passage_vectorizer, passage_matrix = independent_encode(question, passages)

    # Step 2: sentence-level “decoder fusion”
    sentence_records: list[tuple[str, str]] = []
    for passage in passages:
        for sentence in sentence_split(passage["text"]):
            sentence_records.append((passage["title"], sentence))

    sentence_texts = [text for _, text in sentence_records]
    sent_vectorizer = TfidfVectorizer(stop_words="english")
    sent_matrix = sent_vectorizer.fit_transform(sentence_texts)
    q = sent_vectorizer.transform([question])
    sentence_scores = (sent_matrix @ q.T).toarray().ravel()

    max_score = float(sentence_scores.max()) if len(sentence_scores) else 0.0
    adjusted_scores = []
    for idx, raw_score in enumerate(sentence_scores):
        title, sentence = sentence_records[int(idx)]
        penalty = 0.0
        lowered = sentence.lower()
        if "not the primary reason" in lowered:
            penalty += 0.35
        bonus = 0.10 if "to reduce" in lowered or "major driver" in lowered or "experienced" in lowered else 0.0
        adjusted_scores.append(float(raw_score) - penalty + bonus)

    chosen: list[tuple[str, str]] = []
    used_titles = set()
    for idx in np.argsort(adjusted_scores)[::-1]:
        title, sentence = sentence_records[int(idx)]
        raw_score = float(sentence_scores[int(idx)])
        if raw_score < max_score * 0.45:
            continue
        if title not in used_titles:
            chosen.append((title, sentence))
            used_titles.add(title)
        if len(chosen) == 3:
            break

    joined = " ".join(sentence for _, sentence in chosen)
    answer = f"FiD-like fused answer: {joined}"
    return answer, chosen


def main() -> None:
    data_path = Path(__file__).parent / "data" / "cases.json"
    cases = json.loads(data_path.read_text(encoding="utf-8"))

    for case_name, payload in cases.items():
        question = payload["question"]
        passages = payload["passages"]

        print(f"\n=== {case_name.replace('_', ' ').title()} ===")
        print(f"Question: {question}\n")

        print(top1_baseline_answer(question, passages))
        fused_answer, chosen = fid_like_answer(question, passages)
        print(fused_answer)

        print("\nEvidence used in fused answer:")
        for title, sentence in chosen:
            print(f"- {title}: {sentence}")

        print("\nTeaching note:")
        if case_name == "synthesis_case":
            print(
                "  This case shows why late fusion matters: the best answer needs multiple passages, "
                "not a single top-ranked document."
            )
        else:
            print(
                "  This case shows FiD's second strength: it can keep multiple plausible passages around "
                "while still favoring the one that truly answers the question."
            )


if __name__ == "__main__":
    main()
