"""
Demo 01: semantic vs lexical retrieval.

This is a classroom-friendly, lightweight analogue to the chapter 2 repo notebooks:
- w2v_vs_bm25.ipynb
- sentence_transformers.ipynb

The implementation uses:
- a BM25-style lexical scorer
- a tiny distributional semantic model built from synthetic co-occurrence data
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Document:
    id: int
    text: str


class BM25Lite:
    """A small BM25 implementation sufficient for teaching and demo use."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(doc) for doc in self.documents]
        self.avg_len = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(doc) for doc in self.documents]
        self.doc_freq = defaultdict(int)
        for doc in self.documents:
            for term in set(doc):
                self.doc_freq[term] += 1
        self.n_docs = len(self.documents)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        # Standard smoothed BM25-style IDF
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        score = 0.0
        tf = self.term_freqs[doc_index]
        doc_len = self.doc_lens[doc_index]
        for term in query_tokens:
            if term not in tf:
                continue
            freq = tf[term]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query: str) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


class TinyDistributionalSemanticSearch:
    """
    Build a tiny semantic space from co-occurrence counts.
    This is *not* Word2Vec, but it teaches the same key intuition:
    words that appear in similar contexts get similar vectors.
    """

    def __init__(self, training_corpus: Sequence[str], window_size: int = 2, n_components: int = 6):
        tokenized = [tokenize(text) for text in training_corpus]
        vocab = sorted({token for doc in tokenized for token in doc})
        self.vocab = vocab
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}

        matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        for tokens in tokenized:
            for i, word in enumerate(tokens):
                wi = self.word_to_idx[word]
                left = max(0, i - window_size)
                right = min(len(tokens), i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    context_word = tokens[j]
                    cj = self.word_to_idx[context_word]
                    matrix[wi, cj] += 1

        self.word_vectors = self._build_vectors(matrix, n_components=n_components)

    def _build_vectors(self, counts: np.ndarray, n_components: int) -> np.ndarray:
        total = counts.sum()
        row_sums = counts.sum(axis=1, keepdims=True)
        col_sums = counts.sum(axis=0, keepdims=True)

        # Positive PMI
        numerator = counts * total + 1e-9
        denominator = row_sums @ col_sums + 1e-9
        ppmi = np.maximum(np.log(numerator / denominator), 0.0)

        n_components = min(n_components, max(1, ppmi.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        word_vectors = svd.fit_transform(ppmi)
        return normalize(word_vectors)

    def text_vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        vectors = [self.word_vectors[self.word_to_idx[t]] for t in tokens if t in self.word_to_idx]
        if not vectors:
            return np.zeros(self.word_vectors.shape[1], dtype=float)
        return normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]

    def rank(self, query: str, documents: Sequence[Document]) -> list[tuple[int, float]]:
        q = self.text_vector(query)
        scored = []
        for doc in documents:
            dv = self.text_vector(doc.text)
            score = float(dv @ q)
            scored.append((doc.id, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)


def pretty_print(title: str, ranking: Sequence[tuple[int, float]], documents: Sequence[Document]) -> None:
    print(title)
    for doc_id, score in ranking:
        print(f"  Doc {doc_id}: {score:>6.3f}  |  {documents[doc_id].text}")
    print()


def main() -> None:
    documents = [
        Document(0, "My internet connection keeps dropping every few minutes."),
        Document(1, "WiFi connectivity is unstable in the dormitory."),
        Document(2, "Billing question about housing charges."),
        Document(3, "The campus router was restarted to restore the network."),
        Document(4, "Meal plan refund request for cafeteria charges."),
    ]

    # A tiny synthetic corpus that teaches the semantic space how related terms cluster.
    training_corpus = [
        "internet network wifi connectivity router modem",
        "connection connectivity network internet outage drop unstable",
        "wifi internet network issue troubleshooting router",
        "dormitory residence hall campus housing wifi internet",
        "billing charge invoice refund payment account",
    ]

    query = "network connection problems"

    tokenized_docs = [tokenize(doc.text) for doc in documents]
    bm25 = BM25Lite(tokenized_docs)
    semantic = TinyDistributionalSemanticSearch(training_corpus)

    lexical_ranking = bm25.rank(query)
    semantic_ranking = semantic.rank(query, documents)

    print(f"\nQuery: {query!r}\n")
    pretty_print("--- Lexical ranking (BM25-style) ---", lexical_ranking, documents)
    pretty_print("--- Semantic ranking (tiny distributional space) ---", semantic_ranking, documents)



if __name__ == "__main__":
    main()
