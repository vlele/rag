"""
Core helpers for the ANN scale demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchStats:
    exact_time_sec: float
    approx_time_sec: float
    build_time_sec: float
    candidate_fraction: float
    recall_at_10: float
    precision_at_10: float


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_coarse_partition(vectors: np.ndarray, n_centroids: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a coarse partition by sampling random centroids and assigning each vector
    to its nearest centroid by cosine similarity.
    """
    rng = np.random.default_rng(seed)
    centroid_ids = rng.choice(len(vectors), size=n_centroids, replace=False)
    centroids = vectors[centroid_ids]
    assignments = np.argmax(vectors @ centroids.T, axis=1)
    return centroids, assignments


def exact_search(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> np.ndarray:
    scores = (vectors @ query.T).ravel()
    return np.argsort(scores)[::-1][:k]


def approximate_search(
    vectors: np.ndarray,
    query: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    probes: int = 4,
    k: int = 10,
) -> tuple[np.ndarray, int]:
    centroid_scores = (centroids @ query.T).ravel()
    top_buckets = np.argsort(centroid_scores)[::-1][:probes]

    candidate_idx = np.where(np.isin(assignments, top_buckets))[0]
    candidate_scores = (vectors[candidate_idx] @ query.T).ravel()
    top_local = np.argsort(candidate_scores)[::-1][:k]
    return candidate_idx[top_local], len(candidate_idx)


def measure(vectors: np.ndarray, query: np.ndarray, n_centroids: int = 128, probes: int = 4) -> SearchStats:
    build_start = time.time()
    centroids, assignments = build_coarse_partition(vectors, n_centroids=n_centroids)
    build_time = time.time() - build_start

    t0 = time.time()
    exact = exact_search(vectors, query, k=10)
    exact_time = time.time() - t0

    t1 = time.time()
    approx, candidate_count = approximate_search(
        vectors,
        query,
        centroids=centroids,
        assignments=assignments,
        probes=probes,
        k=10,
    )
    approx_time = time.time() - t1

    overlap_count = len(set(map(int, exact)).intersection(set(map(int, approx))))
    recall = overlap_count / 10.0
    precision = overlap_count / len(approx) if len(approx) else 0.0
    candidate_fraction = candidate_count / len(vectors)

    return SearchStats(
        exact_time_sec=exact_time,
        approx_time_sec=approx_time,
        build_time_sec=build_time,
        candidate_fraction=candidate_fraction,
        recall_at_10=recall,
        precision_at_10=precision,
    )
