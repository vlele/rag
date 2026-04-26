"""
Demo 02: ANN scale intuition.

This is a lightweight stand-in for the FAISS notebook in the companion repo.
It uses a coarse partition approach:
- assign each vector to its nearest centroid
- at query time, search only the top few centroid buckets
"""

from __future__ import annotations

import numpy as np

from ann_scale_core import measure, normalize_rows


def main() -> None:
    rng = np.random.default_rng(42)

    n_vectors = 200_000
    dim = 32

    print(f"Generating {n_vectors:,} synthetic vectors of dimension {dim}...")
    vectors = normalize_rows(rng.normal(size=(n_vectors, dim)).astype(np.float32))
    query = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

    stats = measure(vectors, query, n_centroids=128, probes=4)

    print("\n--- Approximate-neighbor search results ---")
    print(f"Index build time:           {stats.build_time_sec:8.4f} s")
    print(f"Exact top-10 search time:   {stats.exact_time_sec:8.4f} s")
    print(f"Approx top-10 search time:  {stats.approx_time_sec:8.4f} s")
    print(f"Candidate fraction searched:{stats.candidate_fraction:8.2%}")
    print(f"Recall@10 vs exact search:  {stats.recall_at_10:8.2f}")
    print(f"Precision@10 vs exact search:{stats.precision_at_10:7.2f}")



if __name__ == "__main__":
    main()
