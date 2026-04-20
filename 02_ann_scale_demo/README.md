# 02 — ANN Scale Demo

## What this demonstrates
This project explains the intuition behind FAISS and approximate nearest-neighbor search:
- exact search checks every vector
- approximate search narrows the candidate set first

To keep the demo lightweight and runnable without `faiss-cpu`, this project uses a simple **coarse partition** strategy:
1. assign vectors to random centroids
2. search only the top centroid buckets
3. compare speed and recall against exact search

## Why it is useful in the talk
Use this demo when explaining:
- why vector search needs specialized infrastructure
- why approximate search is often “good enough”
- the trade-off between speed and recall

## How to run
```bash
python demo_ann_scale.py
```

## What to point out live
- The exact search is still fast on a laptop for small corpora.
- The approximate search becomes more useful as the corpus grows.
- We deliberately search only a small fraction of the full corpus.

## Teaching note
This is **FAISS intuition**, not FAISS itself. The point is to make the cluster-first search idea visible.
