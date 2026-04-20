# 01 — Semantic Search Foundations

## What this demonstrates
This project shows why lexical retrieval and semantic retrieval behave differently.

It is inspired by the companion repository examples:
- `w2v_vs_bm25.ipynb`
- `sentence_transformers.ipynb`

To keep the demo lightweight and runnable in class without model downloads, this version uses:
- a simple **BM25-style lexical scorer**
- a tiny **distributional semantic space** built from a synthetic training corpus

## Why it is useful in the talk
Use this demo when explaining:
- vocabulary mismatch
- dense vs sparse retrieval
- why semantic search can recover conceptually related documents

## How to run
```bash
python demo_semantic_search.py
```

## What to point out live
- The lexical search misses the WiFi ticket because it does not share the exact wording.
- The semantic search recovers it because the training corpus places *internet*, *network*, *wifi*, and *connectivity* near each other.
- Semantic retrieval is broader, which is helpful — but it can also retrieve more loosely related items.

## Classroom presentation note
Say explicitly that this is a **toy semantic space**. The point is the retrieval behavior, not state-of-the-art embedding quality.
