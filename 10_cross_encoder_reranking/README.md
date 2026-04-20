# Demo 10 — Cross-Encoder Reranking

## Concept demonstrated

This demo shows the classic two-stage retrieval pattern:

1. **Fast candidate retrieval** with a bi-encoder-like method. The query and documents are represented independently, so search is cheap and scalable.
2. **Precise reranking** with a cross-encoder-style scorer. The query and each candidate document are read together, so the scorer can inspect conditions, negation, and exact relevance.

The local demo uses TF-IDF for the first stage and a transparent, rule-based **toy cross encoder** for the second stage. The toy scorer is not a neural model; it is deliberately simple so students can see what “joint query-document scoring” means. An optional script is included for a real `sentence-transformers` CrossEncoder model when internet/model access is available.

## Why this matters in the talk

Use this after RAG-Fusion or after the Sentence-BERT section. The teaching point is:

> Bi-encoders are fast because documents can be encoded once. Cross-encoders are slower because each query-document pair must be scored jointly, but they are often better rerankers.

This maps well to production RAG: retrieve broadly, then rerank narrowly.

## What the demo shows

A student asks:

```text
What conditions allow medical documentation after a missed exam to qualify for a make-up exam?
```

The first-stage retriever brings back a lexical decoy that shares many words but actually says documentation **does not** create make-up eligibility. The cross-encoder-style reranker promotes the policy that jointly satisfies the important conditions: missed exam, illness, medical documentation, 48-hour deadline, and make-up eligibility.

## Run the local classroom demo

```bash
pip install -r requirements.txt
python cross_encoder_rerank_demo.py
```

## Optional: run a real cross-encoder model

This requires downloading a Hugging Face model, so it may not be suitable for offline classroom use:

```bash
pip install -r requirements_optional_real_model.txt
python optional_real_cross_encoder.py
```

## How to present live

1. Ask students why a fast retriever may return a passage that shares many terms but gives the wrong answer.
2. Show the first-stage ranking.
3. Show the reranked list and the feature breakdown.
4. Emphasize the trade-off: cross-encoders are powerful because they read the query and document together, but the cost grows with the number of candidates.
