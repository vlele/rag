# 08 — RAPTOR Hierarchical Retrieval Demo

## What this demonstrates
RAPTOR organizes a corpus into a **tree of meaning**:

1. Start with detailed leaf chunks.
2. Group related chunks.
3. Summarize each group into a higher-level node.
4. Repeat recursively so retrieval can search both details and abstractions.

This demo uses TF-IDF + k-means clustering as a lightweight stand-in for embedding clustering
and an extractive summarizer as a stand-in for LLM-generated abstractive summaries. The goal is
to make the architecture visible in a live classroom, not to reproduce a production RAPTOR system.

## Scenario
The corpus mixes three topics:
- Apple revenue growth
- Atlas migration failure
- Security/compliance operations

A flat retriever returns individual chunks. RAPTOR first retrieves a relevant summary node,
then descends into the matching branch to recover details. This lets students see why a
hierarchical index is useful for questions that ask for themes, summaries, or broad explanations.

## How to run
```bash
python raptor_demo.py
```

Try custom queries:
```bash
python raptor_demo.py --query "What were the main drivers of Apple revenue growth?"
python raptor_demo.py --query "Why did the Atlas migration fail?"
```

## What to point out live
- Flat retrieval treats all chunks as equal.
- RAPTOR adds searchable summaries above the chunks.
- The summary node gives orientation; leaf nodes give evidence.
- Summary quality matters: bad summaries can route retrieval to the wrong branch.

## Teaching note
The real RAPTOR idea is not merely summarization. The important idea is **recursive abstraction as part of the retrieval index**.
