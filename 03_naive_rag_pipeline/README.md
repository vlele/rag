# 03 — Naive RAG Pipeline

## What this demonstrates
This project is a tiny, local RAG pipeline:
1. load documents
2. chunk them
3. retrieve top-k chunks
4. synthesize a sourced answer

It is inspired by the companion repo notebook:
- `chapter4/local_rag_chain.ipynb`

## Why it is useful in the talk
Use this project to make the phrase **retrieve-then-generate** concrete.

## How to run
```bash
python rag_pipeline.py
```

You can also show the stuffed prompt-like context:
```bash
python rag_pipeline.py --show-context
```

## What to point out live
- This is **naive RAG**: retrieve top-k chunks, then synthesize from them.
- The output includes sources.
- The answer quality depends on the retrieval stage.

## Teaching note
The synthesizer is intentionally lightweight so the demo runs anywhere. The architecture is the lesson.
