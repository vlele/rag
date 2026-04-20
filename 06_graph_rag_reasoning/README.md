# 06 — Graph RAG Reasoning

## What this demonstrates
This project shows the difference between:
- **flat retrieval** over disconnected text snippets
- **graph-based retrieval** over connected facts

It is inspired by the GraphRAG and planning/correction material in the attached text chapters.

## Scenario
The user asks:
> How did the vendor patch lead to students being locked out of the portal?

The answer is not in one chunk.
It is in a **path**:
vendor patch → authentication service → identity cache → MFA rule → student portal

## How to run
```bash
python graph_rag_demo.py
```

## What to point out live
- Flat retrieval gives fragments.
- Graph retrieval gives a causal chain.
- GraphRAG is useful when relationships and dependencies are the answer.

## Teaching note
Keep repeating: *the answer is in the connection, not only in the chunk*.
