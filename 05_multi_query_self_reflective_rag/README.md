# 05 — Multi-Query + Self-Reflective RAG

## What this demonstrates
This project combines ideas from:
- **RAG-Fusion** — retrieve from multiple query formulations
- **Self-RAG** — critique whether the answer is sufficiently supported
- **Planning / correction** — decompose a broad question into evidence needs

## Scenario
The user asks:
> Why did the Atlas migration project fail?

A single retrieval query tends to be shallow.
The multi-query version decomposes the problem into:
- timeline
- ownership
- technical obstacles
- stakeholder decisions

Then a critique step checks whether those aspects are covered.

## How to run
```bash
python adaptive_rag.py
```

Try the three modes:
```bash
python adaptive_rag.py --mode single
python adaptive_rag.py --mode fusion
python adaptive_rag.py --mode reflective
```

## What to point out live
- Multi-query retrieval improves **coverage**
- The critique loop improves **completeness**
- Planning is often just query decomposition made explicit

## Teaching note
This is a deliberately transparent implementation. It is not trying to hide the control logic inside an LLM.
