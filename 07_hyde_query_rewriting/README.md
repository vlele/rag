# 07 — HyDE Query Rewriting Demo

## What this demonstrates
HyDE stands for **Hypothetical Document Embeddings**. The teaching idea is simple:
when the user's query is short, vague, or phrased differently from the knowledge base,
first generate a plausible **hypothetical answer/document**, then retrieve with that richer text.

In the book's storyline, HyDE belongs to the query-enhancement family: it helps when the
surface wording of the user's question is not the same as the source-document wording.
For classroom use, this demo shows the mechanism without requiring an external LLM API.

## Scenario
The user asks:

> How can we stop customers from being kicked out during upgrades?

A normal lexical retriever over-focuses on words like **customers** and **upgrades** and may
retrieve release-communication material. HyDE imagines what the answer might say:
**session affinity**, **sticky sessions**, **shared session store**, and **rolling deployments**.
That hypothetical answer is closer to the actual runbook language, so retrieval improves.

## How to run
```bash
python hyde_demo.py
```

Try custom queries:
```bash
python hyde_demo.py --query "How do we keep login sessions alive during deployments?"
python hyde_demo.py --query "Why are users signed out after a release?"
```

## What to point out live
- The user query is not always the best retrieval query.
- HyDE improves retrieval by moving the query toward the language of the answer.
- It is useful for vocabulary mismatch and incorrect-specificity problems.
- It can also introduce wrong assumptions if the hypothetical answer is bad, so production systems need reranking and evaluation.

## Teaching note
This is a deliberately transparent, offline demo. In production, the `generate_hypothetical_document()` function would usually call an LLM.
