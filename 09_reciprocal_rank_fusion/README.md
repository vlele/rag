# Demo 09 — Reciprocal Rank Fusion (RRF)

## Concept demonstrated

This demo isolates **Reciprocal Rank Fusion**, the ranking method commonly used in RAG-Fusion style systems. Instead of trusting one query or averaging raw similarity scores, RRF gives a document small amounts of credit every time it appears in a ranked list:

```text
RRF_score(doc) = sum over rankings of 1 / (k + rank(doc))
```

A document that appears near the top across several query formulations can beat a document that appears first for only one narrow formulation.

## Why this matters in the talk

Use this after explaining RAG-Fusion. The teaching point is:

> The user's first wording is not always the best retrieval query. Multi-query retrieval improves recall, and RRF consolidates the evidence without relying on raw similarity scores that may not be calibrated across query variants.

## What the demo shows

The scenario mirrors the financial-services failure pattern from the source material: an employee asks about a denied applicant whose credit score improved after six months. A single query may surface a broad older waiting-period policy. Several alternate query formulations repeatedly surface the more specific fast-track memo.

The script prints:

1. A single-query baseline ranking.
2. The ranking for each generated query variant.
3. The fused RRF ranking.
4. A simple answer based on the top fused document.

## Run

```bash
pip install -r requirements.txt
python rrf_demo.py
```

Try changing the RRF constant:

```bash
python rrf_demo.py --rrf-k 10
python rrf_demo.py --rrf-k 60
```

Lower values reward top ranks more aggressively. Higher values make repeated appearance across lists matter more evenly.

## How to present live

1. Ask students: “Which result should we trust: the highest score from one query, or the document that keeps showing up across multiple query attempts?”
2. Run the single-query baseline.
3. Run the fusion output and point out that the specific memo wins because it is repeatedly supported across rankings.
4. Emphasize that RRF is cheap, transparent, and useful before heavier reranking.
