# Python Demo Projects for the 90-Minute RAG Talk

This bundle contains six classroom-friendly demo projects aligned with the attached source materials and the companion GitHub repository.

## Project Index

1. **01_semantic_search_foundations**  
   Demonstrates vocabulary mismatch, lexical retrieval, and simple semantic retrieval.  
   **Talk fit:** Retrieval foundations (Word2Vec / Sentence-BERT intuition)  
   **Repo inspiration:** `chapter2/w2v_vs_bm25.ipynb`, `chapter2/sentence_transformers.ipynb`

2. **02_ann_scale_demo**  
   Demonstrates why approximate nearest-neighbor style search matters at scale.  
   **Talk fit:** FAISS intuition and scaling  
   **Repo inspiration:** `chapter2/search_with_faiss.ipynb`

3. **03_naive_rag_pipeline**  
   A tiny, fully local RAG pipeline with chunking, retrieval, answer synthesis, and source display.  
   **Talk fit:** Canonical vs naive RAG, retrieve-then-generate flow  
   **Repo inspiration:** `chapter4/local_rag_chain.ipynb`

4. **04_fid_multidoc_fusion**  
   A toy FiD-style multi-document fusion demo, plus an optional T5-based script modeled on the repo chapter 5 code.  
   **Talk fit:** Fusion-in-Decoder  
   **Repo inspiration:** `chapter5/data.py`, `chapter5/encoding_decoding.py`, `chapter5/fid.py`, `chapter5/fine-tuning.py`

5. **05_multi_query_self_reflective_rag**  
   Demonstrates multi-query retrieval, reciprocal-rank-style fusion, planning via decomposition, and a simple critique/retry loop.  
   **Talk fit:** RAG-Fusion, Self-RAG, planning/correction

6. **06_graph_rag_reasoning**  
   Demonstrates graph-based retrieval over a small incident/outage knowledge graph.  
   **Talk fit:** GraphRAG and multi-hop reasoning

## Recommended use in class
- **Fastest live demos:** 01, 03, 05, 06
- **Best conceptual demo:** 04
- **Best “why scale matters” demo:** 02

## Setup
Each project includes its own `requirements.txt`.

Most projects only need:
- numpy
- scikit-learn
- networkx

Project 4 also includes an **optional** script that uses `transformers` if you want a closer match to the companion repository.

## Teaching note
These projects are deliberately simplified for live instruction:
- small local datasets
- deterministic or lightweight synthesis
- clear printed output
- minimal hidden framework magic

They are not meant to be production systems. They are meant to make the architecture legible.

