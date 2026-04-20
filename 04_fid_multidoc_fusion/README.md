# 04 — FiD Multi-Document Fusion

## What this demonstrates
This project explains the core intuition behind **Fusion-in-Decoder (FiD)**:
- encode each passage independently
- fuse evidence later during answer generation

It includes:
1. **`toy_fid.py`** — a fast classroom-friendly analogue
2. **`optional_real_t5_fid.py`** — a closer, repo-style script using T5 if you want a real encoder-decoder example
3. **`optional_fine_tuning_step.py`** — a single FiD-style training step modeled on the repo chapter 5 code

## Why it is useful in the talk
Use this project to explain:
- why naive prompt stuffing breaks on scattered evidence
- why FiD is good for multi-document synthesis
- why FiD shifts the retriever goal from perfect rank-1 precision toward stronger recall

## How to run
```bash
python toy_fid.py
```

Optional repo-style script:
```bash
python optional_real_t5_fid.py
```

## What to point out live
- The **top-1 baseline** behaves like a system that commits too early.
- The **FiD-like answer** combines evidence from multiple passages.
- In the Odyssey example, the model also learns to ignore plausible distractors.

## Teaching note
The toy version is intentionally simple and deterministic. It exists so the architecture is easy to explain in class.
