"""
Optional demo 04C: a single FiD-style fine-tuning step.

This file mirrors the structure of the companion repository's `fine-tuning.py`.
It is optional because it requires `transformers` and `torch`.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
    import torch.optim as optim
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except Exception as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "This optional script requires `transformers` and `torch`.\n"
        "Install them first, then rerun the script."
    ) from exc

from optional_real_t5_fid import encode_passages  # reuse the repo-style helper


def main() -> None:
    data_path = Path(__file__).parent / "data" / "cases.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))["disambiguation_case"]

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    question = payload["question"]
    passages = payload["passages"]
    target_answer = "To reduce the training time to a feasible six-week window."

    optimizer.zero_grad()
    hidden_states, attention_mask = encode_passages(question, passages, tokenizer, model)

    labels = tokenizer(target_answer, return_tensors="pt").input_ids
    num_passages, seq_len, hidden_dim = hidden_states.shape
    fused_hidden = hidden_states.reshape(1, num_passages * seq_len, hidden_dim)
    fused_mask = attention_mask.reshape(1, num_passages * seq_len)

    outputs = model(
        encoder_outputs=(fused_hidden,),
        attention_mask=fused_mask,
        labels=labels,
    )
    loss = outputs.loss
    print(f"Calculated loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()
    print("Completed one FiD-style fine-tuning step.")


if __name__ == "__main__":
    main()
