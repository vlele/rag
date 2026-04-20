"""
Optional demo 04B: a closer, repo-style FiD implementation using T5.

This script adapts the structure of the companion repository's chapter 5 files:
- data.py
- encoding_decoding.py
- fid.py

It requires:
- transformers
- torch
- sentencepiece (T5Tokenizer loads SPM vocab files)

It is intentionally minimal and may download a model the first time you run it.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from transformers.modeling_outputs import BaseModelOutput
except Exception as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "This optional script requires `transformers` and `torch`.\n"
        "Install them first, then rerun the script."
    ) from exc

try:
    import sentencepiece  # noqa: F401 - required when instantiating T5Tokenizer
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "T5Tokenizer needs the `sentencepiece` package.\n"
        "Install: pip install sentencepiece"
    ) from exc


PROMPT_TEMPLATE = (
    "Synthesize an answer for the question by combining relevant information only.\n\n"
    "Title: {title}\n"
    "Context: {context}\n\n"
    "Question: {question}"
)


def encode_passages(question: str, passages: list[dict], tokenizer, model):
    inputs = [
        PROMPT_TEMPLATE.format(title=p["title"], context=p["text"], question=question)
        for p in passages
    ]
    encoded = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")
    encoder_outputs = model.encoder(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        return_dict=True,
    )
    return encoder_outputs.last_hidden_state, encoded.attention_mask


def decode_answer(encoder_hidden_states, attention_mask, tokenizer, model) -> str:
    num_passages, seq_len, hidden_dim = encoder_hidden_states.shape
    fused_hidden_states = encoder_hidden_states.reshape(1, num_passages * seq_len, hidden_dim)
    fused_attention_mask = attention_mask.reshape(1, num_passages * seq_len)
    encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden_states)
    generated = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=fused_attention_mask,
        max_length=64,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    data_path = Path(__file__).parent / "data" / "cases.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))["disambiguation_case"]

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    question = payload["question"]
    passages = payload["passages"]

    print(f"Question: {question}\n")
    hidden, mask = encode_passages(question, passages, tokenizer, model)
    answer = decode_answer(hidden, mask, tokenizer, model)

    print("Repo-style FiD answer:")
    print(answer)


if __name__ == "__main__":
    main()
