"""Teacher-forced next-token and complete-answer probabilities, no generation bias."""

from __future__ import annotations

import math
from typing import Any

import torch

from .v3_dataset import V3Case


def encode_prompt(tokenizer: Any, model: Any, prompt: str) -> list[int]:
    ids = [model.config.bos_token_id]
    ids.extend(tokenizer.encode(prompt, add_special_tokens=False).ids)
    if len(ids) > model.config.max_seq_len:
        raise ValueError(f"Prompt too long ({len(ids)} tokens): {prompt[:40]!r}")
    return ids


def _model_logits(model: Any, ids: list[int], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(tensor, targets=None).logits[0].float()
    if not torch.isfinite(output).all():
        raise ValueError("Non-finite logits returned")
    return output


def _token_top10(logits: torch.Tensor, tokenizer: Any) -> list[dict]:
    log_probs = logits.log_softmax(-1)
    values, ids = log_probs.topk(min(10, log_probs.numel()))
    return [
        {
            "id": int(idx),
            "token": tokenizer.id_to_token(int(idx)),
            "decoded": tokenizer.decode([int(idx)], skip_special_tokens=False),
            "probability": math.exp(float(value)),
            "log_probability": float(value),
        }
        for value, idx in zip(values.tolist(), ids.tolist())
    ]


def score_prompt_choices(model: Any, tokenizer: Any, prompt: str,
                         choices: tuple[str, ...], reference: str,
                         device: torch.device) -> dict:
    """Score tokenized suffixes conditioned on tokenized prompt + BOS.

    Choices are literal continuations (with a leading space), encoded separately
    with special tokens disabled. This defines a precise token-level task; results
    are not interpreted as normalized probabilities over all possible text strings.
    """
    prefix = encode_prompt(tokenizer, model, prompt)
    device = torch.device(device)
    if not choices or reference not in choices:
        raise ValueError("Reference must belong to nonempty choices")
    with torch.inference_mode():
        initial_logits = _model_logits(model, prefix, device)[-1]
        initial_lp = initial_logits.log_softmax(-1)
        top10 = _token_top10(initial_logits, tokenizer)
        scored = []
        for answer in choices:
            suffix = tokenizer.encode(answer, add_special_tokens=False).ids
            if not suffix:
                raise ValueError(f"Empty tokenized candidate: {answer!r}")
            if len(prefix) + len(suffix) - 1 > model.config.max_seq_len:
                raise ValueError("Prompt plus candidate exceeds model context window")
            if any(t in (model.config.bos_token_id, model.config.pad_token_id)
                   for t in suffix):
                raise ValueError("Candidate contains a forbidden special token")
            if len(suffix) == 1:
                parts = [float(initial_lp[suffix[0]].item())]
            else:
                sequence_logits = _model_logits(model, prefix + suffix[:-1], device)
                position = len(prefix) - 1
                positions = sequence_logits[position:position + len(suffix)]
                # positions contains one prediction per candidate token
                log_p = positions.log_softmax(-1)
                parts = log_p.gather(
                    dim=-1,
                    index=torch.tensor(suffix, device=device).unsqueeze(1),
                ).squeeze(1).tolist()
            total = float(sum(parts))
            first_id = suffix[0]
            first_rank = 1 + int((initial_logits > initial_logits[first_id]).sum().item())
            scored.append({
                "text": answer,
                "token_ids": suffix,
                "token_count": len(suffix),
                "log_probability": total,
                "mean_log_probability": total / len(suffix),
                "first_token_rank_vocab": first_rank,
                "token_log_probabilities": [float(p) for p in parts],
            })
        correct = next(entry for entry in scored if entry["text"] == reference)
        rank = 1 + sum(entry["log_probability"] > correct["log_probability"]
                       for entry in scored)
        mean_rank = 1 + sum(entry["mean_log_probability"] > correct["mean_log_probability"]
                            for entry in scored)
        return {
            "top10_next_tokens": top10,
            "reference": correct,
            "reference_rank_by_sum": rank,
            "reference_rank_by_mean": mean_rank,
            "reference_top1": rank == 1,
            "candidates": scored,
        }


def score_case(model: Any, tokenizer: Any, case: V3Case,
               device: str | torch.device) -> dict | None:
    if case.category == "continuation":
        return None
    result = score_prompt_choices(
        model, tokenizer, case.prompt, case.candidates, case.reference, torch.device(device)
    )
    if case.control_prompt is not None:
        baseline = score_prompt_choices(
            model, tokenizer, case.control_prompt,
            case.candidates, case.reference, torch.device(device)
        )
        result["control"] = baseline
        result["reference_logprob_gain_from_context"] = (
            result["reference"]["log_probability"]
            - baseline["reference"]["log_probability"]
        )
        result["reference_rank_gain_from_context"] = (
            baseline["reference_rank_by_sum"] - result["reference_rank_by_sum"]
        )
    return result
