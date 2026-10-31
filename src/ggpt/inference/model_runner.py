"""Text generation from a loaded inference-only model and tokenizer."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from ggpt.inference.generate import GenerationConfig, generate


class GGPTTextRunner:
    def __init__(self, model, tokenizer, device: str | torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def generate(self, prompt: str, settings: GenerationConfig) -> str:
        config = self.model.config
        ids = [config.bos_token_id]
        ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False).ids)
        if len(ids) > config.max_seq_len:
            raise ValueError(f"Prompt too long: {len(ids)} tokens (max {config.max_seq_len})")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
        with autocast:
            output = generate(self.model, tensor, settings)[0].tolist()
        special = {config.bos_token_id, config.eos_token_id, config.pad_token_id}
        continuation = []
        for token in output[len(ids):]:
            if token == config.eos_token_id:
                break
            if token not in special:
                continuation.append(token)
        return self.tokenizer.decode(continuation, skip_special_tokens=True)
