"""Generation using only forward(), config and standard nn.Module APIs."""
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if type(self.do_sample) is not bool:
            raise ValueError("do_sample must be boolean")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive; use do_sample=False for greedy")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")


def generate(model: nn.Module, input_ids: Tensor,
             config: GenerationConfig | None = None) -> Tensor:
    """Return prompt + continuation. Batched prompts must have equal unpadded length.

    Finished rows receive PAD while other rows continue; EOS itself is retained.
    A prompt ending in EOS is already finished. Earlier EOS tokens are allowed.
    """
    config = config or GenerationConfig()
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or 0 in input_ids.shape:
        raise ValueError("input_ids must be a nonempty torch.long [B, T]")
    model_config = model.config
    context_length = model_config.max_seq_len
    if context_length <= 0:
        raise ValueError("model.config.max_seq_len must be positive")
    device = next(model.parameters()).device
    ids = input_ids.to(device=device)
    if bool(((ids < 0) | (ids >= model_config.vocab_size)).any()):
        raise ValueError("prompt contains out-of-vocabulary IDs")
    generator = torch.Generator(device=device).manual_seed(config.seed)
    # Preserve even a deliberately mixed train/eval state of submodules.
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            batch, prompt_length = ids.shape
            result = torch.full((batch, prompt_length + config.max_new_tokens),
                                model_config.pad_token_id, dtype=torch.long, device=device)
            result[:, :prompt_length] = ids
            finished = ids[:, -1] == model_config.eos_token_id
            length = prompt_length
            for _ in range(config.max_new_tokens):
                if bool(finished.all()):
                    break
                window = result[:, max(0, length - context_length):length]
                output = model(input_ids=window, targets=None)
                logits = output.logits[:, -1, :].float()
                active = ~finished
                scores = logits[active]
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("model returned nonfinite logits")
                if config.do_sample:
                    # Subtract maximum before temperature scaling for stability.
                    scores = (scores - scores.max(dim=-1, keepdim=True).values) / config.temperature
                    if config.top_k is not None:
                        values, indices = scores.topk(min(config.top_k, scores.shape[-1]), dim=-1)
                        sampled = torch.multinomial(values.softmax(-1), 1, generator=generator)
                        chosen = indices.gather(-1, sampled).squeeze(-1)
                    else:
                        chosen = torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)
                else:
                    chosen = scores.argmax(dim=-1)
                next_ids = torch.full((batch,), model_config.pad_token_id,
                                      dtype=torch.long, device=device)
                next_ids[active] = chosen
                result[:, length] = next_ids
                length += 1
                finished |= next_ids == model_config.eos_token_id
            return result[:, :length].clone()
    finally:
        for module, training in modes:
            module.training = training


def generate_text(model: nn.Module, prompt: str, tokenizer_path: str | Path,
                  config: GenerationConfig | None = None, *, add_bos: bool = True) -> str:
    """Load the existing Hugging Face tokenizers JSON; return decoded full sequence."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary size differs from model configuration")
    for token_id in (model.config.bos_token_id, model.config.eos_token_id,
                     model.config.pad_token_id):
        if tokenizer.id_to_token(token_id) is None:
            raise ValueError("tokenizer is missing a configured special token ID")
    # Disable JSON postprocessing so it cannot append EOS or duplicate BOS.
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if add_bos:
        ids.insert(0, model.config.bos_token_id)
    if not ids:
        raise ValueError("empty prompt requires add_bos=True")
    output = generate(model, torch.tensor([ids], dtype=torch.long), config)
    special_ids = {model.config.bos_token_id, model.config.eos_token_id, model.config.pad_token_id}
    decoded_ids = [i for i in output[0].tolist() if i not in special_ids]
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)
