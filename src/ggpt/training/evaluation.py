"""Token-weighted validation for models exposing output.loss."""
from collections.abc import Iterable, Mapping
from itertools import islice
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .precision import precision_context, validate_precision


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float
    tokens: int
    batches: int


def evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, Tensor]],
    device: torch.device | str,
    *,
    precision: str = "fp32",
    max_batches: int | None = None,
) -> EvaluationResult:
    """Evaluate mean CE per non-ignored token, not per batch.

    Returns perplexity=exp(loss); overflow becomes infinity.
    Restores original train/eval flags even if validation raises.
    """
    device = torch.device(device)
    validate_precision(device, precision)
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError("max_batches must be a positive integer or None")

    original_modes = [(module, module.training) for module in model.modules()]
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0
    try:
        model.eval()
        with torch.inference_mode():
            selected = islice(loader, max_batches) if max_batches is not None else loader
            for batch in selected:
                batches_seen += 1
                targets_cpu = batch["targets"]
                if not isinstance(targets_cpu, Tensor):
                    raise TypeError("batch['targets'] must be a Tensor")
                valid = int((targets_cpu != -100).sum().item())
                if valid == 0:
                    continue
                inputs = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
                targets = targets_cpu.to(device, non_blocking=device.type == "cuda")
                with precision_context(device, precision):
                    output = model(input_ids=inputs, targets=targets)
                if output.loss is None:
                    raise RuntimeError("model returned no validation loss")
                loss = float(output.loss.detach().float().item())
                if not math.isfinite(loss):
                    raise ValueError("nonfinite validation loss")
                total_loss += loss * valid
                total_tokens += valid
    finally:
        for module, mode in original_modes:
            module.training = mode

    if total_tokens == 0:
        raise ValueError("validation has no valid target tokens")
    mean_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = math.inf
    return EvaluationResult(
        loss=mean_loss,
        perplexity=perplexity,
        tokens=total_tokens,
        batches=batches_seen,
    )
