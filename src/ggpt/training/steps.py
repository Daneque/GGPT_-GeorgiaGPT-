"""One optimizer update, possibly across several microbatches."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from .config import StepConfig
from .precision import precision_context, validate_precision


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    grad_norm: float
    tokens: int
    microbatches: int


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Mapping[str, Tensor]],
    device: torch.device | str,
    config: StepConfig,
) -> TrainStepResult:
    """Run one update with token-weighted microbatch losses.

    Caller tracks consumed DataLoader batches for checkpointing.
    """
    if not 1 <= len(batches) <= config.gradient_accumulation_steps:
        raise ValueError("number of microbatches must be between 1 and gradient_accumulation_steps")
    device = torch.device(device)
    validate_precision(device, config.precision)
    valid_counts: list[int] = []
    for batch in batches:
        targets = batch["targets"]
        if not isinstance(targets, Tensor):
            raise TypeError("batch['targets'] must be a Tensor")
        valid_counts.append(int((targets != -100).sum().item()))
    total_tokens = sum(valid_counts)
    if total_tokens == 0:
        raise ValueError("no valid target tokens in the update")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    try:
        for batch, count in zip(batches, valid_counts, strict=True):
            if count == 0:
                continue
            inputs = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
            targets = batch["targets"].to(device, non_blocking=device.type == "cuda")
            with precision_context(device, config.precision):
                output = model(input_ids=inputs, targets=targets)
                if output.loss is None:
                    raise RuntimeError("model returned no loss for the supplied targets")
                loss = output.loss
            (loss * (count / total_tokens)).backward()
            total_loss += loss.detach().float() * count
        grad_norm_tensor = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(config.max_grad_norm if config.max_grad_norm is not None else math.inf),
            error_if_nonfinite=True,
        )
        optimizer.step()
        result = TrainStepResult(
            loss=(total_loss / total_tokens).item(),
            grad_norm=grad_norm_tensor.item(),
            tokens=total_tokens,
            microbatches=len(batches),
        )
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    return result
