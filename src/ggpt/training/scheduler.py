"""Token-update step warmup followed by cosine learning-rate decay."""

import math

import torch


def lr_multiplier(step: int, *, total_steps: int, warmup_steps: int,
                  min_lr_ratio: float) -> float:
    """Scale base LR before update number `step` (zero-based)."""
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def create_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int,
                     warmup_steps: int, min_lr_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    # Validate immediately rather than waiting for the first scheduler step.
    lr_multiplier(0, total_steps=total_steps, warmup_steps=warmup_steps,
                  min_lr_ratio=min_lr_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
