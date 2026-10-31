"""Warmup/cosine scheduler and smooth continuation after a finished run."""

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
        raise ValueError("min_lr_ratio must be finite in [0, 1]")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def continuation_lr_multiplier(step: int, *, continuation_from: int,
                               total_steps: int, ramp_steps: int,
                               starting_ratio: float, peak_ratio: float,
                               final_ratio: float) -> float:
    """Keep the prior final LR, ramp gently, then cosine-decay to final LR.

    All ratios are relative to the *original* AdamW base LR. The initial
    point matches the preceding run's final LR exactly.
    """
    if not 0 <= continuation_from < total_steps:
        raise ValueError("continuation_from must be in [0, total_steps)")
    if ramp_steps < 1 or continuation_from + ramp_steps >= total_steps:
        raise ValueError("ramp_steps must fit inside the continuation")
    if not all(math.isfinite(r) and 0 <= r <= 1 for r in
               (starting_ratio, peak_ratio, final_ratio)):
        raise ValueError("LR ratios must be finite in [0, 1]")
    if not final_ratio <= starting_ratio <= peak_ratio:
        raise ValueError("require final_ratio <= starting_ratio <= peak_ratio")
    if step <= continuation_from:
        return starting_ratio
    if step < continuation_from + ramp_steps:
        ratio = (step - continuation_from) / ramp_steps
        return starting_ratio + (peak_ratio - starting_ratio) * ratio
    t = min(1.0, (step - continuation_from - ramp_steps)
            / (total_steps - continuation_from - ramp_steps))
    return final_ratio + (peak_ratio - final_ratio) * (1 + math.cos(math.pi * t)) / 2


def create_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int,
                     warmup_steps: int, min_lr_ratio: float = 0.1,
                     continuation_from: int = 0,
                     continuation_start_ratio: float | None = None,
                     continuation_ramp_steps: int = 1000,
                     continuation_peak_ratio: float = 0.3,
                     continuation_final_ratio: float = 0.05
                     ) -> torch.optim.lr_scheduler.LambdaLR:
    # Validate the standard schedule in all cases.
    lr_multiplier(0, total_steps=total_steps, warmup_steps=warmup_steps,
                  min_lr_ratio=min_lr_ratio)
    if continuation_from:
        start_ratio = (min_lr_ratio if continuation_start_ratio is None
                       else continuation_start_ratio)
        continuation_lr_multiplier(
            continuation_from, continuation_from=continuation_from,
            total_steps=total_steps, ramp_steps=continuation_ramp_steps,
            starting_ratio=start_ratio,
            peak_ratio=continuation_peak_ratio,
            final_ratio=continuation_final_ratio,
        )
        multiplier = lambda step: continuation_lr_multiplier(
            step, continuation_from=continuation_from,
            total_steps=total_steps, ramp_steps=continuation_ramp_steps,
            starting_ratio=start_ratio,
            peak_ratio=continuation_peak_ratio,
            final_ratio=continuation_final_ratio,
        )
    else:
        multiplier = lambda step: lr_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
