"""Synchronized optimizer-step timing and PyTorch CUDA allocator measurements."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics
import time

import torch
from torch import Tensor, nn

from .config import StepConfig
from .steps import train_step


@dataclass(frozen=True)
class BenchmarkResult:
    measured_steps: int
    warmup_steps: int
    tokens: int
    mean_step_seconds: float
    median_step_seconds: float
    tokens_per_second: float
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


def benchmark_train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_batches: Sequence[Sequence[Mapping[str, Tensor]]],
    device: torch.device | str,
    config: StepConfig,
    *,
    warmup_steps: int = 1,
    measured_steps: int = 3,
) -> BenchmarkResult:
    """Benchmark actual train_step updates on *already prepared* CPU batches.

    Does not include DataLoader reading/sampling time. Updates model weights.
    CUDA memory peaks include the model, optimizer state and transient tensors;
    they are not total device occupancy as reported by nvidia-smi.
    """
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if type(measured_steps) is not int or measured_steps <= 0:
        raise ValueError("measured_steps must be a positive integer")
    expected = warmup_steps + measured_steps
    if len(update_batches) != expected:
        raise ValueError(f"expected {expected} update groups, got {len(update_batches)}")

    device = torch.device(device)
    is_cuda = device.type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    def synchronize() -> None:
        if is_cuda:
            torch.cuda.synchronize(device)

    for group in update_batches[:warmup_steps]:
        train_step(model, optimizer, group, device, config)
    synchronize()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    total_tokens = 0
    for group in update_batches[warmup_steps:]:
        synchronize()
        start = time.perf_counter()
        result = train_step(model, optimizer, group, device, config)
        synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += result.tokens

    total_time = sum(times)
    return BenchmarkResult(
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        tokens=total_tokens,
        mean_step_seconds=statistics.mean(times),
        median_step_seconds=statistics.median(times),
        tokens_per_second=total_tokens / total_time,
        peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                            if is_cuda else None),
        peak_reserved_mib=(torch.cuda.max_memory_reserved(device) / 2**20
                           if is_cuda else None),
    )
