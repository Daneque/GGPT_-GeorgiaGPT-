"""Autocast policy for FP32 and BF16 training."""
from contextlib import contextmanager
from collections.abc import Iterator
import torch


def validate_precision(device: torch.device | str, precision: str) -> None:
    device = torch.device(device)
    if precision == "fp32":
        return
    if precision != "bf16":
        raise ValueError(f"Unsupported precision: {precision!r}")
    if device.type != "cuda":
        raise ValueError("BF16 training is supported only on CUDA in GGPT")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")


@contextmanager
def precision_context(device: torch.device | str,
                      precision: str) -> Iterator[None]:
    validate_precision(device, precision)
    if precision == "fp32":
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
