"""Validated configuration for optimization and training steps."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool | None = None

    def __post_init__(self) -> None:
        for name in ("learning_rate", "weight_decay", "eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if (not isinstance(self.betas, tuple) or len(self.betas) != 2 or
                any(type(b) not in (int, float) or not math.isfinite(b) or
                    not 0 <= b < 1 for b in self.betas)):
            raise ValueError("betas must be two numbers in [0, 1)")
        if self.fused is not None and type(self.fused) is not bool:
            raise ValueError("fused must be bool or None")


@dataclass(frozen=True)
class StepConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if (type(self.gradient_accumulation_steps) is not int or
                self.gradient_accumulation_steps <= 0):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if self.max_grad_norm is not None:
            value = self.max_grad_norm
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("max_grad_norm must be a positive finite number or None")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be 'fp32' or 'bf16'")
