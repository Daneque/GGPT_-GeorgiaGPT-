"""Validated, immutable model configuration."""
from dataclasses import asdict, dataclass
import json
import math
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "llama"
    vocab_size: int = 16_384
    num_layers: int = 10
    hidden_size: int = 512
    num_heads: int = 8
    intermediate_size: int = 1376
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("model_type must be a nonempty string")
        for name in ("vocab_size", "num_layers", "hidden_size", "num_heads",
                     "intermediate_size", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        for name in ("rope_theta", "dropout", "rms_norm_eps", "initializer_range"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0 or self.initializer_range <= 0:
            raise ValueError("theta, epsilon and initializer_range must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        ids = (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("special token IDs must lie inside the vocabulary")
        if len(set(ids)) != 3:
            raise ValueError("special token IDs must be distinct")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "ModelConfig":
        values = json.loads(value)
        if not isinstance(values, dict):
            raise ValueError("configuration JSON must be an object")
        return cls.from_dict(values)

