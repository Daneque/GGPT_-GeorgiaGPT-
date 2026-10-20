from dataclasses import dataclass
from torch import Tensor


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
