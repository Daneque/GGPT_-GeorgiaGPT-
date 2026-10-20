from torch import nn
from .config import ModelConfig
from .llama import LlamaLM


def create_model(config: ModelConfig) -> nn.Module:
    if config.model_type == "llama":
        return LlamaLM(config)
    raise ValueError(f"Unknown model_type: {config.model_type!r}")
