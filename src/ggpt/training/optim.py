"""AdamW setup with explicit decay policy and tied-weight deduplication."""
import torch
from torch import nn
from .config import OptimizerConfig


def create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for _, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    if not seen:
        raise ValueError("model has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    params = decay + no_decay
    fused = config.fused if config.fused is not None else all(p.is_cuda for p in params)
    if fused and not all(p.is_cuda for p in params):
        raise ValueError("fused=True requires all trainable parameters on CUDA")
    return torch.optim.AdamW(groups, lr=config.learning_rate,
                             betas=config.betas, eps=config.eps, fused=fused)
