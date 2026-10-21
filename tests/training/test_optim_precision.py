import pytest
import torch
from torch import nn

from ggpt.model import ModelConfig, create_model
from ggpt.training import OptimizerConfig, create_optimizer, precision_context, validate_precision


def tiny_model():
    return create_model(ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                                    intermediate_size=24, num_layers=2, max_seq_len=8))


def test_optimizer_groups_and_tied_weight_not_duplicated():
    model = tiny_model()
    optimizer = create_optimizer(model, OptimizerConfig(fused=False, weight_decay=0.12))
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len({id(p) for p in params}) == len(list(model.parameters()))
    assert sum(id(p) == id(model.token_embeddings.weight) for p in params) == 1
    assert len(optimizer.param_groups) == 2
    for group in optimizer.param_groups:
        if group['weight_decay']:
            assert group['weight_decay'] == 0.12
            assert all(p.ndim >= 2 for p in group['params'])
        else:
            assert all(p.ndim == 1 for p in group['params'])


def test_optimizer_rejects_frozen_and_fused_cpu():
    model = tiny_model()
    with pytest.raises(ValueError, match='CUDA'):
        create_optimizer(model, OptimizerConfig(fused=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match='no trainable'):
        create_optimizer(model, OptimizerConfig(fused=False))


@pytest.mark.parametrize('changes', [
    {'learning_rate': 0}, {'weight_decay': -1}, {'eps': float('inf')},
    {'betas': (0.9, 1)}, {'fused': 'yes'},
])
def test_invalid_optimizer_config(changes):
    with pytest.raises(ValueError):
        OptimizerConfig(**changes)


def test_fp32_precision_context_cpu():
    validate_precision('cpu', 'fp32')
    with precision_context('cpu', 'fp32'):
        value = torch.ones(2) + 1
    assert value.dtype == torch.float32
    with pytest.raises(ValueError, match='CUDA'):
        validate_precision('cpu', 'bf16')
    with pytest.raises(ValueError, match='Unsupported'):
        validate_precision('cpu', 'fp16')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bf16_context():
    if not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device lacks BF16')
    with precision_context('cuda', 'bf16'):
        result = torch.ones(4, 4, device='cuda') @ torch.ones(4, 4, device='cuda')
    assert result.dtype == torch.bfloat16
