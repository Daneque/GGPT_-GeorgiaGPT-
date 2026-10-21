import pytest
import torch

from ggpt.model import ModelConfig, create_model
from ggpt.training import OptimizerConfig, StepConfig, create_optimizer, train_step


def tiny_model():
    return create_model(ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                                    intermediate_size=24, num_layers=2, max_seq_len=8))


def test_accumulation_matches_combined_batch_and_updates_weights():
    torch.manual_seed(2026)
    split_model = tiny_model()
    combined_model = tiny_model()
    combined_model.load_state_dict(split_model.state_dict())
    b1 = {'input_ids': torch.randint(0, 32, (2, 5)),
          'targets': torch.randint(0, 32, (2, 5))}
    b2 = {'input_ids': torch.randint(0, 32, (1, 5)),
          'targets': torch.randint(0, 32, (1, 5))}
    b1['targets'][0, 0] = -100
    b2['targets'][0, 1:4] = -100
    merged = {key: torch.cat((b1[key], b2[key]), dim=0) for key in b1}
    opt_config = OptimizerConfig(fused=False, weight_decay=0, learning_rate=1e-3)
    o1 = create_optimizer(split_model, opt_config)
    o2 = create_optimizer(combined_model, opt_config)
    before = split_model.token_embeddings.weight.detach().clone()
    separate = train_step(split_model, o1, [b1, b2], 'cpu',
                          StepConfig(gradient_accumulation_steps=2))
    together = train_step(combined_model, o2, [merged], 'cpu', StepConfig())
    assert separate.tokens == together.tokens == 11
    assert separate.microbatches == 2 and together.microbatches == 1
    assert separate.loss == pytest.approx(together.loss, abs=1e-6)
    assert separate.grad_norm == pytest.approx(together.grad_norm, rel=1e-4, abs=1e-5)
    assert not torch.equal(before, split_model.token_embeddings.weight)
    for a, b in zip(split_model.parameters(), combined_model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
    assert all(p.grad is None for p in split_model.parameters())


def test_partial_accumulation_group_and_ignored_only_batch():
    model = tiny_model()
    opt = create_optimizer(model, OptimizerConfig(fused=False))
    x = torch.randint(0, 32, (1, 4))
    ignored = {'input_ids': x, 'targets': torch.full_like(x, -100)}
    valid = {'input_ids': x, 'targets': torch.roll(x, -1, 1)}
    config = StepConfig(gradient_accumulation_steps=4, max_grad_norm=None)
    result = train_step(model, opt, [ignored, valid], 'cpu', config)
    assert result.tokens == 4 and result.microbatches == 2
    assert result.loss > 0
    with pytest.raises(ValueError, match='no valid'):
        train_step(model, opt, [ignored], 'cpu', config)
    with pytest.raises(ValueError, match='number of microbatches'):
        train_step(model, opt, [valid] * 5, 'cpu', config)


@pytest.mark.parametrize('changes', [
    {'gradient_accumulation_steps': 0}, {'gradient_accumulation_steps': 2.5},
    {'max_grad_norm': 0}, {'max_grad_norm': float('nan')}, {'precision': 'fp16'},
])
def test_invalid_step_config(changes):
    with pytest.raises(ValueError):
        StepConfig(**changes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bf16_backward_and_fused_adamw():
    if not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device lacks BF16')
    model = tiny_model().cuda()
    optimizer = create_optimizer(model, OptimizerConfig(fused=True))
    x = torch.randint(0, 32, (2, 5), dtype=torch.long)
    result = train_step(model, optimizer, [{'input_ids': x, 'targets': x}],
                        'cuda', StepConfig(precision='bf16'))
    assert result.tokens == 10
    assert torch.isfinite(torch.tensor(result.loss))
    assert torch.isfinite(torch.tensor(result.grad_norm))
