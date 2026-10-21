import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ggpt.model import CausalLMOutput, ModelConfig, create_model
from ggpt.training import evaluate


class SyntheticLossModel(nn.Module):
    """Return mean non-ignored target value as loss; track evaluation conditions."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout(0.3)
        self.seen = []

    def forward(self, input_ids, targets=None):
        assert not self.training and not self.child.training
        assert torch.is_inference_mode_enabled()
        valid = targets[targets != -100]
        self.seen.append(len(valid))
        return CausalLMOutput(logits=torch.empty(0, device=input_ids.device),
                              loss=valid.float().mean() + self.anchor * 0)


def batch(values):
    ids = torch.zeros((1, len(values)), dtype=torch.long)
    return {'input_ids': ids, 'targets': torch.tensor([values], dtype=torch.long)}


def test_token_weighted_loss_ppl_and_mode_restoration():
    model = SyntheticLossModel().train()
    model.child.eval()
    original_grad = model.anchor.grad
    output = evaluate(model, [batch([2, 4, -100]), batch([8, -100, -100])], 'cpu')
    assert output.tokens == 3
    assert output.batches == 2
    assert output.loss == pytest.approx(14 / 3)
    assert output.perplexity == pytest.approx(math.exp(14 / 3))
    assert model.training and not model.child.training
    assert model.anchor.grad is original_grad


def test_zero_valid_batches_and_limits():
    model = SyntheticLossModel()
    result = evaluate(model, [batch([-100, -100]), batch([2, 4])], 'cpu')
    assert result.tokens == 2 and result.batches == 2
    with pytest.raises(ValueError, match='no valid'):
        evaluate(model, [batch([-100, -100])], 'cpu')
    with pytest.raises(ValueError, match='no valid'):
        evaluate(model, [batch([-100]), batch([2])], 'cpu', max_batches=1)
    with pytest.raises(ValueError, match='max_batches'):
        evaluate(model, [batch([1])], 'cpu', max_batches=0)


def test_evaluation_matches_manual_token_average_on_real_model():
    torch.manual_seed(7)
    model = create_model(ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                                     intermediate_size=24, num_layers=2, max_seq_len=8)).eval()
    a = batch([5, 3, 1, -100])
    b = batch([7, 2, -100, -100])
    with torch.no_grad():
        loss_a = model(**{'input_ids': a['input_ids'], 'targets': a['targets']}).loss.item()
        loss_b = model(**{'input_ids': b['input_ids'], 'targets': b['targets']}).loss.item()
    expected = (3 * loss_a + 2 * loss_b) / 5
    result = evaluate(model, [a, b], 'cpu')
    assert result.loss == pytest.approx(expected)
    assert not model.training


def test_evaluation_restores_modes_on_failure():
    model = SyntheticLossModel().train()
    model.child.eval()
    with pytest.raises(KeyError):
        evaluate(model, [{'input_ids': torch.zeros(1, 2, dtype=torch.long)}], 'cpu')
    assert model.training and not model.child.training
