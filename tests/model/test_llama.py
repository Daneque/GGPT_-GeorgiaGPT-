import io
import pytest
import torch
import torch.nn.functional as F
from ggpt.model import CausalLMOutput, LlamaLM, ModelConfig, create_model


def tiny_config():
    return ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                       intermediate_size=24, num_layers=2, max_seq_len=12)


def test_shapes_loss_and_backward():
    model = create_model(tiny_config())
    x = torch.randint(0, 32, (2, 7))
    targets = torch.randint(0, 32, (2, 7))
    targets[0, 2] = -100
    output = model(input_ids=x, targets=targets)
    assert isinstance(output, CausalLMOutput)
    assert output.logits.shape == (2, 7, 32)
    assert output.loss.ndim == 0
    torch.testing.assert_close(output.loss, F.cross_entropy(
        output.logits.reshape(-1, 32), targets.reshape(-1), ignore_index=-100))
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(input_ids=x).loss is None


def test_all_ignored_loss():
    model = create_model(tiny_config())
    x = torch.zeros(1, 3, dtype=torch.long)
    loss = model(input_ids=x, targets=torch.full_like(x, -100)).loss
    assert loss.item() == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_causality():
    model = create_model(tiny_config()).eval()
    x = torch.randint(0, 32, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    with torch.no_grad():
        first = model(input_ids=x).logits
        second = model(input_ids=changed).logits
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-5)


def test_tying_factory_and_state_dict():
    model = create_model(tiny_config()).eval()
    assert isinstance(model, LlamaLM)
    assert model.lm_head.weight is model.token_embeddings.weight
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = create_model(ModelConfig.from_json(model.config.to_json())).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.lm_head.weight is restored.token_embeddings.weight
    x = torch.randint(0, 32, (2, 5))
    torch.testing.assert_close(model(input_ids=x).logits, restored(input_ids=x).logits)


def test_base_parameter_count():
    # Meta construction checks registration/counts without allocating real weights.
    with torch.device('meta'):
        model = create_model(ModelConfig())
    count = sum(p.numel() for p in model.parameters())
    assert count == 40_020_480
    assert 30_000_000 <= count <= 50_000_000


@pytest.mark.parametrize('x', [torch.zeros(1, 13, dtype=torch.long),
                              torch.zeros(1, 0, dtype=torch.long),
                              torch.zeros(0, 3, dtype=torch.long),
                              torch.zeros(1, 3), torch.zeros(3, dtype=torch.long)])
def test_bad_inputs(x):
    with pytest.raises(ValueError):
        create_model(tiny_config())(input_ids=x)


def test_bad_targets():
    model = create_model(tiny_config())
    x = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        model(input_ids=x, targets=x[:, :2])
    with pytest.raises(ValueError):
        model(input_ids=x, targets=x.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_autocast():
    model = create_model(tiny_config()).to('cuda')
    x = torch.randint(0, 32, (2, 5), device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(input_ids=x, targets=x)
    assert output.logits.dtype == torch.float16
    assert output.loss.dtype == torch.float32
    output.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
