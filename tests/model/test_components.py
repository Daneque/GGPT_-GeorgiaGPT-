import torch
import torch.nn.functional as F
from ggpt.model import ModelConfig
from ggpt.model.layers import RMSNorm, SwiGLU
from ggpt.model.rope import RotaryEmbedding, apply_rope


def test_rmsnorm():
    x = torch.tensor([[[1., -2., 3., -4.]]], requires_grad=True)
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1., 2., 3., 4.]))
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(x), expected * norm.weight)
    norm(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(norm(torch.zeros_like(x)), torch.zeros_like(x))


def test_swiglu():
    config = ModelConfig(hidden_size=8, num_heads=2, intermediate_size=12, num_layers=1)
    layer = SwiGLU(config).eval()
    x = torch.randn(2, 3, 8)
    expected = F.linear(F.silu(F.linear(x, layer.gate_proj.weight))
                        * F.linear(x, layer.up_proj.weight), layer.down_proj.weight)
    torch.testing.assert_close(layer(x), expected)
    assert all('bias' not in name for name, _ in layer.named_parameters())


def test_rope_formula_norm_and_zero_position():
    rope = RotaryEmbedding(4, 10000.)
    cos, sin = rope(3, torch.device('cpu'))
    x = torch.tensor([[[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]]]])
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :])
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    angle = torch.tensor([1., 0.01])
    expected = torch.stack((x[0, 0, 1, 0::2] * angle.cos() - x[0, 0, 1, 1::2] * angle.sin(),
                            x[0, 0, 1, 0::2] * angle.sin() + x[0, 0, 1, 1::2] * angle.cos()), -1).flatten()
    torch.testing.assert_close(rotated[0, 0, 1], expected)
    low = apply_rope(x.to(torch.bfloat16), cos, sin)
    assert low.dtype == torch.bfloat16
    rope.half()
    assert rope(3, torch.device('cpu'))[0].dtype == torch.float32


def test_rope_relative_position():
    rope = RotaryEmbedding(8)
    cos, sin = rope(5, torch.device('cpu'))
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    q_rot = apply_rope(q.expand(-1, -1, 5, -1), cos, sin)
    k_rot = apply_rope(k.expand(-1, -1, 5, -1), cos, sin)
    torch.testing.assert_close((q_rot[..., 1, :] * k_rot[..., 3, :]).sum(),
                               (q_rot[..., 2, :] * k_rot[..., 4, :]).sum())


def test_rope_fp32_cache_reuse_resize_and_state_dict():
    rope = RotaryEmbedding(8)
    device = torch.device('cpu')
    cos_first, sin_first = rope(8, device)
    cos_again, sin_again = rope(8, device)
    assert cos_first.data_ptr() == cos_again.data_ptr()
    assert sin_first.data_ptr() == sin_again.data_ptr()

    cos_short, sin_short = rope(3, device)
    assert cos_short.data_ptr() == cos_first.data_ptr()
    torch.testing.assert_close(cos_short, cos_first[..., :3, :])
    torch.testing.assert_close(sin_short, sin_first[..., :3, :])

    cos_long, sin_long = rope(13, device)
    assert cos_long.shape == (1, 1, 13, 4)
    torch.testing.assert_close(cos_long[..., :8, :], cos_first)
    torch.testing.assert_close(sin_long[..., :8, :], sin_first)
    assert '_cos_cached' not in rope.state_dict()
    assert '_sin_cached' not in rope.state_dict()


def test_rope_cache_recovers_fp32_after_model_half():
    rope = RotaryEmbedding(8)
    before = rope(6, torch.device('cpu'))[0].clone()
    rope.half()
    assert rope._cos_cached.dtype == torch.float16
    cos, sin = rope(6, torch.device('cpu'))
    assert cos.dtype == sin.dtype == torch.float32
    torch.testing.assert_close(cos, before, rtol=0, atol=0)
    assert cos.data_ptr() == rope(6, torch.device('cpu'))[0].data_ptr()

