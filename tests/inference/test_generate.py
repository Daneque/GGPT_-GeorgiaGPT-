from types import SimpleNamespace
import pytest
import torch
from torch import nn
from ggpt.model import CausalLMOutput
from ggpt.inference import GenerationConfig, generate, generate_text


class FakeLM(nn.Module):
    def __init__(self, mode='next'):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout(0.5)
        self.config = SimpleNamespace(max_seq_len=3, vocab_size=8,
                                      bos_token_id=0, eos_token_id=1, pad_token_id=2)
        self.mode = mode
        self.lengths = []

    def forward(self, input_ids, targets=None):
        assert not self.training and not self.child.training
        assert torch.is_inference_mode_enabled()
        self.lengths.append(input_ids.shape[1])
        if self.mode == 'error':
            raise RuntimeError('deliberate failure')
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        if self.mode == 'next':
            logits.scatter_(-1, ((input_ids + 1) % 8).unsqueeze(-1), 20.)
        elif self.mode == 'eos':
            logits[..., 1] = 20.
        return CausalLMOutput(logits)


def test_greedy_context_and_mode_restoration():
    model = FakeLM().train()
    model.child.eval()
    result = generate(model, torch.tensor([[0, 3, 4, 5]]), GenerationConfig(max_new_tokens=4))
    assert result.tolist() == [[0, 3, 4, 5, 6, 7, 0, 1]]
    assert model.lengths == [3, 3, 3, 3]
    assert model.training and not model.child.training


def test_eos_batch_and_already_finished():
    model = FakeLM('eos').eval()
    result = generate(model, torch.tensor([[0, 3], [0, 1]]))
    assert result.tolist() == [[0, 3, 1], [0, 1, 2]]
    assert not model.training
    model.lengths.clear()
    assert generate(model, torch.tensor([[0, 1]])).tolist() == [[0, 1]]
    assert model.lengths == []


def test_sampling_seed_and_top_k():
    model = FakeLM('uniform')
    prompt = torch.tensor([[0, 3]])
    options = GenerationConfig(max_new_tokens=12, do_sample=True, temperature=0.7, seed=123)
    global_state = torch.random.get_rng_state().clone()
    assert torch.equal(generate(model, prompt, options), generate(model, prompt, options))
    assert torch.equal(global_state, torch.random.get_rng_state())
    model = FakeLM()
    greedy = generate(model, prompt, GenerationConfig(max_new_tokens=5))
    sampled = generate(model, prompt, GenerationConfig(max_new_tokens=5, do_sample=True, top_k=1))
    assert torch.equal(greedy, sampled)
    generate(model, prompt, GenerationConfig(max_new_tokens=1, do_sample=True, top_k=100))


def test_exception_restores_modes():
    model = FakeLM('error').train()
    model.child.eval()
    with pytest.raises(RuntimeError, match='deliberate'):
        generate(model, torch.tensor([[0, 3]]))
    assert model.training and not model.child.training


def test_zero_tokens_and_invalid_prompt():
    model = FakeLM()
    x = torch.tensor([[0, 3]])
    assert torch.equal(generate(model, x, GenerationConfig(max_new_tokens=0)), x)
    assert not model.lengths
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[8]]))
    with pytest.raises(ValueError):
        generate(model, torch.empty((1, 0), dtype=torch.long))


@pytest.mark.parametrize('kwargs', [{'temperature': 0}, {'temperature': float('nan')},
                                   {'top_k': 0}, {'max_new_tokens': -1}, {'seed': -1}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_text_wrapper(tmp_path):
    tokenizers = pytest.importorskip('tokenizers')
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tokenizer = tokenizers.Tokenizer(BPE(vocab={
        '<bos>': 0, '<eos>': 1, '<pad>': 2, 'a': 3,
        'b': 4, 'c': 5, 'd': 6, 'e': 7}, merges=[]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(['<bos>', '<eos>', '<pad>'])
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(path))
    assert generate_text(FakeLM('eos'), 'ab', path) == 'ab'
    assert generate_text(FakeLM('eos'), '', path) == ''
    with pytest.raises(ValueError):
        generate_text(FakeLM('eos'), '', path, add_bos=False)

