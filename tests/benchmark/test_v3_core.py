from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ggpt.benchmark.v3_dataset import load_v3_dataset, V3Case
from ggpt.benchmark.v3_scoring import score_case, score_prompt_choices
from ggpt.inference.generate import GenerationConfig, generate


class CharacterTokenizer:
    symbols = {'X': 3, ' ': 4, 'A': 5, 'B': 6, 'C': 7, 'D': 8}

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[self.symbols[c] for c in text])

    def id_to_token(self, idx):
        return {value: key for key, value in self.symbols.items()}.get(idx, f'<{idx}>')

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.id_to_token(i) for i in ids)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=9, bos_token_id=0, eos_token_id=1,
                                      pad_token_id=2, max_seq_len=20)

    def forward(self, input_ids, targets=None):
        ids = input_ids
        logits = torch.full((*ids.shape, 9), -5.0, device=ids.device)
        for batch in range(ids.shape[0]):
            for pos, token in enumerate(ids[batch].tolist()):
                if token == 3:  # X -> whitespace
                    logits[batch, pos, 4] = 10.0
                elif token == 4:  # whitespace -> A is better than B
                    logits[batch, pos, 5] = 8.0
                    logits[batch, pos, 6] = 6.0
                else:
                    logits[batch, pos, 1] = 9.0
        return SimpleNamespace(logits=logits)


def test_full_54_questions_load():
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'benchmarks/georgian/language_benchmark_v3.json'
    suite = load_v3_dataset(path)
    assert len(suite.cases) == 54
    assert [sum(c.category == name for c in suite.cases)
            for name in ('continuation', 'facts', 'context')] == [18, 18, 18]
    assert sum(c.format == 'qa' for c in suite.cases) == 6
    assert len({c.metadata['genre'] for c in suite.cases if c.category == 'continuation'}) >= 10


def test_teacher_forcing_scores_all_answer_tokens_and_ranks():
    model, tok = ToyLM(), CharacterTokenizer()
    result = score_prompt_choices(model, tok, 'X', (' A', ' B'), ' B', torch.device('cpu'))
    assert result['reference']['token_count'] == 2
    assert result['reference']['first_token_rank_vocab'] == 1
    assert result['reference_rank_by_sum'] == 2
    assert result['top10_next_tokens'][0]['token'] == ' '
    assert abs(result['reference']['log_probability'] - sum(
        result['reference']['token_log_probabilities'])) < 1e-7
    assert result['reference']['log_probability'] < result['candidates'][0]['log_probability']


def test_context_control_runs_both_conditions():
    model, tok = ToyLM(), CharacterTokenizer()
    case = V3Case('test', 'context', 'cloze', 'X', ' B', (' A', ' B'),
                  'X', {})
    score = score_case(model, tok, case, 'cpu')
    assert score['reference_logprob_gain_from_context'] == pytest.approx(0)
    assert score['reference_rank_gain_from_context'] == 0


def test_nucleus_sampling_and_greedy():
    model = ToyLM()
    prefix = torch.tensor([[0, 3]], dtype=torch.long)
    config = GenerationConfig(max_new_tokens=2, do_sample=True, top_p=0.01)
    a = generate(model, prefix, config)
    b = generate(model, prefix, config)
    assert torch.equal(a, b)
    assert a[0, -1].item() == 5
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.1)


def test_v3_collection_and_resume(tmp_path, monkeypatch):
    import json
    from ggpt.benchmark import v3_runner
    from ggpt.benchmark.v3_dataset import V3Suite

    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'manifest.json').write_text(json.dumps({'format': 'ggpt-inference-v1'}))
    monkeypatch.setattr(v3_runner, 'load_model_artifact',
                        lambda path, device: (ToyLM(), CharacterTokenizer(), {'step': 1}))
    suite = V3Suite('toy', (
        V3Case('c', 'continuation', 'free', 'X', None, (), None, {}),
        V3Case('f', 'facts', 'cloze', 'X', ' B', (' A', ' B'), None, {}),
    ), 'toy-sha')
    output = tmp_path / 'result'
    kwargs = {'model_dir': model_dir, 'suite': suite, 'output': output,
              'device': 'cpu', 'variants': ('greedy', 'topp_t08')}
    result = v3_runner.run_v3(**kwargs)
    assert result['scored'] == 1
    assert result['generated'] == 4
    rows = [json.loads(line) for line in (output / 'scores.jsonl').read_text().splitlines()]
    assert rows[0]['measurement']['reference_rank_by_sum'] == 2
    result_again = v3_runner.run_v3(**kwargs, resume=True)
    assert result_again == result
    assert len((output / 'generations.jsonl').read_text().splitlines()) == 4
