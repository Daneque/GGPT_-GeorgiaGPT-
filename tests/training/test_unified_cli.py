"""CPU integration tests of start/resume/extend with the additional-step budget."""

import json
import re
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from ggpt.training.experiment import prepare_extension, run_start, run_resume
from ggpt.training.experiment_config import load_experiment_config
from ggpt.training.logging import reconcile_metrics_on_resume


def config_file(tmp_path: Path, *, name: str, mode: str, additional: int = 0) -> Path:
    root = Path(__file__).resolve().parents[2]
    template = (root / 'configs/training/llama_40m.toml').read_text()
    # Tiny valid LLaMA and tiny uint16 corpus for end-to-end CPU tests.
    replacements = {
        'name = "llama_40m_v1"': f'name = "{name}"',
        'runs_dir = "runs/ggpt"': f'runs_dir = "{tmp_path.as_posix()}"',
        'vocab_size = 16384': 'vocab_size = 32',
        'num_layers = 10': 'num_layers = 2',
        'hidden_size = 512': 'hidden_size = 16',
        'num_heads = 8': 'num_heads = 2',
        'intermediate_size = 1376': 'intermediate_size = 24',
        'max_seq_len = 512': 'max_seq_len = 8',
        'train_path = "data/tokenized/kat_Geor/train.bin"': f'train_path = "{tmp_path / "train.bin"}"',
        'val_path = "data/tokenized/kat_Geor/val.bin"': f'val_path = "{tmp_path / "val.bin"}"',
        'tokenizer_path = "artifacts/tokenizer/bpe_100k_16384/tokenizer.json"': f'tokenizer_path = "{tmp_path / "tokenizer.json"}"',
        'train_windows = 10000': 'train_windows = 12',
        'validation_windows = 256': 'validation_windows = 6',
        'max_steps = 60000': 'max_steps = 6' if mode == 'start' else f'additional_steps = {additional}',
        'micro_batch_size = 4': 'micro_batch_size = 2',
        'accumulation_steps = 4': 'accumulation_steps = 2',
        'precision = "bf16"': 'precision = "fp32"',
        'device = "cuda"': 'device = "cpu"',
        'learning_rate = 0.0002': 'learning_rate = 0.002',
        'warmup_steps = 400': 'warmup_steps = 2',
        'log_every = 20': 'log_every = 1',
        'eval_every = 500': 'eval_every = 3',
        'checkpoint_every = 1000': 'checkpoint_every = 3',
        'generate_every = 1000': 'generate_every = 0',
    }
    for before, after in replacements.items():
        assert before in template, before
        template = template.replace(before, after)
    if mode == 'extend':
        template = template.replace(
            'name = "warmup_cosine"\nwarmup_steps = 2\nmin_lr_ratio = 0.1',
            'name = "continuation_cosine"\nramp_steps = 2\npeak_lr = 0.0006\nfinal_lr = 0.0001',
        )
    file = tmp_path / f'{name}.toml'
    file.write_text(template)
    return file


def test_start_resume_extend_and_extend_again(tmp_path, monkeypatch):
    # Test environment has no tensorboard; exercise the trainer with a no-op writer.
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def add_text(self, *args, **kwargs): pass
        def flush(self): pass
        def close(self): pass
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard",
                        types.SimpleNamespace(SummaryWriter=SummaryWriter))
    (tmp_path / 'tokenizer.json').write_text('placeholder')
    (np.arange(200, dtype='<u2') % 32).tofile(tmp_path / 'train.bin')
    (np.arange(100, dtype='<u2') % 32).tofile(tmp_path / 'val.bin')
    root = Path(__file__).resolve().parents[2]
    original_file = config_file(tmp_path, name='original', mode='start')
    orig_cfg = load_experiment_config(original_file, mode='start', project_root=root)
    assert run_start(orig_cfg, config_path=original_file, stop_after=3)['step'] == 3
    assert run_resume(tmp_path / 'original', device='cpu')['step'] == 6
    original_checkpoint = tmp_path / 'original/checkpoints/latest.pt'
    source_sha = original_checkpoint.read_bytes()
    second_file = config_file(tmp_path, name='extended', mode='extend', additional=8)
    ext_cfg = load_experiment_config(second_file, mode='extend', project_root=root)
    target, start, end = prepare_extension(tmp_path / 'original', ext_cfg, config_path=second_file)
    assert (start, end) == (6, 14)  # Eight ADDITIONAL steps.
    metadata = json.loads((target / 'parent_run.json').read_text())
    assert metadata['additional_steps'] == 8
    assert metadata['end_global_step'] == 14
    cp = torch.load(target / 'checkpoints/latest.pt', weights_only=True)
    assert cp['progress']['step'] == 6
    assert cp['run_spec']['trainer']['continuation_start_ratio'] == pytest.approx(0.1)
    assert run_resume(target, device='cpu', stop_after=9)['step'] == 9
    events = [json.loads(line) for line in (target / 'metrics.jsonl').read_text().splitlines()]
    first_train = next(e for e in events if e['type'] == 'train' and e['step'] == 7)
    assert first_train['lr'] == pytest.approx(0.002 * 0.1)  # No LR jump.
    assert run_resume(target, device='cpu')['step'] == 14
    assert original_checkpoint.read_bytes() == source_sha
    assert (tmp_path / 'original/checkpoints/latest.pt').is_file()
    with pytest.raises(FileExistsError):
        prepare_extension(tmp_path / 'original', ext_cfg, config_path=second_file)

    third_file = config_file(tmp_path, name='stage3', mode='extend', additional=7)
    third_cfg = load_experiment_config(third_file, mode='extend', project_root=root)
    stage3, start3, end3 = prepare_extension(target, third_cfg, config_path=third_file)
    assert (start3, end3) == (14, 21)
    cp3 = torch.load(stage3 / 'checkpoints/latest.pt', weights_only=True)
    assert cp3['run_spec']['trainer']['continuation_start_ratio'] == pytest.approx(0.05)
    assert run_resume(stage3, device='cpu')['step'] == 21


def test_config_rejects_invalid_and_incompatible(tmp_path):
    root = Path(__file__).resolve().parents[2]
    file = config_file(tmp_path, name='good', mode='start')
    cfg = load_experiment_config(file, mode='start', project_root=root)
    assert cfg.trainer.max_steps == 6
    bad = file.read_text().replace('max_steps = 6', 'additional_steps = 6')
    file.write_text(bad)
    with pytest.raises(ValueError, match='start accepts'):
        load_experiment_config(file, mode='start', project_root=root)
    bad = bad.replace('additional_steps = 6', 'max_steps = 6').replace('log_every = 1', 'log_every = 1\nundocumented_flag = true')
    file.write_text(bad)
    with pytest.raises(ValueError, match='Unknown'):
        load_experiment_config(file, mode='start', project_root=root)


def test_jsonl_resume_removes_future_steps_not_past(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    events = [{'type': 'train', 'step': 1}, {'type': 'val', 'step': 2},
              {'type': 'train', 'step': 3}, {'type': 'generation', 'step': 3},
              {'type': 'train', 'step': 4}]
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    assert reconcile_metrics_on_resume(path, 2) == 3
    actual = [json.loads(line) for line in path.read_text().splitlines()]
    assert actual == events[:2]


def test_missing_explicit_parameters_rejected(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = config_file(tmp_path, name='strict', mode='start')
    original = path.read_text()
    for old, expected in (
        ('num_layers = 2\n', 'Missing [model]'),
        ('train_windows = 12\n', 'Missing [data]'),
        ('accumulation_steps = 2\n', 'Missing [training]'),
        ('beta2 = 0.95\n', 'Missing [optimizer]'),
        ('min_lr_ratio = 0.1\n', 'Missing [scheduler]'),
        ('log_every = 1\n', 'Missing [monitoring]'),
    ):
        assert old in original
        path.write_text(original.replace(old, ''))
        with pytest.raises(ValueError, match=re.escape(expected)):
            load_experiment_config(path, mode='start', project_root=root)
    path.write_text(original)
    assert load_experiment_config(path, mode='start', project_root=root).trainer.max_steps == 6


def test_old_run_spec_without_continuation_fields_is_recreated_exactly(tmp_path):
    from ggpt.model import ModelConfig
    from ggpt.training.config import OptimizerConfig
    from ggpt.training.experiment_config import saved_configs
    from ggpt.training.trainer import TrainerConfig, build_run_spec

    for name in ('train.bin', 'val.bin', 'tokenizer.json'):
        (tmp_path / name).write_bytes(b'example')
    paths = {'train_path': tmp_path / 'train.bin', 'val_path': tmp_path / 'val.bin',
             'tokenizer_path': tmp_path / 'tokenizer.json'}
    config = TrainerConfig()
    original = build_run_spec(config=config, model_config=ModelConfig(),
                              optimizer_config=OptimizerConfig(), **paths)
    assert 'continuation_from' not in original['trainer']
    assert 'continuation_start_ratio' not in original['trainer']
    model, opt, trainer = saved_configs(original)
    rebuilt = build_run_spec(config=trainer, model_config=model,
                             optimizer_config=opt, **paths)
    assert rebuilt == original
