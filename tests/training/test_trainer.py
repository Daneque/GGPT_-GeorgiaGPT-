import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ggpt.model import ModelConfig
from ggpt.training.config import OptimizerConfig
from ggpt.training.trainer import TrainerConfig, run_training


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.texts = []
    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, step, value))
    def add_text(self, tag, value, step):
        self.texts.append((tag, step))
    def flush(self):
        pass
    def close(self):
        pass


def training_inputs(tmp_path: Path):
    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    tokenizer = tmp_path / "tokenizer.json"
    np.arange(120, dtype="<u2").__mod__(32).tofile(train)
    np.arange(90, dtype="<u2").__mod__(32).tofile(val)
    tokenizer.write_text("placeholder")
    model_config = ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                               intermediate_size=24, num_layers=2,
                               max_seq_len=8)
    config = TrainerConfig(max_steps=8, micro_batch_size=2, accumulation_steps=2,
                           train_windows=12, validation_windows=6, precision="fp32",
                           num_workers=0, warmup_steps=2, log_every=1, eval_every=3,
                           checkpoint_every=4, generate_every=0)
    optimizer_config = OptimizerConfig(learning_rate=0.002, fused=False)
    return dict(train_path=train, val_path=val, tokenizer_path=tokenizer,
                model_config=model_config, optimizer_config=optimizer_config,
                config=config, device="cpu")


def test_trainer_resume_matches_continuous(tmp_path):
    args = training_inputs(tmp_path)
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    result_direct = run_training(run_dir=direct, writer=DummyWriter(), **args)
    first = run_training(run_dir=resumed, writer=DummyWriter(), stop_after=5, **args)
    assert first["step"] == 5
    result_resumed = run_training(run_dir=resumed, writer=DummyWriter(), resume=True, **args)
    assert result_direct == result_resumed
    cp_direct = torch.load(direct / "checkpoints/latest.pt", weights_only=True)
    cp_resumed = torch.load(resumed / "checkpoints/latest.pt", weights_only=True)
    for key in cp_direct["model"]:
        torch.testing.assert_close(cp_direct["model"][key], cp_resumed["model"][key],
                                   atol=0, rtol=0)
    assert cp_direct["scheduler"]["last_epoch"] == cp_resumed["scheduler"]["last_epoch"]
    assert (resumed / "checkpoints/best.pt").exists()
    events = [json.loads(l) for l in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert any(e["type"] == "resume" for e in events)
    assert any(e["type"] == "train" for e in events)
    assert any(e["type"] == "val" for e in events)
    assert any(e["type"] == "checkpoint" for e in events)


def test_reject_changed_config_and_accidental_overwrite(tmp_path):
    args = training_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_training(run_dir=run_dir, writer=DummyWriter(), stop_after=2, **args)
    with pytest.raises(FileExistsError):
        run_training(run_dir=run_dir, writer=DummyWriter(), **args)
    other = dict(args)
    other["optimizer_config"] = OptimizerConfig(learning_rate=0.004, fused=False)
    with pytest.raises(ValueError, match="configuration"):
        run_training(run_dir=run_dir, writer=DummyWriter(), resume=True, **other)


def test_marker_pauses_after_completed_update(tmp_path, monkeypatch):
    args = training_inputs(tmp_path)
    run_dir = tmp_path / "marked"

    class MarkerWriter(DummyWriter):
        def add_scalar(self, tag, value, step):
            super().add_scalar(tag, value, step)
            if tag == "train/loss" and step == 1:
                (run_dir / "STOP").touch()

    state = run_training(run_dir=run_dir, writer=MarkerWriter(), **args)
    # Marker is created while logging step 1; caught after completed step 2.
    assert state["step"] == 2
    assert not (run_dir / "STOP").exists()
    saved = torch.load(run_dir / "checkpoints/latest.pt", weights_only=True)
    assert saved["progress"]["step"] == 2
    assert run_training(run_dir=run_dir, writer=DummyWriter(), resume=True,
                        **args)["step"] == 8


def test_interrupt_during_update_preserves_last_good_checkpoint(tmp_path, monkeypatch):
    from dataclasses import replace
    import ggpt.training.trainer as trainer_module

    args = training_inputs(tmp_path)
    args["config"] = replace(args["config"], checkpoint_every=1)
    run_dir = tmp_path / "interrupted"
    original = trainer_module.train_step
    attempts = 0

    def interrupt_second(*a, **kw):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise KeyboardInterrupt()
        return original(*a, **kw)

    monkeypatch.setattr(trainer_module, "train_step", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        run_training(run_dir=run_dir, writer=DummyWriter(), **args)
    saved = torch.load(run_dir / "checkpoints/latest.pt", weights_only=True)
    assert saved["progress"]["step"] == 1
    monkeypatch.setattr(trainer_module, "train_step", original)
    assert run_training(run_dir=run_dir, writer=DummyWriter(), resume=True,
                        **args)["step"] == 8
