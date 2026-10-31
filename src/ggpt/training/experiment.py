"""Start/resume/extend orchestration independent of the trainer's update loop."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import torch

from ggpt.utils.logger import get_logger
from .experiment_config import ExperimentConfig, saved_configs
from .trainer import build_run_spec, run_training


def read_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    spec_path = run_dir / "run_spec.json"
    latest = run_dir / "checkpoints/latest.pt"
    if not spec_path.is_file() or not latest.is_file():
        raise FileNotFoundError(f"Expected run_spec.json and checkpoints/latest.pt in {run_dir}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def run_start(config: ExperimentConfig, *, config_path: Path,
              stop_after: int | None = None) -> dict[str, Any]:
    run_dir = config.runs_dir / config.name
    if run_dir.exists():
        raise FileExistsError(f"Experiment already exists: {run_dir}")
    # Fail on missing files before creating an experiment directory.
    build_run_spec(config=config.trainer, model_config=config.model,
                   optimizer_config=config.optimizer, train_path=config.train_path,
                   val_path=config.val_path, tokenizer_path=config.tokenizer_path)
    get_logger("training").info(
        "new run=%s device=%s model=%d layers/%d hidden, train=%s, val=%s, "
        "tokenizer=%s, batch=%d x %d, tokens_per_step=%d, max_steps=%d, "
        "optimizer_lr=%g, precision=%s",
        run_dir, config.device, config.model.num_layers, config.model.hidden_size,
        config.train_path, config.val_path, config.tokenizer_path,
        config.trainer.micro_batch_size, config.trainer.accumulation_steps,
        config.trainer.micro_batch_size * config.trainer.accumulation_steps * config.model.max_seq_len,
        config.trainer.max_steps, config.optimizer.learning_rate, config.trainer.precision,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.toml")
    return run_training(run_dir=run_dir, train_path=config.train_path,
                        val_path=config.val_path, tokenizer_path=config.tokenizer_path,
                        model_config=config.model, optimizer_config=config.optimizer,
                        config=config.trainer, resume=False, stop_after=stop_after,
                        device=config.device)


def run_resume(run_dir: str | Path, *, device: str,
               stop_after: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    spec = read_run(run_dir)
    model, optimizer, trainer = saved_configs(spec)
    get_logger("training").info(
        "resume run=%s device=%s max_steps=%d, train=%s, val=%s, tokenizer=%s",
        run_dir, device, trainer.max_steps, spec["train"]["path"],
        spec["validation"]["path"], spec["tokenizer"]["path"],
    )
    return run_training(run_dir=run_dir, train_path=spec["train"]["path"],
                        val_path=spec["validation"]["path"],
                        tokenizer_path=spec["tokenizer"]["path"],
                        model_config=model, optimizer_config=optimizer,
                        config=trainer, resume=True, stop_after=stop_after,
                        device=device)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_extension(source: str | Path, config: ExperimentConfig, *,
                      config_path: Path) -> tuple[Path, int, int]:
    """Clone a *finished* source run into new stage without changing originals.

    The extend budget is ADDITIONAL, not an absolute target step count.
    Changes to model, optimizer or sampling identity are deliberately rejected.
    """
    if config.additional_steps is None:
        raise ValueError("extend requires training.additional_steps")
    source = Path(source).resolve()
    target = (config.runs_dir / config.name).resolve()
    if source == target or target.exists():
        raise FileExistsError(f"Target exists or equals source: {target}")
    parent_spec = read_run(source)
    parent_model, parent_optimizer, parent_trainer = saved_configs(parent_spec)
    if parent_model != config.model or parent_optimizer != config.optimizer:
        raise ValueError("extend requires identical model and AdamW configuration")
    for key in ("micro_batch_size", "accumulation_steps", "train_windows",
                "validation_windows", "num_workers", "seed", "precision", "grad_clip"):
        if getattr(parent_trainer, key) != getattr(config.trainer, key):
            raise ValueError(f"Incompatible training/sampler parameter: {key}")

    paths = (config.train_path, config.val_path, config.tokenizer_path)
    original_paths = (parent_spec["train"]["path"],
                      parent_spec["validation"]["path"], parent_spec["tokenizer"]["path"])
    if tuple(map(lambda x: str(Path(x).resolve()), paths)) != tuple(map(
            lambda x: str(Path(x).resolve()), original_paths)):
        raise ValueError("Corpus/tokenizer paths differ from parent")
    for name, path in zip(("train", "validation", "tokenizer"), paths):
        if not path.is_file():
            raise FileNotFoundError(path)

    latest = source / "checkpoints/latest.pt"
    original = torch.load(latest, map_location="cpu", weights_only=True)
    if original.get("run_spec") != parent_spec:
        raise ValueError("Parent checkpoint run_spec differs from its run_spec.json")
    checkpoint_step = original["progress"]["step"]
    if checkpoint_step != parent_trainer.max_steps:
        raise ValueError("Parent run must be completed before extend; use resume otherwise")
    if original["scheduler"].get("last_epoch") != checkpoint_step:
        raise ValueError("Scheduler state and checkpoint global step disagree")
    if config.ramp_steps >= config.additional_steps:
        raise ValueError("ramp_steps must be shorter than additional_steps")
    global_target = checkpoint_step + config.additional_steps
    base_lr = parent_optimizer.learning_rate
    group_lrs = [g["lr"] for g in original["optimizer"]["param_groups"]]
    if not group_lrs or any(abs(x - group_lrs[0]) > 1e-10 for x in group_lrs):
        raise ValueError("Unexpected different optimizer group learning rates")
    initial_lr = float(group_lrs[0])
    if not (0 < config.final_lr <= initial_lr <= config.peak_lr <= base_lr):
        raise ValueError("Require 0 < final_lr <= current checkpoint LR <= peak_lr <= base LR")
    updated_trainer = replace(
        config.trainer, max_steps=global_target,
        warmup_steps=parent_trainer.warmup_steps,
        min_lr_ratio=parent_trainer.min_lr_ratio,
        continuation_from=checkpoint_step,
        continuation_start_ratio=initial_lr / base_lr,
        continuation_ramp_steps=config.ramp_steps,
        continuation_peak_ratio=config.peak_lr / base_lr,
        continuation_final_ratio=config.final_lr / base_lr,
    )
    new_spec = build_run_spec(config=updated_trainer, model_config=config.model,
                              optimizer_config=config.optimizer, train_path=config.train_path,
                              val_path=config.val_path, tokenizer_path=config.tokenizer_path)
    for name in ("train", "validation", "tokenizer"):
        if new_spec[name] != parent_spec[name]:
            raise ValueError(f"Parent {name} identity changed")

    # Stage everything before atomic rename to the final run directory.
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.staging")
    try:
        (staging / "checkpoints").mkdir(parents=True, exist_ok=False)
        (staging / "run_spec.json").write_text(
            json.dumps(new_spec, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copyfile(config_path, staging / "config.toml")
        (staging / "parent_run.json").write_text(json.dumps({
            "parent_run": str(source), "parent_latest": str(latest),
            "parent_checkpoint_sha256": _hash_file(latest),
            "start_global_step": checkpoint_step,
            "additional_steps": config.additional_steps,
            "end_global_step": global_target,
            "initial_lr": initial_lr,
            "peak_lr": config.peak_lr,
            "final_lr": config.final_lr,
        }, indent=2), encoding="utf-8")
        original["run_spec"] = new_spec
        torch.save(original, staging / "checkpoints/latest.pt")
        del original
        best_source = source / "checkpoints/best.pt"
        if best_source.exists():
            best = torch.load(best_source, map_location="cpu", weights_only=True)
            if best.get("run_spec") != parent_spec:
                raise ValueError("Parent best checkpoint metadata differs from run_spec")
            best["run_spec"] = new_spec
            torch.save(best, staging / "checkpoints/best.pt")
            del best
        # Destination did not exist at check time; fail if an unrelated process creates it.
        if target.exists():
            raise FileExistsError(f"Target created concurrently: {target}")
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target, checkpoint_step, global_target


def run_extend(source: str | Path, config: ExperimentConfig, *, config_path: Path,
               stop_after: int | None = None) -> dict[str, Any]:
    target, origin, global_target = prepare_extension(source, config,
                                                        config_path=config_path)
    get_logger("training").info(
        "extend parent=%s new_run=%s source_step=%d additional_steps=%d "
        "target_global_step=%d, peak_lr=%g, final_lr=%g",
        Path(source).resolve(), target, origin, config.additional_steps,
        global_target, config.peak_lr, config.final_lr,
    )
    print(f"Extended from global step {origin}; additional_steps={config.additional_steps}; "
          f"target_global_step={global_target}", flush=True)
    return run_resume(target, device=config.device, stop_after=stop_after)
