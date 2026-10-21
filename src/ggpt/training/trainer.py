"""Single-GPU GeorgiaGPT trainer with reproducible batch-index resume."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
import uuid

import numpy as np
import torch

from ggpt.data.token_loader import LoaderConfig, create_dataloaders
from ggpt.inference import GenerationConfig, generate_text
from ggpt.model import ModelConfig, create_model
from .checkpoint import load_checkpoint, save_checkpoint
from .config import OptimizerConfig, StepConfig
from .evaluation import evaluate
from .logging import TrainingLogger
from .optim import create_optimizer
from .scheduler import create_scheduler
from .steps import train_step


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 60_000
    micro_batch_size: int = 4
    accumulation_steps: int = 4
    train_windows: int = 10_000
    validation_windows: int = 256
    num_workers: int = 0
    seed: int = 42
    precision: str = "bf16"
    grad_clip: float = 1.0
    warmup_steps: int = 400
    min_lr_ratio: float = 0.1
    log_every: int = 20
    eval_every: int = 500
    checkpoint_every: int = 1000
    generate_every: int = 1000
    generation_max_tokens: int = 64

    def __post_init__(self) -> None:
        for name in ("max_steps", "micro_batch_size", "accumulation_steps",
                     "train_windows", "validation_windows", "log_every", "eval_every",
                     "checkpoint_every", "generation_max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.generate_every) is not int or self.generate_every < 0:
            raise ValueError("generate_every must be a nonnegative integer")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if self.train_windows < self.micro_batch_size:
            raise ValueError("train_windows must be at least micro_batch_size")
        if not 0 <= self.min_lr_ratio <= 1 or not math.isfinite(self.min_lr_ratio):
            raise ValueError("min_lr_ratio must be finite in [0, 1]")
        StepConfig(self.accumulation_steps, self.grad_clip, self.precision)
        LoaderConfig(context_length=512, batch_size=self.micro_batch_size,
                     train_windows=self.train_windows,
                     validation_windows=self.validation_windows,
                     num_workers=self.num_workers, seed=self.seed)


def _file_identity(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data: dict[str, Any] = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    if not hash_file:
        # Cheap integrity guard without hashing multi-gigabyte corpora at startup.
        with path.open("rb") as handle:
            head = handle.read(1024 * 1024)
            handle.seek(max(0, path.stat().st_size - 1024 * 1024))
            tail = handle.read(1024 * 1024)
        data["edges_sha256"] = hashlib.sha256(head + tail).hexdigest()
    if hash_file:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        data["sha256"] = digest.hexdigest()
    return data


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_samples(model: torch.nn.Module, tokenizer_path: Path,
                      step: int, logger: TrainingLogger,
                      max_new_tokens: int, seed: int) -> None:
    samples = ("საქართველო არის", "ქართული ენა")
    for i, prompt in enumerate(samples):
        options = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, seed=seed)
        text = generate_text(model, prompt, tokenizer_path, options)
        logger.generated(step, label=f"greedy_{i}", prompt=prompt, text=text,
                         generation_config=asdict(options))
        options = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True,
                                   top_k=40, temperature=0.8, seed=seed + i)
        text = generate_text(model, prompt, tokenizer_path, options)
        logger.generated(step, label=f"sample_{i}", prompt=prompt, text=text,
                         generation_config=asdict(options))


def run_training(*, run_dir: str | Path, train_path: str | Path,
                 val_path: str | Path, tokenizer_path: str | Path,
                 model_config: ModelConfig | None = None,
                 optimizer_config: OptimizerConfig | None = None,
                 config: TrainerConfig | None = None, resume: bool = False,
                 stop_after: int | None = None,
                 device: torch.device | str = "cuda",
                 writer: Any | None = None) -> dict[str, Any]:
    """Train to max_steps, or deliberately stop at stop_after and checkpoint.

    stop_after is an absolute global update count and is deliberately excluded
    from run_spec, allowing an identical experiment to resume to max_steps.
    """
    config = config or TrainerConfig()
    model_config = model_config or ModelConfig()
    optimizer_config = optimizer_config or OptimizerConfig(learning_rate=2e-4)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if stop_after is not None and (type(stop_after) is not int or
                                   not 1 <= stop_after <= config.max_steps):
        raise ValueError("stop_after must lie in [1, max_steps]")
    limit = stop_after if stop_after is not None else config.max_steps

    run_dir = Path(run_dir).resolve()
    train_path, val_path = Path(train_path).resolve(), Path(val_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    ckpt_dir = run_dir / "checkpoints"
    latest_path, best_path = ckpt_dir / "latest.pt", ckpt_dir / "best.pt"
    spec_path = run_dir / "run_spec.json"
    run_spec = {
        "model": model_config.to_dict(),
        "optimizer": asdict(optimizer_config),
        "trainer": asdict(config),
        "train": _file_identity(train_path),
        "validation": _file_identity(val_path),
        "tokenizer": _file_identity(tokenizer_path, hash_file=True),
    }
    # Match JSON and checkpoint representations (e.g. optimizer betas tuples).
    run_spec = json.loads(json.dumps(run_spec, ensure_ascii=False))
    if resume:
        if not spec_path.exists() or not latest_path.is_file():
            raise FileNotFoundError("--resume requires run_spec.json and checkpoints/latest.pt")
        if json.loads(spec_path.read_text(encoding="utf-8")) != run_spec:
            raise ValueError("Run configuration differs from saved run_spec.json")
    elif spec_path.exists() or latest_path.exists():
        raise FileExistsError(f"Run already exists: {run_dir}; use --resume or a new run name")

    _seed_all(config.seed)
    loader_cfg = LoaderConfig(context_length=model_config.max_seq_len,
                              batch_size=config.micro_batch_size,
                              train_windows=config.train_windows,
                              validation_windows=config.validation_windows,
                              seed=config.seed, num_workers=config.num_workers,
                              vocab_size=model_config.vocab_size,
                              pin_memory=device.type == "cuda")
    train_loader, val_loader = create_dataloaders(loader_cfg, train_path, val_path)
    sampler = train_loader.batch_sampler
    model = create_model(model_config).to(device)
    optimizer = create_optimizer(model, optimizer_config)
    scheduler = create_scheduler(optimizer, total_steps=config.max_steps,
                                 warmup_steps=config.warmup_steps,
                                 min_lr_ratio=config.min_lr_ratio)
    step_cfg = StepConfig(config.accumulation_steps, config.grad_clip, config.precision)
    progress: dict[str, Any] = {"step": 0, "tokens_seen": 0, "epoch": 0,
                                 "next_batch": 0, "best_val_loss": math.inf}
    if resume:
        progress = load_checkpoint(latest_path, model=model, optimizer=optimizer,
                                   scheduler=scheduler, sampler=sampler,
                                   run_spec=run_spec, map_location=device)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Never write run_spec until all dependencies and objects were initialized.
        spec_path.write_text(json.dumps(run_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_latest() -> None:
        state = sampler.state_dict(next_batch=progress["next_batch"])
        save_checkpoint(latest_path, model=model, optimizer=optimizer,
                        scheduler=scheduler, progress=progress, sampler_state=state,
                        run_spec=run_spec)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if progress["step"] > limit:
        raise ValueError(f"Checkpoint step {progress['step']} is beyond requested stop {limit}")
    stop_file = run_dir / "STOP"
    stop_requested = False
    in_update = False
    with TrainingLogger(run_dir, resume_step=progress["step"] if resume else None,
                        writer=writer) as logger:
        logger.event("start" if not resume else "resume", progress["step"],
                     device=str(device), step_limit=limit,
                     tokens_seen=progress["tokens_seen"])
        if not resume:
            initial = evaluate(model, val_loader, device, precision=config.precision)
            logger.scalars("val", 0, loss=initial.loss, perplexity=initial.perplexity,
                           tokens=initial.tokens)
            if config.generate_every:
                _generate_samples(model, tokenizer_path, 0, logger,
                                  config.generation_max_tokens, config.seed)
            logger.flush()

        try:
            while progress["step"] < limit and not stop_requested:
                # Explicit sampler state is the *committed*, not prefetched, cursor.
                sampler.set_epoch(progress["epoch"])
                sampler.set_start_batch(progress["next_batch"])
                iterator = iter(train_loader)
                while progress["step"] < limit and not stop_requested:
                    group = []
                    for _ in range(config.accumulation_steps):
                        try:
                            group.append(next(iterator))
                        except StopIteration:
                            break
                    if not group:
                        progress["epoch"] += 1
                        progress["next_batch"] = 0
                        sampler.set_epoch(progress["epoch"])
                        break

                    synchronize()
                    started = time.perf_counter()
                    used_lr = optimizer.param_groups[0]["lr"]
                    in_update = True
                    result = train_step(model, optimizer, group, device, step_cfg)
                    scheduler.step()
                    in_update = False
                    synchronize()
                    step_seconds = time.perf_counter() - started
                    progress["step"] += 1
                    progress["tokens_seen"] += result.tokens
                    progress["next_batch"] += len(group)
                    if progress["next_batch"] == sampler.total_batches:
                        progress["epoch"] += 1
                        progress["next_batch"] = 0
                        sampler.set_epoch(progress["epoch"])
                        end_of_epoch = True
                    else:
                        end_of_epoch = False
                    step = progress["step"]
                    if stop_file.exists():
                        stop_requested = True
                        stop_file.unlink()
                    if step == 1 or step % config.log_every == 0:
                        metrics = {
                            "loss": result.loss, "grad_norm": result.grad_norm,
                            "lr": used_lr, "tokens": result.tokens,
                            "tokens_seen": progress["tokens_seen"],
                            "step_seconds": step_seconds,
                            "tokens_per_second": result.tokens / step_seconds,
                            "epoch": progress["epoch"],
                        }
                        if device.type == "cuda":
                            metrics["peak_vram_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                        logger.scalars("train", step, **metrics)
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)

                    is_best = False
                    if step % config.eval_every == 0 or step == limit:
                        sync_start = time.perf_counter()
                        val = evaluate(model, val_loader, device, precision=config.precision)
                        synchronize()
                        eval_seconds = time.perf_counter() - sync_start
                        logger.scalars("val", step, loss=val.loss,
                                       perplexity=val.perplexity,
                                       tokens=val.tokens, seconds=eval_seconds)
                        is_best = val.loss < progress["best_val_loss"]
                        if is_best:
                            progress["best_val_loss"] = val.loss
                    if config.generate_every and step % config.generate_every == 0:
                        _generate_samples(model, tokenizer_path, step, logger,
                                          config.generation_max_tokens, config.seed)
                    if (step % config.checkpoint_every == 0 or
                            step == limit or is_best or stop_requested):
                        save_latest()
                        if is_best:
                            _copy_atomic(latest_path, best_path)
                        logger.event("checkpoint", step, path=str(latest_path), best=is_best)
                        logger.flush()
                    if end_of_epoch or stop_requested:
                        break
            logger.event("finished" if progress["step"] == config.max_steps else "paused",
                         progress["step"], tokens_seen=progress["tokens_seen"])
            logger.flush()
        except KeyboardInterrupt:
            # Never checkpoint a partially executed optimizer update.
            if not in_update:
                save_latest()
            logger.event("interrupted", progress["step"],
                         path=str(latest_path) if latest_path.exists() else None,
                         checkpointed=not in_update)
            logger.flush()
            raise
    return progress
