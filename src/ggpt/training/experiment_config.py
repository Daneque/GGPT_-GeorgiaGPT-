"""Strict, TOML-driven experiment definitions; v1 checkpoints remain readable."""

from dataclasses import dataclass, fields
from pathlib import Path
import math
import tomllib
from typing import Any

from ggpt.model import ModelConfig
from .config import OptimizerConfig
from .trainer import TrainerConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    runs_dir: Path
    device: str
    train_path: Path
    val_path: Path
    tokenizer_path: Path
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    additional_steps: int | None = None
    ramp_steps: int | None = None
    peak_lr: float | None = None
    final_lr: float | None = None


def _section(data: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    if name not in data or not isinstance(data[name], dict):
        raise ValueError(f"Missing or invalid [{name}] section")
    unknown = set(data[name]) - allowed
    if unknown:
        raise ValueError(f"Unknown [{name}] keys: {sorted(unknown)}")
    return dict(data[name])


def _require_fields(section: dict[str, Any], required: set[str], name: str) -> None:
    missing = required - set(section)
    if missing:
        raise ValueError(f"Missing [{name}] keys: {sorted(missing)}")


def _must_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite number")
    return float(value)


def _project_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_experiment_config(file: str | Path, *, mode: str,
                           project_root: Path) -> ExperimentConfig:
    """Parse a start/extend TOML; relative paths resolve from project_root."""
    if mode not in {"start", "extend"}:
        raise ValueError("mode must be start or extend")
    with Path(file).open("rb") as handle:
        data = tomllib.load(handle)
    sections = {"run", "model", "data", "training", "optimizer", "scheduler", "monitoring"}
    if set(data) != sections:
        raise ValueError(f"Expected TOML sections {sorted(sections)}; got {sorted(data)}")

    run = _section(data, "run", {"name", "runs_dir"})
    name = run.get("name")
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("run.name must be a simple directory name")
    _require_fields(run, {"name", "runs_dir"}, "run")
    runs_dir = _project_path(run.get("runs_dir"), project_root, "run.runs_dir")

    model_data = _section(data, "model", {f.name for f in fields(ModelConfig)})
    _require_fields(model_data, {f.name for f in fields(ModelConfig)}, "model")
    model = ModelConfig(**model_data)

    data_cfg = _section(data, "data", {"train_path", "val_path", "tokenizer_path",
                                       "train_windows", "validation_windows", "num_workers"})
    _require_fields(data_cfg, {"train_path", "val_path", "tokenizer_path",
                               "train_windows", "validation_windows", "num_workers"}, "data")
    train_path = _project_path(data_cfg.pop("train_path", None), project_root, "data.train_path")
    val_path = _project_path(data_cfg.pop("val_path", None), project_root, "data.val_path")
    tokenizer_path = _project_path(data_cfg.pop("tokenizer_path", None), project_root, "data.tokenizer_path")

    trainer_keys = {f.name for f in fields(TrainerConfig)}
    runtime = _section(data, "training", {
        "max_steps", "additional_steps", "device", "micro_batch_size",
        "accumulation_steps", "precision", "seed", "grad_clip",
    })
    _require_fields(runtime, {"device", "micro_batch_size", "accumulation_steps",
                              "precision", "seed", "grad_clip"}, "training")
    device = runtime.pop("device")
    if not isinstance(device, str) or not device:
        raise ValueError("training.device must be a nonempty string")
    if mode == "start":
        if "additional_steps" in runtime:
            raise ValueError("start accepts max_steps, not additional_steps")
        max_steps = _must_positive_int(runtime.pop("max_steps", None), "training.max_steps")
        additional_steps = None
    else:
        if "max_steps" in runtime:
            raise ValueError("extend accepts additional_steps, not max_steps")
        additional_steps = _must_positive_int(runtime.pop("additional_steps", None),
                                              "training.additional_steps")
        max_steps = 1  # Replaced by (source checkpoint step + additional_steps).

    opt_cfg = _section(data, "optimizer", {"name", "learning_rate", "weight_decay",
                                            "beta1", "beta2", "eps", "fused"})
    _require_fields(opt_cfg, {"name", "learning_rate", "weight_decay", "beta1",
                              "beta2", "eps"}, "optimizer")
    if opt_cfg.pop("name", None) != "adamw":
        raise ValueError("optimizer.name must be adamw")
    if "beta1" in opt_cfg or "beta2" in opt_cfg:
        if not ("beta1" in opt_cfg and "beta2" in opt_cfg):
            raise ValueError("optimizer.beta1 and beta2 must be provided together")
        opt_cfg["betas"] = (opt_cfg.pop("beta1"), opt_cfg.pop("beta2"))
    optimizer = OptimizerConfig(**opt_cfg)

    sched = _section(data, "scheduler", {"name", "warmup_steps", "min_lr_ratio",
                                         "ramp_steps", "peak_lr", "final_lr"})
    monitoring = _section(data, "monitoring", {"log_every", "eval_every",
                                               "checkpoint_every", "generate_every",
                                               "generation_max_tokens"})
    _require_fields(monitoring, {"log_every", "eval_every", "checkpoint_every",
                                 "generate_every", "generation_max_tokens"}, "monitoring")
    if mode == "start":
        if sched.pop("name", None) != "warmup_cosine":
            raise ValueError("start requires scheduler.name = warmup_cosine")
        if set(sched) - {"warmup_steps", "min_lr_ratio"}:
            raise ValueError("start scheduler accepts only warmup_steps and min_lr_ratio")
        _require_fields(sched, {"warmup_steps", "min_lr_ratio"}, "scheduler")
        ramp_steps = peak_lr = final_lr = None
        trainer_sched = sched
    else:
        if sched.pop("name", None) != "continuation_cosine":
            raise ValueError("extend requires scheduler.name = continuation_cosine")
        if set(sched) != {"ramp_steps", "peak_lr", "final_lr"}:
            raise ValueError("extend scheduler needs ramp_steps, peak_lr and final_lr")
        ramp_steps = _must_positive_int(sched["ramp_steps"], "scheduler.ramp_steps")
        peak_lr = _finite_positive(sched["peak_lr"], "scheduler.peak_lr")
        final_lr = _finite_positive(sched["final_lr"], "scheduler.final_lr")
        trainer_sched = {}

    args = dict(data_cfg, **runtime, **monitoring, **trainer_sched,
                max_steps=max_steps)
    unknown_args = set(args) - trainer_keys
    if unknown_args:
        raise ValueError(f"Unknown TrainerConfig arguments: {sorted(unknown_args)}")
    # The source global step is unknown until its checkpoint is opened.
    # Use a valid provisional horizon only for parsing/validation; it is
    # replaced with (checkpoint_step + additional_steps) before any run starts.
    if mode == "extend":
        provisional = additional_steps + max(TrainerConfig().warmup_steps, ramp_steps) + 1
        args["max_steps"] = provisional
    trainer = TrainerConfig(**args)
    return ExperimentConfig(name=name, runs_dir=runs_dir, device=device,
                            train_path=train_path, val_path=val_path,
                            tokenizer_path=tokenizer_path, model=model,
                            optimizer=optimizer, trainer=trainer,
                            additional_steps=additional_steps, ramp_steps=ramp_steps,
                            peak_lr=peak_lr, final_lr=final_lr)


def saved_configs(spec: dict[str, Any]) -> tuple[ModelConfig, OptimizerConfig, TrainerConfig]:
    """Reconstruct v1/v2 Python configs directly from stored run_spec.json."""
    model = ModelConfig.from_dict(spec["model"])
    opt_data = dict(spec["optimizer"])
    if "betas" in opt_data:
        opt_data["betas"] = tuple(opt_data["betas"])
    optimizer = OptimizerConfig(**opt_data)
    trainer = TrainerConfig(**spec["trainer"])
    return model, optimizer, trainer
