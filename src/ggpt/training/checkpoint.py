"""Atomic, restartable trainer checkpoints with reproducible sampling and RNG."""

from pathlib import Path
import os
import random
import uuid
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos,
                         has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"].to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([
            item.to(device="cpu", dtype=torch.uint8) for item in state["cuda"]
        ])


def save_checkpoint(path: str | Path, *, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    progress: dict[str, Any], sampler_state: dict[str, int],
                    run_spec: dict[str, Any]) -> None:
    """Write a complete checkpoint, replacing the old file only after success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "run_spec": run_spec,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "progress": dict(progress),
        "sampler": dict(sampler_state),
        "rng": capture_rng_state(),
    }
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, *, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    sampler: Any, run_spec: dict[str, Any],
                    map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Restore only a checkpoint created for exactly this experiment.

    `weights_only=True` is intentional: RNG state is stored using primitives and
    tensors, avoiding arbitrary unpickling of checkpoint contents.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format version")
    if checkpoint.get("run_spec") != run_spec:
        raise ValueError("Checkpoint experiment configuration or corpus has changed")
    # Validate sampler compatibility before changing any model/optimizer state.
    state = checkpoint["sampler"]
    for key in ("seed", "dataset_length", "batch_size", "num_samples"):
        if state[key] != getattr(sampler, key):
            raise ValueError(f"Checkpoint sampler mismatch: {key}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    sampler.load_state_dict(state)
    restore_rng_state(checkpoint["rng"])
    return dict(checkpoint["progress"])
