"""Deterministic, restartable minibatches over memory-mapped token windows."""
from dataclasses import dataclass
from pathlib import Path
import math
import random
from collections.abc import Iterator, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from .token_dataset import TokenWindowDataset


@dataclass(frozen=True)
class LoaderConfig:
    context_length: int = 512
    batch_size: int = 8
    train_windows: int = 10_000
    validation_windows: int = 256
    seed: int = 42
    num_workers: int = 0
    vocab_size: int = 16_384
    pin_memory: bool = False

    def __post_init__(self) -> None:
        for name in ("context_length", "batch_size", "train_windows", "validation_windows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if type(self.vocab_size) is not int or not 3 <= self.vocab_size <= 65_536:
            raise ValueError("vocab_size must be in [3, 65536]")
        if type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be boolean")


class ResumableRandomBatchSampler(Sampler[list[int]]):
    """Uniform-with-replacement minibatches, reproducible by (seed, epoch, batch).

    A batch has its own deterministic NumPy RNG. Worker prefetch never consumes
    the state of future batches, so resuming from the *last consumed* batch is
    exact even if num_workers differs. Caller owns the consumed-batch counter.
    """

    def __init__(self, dataset_length: int, batch_size: int, num_samples: int,
                 seed: int, *, epoch: int = 0, start_batch: int = 0) -> None:
        for name, value in (("dataset_length", dataset_length), ("batch_size", batch_size),
                            ("num_samples", num_samples)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        self.set_epoch(epoch)
        self.set_start_batch(start_batch)

    @property
    def total_batches(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index in range(self.start_batch, self.total_batches):
            count = min(self.batch_size, self.num_samples - batch_index * self.batch_size)
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, self.epoch, batch_index]
            ))
            yield rng.integers(0, self.dataset_length, size=count).tolist()

    def set_epoch(self, epoch: int) -> None:
        """Start a new logical pass; train_windows is not the corpus size."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch
        self.start_batch = 0

    def set_start_batch(self, next_batch: int) -> None:
        if type(next_batch) is not int or not 0 <= next_batch <= self.total_batches:
            raise ValueError(f"next_batch must be in [0, {self.total_batches}]")
        self.start_batch = next_batch

    def state_dict(self, *, next_batch: int) -> dict[str, int]:
        """Snapshot an EXPLICIT count of batches handed to/used by the trainer.

        Do not snapshot iterator/sampler progress: multiprocessing can prefetch
        batches that training has not consumed. Call after a completed batch.
        """
        if type(next_batch) is not int or not self.start_batch <= next_batch <= self.total_batches:
            raise ValueError(f"next_batch must be in [{self.start_batch}, {self.total_batches}]")
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch": next_batch,
            "dataset_length": self.dataset_length,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        """Restore batch boundary; reject mismatched sampling configuration."""
        required = {"seed", "epoch", "next_batch", "dataset_length", "batch_size", "num_samples"}
        if set(state) != required:
            raise ValueError(f"sampler state must have keys {sorted(required)}")
        for field in ("seed", "dataset_length", "batch_size", "num_samples"):
            if type(state[field]) is not int or state[field] != getattr(self, field):
                raise ValueError(f"sampler state mismatch: {field}")
        if type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("invalid sampler epoch")
        if type(state["next_batch"]) is not int or not 0 <= state["next_batch"] <= self.total_batches:
            raise ValueError("invalid sampler next_batch")
        self.set_epoch(state["epoch"])
        self.set_start_batch(state["next_batch"])


def seed_worker(worker_id: int) -> None:
    """Top-level function is picklable with multiprocessing spawn."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(
    config: LoaderConfig,
    train_path: str | Path = "data/tokenized/kat_Geor/train.bin",
    val_path: str | Path = "data/tokenized/kat_Geor/val.bin",
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
    train_path, val_path = Path(train_path).resolve(), Path(val_path).resolve()
    if train_path == val_path or train_path.samefile(val_path):
        raise ValueError("train and validation must be different files")
    train = TokenWindowDataset(train_path, config.context_length, config.vocab_size)
    val = TokenWindowDataset(val_path, config.context_length, config.vocab_size,
                             validation_windows=config.validation_windows)
    sampler = ResumableRandomBatchSampler(
        len(train), config.batch_size, config.train_windows, config.seed
    )
    common = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
                  worker_init_fn=seed_worker)
    train_loader = DataLoader(train, batch_sampler=sampler, **common,
                              generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False,
                            drop_last=False, **common,
                            generator=torch.Generator().manual_seed(config.seed))
    return train_loader, val_loader

