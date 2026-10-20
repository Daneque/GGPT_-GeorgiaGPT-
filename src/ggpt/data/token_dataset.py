"""Read-only, lazy, process-local memory mapping of little-endian uint16."""
from pathlib import Path
import os
import operator
from typing import Any
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset[dict[str, Tensor]]):
    """All possible windows, or a fixed evenly spaced validation subset."""

    def __init__(self, path: str | Path, context_length: int = 512,
                 vocab_size: int = 16_384, validation_windows: int | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if type(context_length) is not int or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        if type(vocab_size) is not int or not 3 <= vocab_size <= 65_536:
            raise ValueError("vocab_size must be in [3, 65536]")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        size = self.path.stat().st_size
        if size % 2:
            raise ValueError("uint16 file size must be divisible by two")
        self.num_tokens = size // 2
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.num_starts = self.num_tokens - context_length
        if self.num_starts <= 0:
            raise ValueError("file must contain at least context_length + 1 tokens")
        if validation_windows is not None:
            if type(validation_windows) is not int or validation_windows <= 0:
                raise ValueError("validation_windows must be a positive integer")
        self.validation_windows = validation_windows
        self._length = (self.num_starts if validation_windows is None
                        else min(validation_windows, self.num_starts))
        self._mapping: np.memmap | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return self._length

    def window_start(self, index: int) -> int:
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.validation_windows is None or len(self) == 1:
            return index
        return index * (self.num_starts - 1) // (len(self) - 1)

    def _tokens(self) -> np.memmap:
        pid = os.getpid()
        if self._mapping is None or self._pid != pid:
            if self.path.stat().st_size != self.num_tokens * 2:
                raise RuntimeError("token file size changed after dataset construction")
            self._mapping = np.memmap(self.path, dtype="<u2", mode="r",
                                      shape=(self.num_tokens,))
            self._pid = pid
        return self._mapping

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.window_start(index)
        # One small conversion per window; neither view aliases read-only mmap bytes.
        values = np.array(self._tokens()[start:start + self.context_length + 1],
                          dtype=np.int64, copy=True)
        if values.max() >= self.vocab_size:
            raise ValueError(f"out-of-vocabulary token in window starting at {start}")
        tokens = torch.from_numpy(values)
        return {"input_ids": tokens[:-1], "targets": tokens[1:]}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mapping"] = None
        state["_pid"] = None
        return state

