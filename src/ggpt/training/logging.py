"""Append-only JSONL and TensorBoard metrics for long training runs."""

from datetime import datetime, timezone
import os
import uuid
import json
from pathlib import Path
from typing import Any

from ggpt.utils.logger import get_logger


def reconcile_metrics_on_resume(path: Path, resume_step: int) -> int:
    """Drop superseded JSONL events after checkpoint rollback, atomically.

    TensorBoard handles the equivalent via SummaryWriter(purge_step=...).
    This prevents duplicate future-step metrics from corrupting later analysis.
    """
    if not path.exists():
        return 0
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    removed = 0
    try:
        with path.open("r", encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip():
                    continue
                entry = json.loads(line)
                step = entry.get("step")
                if type(step) is int and step > resume_step:
                    removed += 1
                else:
                    out.write(line)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return removed


class TrainingLogger:
    def __init__(self, run_dir: str | Path, *, resume_step: int | None = None,
                 writer: Any | None = None) -> None:
        self.logger = get_logger("training")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.run_dir / "metrics.jsonl"
        if resume_step is not None:
            reconcile_metrics_on_resume(metrics_path, resume_step)
        self.stream = metrics_path.open("a", encoding="utf-8", buffering=1)
        if writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                self.stream.close()
                raise RuntimeError(
                    'Install TensorBoard: uv pip install --python "$(which python)" tensorboard'
                ) from exc
            kwargs = {"purge_step": resume_step + 1} if resume_step is not None else {}
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"), **kwargs)
        else:
            self.writer = writer

    def event(self, kind: str, step: int, **values: Any) -> None:
        obj = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "type": kind, "step": step, **values}
        self.stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        if kind in {"start", "resume", "checkpoint", "finished", "paused", "interrupted"}:
            self.logger.info("%s step=%d %s", kind, step, values)

    def scalars(self, kind: str, step: int, **metrics: float | int) -> None:
        self.event(kind, step, **metrics)
        if kind in {"train", "val"}:
            self.logger.info("%s step=%d %s", kind, step, " ".join(f"{k}={v}" for k, v in metrics.items()))
        for name, value in metrics.items():
            self.writer.add_scalar(f"{kind}/{name}", value, step)

    def generated(self, step: int, *, label: str, prompt: str, text: str,
                  generation_config: dict[str, Any]) -> None:
        self.event("generation", step, label=label, prompt=prompt,
                   text=text, generation_config=generation_config)
        self.writer.add_text(f"generation/{label}",
                             f"Prompt: {prompt}\n\nResult: {text}", step)

    def flush(self) -> None:
        self.stream.flush()
        self.writer.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.writer.close()
            self.stream.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
