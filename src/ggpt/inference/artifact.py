"""Self-contained, inference-only GGPT artifact; independent of benchmarking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from ggpt.model import ModelConfig, create_model


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(run: str | Path, output: str | Path, *, checkpoint: str = "latest") -> Path:
    """Export weights/config/tokenizer only; never optimizer, scheduler or training RNG.

    Requires an existing completed run directory. Output must not exist.
    """
    if checkpoint not in {"latest", "best"}:
        raise ValueError("checkpoint must be latest or best")
    run = Path(run).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    spec = json.loads((run / "run_spec.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(spec["model"])
    tokenizer = Path(spec["tokenizer"]["path"]).expanduser().resolve()
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer}")
    path = run / "checkpoints" / f"{checkpoint}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload["run_spec"] != spec:
        raise ValueError("Checkpoint/run_spec mismatch")
    state = payload["model"]
    if not isinstance(state, dict) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Checkpoint does not contain a tensor-only model state_dict")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as folder:
        tmp = Path(folder)
        torch.save({key: value.detach().cpu() for key, value in state.items()}, tmp / "model.pt")
        shutil.copyfile(tokenizer, tmp / "tokenizer.json")
        metadata: dict[str, Any] = {
            "format": "ggpt-inference-v1",
            "model": config.to_dict(),
            "step": int(payload["progress"]["step"]),
            "source_checkpoint": checkpoint,
            "files": {
                "model.pt": file_sha256(tmp / "model.pt"),
                "tokenizer.json": file_sha256(tmp / "tokenizer.json"),
            },
        }
        (tmp / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Rename only once all files have been written; prevent partial exports.
        tmp.rename(output)
    return output


def load_model_artifact(path: str | Path, *, device: str = "cpu") -> tuple[Any, Any, dict]:
    """Return (eval model, tokenizer, verified manifest) from exported files."""
    from tokenizers import Tokenizer

    root = Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "ggpt-inference-v1":
        raise ValueError("Unsupported GGPT inference artifact format")
    for name in ("model.pt", "tokenizer.json"):
        expected = manifest["files"][name]
        if file_sha256(root / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    config = ModelConfig.from_dict(manifest["model"])
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    destination = torch.device(device)
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = create_model(config)
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.to(destination).eval()
    return model, tokenizer, manifest
