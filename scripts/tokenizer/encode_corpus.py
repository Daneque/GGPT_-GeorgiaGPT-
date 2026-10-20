"""Encode FineWeb2 train into resumable uint16 token streams."""

import argparse
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

from ggpt.utils.logger import get_logger, setup_logging


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/raw/fineweb2/data/kat_Geor/train"
DOWNLOAD_MANIFEST = ROOT / "data/raw/fineweb2/download_manifest.json"
SPLIT_DIR = ROOT / "data/splits/fineweb2/kat_Geor"
TOKENIZER_DIR = ROOT / "artifacts/tokenizer/bpe_100k_16384"
OUTPUT_DIR = ROOT / "data/tokenized/kat_Geor"
LOG_FILE = ROOT / "runs/logs/tokenizer/encode_corpus.log"

logger = get_logger("scripts.tokenizer.encode_corpus")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def checkpoint(train_file, val_file, state_path: Path, state: dict) -> None:
    for file in (train_file, val_file):
        file.flush()
        os.fsync(file.fileno())
    save_json(state_path, state)


def encode_corpus(output_dir: Path, batch_size: int,
                  checkpoint_groups: int, max_groups: int | None) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("This encoder requires a little-endian machine")
    if batch_size <= 0 or checkpoint_groups <= 0:
        raise ValueError("Batch size and checkpoint groups must be positive")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive")

    split = read_json(SPLIT_DIR / "manifest.json")
    download = read_json(DOWNLOAD_MANIFEST)
    model_manifest = read_json(TOKENIZER_DIR / "manifest.json")
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != 16_384 or model_manifest["actual_vocab_size"] != vocab_size:
        raise RuntimeError("Unexpected BPE vocabulary size")
    if vocab_size >= 65_536:
        raise RuntimeError("Vocabulary is too large for uint16")
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    pad = tokenizer.token_to_id("<|pad|>")
    if (bos, eos, pad) != (0, 1, 2):
        raise RuntimeError("Unexpected special token IDs")
    if split["revision"] != download["revision"]:
        raise RuntimeError("Dataset revision mismatch")
    if (split["dataset"], split["subset"]) != (download["repo_id"], download["subset"]):
        raise RuntimeError("Dataset identity mismatch")

    validation_path = SPLIT_DIR / "validation_indices.npy"
    validation = np.load(validation_path, allow_pickle=False)
    if (len(validation) != split["validation_documents"]
            or np.any(np.diff(validation) <= 0)
            or validation[0] < 0
            or validation[-1] >= split["total_documents"]):
        raise RuntimeError("Invalid validation split")

    groups = []
    source_files = []
    for source in split["source_files"]:
        path = SOURCE_DIR / source["path"]
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != source["documents"]:
                raise RuntimeError(f"Source size mismatch: {path}")
            for rg in range(parquet.num_row_groups):
                groups.append((path, rg, parquet.metadata.row_group(rg).num_rows))
        source_files.append({
            "path": source["path"],
            "rows": source["documents"],
            "bytes": path.stat().st_size,
        })

    if sum(g[2] for g in groups) != split["total_documents"]:
        raise RuntimeError("Incorrect source row-group totals")

    target_groups = min(len(groups), max_groups or len(groups))
    expected_docs = sum(g[2] for g in groups[:target_groups])
    signature_data = {
        "encoder_source_sha256": sha256(Path(__file__)),
        "tokenizer_sha256": sha256(tokenizer_path),
        "tokenizer_manifest_sha256": sha256(TOKENIZER_DIR / "manifest.json"),
        "validation_sha256": sha256(validation_path),
        "dataset_revision": download["revision"],
        "source_files": source_files,
        "target_groups": target_groups,
        "encoding": "uint16_le_bos_text_eos_v1",
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    for name in ("progress.json.tmp", "manifest.json.tmp"):
        (output_dir / name).unlink(missing_ok=True)

    if state_path.exists():
        state = read_json(state_path)
        if state.get("signature") != signature:
            raise RuntimeError("Encoding inputs changed: use a new output directory")
        if state.get("status") == "complete":
            if (not manifest_path.is_file()
                    or train_path.stat().st_size != state["train_tokens"] * 2
                    or val_path.stat().st_size != state["val_tokens"] * 2):
                raise RuntimeError("Completed encoding is inconsistent")
            logger.info("Encoding already complete: %s", output_dir)
            print("Encoding already complete", output_dir)
            return
        if state["completed_groups"] > target_groups:
            raise RuntimeError("Invalid checkpoint group index")
        if sum(g[2] for g in groups[:state["completed_groups"]]) != state["processed_documents"]:
            raise RuntimeError("Checkpoint document position mismatch")
        if state["train_documents"] + state["val_documents"] != state["processed_documents"]:
            raise RuntimeError("Checkpoint document counts mismatch")
        logger.info("Resuming after %d row groups", state["completed_groups"])
    else:
        if any(p.exists() for p in (train_path, val_path, manifest_path)):
            raise RuntimeError("Output directory contains data without a checkpoint")
        state = {
            "signature": signature,
            "status": "running",
            "completed_groups": 0,
            "processed_documents": 0,
            "train_documents": 0,
            "val_documents": 0,
            "train_tokens": 0,
            "val_tokens": 0,
        }
        save_json(state_path, state)

    if not train_path.exists() or not val_path.exists():
        if state["completed_groups"]:
            raise RuntimeError("Encoded files missing for existing checkpoint")
        train_path.touch(exist_ok=True)
        val_path.touch(exist_ok=True)

    for path, token_count in ((train_path, state["train_tokens"]),
                              (val_path, state["val_tokens"])):
        if path.stat().st_size < token_count * 2:
            raise RuntimeError(f"Truncated encoding file: {path}")

    manifest_path.unlink(missing_ok=True)
    started = perf_counter()
    logger.info("Start: %d groups, %d documents, batch_size=%d, checkpoint_groups=%d",
                target_groups, expected_docs, batch_size, checkpoint_groups)

    with train_path.open("r+b") as train_file, val_path.open("r+b") as val_file:
        train_file.truncate(state["train_tokens"] * 2)
        val_file.truncate(state["val_tokens"] * 2)
        train_file.seek(0, os.SEEK_END)
        val_file.seek(0, os.SEEK_END)
        global_row = state["processed_documents"]
        group_index = 0

        with tqdm(total=target_groups, initial=state["completed_groups"],
                  unit="groups", desc="Encoding FineWeb2") as progress:
            for source in split["source_files"]:
                path = SOURCE_DIR / source["path"]
                with pq.ParquetFile(path) as parquet:
                    for rg in range(parquet.num_row_groups):
                        if group_index < state["completed_groups"]:
                            group_index += 1
                            continue
                        if group_index >= target_groups:
                            break
                        row_group_start = global_row
                        for batch in parquet.iter_batches(
                            row_groups=[rg], batch_size=batch_size,
                            columns=["text"],
                        ):
                            texts = batch.column(0).to_pylist()
                            if any(t is None for t in texts):
                                raise RuntimeError("Null document text")
                            end = global_row + len(texts)
                            lo = int(np.searchsorted(validation, global_row))
                            hi = int(np.searchsorted(validation, end))
                            is_val = np.zeros(len(texts), dtype=bool)
                            is_val[validation[lo:hi] - global_row] = True

                            encodings = tokenizer.encode_batch(
                                texts, add_special_tokens=False
                            )
                            train_ids, val_ids = array("H"), array("H")
                            for encoding, validation_document in zip(encodings, is_val):
                                target = val_ids if validation_document else train_ids
                                target.append(bos)
                                target.extend(encoding.ids)
                                target.append(eos)
                            train_file.write(train_ids.tobytes())
                            val_file.write(val_ids.tobytes())
                            n_val = int(is_val.sum())
                            state["val_documents"] += n_val
                            state["train_documents"] += len(texts) - n_val
                            state["train_tokens"] += len(train_ids)
                            state["val_tokens"] += len(val_ids)
                            global_row = end
                        if global_row - row_group_start != groups[group_index][2]:
                            raise RuntimeError("Row group document count mismatch")
                        group_index += 1
                        state["processed_documents"] = global_row
                        progress.update(1)
                        if (group_index % checkpoint_groups == 0
                                or group_index == target_groups):
                            state["completed_groups"] = group_index
                            checkpoint(train_file, val_file, state_path, state)
                            logger.info("Checkpoint: groups=%d/%d, docs=%d, train_tokens=%d, val_tokens=%d",
                                        group_index, target_groups, global_row,
                                        state["train_tokens"], state["val_tokens"])
                if group_index >= target_groups:
                    break

        if state["processed_documents"] != expected_docs:
            raise RuntimeError("Incomplete encoding")
        if state["train_documents"] + state["val_documents"] != expected_docs:
            raise RuntimeError("Incorrect split document counts")
        if target_groups == len(groups):
            if state["train_documents"] != split["train_documents"]:
                raise RuntimeError("Train document count mismatch")
            if state["val_documents"] != split["validation_documents"]:
                raise RuntimeError("Validation document count mismatch")
        checkpoint(train_file, val_file, state_path, state)

    manifest = {
        "dataset": download["repo_id"],
        "subset": download["subset"],
        "revision": download["revision"],
        "tokenizer": str(tokenizer_path.relative_to(ROOT)),
        "tokenizer_sha256": signature_data["tokenizer_sha256"],
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "endianness": "little",
        "bos_id": bos,
        "eos_id": eos,
        "pad_id": pad,
        "encoded_groups": target_groups,
        "processed_documents": expected_docs,
        "train_documents": state["train_documents"],
        "validation_documents": state["val_documents"],
        "train_tokens": state["train_tokens"],
        "validation_tokens": state["val_tokens"],
        "signature": signature,
    }
    save_json(manifest_path, manifest)
    state["status"] = "complete"
    save_json(state_path, state)
    elapsed = perf_counter() - started
    logger.info("Completed: elapsed=%.1fs, train_tokens=%d, val_tokens=%d",
                elapsed, state["train_tokens"], state["val_tokens"])
    print("Encoding: OK")
    print(f"Documents: {expected_docs:,} (train={state['train_documents']:,}, val={state['val_documents']:,})")
    print(f"Tokens: train={state['train_tokens']:,}, val={state['val_tokens']:,}")
    print(f"Elapsed this run: {elapsed:.1f} s")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/tokenized/kat_Geor"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-groups", type=int, default=20)
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args()
    destination = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    encode_corpus(destination, args.batch_size,
                  args.checkpoint_groups, args.max_groups)


if __name__ == "__main__":
    setup_logging(log_file=LOG_FILE, console=False)
    logger = get_logger("scripts.tokenizer.encode_corpus")
    try:
        main()
    except Exception:
        logger.exception("Corpus encoding failed")
        raise
