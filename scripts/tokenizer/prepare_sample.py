
"""Prepare a reproducible training sample for the BPE tokenizer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ggpt.utils.logger import get_logger, setup_logging


ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = ROOT / "data/raw/fineweb2/data/kat_Geor/train"
DOWNLOAD_MANIFEST = ROOT / "data/raw/fineweb2/download_manifest.json"
SPLIT_DIR = ROOT / "data/splits/fineweb2/kat_Geor"

OUTPUT_DIR = ROOT / "data/interim/tokenizer/bpe_sample"
LOG_FILE = ROOT / "runs/logs/tokenizer/prepare_sample.log"

logger = get_logger("scripts.tokenizer.prepare_sample")


def prepare_sample(
    n_docs: int,
    seed: int,
    batch_size: int,
) -> None:
    if n_docs <= 0 or batch_size <= 0:
        raise ValueError("n_docs and batch_size must be positive")

    started = perf_counter()

    logger.info(
        "Starting sample preparation: n_docs=%d, seed=%d, batch_size=%d",
        n_docs,
        seed,
        batch_size,
    )

    files = sorted(SOURCE_DIR.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(SOURCE_DIR)

    source_files = [
        {
            "path": path.relative_to(SOURCE_DIR).as_posix(),
            "documents": pq.read_metadata(path).num_rows,
        }
        for path in files
    ]

    total_documents = sum(
        file_info["documents"]
        for file_info in source_files
    )

    split_manifest = json.loads(
        (SPLIT_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    download_manifest = json.loads(
        DOWNLOAD_MANIFEST.read_text(encoding="utf-8")
    )

    if total_documents != split_manifest["total_documents"]:
        raise RuntimeError("Corpus size does not match the split")

    if source_files != split_manifest["source_files"]:
        raise RuntimeError("Source file order differs from the split")

    validation_path = SPLIT_DIR / "validation_indices.npy"

    validation_indices = np.load(
        validation_path,
        allow_pickle=False,
    )

    if (
        len(validation_indices) != split_manifest["validation_documents"]
        or np.any(validation_indices < 0)
        or np.any(validation_indices >= total_documents)
        or np.any(np.diff(validation_indices) <= 0)
    ):
        raise RuntimeError("Invalid validation indices")

    train_indices = np.setdiff1d(
        np.arange(total_documents, dtype=np.int64),
        validation_indices,
        assume_unique=True,
    )

    if len(train_indices) != split_manifest["train_documents"]:
        raise RuntimeError("Unexpected train size")

    if n_docs > len(train_indices):
        raise ValueError("Requested sample exceeds train size")

    logger.info(
        "Corpus: total=%d, train=%d, validation=%d",
        total_documents,
        len(train_indices),
        len(validation_indices),
    )

    rng = np.random.default_rng(seed)

    selected = np.sort(
        rng.choice(
            train_indices,
            size=n_docs,
            replace=False,
        )
    )

    if np.intersect1d(selected, validation_indices).size:
        raise RuntimeError("Validation leakage detected")

    logger.info("Selected %d unique train documents", len(selected))

    temporary_dir = OUTPUT_DIR.with_name(
        OUTPUT_DIR.name + ".partial"
    )

    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)

    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    temporary_dir.mkdir(parents=True)

    schema = pa.schema([
        ("_global_row", pa.int64()),
        ("text", pa.string()),
    ])

    offset = 0
    cursor = 0

    try:
        with pq.ParquetWriter(
            temporary_dir / "sample.parquet",
            schema=schema,
            compression="zstd",
        ) as writer:
            with tqdm(
                total=total_documents,
                desc="Preparing BPE sample",
                unit="docs",
            ) as progress:
                for path in files:
                    with pq.ParquetFile(path) as parquet:
                        for batch in parquet.iter_batches(
                            batch_size=batch_size,
                            columns=["text"],
                        ):
                            end = offset + batch.num_rows

                            next_cursor = int(
                                np.searchsorted(selected, end)
                            )

                            if next_cursor > cursor:
                                chosen_rows = selected[cursor:next_cursor]
                                positions = chosen_rows - offset

                                texts = batch.column(0).take(
                                    pa.array(
                                        positions,
                                        type=pa.int64(),
                                    )
                                )

                                if texts.null_count:
                                    raise RuntimeError(
                                        "Null text in selected documents"
                                    )

                                output_table = pa.Table.from_arrays(
                                    [
                                        pa.array(
                                            chosen_rows,
                                            type=pa.int64(),
                                        ),
                                        texts,
                                    ],
                                    schema=schema,
                                )

                                writer.write_table(output_table)
                                cursor = next_cursor

                            offset = end
                            progress.update(batch.num_rows)

                    logger.info(
                        "Processed %s: scanned=%d, selected=%d",
                        path.name,
                        offset,
                        cursor,
                    )

        if offset != total_documents:
            raise RuntimeError("Incomplete corpus scan")

        if cursor != n_docs:
            raise RuntimeError("Incomplete sample extraction")

        sample_path = temporary_dir / "sample.parquet"

        if pq.read_metadata(sample_path).num_rows != n_docs:
            raise RuntimeError("Incorrect output row count")

        elapsed = perf_counter() - started

        manifest = {
            "dataset": download_manifest["repo_id"],
            "subset": download_manifest["subset"],
            "revision": download_manifest["revision"],
            "source_files": source_files,
            "source_documents": total_documents,
            "train_documents": len(train_indices),
            "sample_documents": n_docs,
            "seed": seed,
            "method": "uniform_without_replacement_from_train",
            "validation_indices_sha256": hashlib.sha256(
                validation_path.read_bytes()
            ).hexdigest(),
            "batch_size": batch_size,
            "sample_file": "sample.parquet",
            "pyarrow_version": pa.__version__,
        }

        (temporary_dir / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_dir.rename(OUTPUT_DIR)

    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    logger.info(
        "Sample preparation completed: documents=%d, elapsed=%.2f s",
        n_docs,
        elapsed,
    )

    logger.info("Output: %s", OUTPUT_DIR)

    print(f"Sample preparation: OK")
    print(f"Documents: {n_docs:,}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Output: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-docs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=4096)

    args = parser.parse_args()

    prepare_sample(
        n_docs=args.n_docs,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    setup_logging(
        log_file=LOG_FILE,
        console=False,
    )

    try:
        main()
    except Exception:
        logger.exception("BPE sample preparation failed")
        raise
