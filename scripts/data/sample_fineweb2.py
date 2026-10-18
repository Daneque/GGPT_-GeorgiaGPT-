
"""Create a reproducible uniform sample from FineWeb2."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from ggpt.data.parquet_corpus import ParquetCorpus
from ggpt.utils.logger import get_logger, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
FINEWEB2_ROOT = DATA_ROOT / "raw" / "fineweb2"
SAMPLES_ROOT = DATA_ROOT / "samples" / "fineweb2" / "kat_Geor"

SUBSET = "kat_Geor"
SPLIT = "train"

logger = get_logger("scripts.data.sample_fineweb2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample FineWeb2 documents uniformly."
    )

    parser.add_argument(
        "--n-docs",
        type=int,
        default=10_000,
        help="Number of documents to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Parquet reading batch size.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(
        log_file=PROJECT_ROOT / "runs/logs/sample_fineweb2.log",
        console=False,
    )

    if args.n_docs <= 0:
        raise ValueError("--n-docs must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    source_dir = FINEWEB2_ROOT / "data" / SUBSET / SPLIT

    output_dir = (
        SAMPLES_ROOT
        / SPLIT
        / f"n{args.n_docs}_seed{args.seed}"
    )

    sample_path = output_dir / "sample.parquet"
    manifest_path = output_dir / "manifest.json"

    if sample_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Sample already exists in {output_dir}"
        )

    corpus = ParquetCorpus(source_dir)

    logger.info(
        "Sampling %d documents from %d train documents",
        args.n_docs,
        corpus.total_documents,
    )

    started_at = perf_counter()

    try:
        sample = corpus.sample(
            n_docs=args.n_docs,
            seed=args.seed,
            batch_size=args.batch_size,
            show_progress=True,
        )

        if len(sample) != args.n_docs:
            raise RuntimeError("Incorrect sample size")

        if not sample["_global_row"].is_unique:
            raise RuntimeError("Duplicate global row indices")

        output_dir.mkdir(parents=True, exist_ok=True)

        temporary_path = output_dir / "sample.parquet.partial"

        if temporary_path.exists():
            raise FileExistsError(temporary_path)

        sample.to_parquet(
            temporary_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

        saved_rows = pq.read_metadata(
            temporary_path
        ).num_rows

        if saved_rows != args.n_docs:
            raise RuntimeError(
                f"Expected {args.n_docs} rows, "
                f"saved {saved_rows}"
            )

        temporary_path.replace(sample_path)

        download_manifest_path = (
            FINEWEB2_ROOT / "download_manifest.json"
        )

        download_manifest = (
            json.loads(
                download_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            if download_manifest_path.exists()
            else {}
        )

        elapsed = perf_counter() - started_at

        manifest = {
            "repo_id": download_manifest.get("repo_id"),
            "revision": download_manifest.get("revision"),
            "subset": SUBSET,
            "split": SPLIT,
            "source_documents": corpus.total_documents,
            "sample_documents": len(sample),
            "seed": args.seed,
            "method": "uniform_without_replacement",
            "batch_size": args.batch_size,
            "columns": sample.columns.tolist(),
            "source_file_counts": {
                str(name): int(count)
                for name, count in sample[
                    "_source_file"
                ].value_counts().items()
            },
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        }

        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Sample saved: %s (%d documents, %.2f seconds)",
            sample_path,
            len(sample),
            elapsed,
        )

        print("\nSample created successfully")
        print(f"Documents: {len(sample):,}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Sample: {sample_path}")
        print(f"Manifest: {manifest_path}")
        print(
            f"Parquet size: "
            f"{sample_path.stat().st_size / 1024**2:.2f} MiB"
        )

    except Exception:
        logger.exception("Failed to create FineWeb2 sample")
        raise


if __name__ == "__main__":
    main()
