
"""Reproducible document-level train/validation split for FineWeb2."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data/raw/fineweb2/data/kat_Geor/train"
)

DOWNLOAD_MANIFEST = (
    PROJECT_ROOT
    / "data/raw/fineweb2/download_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/splits/fineweb2/kat_Geor"
)

VALIDATION_FRACTION = 0.01
SEED = 42
EXPECTED_DOCUMENTS = 3_706_659


def main() -> None:
    files = sorted(SOURCE_DIR.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {SOURCE_DIR}"
        )

    with DOWNLOAD_MANIFEST.open(encoding="utf-8") as file:
        download_manifest = json.load(file)

    file_info = [
        {
            "path": path.relative_to(SOURCE_DIR).as_posix(),
            "documents": pq.read_metadata(path).num_rows,
        }
        for path in files
    ]

    total_documents = sum(
        item["documents"] for item in file_info
    )

    if total_documents != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Unexpected corpus size: {total_documents}"
        )

    validation_size = round(
        total_documents * VALIDATION_FRACTION
    )

    rng = np.random.default_rng(SEED)

    validation_indices = np.sort(
        rng.choice(
            total_documents,
            size=validation_size,
            replace=False,
        )
    )

    train_size = total_documents - validation_size

    manifest = {
        "dataset": download_manifest["repo_id"],
        "subset": download_manifest["subset"],
        "revision": download_manifest["revision"],
        "method": "uniform_without_replacement",
        "index_type": "global_row",
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "total_documents": total_documents,
        "train_documents": train_size,
        "validation_documents": validation_size,
        "source_files": file_info,
        "validation_indices_file": "validation_indices.npy",
    }

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Split already exists: {OUTPUT_DIR}"
        )

    temporary_dir = OUTPUT_DIR.with_name(
        OUTPUT_DIR.name + ".partial"
    )

    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    temporary_dir.mkdir(parents=True)

    np.save(
        temporary_dir / "validation_indices.npy",
        validation_indices,
    )

    with (temporary_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_dir.rename(OUTPUT_DIR)

    print("Split completed")
    print(f"Total:      {total_documents:,}")
    print(f"Train:      {train_size:,}")
    print(f"Validation: {validation_size:,}")
    print(f"Seed:       {SEED}")
    print(f"Output:     {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
