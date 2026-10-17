"""Download the complete filtered Georgian FineWeb2 subset."""


import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "HuggingFaceFW/fineweb-2"
SUBSET = "kat_Geor"
SPLITS = ("train", "test")
MAX_WORKERS = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "fineweb2"
MANIFEST_PATH = OUTPUT_DIR / "download_manifest.json"


def get_revision() -> str:
    """Reuse the pinned dataset revision on repeated runs."""
    if MANIFEST_PATH.exists():
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        if (
            manifest.get("repo_id") != REPO_ID
            or manifest.get("subset") != SUBSET
        ):
            raise ValueError("Existing manifest belongs to another dataset.")

        return manifest["revision"]

    return HfApi().dataset_info(REPO_ID).sha


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    revision = get_revision()

    patterns = [
        f"data/{SUBSET}/{split}/*.parquet"
        for split in SPLITS
    ]

    print(f"Dataset: {REPO_ID} / {SUBSET}")
    print(f"Revision: {revision}")
    print(f"Destination: {OUTPUT_DIR}")
    print("Downloading compressed Parquet files.")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=OUTPUT_DIR,
        allow_patterns=patterns,
        max_workers=MAX_WORKERS
    )

    counts = {}

    for split in SPLITS:
        split_dir = OUTPUT_DIR / "data" / SUBSET / split
        files = sorted(split_dir.rglob("*.parquet"))

        if not files:
            raise RuntimeError(f"No parquets files: {split_dir}")

        counts[split] = len(files)
        print(f"{split}: {len(files)} files.")

    manifest = {
        "repo_id": REPO_ID,
        "subset": SUBSET,
        "revision": revision,
        "splits": list(SPLITS),
        "file_counts": counts,
        "downloaded_at_utc": datetime.now(
            timezone.uts
        ).isoformat(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"Download complete. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
