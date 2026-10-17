"""Download the complete filtered Georgian FineWeb2 subset."""


import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


from huggingface_hub import HfApi, snapshot_download

from ggpt.utils.logger import setup_logging, get_logger


REPO_ID = "HuggingFaceFW/fineweb-2"
SUBSET = "kat_Geor"
SPLITS = ("train", "test")
MAX_WORKERS = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "fineweb2"
MANIFEST_PATH = OUTPUT_DIR / "download_manifest.json"

LOG_PATH = PROJECT_ROOT / "runs" / "logs" / "download_fineweb2.log"
logger = get_logger("scripts.data.download_fineweb2")


def get_revision() -> str:
    """Reuse the pinned dataset revision on repeated runs."""
    if MANIFEST_PATH.exists():

        logger.info("Existing download manifest found")

        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        if (
            manifest.get("repo_id") != REPO_ID
            or manifest.get("subset") != SUBSET
        ):
            raise ValueError("Existing manifest belongs to another dataset.")

        logger.info("Reusing pinned revision: %s", manifest["revision"])

        return manifest["revision"]

    logger.info("Fetching latest dataset revision")

    revision = HfApi().dataset_info(REPO_ID).sha

    logger.info(
        "Resolved dataset revision: %s",
        revision,
    )

    return revision


def main() -> None:
    """Download Georgian FineWeb2 and save its metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    revision = get_revision()

    patterns = [
        f"data/{SUBSET}/{split}/*.parquet"
        for split in SPLITS
    ]

    logger.info("Starting FineWeb2 download")
    logger.info("Dataset: %s / %s", REPO_ID, SUBSET)
    logger.info("Revision: %s", revision)
    logger.info("Destination: %s", OUTPUT_DIR)

    logger.debug("File patterns: %s", patterns)
    logger.debug("Max workers: %d", MAX_WORKERS)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=OUTPUT_DIR,
        allow_patterns=patterns,
        max_workers=MAX_WORKERS
    )

    logger.info("Snapshot download completed")

    counts = {}

    for split in SPLITS:
        split_dir = OUTPUT_DIR / "data" / SUBSET / split
        files = sorted(split_dir.rglob("*.parquet"))

        if not files:
            raise RuntimeError(f"No parquets files: {split_dir}")

        counts[split] = len(files)

        total_bytes = sum(
            path.stat().st_size
            for path in files
        )

        logger.info(
            "Split %s: %d files, %.2f GiB",
            split,
            len(files),
            total_bytes / 1024**3,
        )

    manifest = {
        "repo_id": REPO_ID,
        "subset": SUBSET,
        "revision": revision,
        "splits": list(SPLITS),
        "file_counts": counts,
        "downloaded_at_utc": datetime.now(
            timezone.utc
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

    logger.info("Download complete")
    logger.info("Manifest saved: %s", MANIFEST_PATH)


if __name__ == "__main__":
    setup_logging(
        log_file=LOG_PATH,
        console=False,
    )

    started_at = perf_counter()

    try:
        main()

    except Exception:
        logger.exception("FineWeb2 download failed")
        raise

    else:
        elapsed = perf_counter() - started_at

        logger.info(
            "Download completed successfully in %.2f seconds",
            elapsed,
        )
