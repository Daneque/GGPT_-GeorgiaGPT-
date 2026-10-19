
"""Streaming EDA pipeline for FineWeb2 sample and full train corpus."""

import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from ggpt.analysis.metrics import DocumentProfiler, profile_batch
from ggpt.analysis.statistics import CorpusStatistics
from ggpt.analysis.storage import DocumentMetricsWriter
from ggpt.data.parquet_corpus import ParquetCorpus
from ggpt.utils.logger import get_logger


logger = get_logger(__name__)

EDAMode = Literal["sample", "full"]

INPUT_COLUMNS = (
    "text",
    "id",
    "url",
    "date",
    "dump",
    "language_score",
    "language_script",
    "minhash_cluster_size",
    "top_langs",
)

PROVENANCE_COLUMNS = (
    "_global_row",
    "_source_file",
)


def _validate_columns(
    corpus: ParquetCorpus,
    columns: list[str],
) -> None:
    for path in corpus.files:
        available = set(pq.read_schema(path).names)
        missing = set(columns) - available

        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {sorted(missing)}"
            )


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(
        frame.to_json(orient="records")
    )


def _build_summary(
    statistics: CorpusStatistics,
) -> dict:
    histograms = {}

    for name in statistics.histograms:
        rows = _records(statistics.histogram(name))

        for row in rows:
            if row["right"] is None:
                row["right"] = "+inf"

        histograms[name] = rows

    return {
        "total_documents": statistics.total_documents,
        "numeric": _records(
            statistics.numeric_summary().reset_index()
        ),
        "flags": _records(
            statistics.flag_summary().reset_index()
        ),
        "histograms": histograms,
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(
        path.name + ".partial"
    )

    if temporary_path.exists():
        raise FileExistsError(temporary_path)

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        temporary_path.replace(path)

    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run_eda(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    mode: EDAMode,
    batch_size: int = 512,
    show_progress: bool = True,
    dataset_revision: str | None = None,
) -> dict:
    """Profile one corpus and persist document metrics and summaries."""

    if mode not in ("sample", "full"):
        raise ValueError(f"Unsupported EDA mode: {mode}")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )

    corpus = ParquetCorpus(source_dir)

    columns = list(INPUT_COLUMNS)

    if mode == "sample":
        columns.extend(PROVENANCE_COLUMNS)

    _validate_columns(corpus, columns)

    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = DocumentProfiler()
    statistics = CorpusStatistics()

    metrics_path = output_dir / "metrics.parquet"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    timings = {
        "reading": 0.0,
        "profiling": 0.0,
        "aggregation": 0.0,
        "writing": 0.0,
    }

    logger.info(
        "Starting EDA: mode=%s, documents=%d, batch_size=%d",
        mode,
        corpus.total_documents,
        batch_size,
    )

    started_at = perf_counter()

    batches = corpus.iter_batches(
        batch_size=batch_size,
        columns=columns,
        include_provenance=(mode == "full"),
        show_progress=show_progress,
    )

    with closing(batches):
        with DocumentMetricsWriter(metrics_path) as writer:
            iterator = iter(batches)

            while True:
                started = perf_counter()
                batch = next(iterator, None)
                timings["reading"] += perf_counter() - started

                if batch is None:
                    break

                started = perf_counter()
                metrics = profile_batch(
                    batch["text"],
                    profiler,
                ).reset_index(drop=True)

                timings["profiling"] += perf_counter() - started

                started = perf_counter()
                statistics.update(metrics)
                timings["aggregation"] += perf_counter() - started

                started = perf_counter()

                metadata = batch.drop(
                    columns=["text"]
                ).reset_index(drop=True)

                document_metrics = pd.concat(
                    [metadata, metrics],
                    axis=1,
                )

                writer.write(document_metrics)

                timings["writing"] += perf_counter() - started

    if statistics.total_documents != corpus.total_documents:
        raise RuntimeError(
            f"Processed {statistics.total_documents} documents, "
            f"expected {corpus.total_documents}"
        )

    if writer.rows_written != statistics.total_documents:
        raise RuntimeError(
            "Written document count does not match EDA statistics"
        )

    elapsed = perf_counter() - started_at

    summary = _build_summary(statistics)
    summary["stage_timings_seconds"] = {
        name: round(value, 3)
        for name, value in timings.items()
    }

    _write_json(summary_path, summary)

    manifest = {
        "mode": mode,
        "dataset_revision": dataset_revision,
        "source_dir": str(source_dir),
        "source_files": [
            path.relative_to(source_dir).as_posix()
            for path in corpus.files
        ],
        "source_documents": corpus.total_documents,
        "processed_documents": statistics.total_documents,
        "batch_size": batch_size,
        "input_columns": columns,
        "metrics_file": metrics_path.name,
        "summary_file": summary_path.name,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "status": "complete",
    }

    _write_json(manifest_path, manifest)

    logger.info(
        "EDA completed: %d documents in %.2f seconds",
        statistics.total_documents,
        elapsed,
    )

    return {
        "mode": mode,
        "documents": statistics.total_documents,
        "elapsed_seconds": elapsed,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }
