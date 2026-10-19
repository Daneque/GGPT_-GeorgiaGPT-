"""Resumable full-corpus EDA using Parquet row-group checkpoints."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ggpt.analysis import metrics as metrics_module
from ggpt.analysis.eda import INPUT_COLUMNS, _build_summary, _write_json
from ggpt.analysis.metrics import DocumentProfiler, profile_batch
from ggpt.analysis.statistics import (
    CorpusStatistics,
    FLAG_COLUMNS,
    NUMERIC_COLUMNS,
)
from ggpt.analysis.storage import DocumentMetricsWriter


def run_full_eda(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_revision: str,
    batch_size: int = 512,
    row_groups_per_part: int = 100,
    max_new_parts: int | None = None,
) -> dict:
    if batch_size <= 0 or row_groups_per_part <= 0:
        raise ValueError("Batch and partition sizes must be positive")
    if max_new_parts is not None and max_new_parts <= 0:
        raise ValueError("max_new_parts must be positive")

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(source_dir)

    chunks = []
    global_start = 0
    for source in files:
        metadata = pq.read_metadata(source)
        if set(INPUT_COLUMNS) - set(metadata.schema.to_arrow_schema().names):
            raise ValueError(f"Missing input columns in {source}")
        local_start = 0
        for first in range(0, metadata.num_row_groups, row_groups_per_part):
            stop = min(first + row_groups_per_part, metadata.num_row_groups)
            count = sum(metadata.row_group(i).num_rows for i in range(first, stop))
            chunks.append((source, first, stop, global_start + local_start, count))
            local_start += count
        global_start += metadata.num_rows

    fingerprint = hashlib.sha256(
        Path(metrics_module.__file__).read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()
    final_path = output_dir / "metrics.parquet"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("dataset_revision") != dataset_revision
            or manifest.get("source_documents") != global_start
            or manifest.get("code_fingerprint") != fingerprint
            or not final_path.exists()
            or not summary_path.exists()
            or pq.read_metadata(final_path).num_rows != global_start
        ):
            raise RuntimeError("Existing complete EDA does not match inputs")
        return {
            "status": "complete",
            "documents": global_start,
            "metrics_path": final_path,
            "summary_path": summary_path,
            "manifest_path": manifest_path,
        }

    started = perf_counter()
    new_parts = 0
    all_part_paths = []
    all_timings = {name: 0.0 for name in ("reading", "profiling", "aggregation", "writing")}

    for part_no, (source, first, stop, start_row, count) in enumerate(chunks):
        part_path = parts_dir / f"part_{part_no:04d}.parquet"
        done_path = parts_dir / f"part_{part_no:04d}.json"
        expected = {
            "source_file": source.name,
            "row_group_start": first,
            "row_group_stop": stop,
            "global_row_start": start_row,
            "documents": count,
            "dataset_revision": dataset_revision,
            "code_fingerprint": fingerprint,
        }

        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                not all(done.get(key) == value for key, value in expected.items())
                or not part_path.exists()
                or pq.read_metadata(part_path).num_rows != count
            ):
                raise RuntimeError(f"Invalid completed partition {part_no}")
            for name in all_timings:
                all_timings[name] += done["timings_seconds"][name]
            all_part_paths.append(part_path)
            continue

        part_path.unlink(missing_ok=True)
        part_path.with_name(part_path.name + ".partial").unlink(missing_ok=True)
        done_path.with_name(done_path.name + ".partial").unlink(missing_ok=True)

        profiler = DocumentProfiler()
        timings = {name: 0.0 for name in all_timings}
        row = start_row

        with pq.ParquetFile(source) as parquet_file:
            iterator = iter(parquet_file.iter_batches(
                row_groups=list(range(first, stop)),
                batch_size=batch_size,
                columns=list(INPUT_COLUMNS),
            ))
            with DocumentMetricsWriter(part_path) as writer:
                with tqdm(total=count, desc=f"Part {part_no + 1}/{len(chunks)}", unit="docs") as progress:
                    while True:
                        t = perf_counter()
                        batch = next(iterator, None)
                        timings["reading"] += perf_counter() - t
                        if batch is None:
                            break
                        frame = batch.to_pandas()
                        size = len(frame)
                        frame["_global_row"] = np.arange(row, row + size, dtype=np.int64)
                        frame["_source_file"] = source.name
                        row += size

                        t = perf_counter()
                        metrics = profile_batch(frame["text"], profiler).reset_index(drop=True)
                        timings["profiling"] += perf_counter() - t

                        t = perf_counter()
                        result = pd.concat(
                            [frame.drop(columns="text").reset_index(drop=True), metrics],
                            axis=1,
                        )
                        timings["aggregation"] += perf_counter() - t

                        t = perf_counter()
                        writer.write(result)
                        timings["writing"] += perf_counter() - t
                        progress.update(size)

        if row != start_row + count or writer.rows_written != count:
            raise RuntimeError(f"Wrong row count in partition {part_no}")

        _write_json(done_path, {**expected, "timings_seconds": timings})
        for name in all_timings:
            all_timings[name] += timings[name]
        all_part_paths.append(part_path)
        new_parts += 1
        print(f"Saved checkpoint {part_no + 1}/{len(chunks)}: {count:,} rows")

        if max_new_parts is not None and new_parts >= max_new_parts:
            return {
                "status": "partial",
                "completed_parts": len(all_part_paths),
                "total_parts": len(chunks),
                "part_path": part_path,
            }

    tmp_final = final_path.with_name(final_path.name + ".partial")
    tmp_final.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)
    with pq.ParquetWriter(
        tmp_final,
        schema=pq.read_schema(all_part_paths[0]),
        compression="zstd",
    ) as writer:
        for part_path in all_part_paths:
            writer.write_table(pq.read_table(part_path))
    if pq.read_metadata(tmp_final).num_rows != global_start:
        raise RuntimeError("Combined metrics have an unexpected row count")
    tmp_final.replace(final_path)

    statistics = CorpusStatistics()
    for part_path in all_part_paths:
        columns = list(NUMERIC_COLUMNS) + list(FLAG_COLUMNS)
        statistics.update(pq.read_table(part_path, columns=columns).to_pandas())
    if statistics.total_documents != global_start:
        raise RuntimeError("Aggregated statistics have an unexpected row count")

    summary = _build_summary(statistics)
    summary["stage_timings_seconds"] = {
        key: round(value, 3) for key, value in all_timings.items()
    }
    _write_json(summary_path, summary)

    _write_json(manifest_path, {
        "status": "complete",
        "mode": "full",
        "dataset_revision": dataset_revision,
        "source_dir": str(source_dir),
        "source_documents": global_start,
        "processed_documents": global_start,
        "batch_size": batch_size,
        "row_groups_per_part": row_groups_per_part,
        "parts": len(all_part_paths),
        "code_fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    })
    return {
        "status": "complete",
        "documents": global_start,
        "metrics_path": final_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }
