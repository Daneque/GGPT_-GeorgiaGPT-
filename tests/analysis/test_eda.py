
"""Tests for the unified streaming EDA pipeline."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ggpt.analysis.eda import run_eda
from ggpt.data.parquet_corpus import ParquetCorpus


@pytest.fixture
def eda_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    sample_dir = tmp_path / "sample"

    raw_dir.mkdir()
    sample_dir.mkdir()

    texts = [
        "გამარჯობა საქართველო",
        "თბილისი არის ქალაქი",
        "Hello world",
        "ქართული ტექსტი",
        "",
        "საქართველო\nსაქართველო",
        "<div>გამარჯობა</div>",
        "საქართველო https://example.com",
    ]

    rows = [
        {
            "text": text,
            "id": f"doc-{i}",
            "url": f"https://example.com/{i}",
            "date": "2025-01-01",
            "dump": "CC-MAIN-2025-01",
            "language_score": 0.95,
            "language_script": "Geor",
            "minhash_cluster_size": 1,
            "top_langs": "[]",
        }
        for i, text in enumerate(texts)
    ]

    for index, start in enumerate((0, 4)):
        table = pa.Table.from_pylist(
            rows[start:start + 4]
        )

        pq.write_table(
            table,
            raw_dir / f"part_{index}.parquet",
            row_group_size=2,
        )

    corpus = ParquetCorpus(raw_dir)

    full_data = pd.concat(
        corpus.iter_batches(batch_size=3),
        ignore_index=True,
    )

    full_data.to_parquet(
        sample_dir / "sample.parquet",
        index=False,
    )

    return raw_dir, sample_dir


def test_sample_and_full_agree(tmp_path, eda_sources):
    raw_dir, sample_dir = eda_sources

    full_result = run_eda(
        raw_dir,
        tmp_path / "full_output",
        mode="full",
        batch_size=3,
        show_progress=False,
    )

    sample_result = run_eda(
        sample_dir,
        tmp_path / "sample_output",
        mode="sample",
        batch_size=4,
        show_progress=False,
    )

    full_metrics = pd.read_parquet(
        full_result["metrics_path"]
    )
    sample_metrics = pd.read_parquet(
        sample_result["metrics_path"]
    )

    pd.testing.assert_frame_equal(
        full_metrics,
        sample_metrics,
    )

    assert "text" not in full_metrics.columns
    assert len(full_metrics) == 8

    with open(
        full_result["summary_path"],
        encoding="utf-8",
    ) as stream:
        full_summary = json.load(stream)

    with open(
        sample_result["summary_path"],
        encoding="utf-8",
    ) as stream:
        sample_summary = json.load(stream)

    assert full_summary["total_documents"] == 8
    assert sample_summary["total_documents"] == 8

    assert full_summary["flags"] == sample_summary["flags"]
    assert full_summary["histograms"] == sample_summary["histograms"]

    full_numeric = {
        row["metric"]: row
        for row in full_summary["numeric"]
    }

    sample_numeric = {
        row["metric"]: row
        for row in sample_summary["numeric"]
    }

    for metric in full_numeric:
        for statistic in ("count", "sum", "mean", "std", "min", "max"):
            assert full_numeric[metric][statistic] == pytest.approx(
                sample_numeric[metric][statistic],
                nan_ok=True,
            )


def test_manifest_and_output_guard(tmp_path, eda_sources):
    raw_dir, _ = eda_sources
    output_dir = tmp_path / "eda_result"

    result = run_eda(
        raw_dir,
        output_dir,
        mode="full",
        show_progress=False,
    )

    with open(
        result["manifest_path"],
        encoding="utf-8",
    ) as stream:
        manifest = json.load(stream)

    assert manifest["status"] == "complete"
    assert manifest["processed_documents"] == 8
    assert pq.read_metadata(
        result["metrics_path"]
    ).num_rows == 8

    with pytest.raises(FileExistsError):
        run_eda(
            raw_dir,
            output_dir,
            mode="full",
            show_progress=False,
        )
