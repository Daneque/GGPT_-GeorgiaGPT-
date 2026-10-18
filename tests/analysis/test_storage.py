
"""Tests for incremental document metrics storage."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ggpt.analysis.storage import DocumentMetricsWriter


@pytest.fixture
def metrics():
    return pd.DataFrame({
        "_global_row": [0, 1, 2, 3],
        "n_words": [10, 20, 30, 40],
        "georgian_share": [0.9, 0.8, 0.7, 0.6],
    })


def test_write_multiple_batches(tmp_path, metrics):
    path = tmp_path / "metrics.parquet"

    with DocumentMetricsWriter(path) as writer:
        writer.write(metrics.iloc[:2])
        writer.write(metrics.iloc[2:])

        assert writer.rows_written == 4

    assert path.exists()

    result = pd.read_parquet(path)

    pd.testing.assert_frame_equal(
        result,
        metrics.reset_index(drop=True),
    )

    assert pq.read_metadata(path).num_rows == 4


def test_refuses_existing_file(tmp_path, metrics):
    path = tmp_path / "metrics.parquet"

    with DocumentMetricsWriter(path) as writer:
        writer.write(metrics)

    with pytest.raises(FileExistsError):
        with DocumentMetricsWriter(path):
            pass


def test_removes_partial_file_on_error(tmp_path, metrics):
    path = tmp_path / "metrics.parquet"

    with pytest.raises(RuntimeError):
        with DocumentMetricsWriter(path) as writer:
            writer.write(metrics)
            raise RuntimeError("Simulated failure")

    assert not path.exists()
    assert not (tmp_path / "metrics.parquet.partial").exists()


def test_rejects_raw_text(tmp_path, metrics):
    path = tmp_path / "metrics.parquet"
    invalid = metrics.assign(text=["a", "b", "c", "d"])

    with pytest.raises(ValueError):
        with DocumentMetricsWriter(path) as writer:
            writer.write(invalid)

    assert not path.exists()


def test_rejects_schema_changes(tmp_path, metrics):
    path = tmp_path / "metrics.parquet"

    with pytest.raises(ValueError):
        with DocumentMetricsWriter(path) as writer:
            writer.write(metrics.iloc[:2])
            writer.write(metrics.iloc[2:].drop(columns=["n_words"]))

    assert not path.exists()
