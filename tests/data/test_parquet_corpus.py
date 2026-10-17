
"""Tests for streaming and sampling a Parquet corpus."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ggpt.data.parquet_corpus import ParquetCorpus


@pytest.fixture
def corpus_dir(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir()

    for file_id, (start, end) in enumerate([(0, 5), (5, 12)]):
        table = pa.table({
            "text": [f"document_{i}" for i in range(start, end)],
            "position": list(range(start, end)),
        })

        pq.write_table(
            table,
            directory / f"part_{file_id}.parquet",
            row_group_size=3,
        )

    return directory


def test_corpus_metadata(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    assert len(corpus.files) == 2
    assert corpus.total_documents == 12


def test_iter_batches_reads_every_document(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    batches = list(
        corpus.iter_batches(
            batch_size=2,
        )
    )

    assert all(len(batch) <= 2 for batch in batches)

    result = pd.concat(batches, ignore_index=True)

    assert len(result) == 12
    assert result["position"].tolist() == list(range(12))
    assert result["_global_row"].tolist() == list(range(12))
    assert result["_source_file"].nunique() == 2


def test_column_selection(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    batches = list(
        corpus.iter_batches(
            batch_size=3,
            columns=["text"],
        )
    )

    for batch in batches:
        assert "text" in batch.columns
        assert "position" not in batch.columns


def test_sample_reproducibility(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    first = corpus.sample(n_docs=6, seed=42, batch_size=2)
    second = corpus.sample(n_docs=6, seed=42, batch_size=2)

    pd.testing.assert_frame_equal(first, second)

    assert first["_global_row"].is_unique

    for _, row in first.iterrows():
        assert row["position"] == row["_global_row"]


def test_full_sample(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    result = corpus.sample(
        n_docs=12,
        seed=42,
        batch_size=2,
    )

    assert len(result) == 12
    assert result["position"].tolist() == list(range(12))


@pytest.mark.parametrize("n_docs", [0, -1, 13])
def test_invalid_sample_size(corpus_dir, n_docs):
    corpus = ParquetCorpus(corpus_dir)

    with pytest.raises(ValueError):
        corpus.sample(n_docs=n_docs)


def test_invalid_batch_size(corpus_dir):
    corpus = ParquetCorpus(corpus_dir)

    with pytest.raises(ValueError):
        list(corpus.iter_batches(batch_size=0))


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParquetCorpus(tmp_path)
