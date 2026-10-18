
"""Tests for streaming corpus statistics."""

import numpy as np
import pandas as pd
import pytest

from ggpt.analysis.metrics import (
    DocumentProfiler,
    profile_batch,
)
from ggpt.analysis.statistics import CorpusStatistics


@pytest.fixture
def document_metrics():
    profiler = DocumentProfiler()

    texts = pd.Series([
        "გამარჯობა საქართველო",
        "Hello world",
        "",
        "საქართველო\nსაქართველო",
        "<div>ქართული</div>",
        None,
        "თბილისი https://example.com",
    ])

    return profile_batch(texts, profiler)


def test_document_count(document_metrics):
    statistics = CorpusStatistics()

    statistics.update(document_metrics.iloc[:3])
    statistics.update(document_metrics.iloc[3:])

    assert statistics.total_documents == 7


def test_chunked_statistics_match_single_batch(document_metrics):
    complete = CorpusStatistics()
    complete.update(document_metrics)

    chunked = CorpusStatistics()

    for start in range(0, len(document_metrics), 2):
        chunked.update(
            document_metrics.iloc[start:start + 2]
        )

    expected = complete.numeric_summary()
    actual = chunked.numeric_summary()

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    pd.testing.assert_frame_equal(
        complete.flag_summary(),
        chunked.flag_summary(),
    )

    for metric in complete.histograms:
        pd.testing.assert_frame_equal(
            complete.histogram(metric),
            chunked.histogram(metric),
        )


def test_mean_and_standard_deviation(document_metrics):
    statistics = CorpusStatistics()
    statistics.update(document_metrics)

    summary = statistics.numeric_summary()

    assert summary.loc["n_words", "mean"] == pytest.approx(
        document_metrics["n_words"].mean()
    )

    assert summary.loc["n_words", "std"] == pytest.approx(
        document_metrics["n_words"].std(ddof=1)
    )

    assert summary.loc["n_words", "sum"] == pytest.approx(
        document_metrics["n_words"].sum()
    )


def test_histogram_counts(document_metrics):
    statistics = CorpusStatistics()
    statistics.update(document_metrics)

    histogram = statistics.histogram("n_words")

    assert histogram["documents"].sum() == len(document_metrics)


def test_missing_columns(document_metrics):
    statistics = CorpusStatistics()

    with pytest.raises(ValueError):
        statistics.update(
            document_metrics.drop(columns=["n_words"])
        )


def test_empty_statistics():
    statistics = CorpusStatistics()

    assert statistics.total_documents == 0
    assert np.isnan(
        statistics.numeric_summary().loc["n_words", "mean"]
    )
