
"""Tests for document-level text profiling."""

import hashlib

import pandas as pd
import pytest

from ggpt.analysis.metrics import (
    DocumentProfiler,
    profile_batch,
)


@pytest.fixture
def profiler():
    return DocumentProfiler()


def test_georgian_and_latin(profiler):
    text = "გამარჯობა, საქართველო! Hello 123"

    metrics = profiler.profile(text)

    assert metrics["n_words"] == 3
    assert metrics["n_georgian_letters"] == len(
        "გამარჯობასაქართველო"
    )
    assert metrics["n_latin_letters"] == 5
    assert metrics["n_cyrillic_letters"] == 0

    assert metrics["n_letters"] == (
        metrics["n_georgian_letters"] + 5
    )

    assert metrics["georgian_share"] == pytest.approx(
        metrics["n_georgian_letters"] / metrics["n_letters"]
    )


def test_repeated_lines(profiler):
    text = "ერთია\nსხვა\nერთია\n\n სხვა "

    metrics = profiler.profile(text)

    assert metrics["n_lines"] == 4
    assert metrics["repeated_line_share"] == pytest.approx(0.5)


def test_missing_and_whitespace(profiler):
    for text in [None, "   \n "]:
        metrics = profiler.profile(text)

        assert metrics["is_empty_text"]
        assert metrics["n_letters"] == 0
        assert metrics["georgian_share"] == 0

    assert profiler.profile(None)["is_missing_text"]
    assert not profiler.profile("   ")["is_missing_text"]


def test_artifacts(profiler):
    text = "<div>ქართული</div> https://example.org \ufffd"

    metrics = profiler.profile(text)

    assert metrics["has_html_like"]
    assert metrics["n_url_mentions"] == 1
    assert metrics["n_replacement_chars"] == 1

    assert metrics["sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    assert metrics["n_bytes"] == len(text.encode("utf-8"))


def test_batch_preserves_index(profiler):
    texts = pd.Series(
        ["ქართული", None, "Hello"],
        index=[10, 20, 30],
    )

    metrics = profile_batch(texts, profiler)

    assert len(metrics) == 3
    assert metrics.index.tolist() == [10, 20, 30]
    assert metrics.loc[20, "is_empty_text"]
