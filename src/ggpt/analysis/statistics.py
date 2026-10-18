
"""Incremental descriptive statistics for document metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "n_chars",
    "n_bytes",
    "n_words",
    "n_lines",
    "n_letters",
    "n_georgian_letters",
    "n_latin_letters",
    "n_cyrillic_letters",
    "georgian_share",
    "latin_share",
    "cyrillic_share",
    "repeated_line_share",
    "n_url_mentions",
    "n_replacement_chars",
)

FLAG_COLUMNS = (
    "is_missing_text",
    "is_empty_text",
    "has_html_like",
)


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    return values[np.isfinite(values)]


@dataclass
class RunningMoments:
    """Accumulate moments without retaining individual values."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(np.sum((values - batch_mean) ** 2))

        if self.count:
            delta = batch_mean - self.mean
            combined = self.count + batch_count

            self.m2 += (
                batch_m2
                + delta**2
                * self.count
                * batch_count
                / combined
            )

            self.mean += delta * batch_count / combined
            self.count = combined

        else:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2

        self.total += float(values.sum())

        self.minimum = min(
            self.minimum,
            float(values.min()),
        )

        self.maximum = max(
            self.maximum,
            float(values.max()),
        )

    def summary(self, total_documents: int) -> dict:
        return {
            "count": self.count,
            "missing": total_documents - self.count,
            "sum": self.total,
            "mean": self.mean if self.count else np.nan,
            "std": (
                np.sqrt(self.m2 / (self.count - 1))
                if self.count > 1
                else np.nan
            ),
            "min": self.minimum if self.count else np.nan,
            "max": self.maximum if self.count else np.nan,
        }


class HistogramAccumulator:
    """Accumulate histogram counts across batches."""

    def __init__(self, edges: np.ndarray):
        self.edges = np.asarray(
            edges,
            dtype=np.float64,
        )

        if (
            len(self.edges) < 2
            or np.any(np.diff(self.edges) <= 0)
        ):
            raise ValueError(
                "Histogram edges must be strictly increasing"
            )

        self.counts = np.zeros(
            len(self.edges) - 1,
            dtype=np.int64,
        )

    def update(self, values: np.ndarray) -> None:
        self.counts += np.histogram(
            values,
            bins=self.edges,
        )[0]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "left": self.edges[:-1],
            "right": self.edges[1:],
            "documents": self.counts,
        })


class CorpusStatistics:
    """Accumulate corpus statistics from document-metric batches."""

    def __init__(self) -> None:
        self.total_documents = 0

        self.numeric = {
            name: RunningMoments()
            for name in NUMERIC_COLUMNS
        }

        self.flags = {
            name: 0
            for name in FLAG_COLUMNS
        }

        self.histograms = {
            "n_words": HistogramAccumulator(
                np.array([
                    0, 1, 10, 25, 50, 100,
                    200, 500, 1000, 2000,
                    5000, 10000, 50000, np.inf,
                ])
            ),
            "georgian_share": HistogramAccumulator(
                np.linspace(0, 1, 21)
            ),
            "latin_share": HistogramAccumulator(
                np.linspace(0, 1, 21)
            ),
            "repeated_line_share": HistogramAccumulator(
                np.linspace(0, 1, 21)
            ),
        }

    def update(self, metrics: pd.DataFrame) -> None:
        required = set(NUMERIC_COLUMNS) | set(FLAG_COLUMNS)
        missing = required - set(metrics.columns)

        if missing:
            raise ValueError(
                f"Missing document metrics: {sorted(missing)}"
            )

        self.total_documents += len(metrics)

        for column, accumulator in self.numeric.items():
            accumulator.update(
                _finite_values(metrics[column])
            )

        for column in FLAG_COLUMNS:
            self.flags[column] += int(
                metrics[column].fillna(False).sum()
            )

        for column, histogram in self.histograms.items():
            histogram.update(
                _finite_values(metrics[column])
            )

    def numeric_summary(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                name: moment.summary(self.total_documents)
                for name, moment in self.numeric.items()
            },
            orient="index",
        ).rename_axis("metric")

    def flag_summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "flag": flag,
                "documents": count,
                "share": (
                    count / self.total_documents
                    if self.total_documents
                    else np.nan
                ),
            }
            for flag, count in self.flags.items()
        ]).set_index("flag")

    def histogram(self, metric: str) -> pd.DataFrame:
        return self.histograms[metric].to_frame()
