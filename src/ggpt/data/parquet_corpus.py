"""Streaming access to local Parquet corpora"""


from contextlib import closing
from pathlib import Path
from typing import Iterator, Sequence


import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ggpt.utils.logger import get_logger


logger = get_logger(__name__)


class ParquetCorpus:
    """Read a collection of Parquet files without loading it all into RAM."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.files = sorted(self.directory.rglob("*.parquet"))

        if not self.files:
            raise FileNotFoundError(f"No parquet files in {self.directory}")

        self.total_documents = sum(
            pq.read_metadata(path).num_rows for path in self.files
        )

        if self.total_documents == 0:
            raise ValueError("Empty corpus.")

        logger.info(
            "Indexed %d Parquet files with %d documents",
            len(self.files),
            self.total_documents
        )

    def _iter_record_batches(
            self,
            batch_size: int,
            columns: Sequence[str] | None,
            show_progress: bool
    ) -> Iterator[tuple[Path, int, pa.RecordBatch]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if (
            columns is not None
            and (isinstance(columns, str)
            or not hasattr(columns, "__iter__")
            or not columns)
        ):
            raise ValueError("columns must be a non-empty sequence of names")

        selected_columns = list(columns) if columns is not None else None
        offset = 0

        with tqdm(
            total=self.total_documents,
            desc="Reading Parquet",
            unit="docs",
            disable=not show_progress,
        ) as progress:
            for path in self.files:
                logger.debug("Reading %s", path)

                with pq.ParquetFile(path) as parquet_file:
                    for batch in parquet_file.iter_batches(
                        batch_size=batch_size,
                        columns=selected_columns
                    ):
                        yield path, offset, batch
                        offset += batch.num_rows
                        progress.update(batch.num_rows)

        if offset != self.total_documents:
            raise RuntimeError(f"Expected {self.total_documents} rows, read {offset}")

    def _to_dataframe(
        self,
        batch: pa.RecordBatch,
        path: Path,
        row_ids: np.ndarray | None,
    ) -> pd.DataFrame:
        frame = batch.to_pandas()

        if row_ids is not None:
            reserved = {"_global_row", "_source_file"}

            if reserved.intersection(frame.columns):
                raise ValueError("Input already contains provenance columns")

            frame["_global_row"] = row_ids
            frame["_source_file"] = path.relative_to(self.directory).as_posix()

        return frame

    def iter_batches(
            self,
            batch_size: int = 1024,
            columns: Sequence[str] | None = None,
            *,
            include_provenance: bool = True,
            show_progress: bool = False
    ) -> Iterator[pd.DataFrame]:
        """Stream the complete corpus as pandas batches."""

        for path, offset, batch in self._iter_record_batches(
            batch_size, columns, show_progress
        ):
            row_ids = (
                np.arange(
                    offset,
                    offset + batch.num_rows,
                    dtype=np.int64
                )
                if include_provenance else None
            )

            yield self._to_dataframe(batch, path, row_ids)

    def sample(
        self,
        n_docs: int,
        seed: int = 42,
        *,
        batch_size: int = 1024,
        columns: Sequence[str] | None = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Choose documents uniformly without replacement in one pass."""

        if not 0 < n_docs <= self.total_documents:
            raise ValueError(
                f"n_docs must be between 1 and {self.total_documents}"
            )

        selected_ids = np.sort(
            np.random.default_rng(seed).choice(
                self.total_documents,
                size=n_docs,
                replace=False,
            )
        )

        frames = []
        cursor = 0

        logger.info(
            "Sampling %d of %d documents (seed=%d)",
            n_docs,
            self.total_documents,
            seed,
        )

        with closing(
            self._iter_record_batches(batch_size, columns, show_progress)
        ) as batches:
            for path, offset, batch in batches:
                end = offset + batch.num_rows
                next_cursor = int(np.searchsorted(selected_ids, end))

                if next_cursor > cursor:
                    chosen_ids = selected_ids[cursor:next_cursor]
                    positions = chosen_ids - offset

                    chosen_batch = batch.take(
                        pa.array(positions, type=pa.int64())
                    )

                    frames.append(
                        self._to_dataframe(
                            chosen_batch,
                            path,
                            chosen_ids,
                        )
                    )

                    cursor = next_cursor

                if cursor == n_docs:
                    break

        if cursor != n_docs:
            raise RuntimeError(
                f"Expected {n_docs} rows, extracted {cursor}"
            )

        result = pd.concat(frames, ignore_index=True)

        logger.info("Sample contains %d documents", len(result))

        return result
