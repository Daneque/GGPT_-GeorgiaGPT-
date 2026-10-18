
"""Incremental storage of document-level metrics."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ggpt.utils.logger import get_logger


logger = get_logger(__name__)


class DocumentMetricsWriter:
    """Write document metrics to Parquet incrementally."""

    def __init__(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_name(
            self.path.name + ".partial"
        )

        self.overwrite = overwrite
        self.rows_written = 0

        self._writer = None
        self._schema = None
        self._active = False

    def __enter__(self):
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output already exists: {self.path}"
            )

        if self.partial_path.exists():
            raise FileExistsError(
                f"Unfinished output exists: {self.partial_path}"
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._active = True
        return self

    def write(self, metrics: pd.DataFrame) -> None:
        """Append one DataFrame of document metrics."""
        if not self._active:
            raise RuntimeError(
                "Use DocumentMetricsWriter inside a with block"
            )

        if metrics.empty:
            return

        if "text" in metrics.columns:
            raise ValueError(
                "Raw document texts must not be stored in metrics"
            )

        table = pa.Table.from_pandas(
            metrics,
            preserve_index=False,
        )

        if self._writer is None:
            self._schema = table.schema

            self._writer = pq.ParquetWriter(
                self.partial_path,
                schema=self._schema,
                compression="zstd",
            )

        elif not table.schema.equals(
            self._schema,
            check_metadata=False,
        ):
            raise ValueError(
                "Document metrics schema changed between batches"
            )

        self._writer.write_table(table)
        self.rows_written += len(metrics)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._writer is not None:
                self._writer.close()

            if exc_type is None:
                if self.rows_written == 0:
                    raise ValueError(
                        "No document metrics were written"
                    )

                self.partial_path.replace(self.path)

                logger.info(
                    "Saved %d document metrics to %s",
                    self.rows_written,
                    self.path,
                )

            else:
                self.partial_path.unlink(missing_ok=True)

        except BaseException:
            self.partial_path.unlink(missing_ok=True)
            raise

        finally:
            self._writer = None
            self._active = False

        return False
