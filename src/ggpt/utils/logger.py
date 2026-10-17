"""Common logging configuration for GGPT."""


import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "ggpt"

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
        log_file: str | Path | None = None,
        *,
        console: bool = True,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 20 * 1024 * 1024,
        backup_count: int = 3
) -> logging.Logger:
    """Configure logging for the GGPT application.

    Call once at the entry point of a script.

    Args:
        log_file: Optional destination for file logs.
        console: Whether to output logs to the console.
        console_level: Minimum severity for console output.
        file_level: Minimum severity for file output.
        max_bytes: Maximum size of one log file in bytes.
        backup_count: Number of rotated files to retain.

    Returns:
        Configured parent GGPT logger.
    """

    if not console and log_file is None:
        raise ValueError(
            "At least one logging output must be enabled"
        )

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    if backup_count <= 0:
        raise ValueError("backup_count cannot be negative")

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the GGPT namespace.

    Examples:
        get_logger("data.download")
        get_logger("ggpt.data.parquet_corpus")
        get_logger(__name__)
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)

    if name == LOGGER_NAME or name.startswith(
        f"{LOGGER_NAME}."
    ):
        return logging.getLogger(name)

    return logging.getLogger(
        f"{LOGGER_NAME}.{name}"
    )
