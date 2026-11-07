"""Structured logging configuration for OlfactAI."""

import logging
import sys
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_format: Optional[str] = "text",
) -> None:
    """Configure root and application loggers.

    Args:
        log_level: Minimum logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_format: Format style ('text' or 'json').
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    if log_format == "json":
        format_str = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", '
            '"message": "%(message)s"}'
        )
    else:
        format_str = (
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        )

    formatter = logging.Formatter(fmt=format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence overly noisy third-party libraries if needed
    logging.getLogger("uvicorn.access").setLevel(numeric_level)
    logging.getLogger("rdkit").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance configured for the given component name.

    Args:
        name: Name of module or component.
    """
    return logging.getLogger(f"olfactai.{name}")

