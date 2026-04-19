"""
Shared logging configuration for application containers.

The API and UI run in separate Docker images, so logging setup must live in a
package copied into both images. Keep API-specific logging middleware in
``api/``; this module only configures the Python root logger.
"""

import logging
import os
import sys


def setup_logging() -> None:
    """
    Configure the root logger with either JSON or plain-text formatting.

    Reads two environment variables:
      LOG_LEVEL  - minimum severity to emit (default: INFO)
      LOG_FORMAT - "json" for structured output, anything else for plain text
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = os.getenv("LOG_FORMAT", "text").lower()

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    formatter: logging.Formatter
    if log_format == "json":
        try:
            from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[import-untyped]

            formatter = JsonFormatter(  # type: ignore[assignment]
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        except ImportError:
            formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
