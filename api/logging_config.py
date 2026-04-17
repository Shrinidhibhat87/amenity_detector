"""
Structured JSON logging configuration — Phase 4 observability.

Why structured (JSON) logging?
  Plain text log lines like "2024-01-01 INFO Starting up" are human-readable but
  hard for log aggregators (Loki, CloudWatch, Datadog) to parse and query.
  JSON logs emit every field as a key-value pair so you can filter by, for example,
  ``status_code=500`` or ``property_id=abc-123`` without writing regex patterns.

Usage:
  Call ``setup_logging()`` once at application startup (in api/main.py before the
  FastAPI app is created). After that, every ``logging.getLogger(__name__)`` call
  in the codebase emits structured JSON automatically.

  In development (LOG_FORMAT=text or unset) you still get readable text output.
  Set LOG_FORMAT=json in docker-compose or your .env to switch to JSON.

Log levels:
  Controlled by the LOG_LEVEL environment variable (default: INFO).
  Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import logging
import os
import sys


def setup_logging() -> None:
    """
    Configure the root logger with either JSON or plain-text formatting.

    Reads two environment variables:
      LOG_LEVEL  — minimum severity to emit (default: INFO)
      LOG_FORMAT — "json" for structured output, anything else for plain text (default: text)

    The JSON formatter is provided by ``python-json-logger``. Each log record
    becomes a single JSON line with fields: timestamp, level, name (module), message,
    plus any extra fields passed via ``logger.info(..., extra={...})``.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = os.getenv("LOG_FORMAT", "text").lower()

    # Remove any existing handlers so we don't get duplicate log lines
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    formatter: logging.Formatter
    if log_format == "json":
        # Import here so the plain-text path doesn't require python-json-logger
        try:
            from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[import-untyped]

            formatter = JsonFormatter(  # type: ignore[assignment]
                # Fields to include in every JSON log line
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        except ImportError:
            # Graceful fallback: if the library is missing, use plain text
            formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    else:
        # Human-readable format for local development
        formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
