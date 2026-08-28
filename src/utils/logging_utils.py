"""Structured (JSON) logging setup shared by the pipeline modules."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonLogFormatter(logging.Formatter):
    """Renders log records as single-line JSON for log-aggregator ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS
        }
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit structured JSON to stdout.

    Idempotent: safe to call multiple times (e.g. from tests) without
    accumulating duplicate handlers.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
