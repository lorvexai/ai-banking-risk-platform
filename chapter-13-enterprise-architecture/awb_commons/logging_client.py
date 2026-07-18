"""awb_commons/logging_client.py
Structured JSON logger — mandatory for all AWB AI services.
Produces CloudWatch-indexed entries with 8 mandatory fields.
7-year retention via CloudWatch → S3 Glacier (FCA COBS 9).
"""
import json
import logging
import time
import uuid
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Emit log records as JSON for CloudWatch Insights."""

    MANDATORY_FIELDS = (
        "timestamp", "service_name", "service_version",
        "log_level", "correlation_id", "user_id",
        "event_type", "message",
    )

    def __init__(
        self,
        service_name: str = "unknown",
        service_version: str = "0.0.0",
    ) -> None:
        super().__init__()
        self._service_name = service_name
        self._service_version = service_version

    def format(self, record: logging.LogRecord) -> str:
        extra: dict[str, Any] = getattr(record, "__dict__", {})
        entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(record.created)
            ),
            "service_name": self._service_name,
            "service_version": self._service_version,
            "log_level": record.levelname,
            "correlation_id": extra.get(
                "correlation_id", str(uuid.uuid4())
            ),
            "user_id": extra.get("user_id", "system"),
            "event_type": extra.get("event_type", record.funcName),
            "message": record.getMessage(),
        }
        # Optional fields — include when present
        for opt in ("model_id", "decision_outcome", "duration_ms"):
            if opt in extra:
                entry[opt] = extra[opt]
        return json.dumps(entry, default=str)


def get_structured_logger(
    name: str,
    service_name: str = "awb-service",
    version: str = "2.3.1",
) -> logging.Logger:
    """Return a logger configured with StructuredFormatter.

    Args:
        name:         Module __name__.
        service_name: DORA ICT asset name.
        version:      awb_commons or service version.

    Returns:
        Configured logging.Logger instance.
    """
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            StructuredFormatter(service_name, version)
        )
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log
