"""Small structured logging setup with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

from yachiyo.config.models import LoggingSettings

_AUTHORIZATION = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?[A-Za-z][A-Za-z0-9_-]*\s+)"
    r"[^\s,;\"'}]+"
)
_SECRET = re.compile(
    r"(?i)((?:api[_-]?key|bot[_-]?token|password|secret)[\"']?\s*[:=]\s*[\"']?)"
    r"[^\s,;\"'}]+"
)
_TOKEN = re.compile(
    r"\b(?:sk|ghp|gho|github_pat|xox[baprs])[-_A-Za-z0-9]{16,}\b|"
    r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b",
    re.I,
)


def redact(value: str) -> str:
    value = _AUTHORIZATION.sub(r"\1[REDACTED]", value)
    value = _SECRET.sub(r"\1[REDACTED]", value)
    return _TOKEN.sub("[REDACTED]", value)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        for key in ("job_id", "run_id", "conversation_id", "transport"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = str(value)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(settings: LoggingSettings) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if settings.json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(settings.level)
