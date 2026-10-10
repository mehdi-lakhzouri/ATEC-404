"""Structured logging with structlog: JSON lines (or console), UTC ISO timestamps, request / mission context via
contextvars, and the standard-library loggers (uvicorn, pymongo) routed through the same processors."""

from __future__ import annotations

import logging
import sys

import structlog

_SHARED = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
]


def configure(level: str = "INFO", json_logs: bool = True) -> None:
    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[*_SHARED, structlog.processors.format_exc_info, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(foreign_pre_chain=_SHARED, processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta, structlog.processors.format_exc_info, renderer]))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("pymongo").setLevel(logging.WARNING)


def get(name: str = "artemis.ground"):
    return structlog.get_logger(name)
