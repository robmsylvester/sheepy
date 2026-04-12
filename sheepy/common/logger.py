"""Structured logging via ``structlog`` bound to the stdlib logging stack."""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging for the sheepy library."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        root.addHandler(handler)


def get_logger(name: str = "sheepy") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given name."""
    configure_logging()
    return structlog.get_logger(name)


# Backward-compatible alias
def get_std_out_logger(log_level: int = logging.INFO) -> structlog.stdlib.BoundLogger:
    """Like ``get_logger`` but sets the root log level; kept for older call sites."""
    configure_logging(log_level)
    return structlog.get_logger("sheepy")
