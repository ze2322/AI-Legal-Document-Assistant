"""Small logging helper so every module logs in the same format."""

from __future__ import annotations

import logging
import sys

from utils.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    # On Windows the console defaults to cp1252, which raises
    # UnicodeEncodeError when a log record contains a non-Latin-1 character
    # (for example a Cyrillic document name). Force UTF-8 where possible.
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # pragma: no cover - depends on the console
            pass

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger("legal_assistant")
    root.setLevel(settings.log_level)
    root.addHandler(handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger configured with the app-wide handler."""
    _configure_root()
    return logging.getLogger(f"legal_assistant.{name}")
