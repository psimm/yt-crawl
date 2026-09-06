"""Best-effort remote observability without terminal or local-file logging."""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import logfire
from loguru import logger
from rich.console import Console
from rich.text import Text

LOGFIRE_URL = "https://logfire.pydantic.dev/"
_WARNING_LOGGER = logging.getLogger("py.warnings")
_APP_LOGGER = logging.getLogger("yt_crawl")
_quiet_warning_handler = logging.NullHandler()
_quiet_app_handler = logging.NullHandler()
_logfire_warning_handler: logging.Handler | None = None
_configured = False
_LITELLM_INSTRUMENTED_ATTR = "_yt_crawl_logfire_litellm_instrumented"


def _quiet_terminal_logging() -> None:
    """Remove only sinks owned by this app and contain captured warnings."""

    logger.remove()
    if _quiet_warning_handler not in _WARNING_LOGGER.handlers:
        _WARNING_LOGGER.addHandler(_quiet_warning_handler)
    if _quiet_app_handler not in _APP_LOGGER.handlers:
        _APP_LOGGER.addHandler(_quiet_app_handler)
    _WARNING_LOGGER.propagate = False
    _APP_LOGGER.propagate = False
    logging.captureWarnings(True)


def configure_observability() -> bool:
    """Connect Loguru and warnings to Logfire when configuration is available.

    Any setup failure leaves the application with quiet no-op logging. The
    crawler's durable JSONL audit is independent of this optional telemetry.
    """

    global _configured, _logfire_warning_handler
    if _configured:
        return True
    if _logfire_warning_handler is not None:
        _WARNING_LOGGER.removeHandler(_logfire_warning_handler)
        _APP_LOGGER.removeHandler(_logfire_warning_handler)
        _logfire_warning_handler = None
    _quiet_terminal_logging()
    try:
        send_to_logfire: bool | str = (
            False
            if os.getenv("YT_CRAWL_DISABLE_TELEMETRY") == "1"
            else "if-token-present"
        )
        logfire.configure(
            send_to_logfire=send_to_logfire,
            console=False,
            service_name="yt-crawl",
        )
        logger.add(
            logfire.loguru_handler(),
            level="DEBUG",
            catch=True,
            backtrace=False,
            diagnose=False,
        )
        warning_handler = logfire.LogfireLoggingHandler(fallback=logging.NullHandler())
        _logfire_warning_handler = warning_handler
        _WARNING_LOGGER.addHandler(warning_handler)
        _APP_LOGGER.addHandler(warning_handler)
        _configured = True
        instrument_litellm()
    except Exception:
        # Observability must never determine whether research can proceed.
        logger.remove()
        if _logfire_warning_handler is not None:
            _WARNING_LOGGER.removeHandler(_logfire_warning_handler)
            _APP_LOGGER.removeHandler(_logfire_warning_handler)
            _logfire_warning_handler = None
        _configured = False
    return _configured


def instrument_litellm() -> bool:
    """Best-effort process-level Logfire instrumentation for LiteLLM."""

    if not _configured:
        return False
    if getattr(logfire, _LITELLM_INSTRUMENTED_ATTR, False):
        return True
    try:
        logfire.instrument_litellm()
    except Exception:
        return False
    try:
        setattr(logfire, _LITELLM_INSTRUMENTED_ATTR, True)
    except Exception:
        pass
    return True


def logfire_link() -> Text:
    """Return a clickable label followed by a visible copyable fallback URL."""

    text = Text()
    text.append("View logs in Logfire", style=f"bold cyan link {LOGFIRE_URL}")
    text.append(f" — {LOGFIRE_URL}", style="dim")
    return text


def print_logfire_link(console: Console) -> None:
    """Print the stable Logfire control without relying on private project URLs."""

    console.print(logfire_link())


class RunSessionSpan(AbstractContextManager["RunSessionSpan"]):
    """One failure-isolated manual span for a start or resume invocation."""

    def __init__(
        self,
        *,
        run_id: str,
        project: str | Path,
        action: str,
        planned_credits: int,
        credits_added: int = 0,
        controls: dict[str, int] | None = None,
    ) -> None:
        self._span: Any | None = None
        self._entered = False
        self._closed = False
        attributes: dict[str, Any] = {
            "run_id": run_id,
            "project": str(Path(project).expanduser().resolve()),
            "action": action,
            "planned_credits": planned_credits,
            "credits_added": credits_added,
        }
        attributes.update(controls or {})
        try:
            self._span = logfire.span(
                "yt-crawl {action} session",
                **attributes,
            )
            self._span.__enter__()
            self._entered = True
        except Exception:
            self._span = None
            self._entered = False

    def __enter__(self) -> Self:
        return self

    def set_outcome(self, status: str, stop_reason: str | None = None) -> None:
        """Attach non-secret completion metadata when the span is available."""

        if not self._entered or self._span is None:
            return
        try:
            self._span.set_attribute("status", status)
            if stop_reason:
                self._span.set_attribute("stop_reason", stop_reason)
        except Exception:
            pass

    def close(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Close and briefly flush telemetry without surfacing exporter errors."""

        if self._closed:
            return
        self._closed = True
        if self._entered and self._span is not None:
            try:
                self._span.__exit__(exc_type, exc, traceback)
            except Exception:
                pass
        try:
            logfire.force_flush(timeout_millis=500)
        except Exception:
            pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.set_outcome("failed", type(exc).__name__)
        self.close(exc_type, exc, traceback)


_quiet_terminal_logging()


__all__ = [
    "LOGFIRE_URL",
    "RunSessionSpan",
    "configure_observability",
    "instrument_litellm",
    "logfire_link",
    "print_logfire_link",
]
