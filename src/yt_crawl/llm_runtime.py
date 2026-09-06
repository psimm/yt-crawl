"""Logged OpenAI Responses wrapper used by crawler projects."""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Iterator, Literal

from loguru import logger

from yt_crawl.records import ApiCallRecord
from yt_crawl.runtime_events import RuntimeEvent, RuntimeEventCallback
from yt_crawl.storage import JsonlRunWriter


@dataclass(frozen=True, slots=True)
class LlmCallContext:
    pool: Literal["discovery", "transcript"]
    purpose: str


class StructuredOutputError(RuntimeError):
    """A concise provider structured-output failure safe to show in the CLI."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        response_id: str | None = None,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.response_id = response_id
        identity = f" (response {response_id})" if response_id else ""
        super().__init__(
            f"OpenAI structured output failed during {operation}: {reason}{identity}"
        )


class LoggedOpenAIClient:
    """Responses facade with JSONL logging and safe structured parsing."""

    def __init__(
        self,
        client: Any,
        writer: JsonlRunWriter,
        *,
        on_event: RuntimeEventCallback | None = None,
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._client = client
        self._writer = writer
        self._context: ContextVar[LlmCallContext | None] = ContextVar(
            f"yt_crawl_llm_context_{id(self)}",
            default=None,
        )
        self._request_slots = BoundedSemaphore(max_concurrency)
        self.max_concurrency = max_concurrency
        self._on_event = on_event or (lambda _event: None)
        self.responses = _LoggedResponses(self)

    def _emit_event(self, event: RuntimeEvent) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            logger.error(
                "Runtime display callback failed provider={} operation={} "
                "phase={} error_type={}",
                event.provider,
                event.operation,
                event.phase,
                type(exc).__name__,
            )

    @contextmanager
    def call_context(
        self,
        pool: Literal["discovery", "transcript"],
        purpose: str,
    ) -> Iterator[None]:
        if self._context.get() is not None:
            raise RuntimeError("nested LLM call contexts are not supported")
        token = self._context.set(LlmCallContext(pool, purpose))
        try:
            yield
        finally:
            self._context.reset(token)


class _LoggedResponses:
    def __init__(self, owner: LoggedOpenAIClient) -> None:
        self._owner = owner

    def parse(self, **kwargs: Any) -> Any:
        context = self._owner._context.get()
        if context is None:
            raise RuntimeError("responses.parse requires an active LLM call_context")
        with self._owner._request_slots:
            return self._parse_one(context, kwargs)

    def _parse_one(self, context: LlmCallContext, kwargs: dict[str, Any]) -> Any:
        self._owner._emit_event(
            RuntimeEvent(
                provider="openai",
                operation=context.purpose,
                phase="started",
            )
        )
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"^Pydantic serializer warnings:\s+"
                        r"PydanticSerializationUnexpectedValue\("
                    ),
                    category=UserWarning,
                    module=r"^pydantic\.main$",
                )
                raw = self._owner._client.responses.with_raw_response.parse(**kwargs)
        except Exception as exc:
            envelope = _exception_envelope(exc)
            self._finish(
                context,
                started=started,
                status="error",
                response_id=_response_id(envelope, getattr(exc, "request_id", None)),
                model=_response_model(envelope, kwargs.get("model")),
                error=_request_error_reason(exc),
            )
            raise

        envelope = _raw_envelope(raw)
        response_id = _response_id(envelope, getattr(raw, "request_id", None))
        model = _response_model(envelope, kwargs.get("model"))

        problem = _envelope_problem(envelope)
        if problem is not None:
            error = StructuredOutputError(
                context.purpose,
                problem,
                response_id=response_id,
            )
            self._finish(
                context,
                started=started,
                status="error",
                response_id=response_id,
                model=model,
                error=str(error),
            )
            raise error

        try:
            response = raw.parse()
        except Exception:
            error = StructuredOutputError(
                context.purpose,
                "completed response was not valid structured output",
                response_id=response_id,
            )
            self._finish(
                context,
                started=started,
                status="error",
                response_id=response_id,
                model=model,
                error=str(error),
            )
            raise error from None

        if getattr(response, "output_parsed", None) is None:
            error = StructuredOutputError(
                context.purpose,
                "completed response contained no structured output",
                response_id=response_id,
            )
            self._finish(
                context,
                started=started,
                status="error",
                response_id=response_id,
                model=model,
                error=str(error),
            )
            raise error

        self._finish(
            context,
            started=started,
            status="success",
            response_id=response_id,
            model=model,
            error=None,
        )
        return response

    def _finish(
        self,
        context: LlmCallContext,
        *,
        started: float,
        status: Literal["success", "error"],
        response_id: str | None,
        model: str | None,
        error: str | None,
    ) -> None:
        self._owner._writer.append(
            ApiCallRecord(
                run_id=self._owner._writer.run_id,
                provider="openai",
                operation=context.purpose,
                request_id=response_id,
                status=status,
                llm_model=model,
                latency_seconds=time.perf_counter() - started,
                error=error,
            )
        )
        self._owner._emit_event(
            RuntimeEvent(
                provider="openai",
                operation=context.purpose,
                phase="finished",
                status=status,
                error=error,
            )
        )


def _exception_envelope(exc: Exception) -> Mapping[str, Any] | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        value = response.json()
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _raw_envelope(raw: Any) -> Mapping[str, Any] | None:
    """Read a raw SDK response without invoking its typed post-parser."""

    read_json = getattr(raw, "json", None)
    if not callable(read_json):
        # OpenAI SDK 2.x LegacyAPIResponse exposes the underlying public
        # httpx.Response; newer raw response types expose json() directly.
        read_json = getattr(getattr(raw, "http_response", None), "json", None)
    if not callable(read_json):
        return None
    try:
        value = read_json()
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _request_error_reason(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return f"OpenAI request failed with HTTP {status_code}"
    return f"OpenAI request failed: {type(exc).__name__}"


def _response_id(
    envelope: Mapping[str, Any] | Any,
    request_id: str | None,
) -> str | None:
    if isinstance(envelope, Mapping):
        response_id = envelope.get("id")
        if isinstance(response_id, str) and response_id:
            return response_id
    return request_id


def _response_model(envelope: Mapping[str, Any] | Any, requested: Any) -> str | None:
    if isinstance(envelope, Mapping):
        model = envelope.get("model")
        if isinstance(model, str) and model:
            return model
    return requested if isinstance(requested, str) and requested else None


def _envelope_problem(envelope: Mapping[str, Any] | Any) -> str | None:
    if not isinstance(envelope, Mapping):
        return "provider returned an unreadable response envelope"
    status = envelope.get("status")
    if status != "completed":
        if status == "incomplete":
            details = envelope.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, Mapping) else None
            suffix = f" ({reason})" if isinstance(reason, str) and reason else ""
            return f"response was incomplete{suffix}"
        if status == "failed":
            error = envelope.get("error")
            code = error.get("code") if isinstance(error, Mapping) else None
            suffix = f" ({code})" if isinstance(code, str) and code else ""
            return f"response failed{suffix}"
        status_name = str(status) if status is not None else "missing"
        return f"response status was {status_name}"
    if _contains_refusal(envelope.get("output")):
        return "model refused the structured response"
    return None


def _contains_refusal(output: Any) -> bool:
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "refusal":
            return True
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") == "refusal"
            for part in content
        ):
            return True
    return False


__all__ = [
    "LoggedOpenAIClient",
    "LlmCallContext",
    "StructuredOutputError",
]
