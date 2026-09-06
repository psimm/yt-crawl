"""Logged LiteLLM wrapper used by crawler projects."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Iterator, Literal

import litellm
from loguru import logger

from yt_crawl.records import ApiCallRecord
from yt_crawl.runtime_events import RuntimeEvent, RuntimeEventCallback
from yt_crawl.storage import JsonlRunWriter

LLM_TIMEOUT_SECONDS = 90.0
LLM_NUM_RETRIES = 3


@dataclass(frozen=True, slots=True)
class LlmCallContext:
    pool: Literal["discovery", "transcript"]
    purpose: str


@dataclass(frozen=True, slots=True)
class ParsedLlmResponse:
    id: str | None
    output_parsed: Any
    usage: Any
    model: str | None


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
            f"LLM structured output failed during {operation}: {reason}{identity}"
        )


class LoggedLlmClient:
    """Chat-completions facade with JSONL logging and safe structured parsing."""

    def __init__(
        self,
        writer: JsonlRunWriter,
        *,
        on_event: RuntimeEventCallback | None = None,
        max_concurrency: int = 1,
        api_base: str | None = None,
        timeout: float = LLM_TIMEOUT_SECONDS,
        num_retries: int = LLM_NUM_RETRIES,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._writer = writer
        self._api_base = api_base
        self._timeout = timeout
        self._num_retries = num_retries
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
    def __init__(self, owner: LoggedLlmClient) -> None:
        self._owner = owner

    def parse(self, **kwargs: Any) -> ParsedLlmResponse:
        context = self._owner._context.get()
        if context is None:
            raise RuntimeError("responses.parse requires an active LLM call_context")
        with self._owner._request_slots:
            return self._parse_one(context, kwargs)

    def _parse_one(
        self, context: LlmCallContext, kwargs: dict[str, Any]
    ) -> ParsedLlmResponse:
        self._owner._emit_event(
            RuntimeEvent(
                provider="llm",
                operation=context.purpose,
                phase="started",
            )
        )
        started = time.perf_counter()
        requested_model = kwargs.get("model")
        try:
            raw = _complete(self._owner, kwargs)
        except Exception as exc:
            self._finish(
                context,
                started=started,
                status="error",
                response_id=getattr(exc, "request_id", None),
                model=_as_optional_str(requested_model),
                error=_request_error_reason(exc),
            )
            raise

        response_id = _as_optional_str(getattr(raw, "id", None))
        model = _as_optional_str(getattr(raw, "model", None)) or _as_optional_str(
            requested_model
        )
        try:
            parsed = _parse_completion(raw, kwargs.get("text_format"), context.purpose)
        except StructuredOutputError as error:
            self._finish(
                context,
                started=started,
                status="error",
                response_id=error.response_id or response_id,
                model=model,
                error=str(error),
            )
            raise

        self._finish(
            context,
            started=started,
            status="success",
            response_id=parsed.id or response_id,
            model=parsed.model or model,
            error=None,
        )
        return parsed

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
                provider="llm",
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
                provider="llm",
                operation=context.purpose,
                phase="finished",
                status=status,
                error=error,
            )
        )


def _complete(owner: LoggedLlmClient, kwargs: dict[str, Any]) -> Any:
    model = kwargs.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("responses.parse requires a model")
    text_format = kwargs.get("text_format")
    if text_format is None:
        raise ValueError("responses.parse requires text_format")
    request: dict[str, Any] = {
        "model": model,
        "messages": _as_messages(
            instructions=kwargs.get("instructions"),
            input_value=kwargs.get("input"),
        ),
        "response_format": text_format,
        "timeout": owner._timeout,
        "num_retries": owner._num_retries,
    }
    if owner._api_base:
        request["api_base"] = owner._api_base
    return litellm.completion(**request)


def _parse_completion(raw: Any, schema: Any, operation: str) -> ParsedLlmResponse:
    response_id = _as_optional_str(getattr(raw, "id", None))
    model = _as_optional_str(getattr(raw, "model", None))
    choices = getattr(raw, "choices", None)
    if not choices:
        raise StructuredOutputError(
            operation,
            "completed response contained no structured output",
            response_id=response_id,
        )
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "content_filter":
        raise StructuredOutputError(
            operation,
            "model refused the structured response",
            response_id=response_id,
        )
    message = getattr(choice, "message", None)
    refusal = getattr(message, "refusal", None) if message is not None else None
    if isinstance(refusal, str) and refusal.strip():
        raise StructuredOutputError(
            operation,
            "model refused the structured response",
            response_id=response_id,
        )
    content = getattr(message, "content", None) if message is not None else None
    if content is None or content == "":
        raise StructuredOutputError(
            operation,
            "completed response contained no structured output",
            response_id=response_id,
        )
    try:
        parsed = (
            schema.model_validate_json(content)
            if isinstance(content, str)
            else schema.model_validate(content)
        )
    except Exception:
        raise StructuredOutputError(
            operation,
            "completed response was not valid structured output",
            response_id=response_id,
        ) from None
    return ParsedLlmResponse(
        id=response_id,
        output_parsed=parsed,
        usage=getattr(raw, "usage", None),
        model=model,
    )


def _as_messages(*, instructions: Any, input_value: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages
    if isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            if role == "developer":
                role = "system"
            elif role not in {"system", "user", "assistant"}:
                role = "user"
            content = _flatten_content(item.get("content"))
            if content:
                messages.append({"role": str(role), "content": content})
        return messages
    if input_value is not None:
        messages.append({"role": "user", "content": str(input_value)})
    return messages


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _request_error_reason(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return f"LLM request failed with HTTP {status_code}"
    return f"LLM request failed: {type(exc).__name__}"


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "LLM_NUM_RETRIES",
    "LLM_TIMEOUT_SECONDS",
    "LoggedLlmClient",
    "LlmCallContext",
    "ParsedLlmResponse",
    "StructuredOutputError",
]
