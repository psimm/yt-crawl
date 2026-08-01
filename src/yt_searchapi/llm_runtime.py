"""Audited OpenAI Responses wrapper used by crawler projects."""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from threading import BoundedSemaphore
from typing import Any, Iterator, Literal

from loguru import logger

from yt_searchapi.records import ApiCallRecord
from yt_searchapi.runtime_events import RuntimeEvent, RuntimeEventCallback
from yt_searchapi.storage import JsonlRunWriter

GPT56_LUNA_MODEL = "gpt-5.6-luna"
_TOKENS_PER_MILLION = Decimal(1_000_000)
_LUNA_SHORT_CONTEXT_RATES = {
    "input": Decimal("0.20"),
    "cached_input": Decimal("0.02"),
    "cache_write": Decimal("0.25"),
    "output": Decimal("1.20"),
}
_LUNA_LONG_CONTEXT_RATES = {
    "input": Decimal("0.40"),
    "cached_input": Decimal("0.04"),
    "cache_write": Decimal("0.50"),
    "output": Decimal("1.80"),
}
_LONG_CONTEXT_THRESHOLD = 272_000


@dataclass(frozen=True, slots=True)
class LlmCallContext:
    pool: Literal["discovery", "transcript"]
    purpose: str


@dataclass(frozen=True, slots=True)
class LlmTokenMetrics:
    """Provider-reported token categories used for audit and cost estimates."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


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


class AuditedOpenAIClient:
    """Small Responses facade with JSONL auditing and safe structured parsing.

    SearchAPI credits are the only local hard budget. Provider-reported OpenAI
    usage remains visible in the append-only API-call audit.
    """

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
            f"yt_searchapi_llm_context_{id(self)}",
            default=None,
        )
        self._request_slots = BoundedSemaphore(max_concurrency)
        self.max_concurrency = max_concurrency
        self._on_event = on_event or (lambda _event: None)
        self.responses = _AuditedResponses(self)

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


class _AuditedResponses:
    def __init__(self, owner: AuditedOpenAIClient) -> None:
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
            response_id = _response_id(envelope, getattr(exc, "request_id", None))
            usage = _usage_tokens(envelope)
            self._finish(
                context,
                started=started,
                status="error",
                response_id=response_id,
                usage=usage,
                model=_response_model(envelope, kwargs.get("model")),
                error=_request_error_reason(exc),
            )
            raise

        envelope = _raw_envelope(raw)
        response_id = _response_id(envelope, getattr(raw, "request_id", None))
        usage = _usage_tokens(envelope)
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
                usage=usage,
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
                usage=usage,
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
                usage=usage,
                model=model,
                error=str(error),
            )
            raise error

        self._finish(
            context,
            started=started,
            status="success",
            response_id=response_id,
            usage=usage,
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
        usage: LlmTokenMetrics,
        model: str | None,
        error: str | None,
    ) -> None:
        estimated_cost = estimate_gpt56_luna_standard_cost(
            usage,
            model=model,
        )
        self._owner._writer.append(
            ApiCallRecord(
                run_id=self._owner._writer.run_id,
                provider="openai",
                operation=context.purpose,
                request_id=response_id,
                status=status,
                llm_input_tokens=usage.input_tokens,
                llm_cached_input_tokens=usage.cached_input_tokens,
                llm_cache_write_tokens=usage.cache_write_tokens,
                llm_output_tokens=usage.output_tokens,
                llm_model=model,
                llm_estimated_cost_usd=estimated_cost,
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
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                estimated_cost_usd=estimated_cost,
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


def _usage_tokens(envelope: Mapping[str, Any] | Any) -> LlmTokenMetrics:
    if not isinstance(envelope, Mapping):
        return LlmTokenMetrics()
    usage = envelope.get("usage")
    if not isinstance(usage, Mapping):
        return LlmTokenMetrics()
    input_tokens = _nonnegative_int(usage.get("input_tokens"))
    details = usage.get("input_tokens_details")
    cached_input_tokens = (
        _nonnegative_int(details.get("cached_tokens"))
        if isinstance(details, Mapping)
        else 0
    )
    cached_input_tokens = min(cached_input_tokens, input_tokens)
    cache_write_tokens = (
        _nonnegative_int(details.get("cache_write_tokens"))
        if isinstance(details, Mapping)
        else 0
    )
    cache_write_tokens = min(cache_write_tokens, input_tokens - cached_input_tokens)
    return LlmTokenMetrics(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=_nonnegative_int(usage.get("output_tokens")),
    )


def _response_model(envelope: Mapping[str, Any] | Any, requested: Any) -> str | None:
    if isinstance(envelope, Mapping):
        model = envelope.get("model")
        if isinstance(model, str) and model:
            return model
    return requested if isinstance(requested, str) and requested else None


def estimate_gpt56_luna_standard_cost(
    usage: LlmTokenMetrics,
    *,
    model: str | None = GPT56_LUNA_MODEL,
) -> float | None:
    """Estimate direct Standard-tier cost using the current Luna rate card.

    Inputs above 272K use the full-request long-context rates published for
    GPT-5.6 Luna. Regional-processing uplifts and non-Standard service tiers are
    intentionally outside this estimate.
    """

    if model is None or not model.startswith(GPT56_LUNA_MODEL):
        return None
    rates = (
        _LUNA_LONG_CONTEXT_RATES
        if usage.input_tokens > _LONG_CONTEXT_THRESHOLD
        else _LUNA_SHORT_CONTEXT_RATES
    )
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    writes = min(usage.cache_write_tokens, usage.input_tokens - cached)
    uncached = usage.input_tokens - cached - writes
    cost = (
        Decimal(uncached) * rates["input"]
        + Decimal(cached) * rates["cached_input"]
        + Decimal(writes) * rates["cache_write"]
        + Decimal(usage.output_tokens) * rates["output"]
    ) / _TOKENS_PER_MILLION
    return float(cost)


def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


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
    "GPT56_LUNA_MODEL",
    "AuditedOpenAIClient",
    "LlmCallContext",
    "LlmTokenMetrics",
    "StructuredOutputError",
    "estimate_gpt56_luna_standard_cost",
]
