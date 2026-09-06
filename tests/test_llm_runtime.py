from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import OpenAI

from yt_crawl import observability
from yt_crawl.classifier import (
    RelevanceClassifier,
    RelevanceDecision,
    VideoCandidate,
)
from yt_crawl.llm_runtime import (
    AuditedOpenAIClient,
    LlmTokenMetrics,
    StructuredOutputError,
    estimate_gpt56_luna_standard_cost,
)
from yt_crawl.prompts import CompiledClassifierPrompt, TopicExpansion
from yt_crawl.storage import JsonlRunWriter

VALID_EXPANSION = {
    "topic_interpretation": "Practical heat-pump retrofits.",
    "inclusion_criteria": ["Concrete retrofit experience."],
    "exclusion_criteria": ["Passing mentions."],
    "search_queries": ["Wärmepumpe Mehrfamilienhaus"],
    "channel_discovery_queries": ["Wärmepumpe Praxis Kanal"],
    "ambiguity_rules": ["Require substantive coverage."],
}


def _envelope(
    *,
    response_id: str = "resp_test",
    status: str = "completed",
    output: list[dict[str, Any]] | None = None,
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
    input_tokens: int = 120,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    output_tokens: int = 30,
) -> dict[str, Any]:
    if output is None:
        output = [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(VALID_EXPANSION),
                        "annotations": [],
                    }
                ],
            }
        ]
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.6-luna",
        "status": status,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason else None
        ),
        "error": error,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _openai_client(handler, *, max_retries: int = 0) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        base_url="https://openai.test/v1",
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _audited_client(tmp_path: Path, handler, *, on_event=None, max_retries: int = 0):
    writer = JsonlRunWriter(tmp_path, "run-audited")
    inner = _openai_client(handler, max_retries=max_retries)
    return AuditedOpenAIClient(inner, writer, on_event=on_event), writer, inner


def _audit_rows(writer: JsonlRunWriter) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (writer.run_dir / "api_call.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _parse_expansion(client: AuditedOpenAIClient):
    with client.call_context("discovery", "expand_topic_queries"):
        return client.responses.parse(
            model="gpt-5.6-luna",
            input="heat pumps",
            text_format=TopicExpansion,
        )


def test_valid_completed_response_is_parsed_and_audited_once(tmp_path) -> None:
    request_bodies = []
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_envelope(cached_tokens=20, cache_write_tokens=40),
            headers={"x-request-id": "req_test"},
            request=request,
        )

    client, writer, inner = _audited_client(tmp_path, handler, on_event=events.append)
    try:
        response = _parse_expansion(client)
    finally:
        inner.close()

    assert response.output_parsed == TopicExpansion.model_validate(VALID_EXPANSION)
    assert "max_output_tokens" not in request_bodies[0]
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "success"
    assert (events[-1].input_tokens, events[-1].output_tokens) == (120, 30)
    assert events[-1].cached_input_tokens == 20
    assert events[-1].cache_write_tokens == 40
    assert events[-1].estimated_cost_usd == pytest.approx(0.0000584)
    (audit,) = _audit_rows(writer)
    assert audit["request_id"] == "resp_test"
    assert audit["status"] == "success"
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (120, 30)
    assert audit["llm_cached_input_tokens"] == 20
    assert audit["llm_cache_write_tokens"] == 40
    assert audit["llm_model"] == "gpt-5.6-luna"
    assert audit["llm_estimated_cost_usd"] == pytest.approx(0.0000584)
    assert audit["error"] is None


def test_classifier_cache_controls_reach_the_sdk_http_body(tmp_path) -> None:
    request_bodies: list[dict[str, Any]] = []
    decision = {
        "decision": "irrelevant",
        "language_match": "mismatch",
        "detected_language": "en",
        "primary_reason": "wrong_language",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        output = [
            {
                "id": "msg_classifier",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(decision),
                        "annotations": [],
                    }
                ],
            }
        ]
        return httpx.Response(
            200,
            json=_envelope(response_id="resp_classifier", output=output),
            request=request,
        )

    client, _writer, inner = _audited_client(tmp_path, handler)
    prompt = CompiledClassifierPrompt(
        system_prompt="Stable classifier instructions and examples.",
        prompt_sha256="a" * 64,
    )
    try:
        with client.call_context("discovery", "classify_video_metadata"):
            result = RelevanceClassifier(client).classify(
                VideoCandidate(
                    video_id="video-1",
                    title="Candidate",
                    discovery_source="search",
                    discovery_reference="query",
                ),
                prompt,
                stage="metadata",
            )
    finally:
        inner.close()

    assert result == RelevanceDecision.model_validate(decision)
    (body,) = request_bodies
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert body["prompt_cache_key"] == prompt.prompt_sha256
    assert "instructions" not in body
    assert "max_output_tokens" not in body
    assert body["input"][0]["role"] == "developer"
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert body["input"][1]["role"] == "user"
    assert (
        sum(
            "prompt_cache_breakpoint" in block
            for message in body["input"]
            for block in message["content"]
        )
        == 1
    )


def test_logfire_instrumented_sdk_response_is_warning_clean_and_audited_once(
    tmp_path,
    monkeypatch,
) -> None:
    requests = []
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_envelope(response_id="resp_instrumented"),
            headers={"x-request-id": "req_instrumented"},
            request=request,
        )

    observability.logfire.configure(send_to_logfire=False, console=False)
    monkeypatch.setattr(observability, "_configured", True)
    client, writer, inner = _audited_client(tmp_path, handler, on_event=events.append)
    try:
        assert observability.instrument_openai_client(inner) is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            response = _parse_expansion(client)
    finally:
        inner.close()

    assert response.output_parsed == TopicExpansion.model_validate(VALID_EXPANSION)
    assert len(requests) == 1
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "success"
    (audit,) = _audit_rows(writer)
    assert audit["request_id"] == "resp_instrumented"
    assert audit["status"] == "success"
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (120, 30)


def test_only_known_logfire_pydantic_serializer_warning_is_suppressed(
    tmp_path,
) -> None:
    class WarningRawResponse:
        request_id = "req_warning"

        def json(self):
            return _envelope(response_id="resp_warning")

        def parse(self):
            return SimpleNamespace(
                id="resp_warning",
                output_parsed=TopicExpansion.model_validate(VALID_EXPANSION),
            )

    class WarningResponses:
        with_raw_response = None

        def __init__(self):
            self.with_raw_response = self

        def parse(self, **_kwargs):
            warnings.warn_explicit(
                "Pydantic serializer warnings:\n"
                "  PydanticSerializationUnexpectedValue(Expected `none`)",
                UserWarning,
                filename="pydantic/main.py",
                lineno=475,
                module="pydantic.main",
            )
            return WarningRawResponse()

    writer = JsonlRunWriter(tmp_path, "run-warning")
    client = AuditedOpenAIClient(
        SimpleNamespace(responses=WarningResponses()),
        writer,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = _parse_expansion(client)

    assert response.output_parsed == TopicExpansion.model_validate(VALID_EXPANSION)
    assert len(_audit_rows(writer)) == 1


@pytest.mark.parametrize(
    ("message", "module"),
    [
        ("unrelated user warning", "pydantic.main"),
        (
            "Pydantic serializer warnings:\n  DifferentSerializerWarning(",
            "pydantic.main",
        ),
        (
            "Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected `none`)",
            "some_other_module",
        ),
    ],
)
def test_unrelated_or_changed_warnings_remain_visible(
    tmp_path,
    message,
    module,
) -> None:
    class VisibleWarningResponses:
        with_raw_response = None

        def __init__(self):
            self.with_raw_response = self

        def parse(self, **_kwargs):
            warnings.warn_explicit(
                message,
                UserWarning,
                filename=f"{module.replace('.', '/')}.py",
                lineno=1,
                module=module,
            )
            raise RuntimeError("warning did not become an error")

    writer = JsonlRunWriter(tmp_path, f"run-visible-{module.replace('.', '-')}")
    client = AuditedOpenAIClient(
        SimpleNamespace(responses=VisibleWarningResponses()),
        writer,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="^" + message.splitlines()[0]):
            _parse_expansion(client)

    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"


def test_new_style_raw_json_is_used_without_text_property(tmp_path) -> None:
    class JsonRawResponse:
        request_id = "req_new_style"

        def json(self):
            return _envelope(response_id="resp_new_style")

        def parse(self):
            return SimpleNamespace(
                id="resp_new_style",
                output_parsed=TopicExpansion.model_validate(VALID_EXPANSION),
            )

    class RawResponses:
        with_raw_response = None

        def __init__(self):
            self.with_raw_response = self

        def parse(self, **_kwargs):
            return JsonRawResponse()

    writer = JsonlRunWriter(tmp_path, "run-new-style")
    client = AuditedOpenAIClient(SimpleNamespace(responses=RawResponses()), writer)

    response = _parse_expansion(client)

    assert response.id == "resp_new_style"
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "success"
    assert audit["request_id"] == "resp_new_style"
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (120, 30)


def test_unreadable_raw_json_is_sanitized_and_audited_once(tmp_path) -> None:
    events = []

    class UnreadableRawResponse:
        request_id = "req_unreadable"

        def json(self):
            raise ValueError("private malformed body")

        def parse(self):
            raise AssertionError("typed parser must not run")

    class RawResponses:
        with_raw_response = None

        def __init__(self):
            self.with_raw_response = self

        def parse(self, **_kwargs):
            return UnreadableRawResponse()

    writer = JsonlRunWriter(tmp_path, "run-unreadable")
    client = AuditedOpenAIClient(
        SimpleNamespace(responses=RawResponses()),
        writer,
        on_event=events.append,
    )

    with pytest.raises(StructuredOutputError) as caught:
        _parse_expansion(client)

    message = str(caught.value)
    assert "provider returned an unreadable response envelope" in message
    assert "req_unreadable" in message
    assert "private malformed body" not in message
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] == "req_unreadable"
    assert "private malformed body" not in (audit["error"] or "")


def test_truncated_incomplete_response_is_cleanly_rejected_and_audited(
    tmp_path,
) -> None:
    events = []
    partial = '{"topic_interpretation":"unfinished'

    def handler(request: httpx.Request) -> httpx.Response:
        output = [
            {
                "id": "msg_partial",
                "type": "message",
                "role": "assistant",
                "status": "incomplete",
                "content": [
                    {"type": "output_text", "text": partial, "annotations": []}
                ],
            }
        ]
        return httpx.Response(
            200,
            json=_envelope(
                response_id="resp_partial",
                status="incomplete",
                output=output,
                incomplete_reason="max_output_tokens",
                input_tokens=410,
                output_tokens=900,
            ),
            request=request,
        )

    client, writer, inner = _audited_client(tmp_path, handler, on_event=events.append)
    try:
        with pytest.raises(StructuredOutputError) as caught:
            _parse_expansion(client)
    finally:
        inner.close()

    message = str(caught.value)
    assert "response was incomplete (max_output_tokens)" in message
    assert "resp_partial" in message
    assert partial not in message
    assert "EOF" not in message
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    assert (events[-1].input_tokens, events[-1].output_tokens) == (410, 900)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] == "resp_partial"
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (410, 900)


@pytest.mark.parametrize(
    ("envelope", "reason"),
    [
        (
            _envelope(
                response_id="resp_refusal",
                output=[
                    {
                        "id": "msg_refusal",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "refusal", "refusal": "sensitive details"}
                        ],
                    }
                ],
            ),
            "model refused the structured response",
        ),
        (
            _envelope(
                response_id="resp_filtered",
                status="incomplete",
                incomplete_reason="content_filter",
            ),
            "response was incomplete (content_filter)",
        ),
        (
            _envelope(
                response_id="resp_failed",
                status="failed",
                error={"code": "server_error", "message": "private provider text"},
            ),
            "response failed (server_error)",
        ),
    ],
)
def test_noncompleted_and_refusal_responses_are_safe_errors(
    tmp_path,
    envelope,
    reason,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope, request=request)

    client, writer, inner = _audited_client(tmp_path, handler)
    try:
        with pytest.raises(StructuredOutputError) as caught:
            _parse_expansion(client)
    finally:
        inner.close()

    (audit,) = _audit_rows(writer)
    assert reason in str(caught.value)
    assert audit["status"] == "error"
    assert audit["request_id"] == envelope["id"]
    assert "private provider text" not in (audit["error"] or "")
    assert "sensitive details" not in (audit["error"] or "")


def test_completed_malformed_structured_output_hides_parser_details(tmp_path) -> None:
    malformed = '{"topic_interpretation":"unfinished'

    def handler(request: httpx.Request) -> httpx.Response:
        output = [
            {
                "id": "msg_bad_json",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": malformed, "annotations": []}
                ],
            }
        ]
        return httpx.Response(
            200,
            json=_envelope(response_id="resp_bad_json", output=output),
            request=request,
        )

    client, writer, inner = _audited_client(tmp_path, handler)
    try:
        with pytest.raises(StructuredOutputError) as caught:
            _parse_expansion(client)
    finally:
        inner.close()

    message = str(caught.value)
    assert "completed response was not valid structured output" in message
    assert "resp_bad_json" in message
    assert malformed not in message
    assert "EOF" not in message
    assert len(_audit_rows(writer)) == 1
    assert _audit_rows(writer)[0]["status"] == "error"


def test_completed_response_without_structured_output_is_an_error(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(response_id="resp_empty", output=[]),
            request=request,
        )

    client, writer, inner = _audited_client(tmp_path, handler)
    try:
        with pytest.raises(StructuredOutputError) as caught:
            _parse_expansion(client)
    finally:
        inner.close()

    assert "completed response contained no structured output" in str(caught.value)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] == "resp_empty"


def test_network_error_balances_events_and_records_zero_usage(tmp_path) -> None:
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    client, writer, inner = _audited_client(tmp_path, handler, on_event=events.append)
    try:
        with pytest.raises(Exception, match="Connection error"):
            _parse_expansion(client)
    finally:
        inner.close()

    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    assert (events[-1].input_tokens, events[-1].output_tokens) == (0, 0)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] is None
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (0, 0)


def test_timeout_balances_events_and_records_zero_usage(tmp_path) -> None:
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("response stalled", request=request)

    client, writer, inner = _audited_client(tmp_path, handler, on_event=events.append)
    try:
        with pytest.raises(Exception, match="Request timed out"):
            _parse_expansion(client)
    finally:
        inner.close()

    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    assert (events[-1].input_tokens, events[-1].output_tokens) == (0, 0)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] is None
    assert (audit["llm_input_tokens"], audit["llm_output_tokens"]) == (0, 0)


def test_transient_network_failure_succeeds_within_one_audited_call(tmp_path) -> None:
    attempts = 0
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary network failure", request=request)
        return httpx.Response(200, json=_envelope(), request=request)

    client, writer, inner = _audited_client(
        tmp_path,
        handler,
        on_event=events.append,
        max_retries=3,
    )
    try:
        response = _parse_expansion(client)
    finally:
        inner.close()

    assert response.output_parsed == TopicExpansion.model_validate(VALID_EXPANSION)
    assert attempts == 3
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "success"
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "success"


def test_persistent_sdk_failure_has_one_balanced_audit_error(tmp_path) -> None:
    attempts = 0
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            500,
            json={"error": {"message": "persistent failure"}},
            headers={"retry-after-ms": "1"},
            request=request,
        )

    client, writer, inner = _audited_client(
        tmp_path,
        handler,
        on_event=events.append,
        max_retries=3,
    )
    try:
        with pytest.raises(Exception, match="persistent failure"):
            _parse_expansion(client)
    finally:
        inner.close()

    assert attempts == 4
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"


def test_unscoped_call_is_not_sent(tmp_path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_envelope(), request=request)

    client, _writer, inner = _audited_client(tmp_path, handler)
    try:
        with pytest.raises(RuntimeError):
            client.responses.parse(input="not audited")
    finally:
        inner.close()
    assert calls == []


def test_runtime_display_callback_cannot_fail_provider_call(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(), request=request)

    def broken_display(_event):
        raise RuntimeError("display unavailable")

    client, writer, inner = _audited_client(
        tmp_path,
        handler,
        on_event=broken_display,
    )
    try:
        response = _parse_expansion(client)
    finally:
        inner.close()

    assert response.id == "resp_test"
    assert len(_audit_rows(writer)) == 1


def test_concurrent_call_contexts_are_isolated_and_provider_calls_are_bounded(
    tmp_path,
) -> None:
    lock = Lock()
    active = 0
    max_active = 0

    class ConcurrentRawResponse:
        def __init__(self, index: int) -> None:
            self.index = index
            self.request_id = f"req-{index}"

        def json(self):
            return _envelope(response_id=f"resp-{self.index}")

        def parse(self):
            return SimpleNamespace(
                id=f"resp-{self.index}",
                output_parsed=TopicExpansion.model_validate(VALID_EXPANSION),
            )

    class ConcurrentResponses:
        with_raw_response = None

        def __init__(self) -> None:
            self.with_raw_response = self

        def parse(self, **kwargs):
            nonlocal active, max_active
            index = int(kwargs["input"])
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                sleep(0.03)
                return ConcurrentRawResponse(index)
            finally:
                with lock:
                    active -= 1

    writer = JsonlRunWriter(tmp_path, "run-concurrent")
    client = AuditedOpenAIClient(
        SimpleNamespace(responses=ConcurrentResponses()),
        writer,
        max_concurrency=2,
    )

    def invoke(index: int) -> str:
        with client.call_context("discovery", f"operation-{index}"):
            response = client.responses.parse(
                model="gpt-5.6-luna",
                input=str(index),
                text_format=TopicExpansion,
            )
        return response.id

    with ThreadPoolExecutor(max_workers=4) as executor:
        response_ids = tuple(executor.map(invoke, range(4)))

    assert response_ids == ("resp-0", "resp-1", "resp-2", "resp-3")
    assert max_active == 2
    rows = _audit_rows(writer)
    assert {row["operation"] for row in rows} == {
        "operation-0",
        "operation-1",
        "operation-2",
        "operation-3",
    }
    assert {row["request_id"] for row in rows} == {
        "resp-0",
        "resp-1",
        "resp-2",
        "resp-3",
    }


def test_luna_standard_cost_uses_cache_read_write_and_context_tiers() -> None:
    short = estimate_gpt56_luna_standard_cost(
        LlmTokenMetrics(
            input_tokens=120,
            cached_input_tokens=20,
            cache_write_tokens=40,
            output_tokens=30,
        ),
        model="gpt-5.6-luna",
    )
    long = estimate_gpt56_luna_standard_cost(
        LlmTokenMetrics(
            input_tokens=300_000,
            cached_input_tokens=100_000,
            cache_write_tokens=50_000,
            output_tokens=1_000,
        ),
        model="gpt-5.6-luna",
    )

    assert short == pytest.approx(0.0000584)
    assert long == pytest.approx(0.0908)
    assert (
        estimate_gpt56_luna_standard_cost(
            LlmTokenMetrics(input_tokens=100),
            model="gpt-5.6-terra",
        )
        is None
    )


def test_audited_client_rejects_invalid_concurrency(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        AuditedOpenAIClient(
            object(),
            JsonlRunWriter(tmp_path, "invalid"),
            max_concurrency=0,
        )


def test_application_never_supplies_an_openai_output_cap() -> None:
    source_root = Path(__file__).parents[1] / "src" / "yt_crawl"
    offenders = [
        path
        for path in source_root.rglob("*.py")
        if "max_output_tokens" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
