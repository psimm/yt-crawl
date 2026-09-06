from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest

from yt_crawl.classifier import (
    RelevanceClassifier,
    RelevanceDecision,
    VideoCandidate,
)
from yt_crawl.llm_runtime import (
    LLM_NUM_RETRIES,
    LLM_TIMEOUT_SECONDS,
    LoggedLlmClient,
    StructuredOutputError,
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


def _completion_response(
    *,
    content: str | None = None,
    response_id: str = "resp_test",
    model: str = "openai/gpt-5.6-luna",
    refusal: str | None = None,
    finish_reason: str | None = "stop",
) -> SimpleNamespace:
    if content is None:
        content = json.dumps(VALID_EXPANSION)
    return SimpleNamespace(
        id=response_id,
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        ),
    )


def _audited_client(tmp_path: Path, *, on_event=None, api_base=None, max_concurrency=1):
    writer = JsonlRunWriter(tmp_path, "run-audited")
    client = LoggedLlmClient(
        writer,
        on_event=on_event,
        max_concurrency=max_concurrency,
        api_base=api_base,
    )
    return client, writer


def _audit_rows(writer: JsonlRunWriter) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (writer.run_dir / "api_call.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _parse_expansion(client: LoggedLlmClient):
    with client.call_context("discovery", "expand_topic_queries"):
        return client.responses.parse(
            model="openai/gpt-5.6-luna",
            input="heat pumps",
            text_format=TopicExpansion,
        )


def test_valid_completed_response_is_parsed_and_audited_once(
    tmp_path, monkeypatch
) -> None:
    calls = []
    events = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _completion_response()

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path, on_event=events.append)
    response = _parse_expansion(client)

    assert response.output_parsed == TopicExpansion.model_validate(VALID_EXPANSION)
    (call,) = calls
    assert call["timeout"] == LLM_TIMEOUT_SECONDS
    assert call["num_retries"] == LLM_NUM_RETRIES
    assert "max_output_tokens" not in call
    assert "api_base" not in call
    assert call["messages"] == [{"role": "user", "content": "heat pumps"}]
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "success"
    (audit,) = _audit_rows(writer)
    assert audit["provider"] == "llm"
    assert audit["request_id"] == "resp_test"
    assert audit["status"] == "success"
    assert audit["llm_model"] == "openai/gpt-5.6-luna"
    assert audit["error"] is None


def test_classifier_messages_convert_developer_role(tmp_path, monkeypatch) -> None:
    calls = []
    decision = {
        "decision": "irrelevant",
        "language_match": "mismatch",
        "detected_language": "en",
        "primary_reason": "wrong_language",
    }

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _completion_response(content=json.dumps(decision))

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, _writer = _audited_client(tmp_path)
    prompt = CompiledClassifierPrompt(
        system_prompt="Stable classifier instructions and examples.",
        prompt_sha256="a" * 64,
    )
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

    assert result == RelevanceDecision.model_validate(decision)
    (call,) = calls
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"] == prompt.system_prompt
    assert call["messages"][1]["role"] == "user"
    assert "prompt_cache_key" not in call
    assert "prompt_cache_options" not in call


def test_instructions_become_a_system_message(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _completion_response()

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, _writer = _audited_client(tmp_path)
    with client.call_context("discovery", "expand_topic_queries"):
        client.responses.parse(
            model="openai/gpt-5.6-luna",
            instructions="Expand the brief.",
            input="heat pumps",
            text_format=TopicExpansion,
        )

    assert calls[0]["messages"] == [
        {"role": "system", "content": "Expand the brief."},
        {"role": "user", "content": "heat pumps"},
    ]


def test_api_base_is_forwarded_when_configured(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _completion_response()

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, _writer = _audited_client(
        tmp_path, api_base="http://127.0.0.1:11434/v1"
    )
    _parse_expansion(client)
    assert calls[0]["api_base"] == "http://127.0.0.1:11434/v1"


def test_malformed_structured_output_hides_parser_details(
    tmp_path, monkeypatch
) -> None:
    malformed = '{"topic_interpretation":"unfinished'

    def fake_completion(**_kwargs):
        return _completion_response(content=malformed, response_id="resp_bad_json")

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path)
    with pytest.raises(StructuredOutputError) as caught:
        _parse_expansion(client)

    message = str(caught.value)
    assert "completed response was not valid structured output" in message
    assert "resp_bad_json" in message
    assert malformed not in message
    assert "EOF" not in message
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"


def test_empty_content_is_an_error(tmp_path, monkeypatch) -> None:
    def fake_completion(**_kwargs):
        return _completion_response(content="", response_id="resp_empty")

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path)
    with pytest.raises(StructuredOutputError) as caught:
        _parse_expansion(client)

    assert "completed response contained no structured output" in str(caught.value)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["request_id"] == "resp_empty"


def test_refusal_is_a_safe_error(tmp_path, monkeypatch) -> None:
    def fake_completion(**_kwargs):
        return _completion_response(
            content=None,
            refusal="sensitive details",
            response_id="resp_refusal",
        )

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path)
    with pytest.raises(StructuredOutputError) as caught:
        _parse_expansion(client)

    assert "model refused the structured response" in str(caught.value)
    assert "sensitive details" not in str(caught.value)
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert "sensitive details" not in (audit["error"] or "")


def test_http_error_is_sanitized_and_audited(tmp_path, monkeypatch) -> None:
    events = []

    def fake_completion(**_kwargs):
        error = RuntimeError("private provider text")
        error.status_code = 500
        raise error

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path, on_event=events.append)
    with pytest.raises(RuntimeError):
        _parse_expansion(client)

    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].status == "error"
    (audit,) = _audit_rows(writer)
    assert audit["status"] == "error"
    assert audit["error"] == "LLM request failed with HTTP 500"
    assert "private provider text" not in (audit["error"] or "")


def test_unscoped_call_is_not_sent(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _completion_response()

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, _writer = _audited_client(tmp_path)
    with pytest.raises(RuntimeError):
        client.responses.parse(input="not audited", text_format=TopicExpansion)
    assert calls == []


def test_runtime_display_callback_cannot_fail_provider_call(
    tmp_path, monkeypatch
) -> None:
    def fake_completion(**_kwargs):
        return _completion_response()

    def broken_display(_event):
        raise RuntimeError("display unavailable")

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path, on_event=broken_display)
    response = _parse_expansion(client)
    assert response.id == "resp_test"
    assert len(_audit_rows(writer)) == 1


def test_concurrent_call_contexts_are_isolated_and_provider_calls_are_bounded(
    tmp_path, monkeypatch
) -> None:
    lock = Lock()
    active = 0
    max_active = 0

    def fake_completion(**kwargs):
        nonlocal active, max_active
        index = kwargs["messages"][0]["content"]
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            sleep(0.03)
            return _completion_response(response_id=f"resp-{index}")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("yt_crawl.llm_runtime.litellm.completion", fake_completion)
    client, writer = _audited_client(tmp_path, max_concurrency=2)

    def invoke(index: int) -> str:
        with client.call_context("discovery", f"operation-{index}"):
            response = client.responses.parse(
                model="openai/gpt-5.6-luna",
                input=str(index),
                text_format=TopicExpansion,
            )
        return response.id

    with ThreadPoolExecutor(max_workers=4) as executor:
        response_ids = tuple(executor.map(invoke, range(4)))

    assert set(response_ids) == {"resp-0", "resp-1", "resp-2", "resp-3"}
    assert max_active == 2
    rows = _audit_rows(writer)
    assert {row["operation"] for row in rows} == {
        "operation-0",
        "operation-1",
        "operation-2",
        "operation-3",
    }


def test_logged_client_rejects_invalid_concurrency(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        LoggedLlmClient(
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
