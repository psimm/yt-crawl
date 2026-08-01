import logging
import warnings
from types import SimpleNamespace

import pytest
from loguru import logger
from rich.style import Style

from yt_searchapi import observability


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_application_logging_and_warnings_do_not_leak_to_terminal(capsys) -> None:
    logger.info("quiet loguru message")
    with pytest.warns(UserWarning, match="quiet warning message"):
        warnings.warn("quiet warning message", UserWarning, stacklevel=1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_logfire_control_is_clickable_and_keeps_visible_fallback_url() -> None:
    link = observability.logfire_link()

    assert str(link).startswith("View logs in Logfire")
    assert observability.LOGFIRE_URL in str(link)
    assert any(
        span.style and Style.parse(str(span.style)).link == observability.LOGFIRE_URL
        for span in link.spans
    )


def test_configure_uses_official_logfire_bridge_and_required_options(
    monkeypatch,
) -> None:
    config_calls = []
    bridge_calls = []
    handler = _RecordingHandler()

    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.delenv("YT_SEARCHAPI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(
        observability.logfire,
        "configure",
        lambda **kwargs: config_calls.append(kwargs),
    )
    monkeypatch.setattr(
        observability.logfire,
        "loguru_handler",
        lambda: bridge_calls.append(True) or handler,
    )
    monkeypatch.setattr(
        observability.logfire,
        "LogfireLoggingHandler",
        lambda **_kwargs: _RecordingHandler(),
    )

    assert observability.configure_observability() is True
    logger.info("bridged message")

    assert config_calls == [
        {
            "send_to_logfire": "if-token-present",
            "console": False,
            "service_name": "yt-searchapi",
        }
    ]
    assert bridge_calls == [True]
    assert len(handler.messages) == 1
    assert handler.messages[0].endswith("bridged message")


def test_unreachable_configuration_is_nonfatal_and_has_no_terminal_sink(
    monkeypatch, capsys
) -> None:
    unrelated = _RecordingHandler()
    root = logging.getLogger()
    root.addHandler(unrelated)
    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.setattr(
        observability.logfire,
        "configure",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    assert observability.configure_observability() is False
    logger.error("must remain quiet")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert unrelated in root.handlers
    root.removeHandler(unrelated)


def test_test_disable_switch_prevents_export(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.setenv("YT_SEARCHAPI_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(
        observability.logfire,
        "configure",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        observability.logfire, "loguru_handler", lambda: _RecordingHandler()
    )
    monkeypatch.setattr(
        observability.logfire,
        "LogfireLoggingHandler",
        lambda **_kwargs: _RecordingHandler(),
    )

    assert observability.configure_observability() is True
    assert calls[0]["send_to_logfire"] is False


def test_manual_session_span_contains_only_intended_metadata(monkeypatch) -> None:
    calls = []
    attributes = {}

    class FakeSpan:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *_args):
            calls.append("exit")

        def set_attribute(self, name, value):
            attributes[name] = value

    def fake_span(template, **kwargs):
        calls.append((template, kwargs))
        return FakeSpan()

    monkeypatch.setattr(observability.logfire, "span", fake_span)
    monkeypatch.setattr(
        observability.logfire,
        "force_flush",
        lambda **kwargs: calls.append(("flush", kwargs)) or True,
    )

    with observability.RunSessionSpan(
        run_id="run-17",
        project="runs/example",
        action="resume",
        planned_credits=12,
        credits_added=8,
        controls={"max_queries": 6, "max_depth": 2},
    ) as session:
        session.set_outcome("completed", "frontier exhausted")

    template, sent = calls[0]
    assert template == "yt-searchapi {action} session"
    assert sent["run_id"] == "run-17"
    assert sent["action"] == "resume"
    assert sent["planned_credits"] == 12
    assert sent["credits_added"] == 8
    assert sent["max_queries"] == 6
    assert sent["max_depth"] == 2
    assert not {
        "api_key",
        "token",
        "prompt",
        "transcript",
        "provider_body",
    }.intersection(sent)
    assert attributes == {
        "status": "completed",
        "stop_reason": "frontier exhausted",
    }
    assert calls[-2:] == ["exit", ("flush", {"timeout_millis": 500})]


def test_span_and_flush_failures_never_escape(monkeypatch) -> None:
    monkeypatch.setattr(
        observability.logfire,
        "span",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    monkeypatch.setattr(
        observability.logfire,
        "force_flush",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    with observability.RunSessionSpan(
        run_id="run",
        project="project",
        action="start",
        planned_credits=4,
    ) as session:
        session.set_outcome("completed")


def test_configuration_does_not_globally_instrument_openai_or_pydantic(
    monkeypatch,
) -> None:
    calls = SimpleNamespace(openai=0, pydantic=0)

    def instrument_openai(*_args, **_kwargs):
        calls.openai += 1

    def instrument_pydantic(*_args, **_kwargs):
        calls.pydantic += 1

    monkeypatch.setattr(
        observability.logfire,
        "instrument_openai",
        instrument_openai,
        raising=False,
    )
    monkeypatch.setattr(
        observability.logfire,
        "instrument_pydantic",
        instrument_pydantic,
        raising=False,
    )

    # Configuration alone uses only the manual bridges above. The shared CLI
    # factory instruments its concrete client separately.
    observability.configure_observability()
    assert calls.openai == 0
    assert calls.pydantic == 0


def test_openai_instance_instrumentation_is_configured_and_idempotent(
    monkeypatch,
) -> None:
    calls = []
    client = SimpleNamespace()
    monkeypatch.setattr(observability, "_configured", True)
    monkeypatch.setattr(
        observability.logfire,
        "instrument_openai",
        lambda value: calls.append(value),
    )

    assert observability.instrument_openai_client(client) is True
    assert observability.instrument_openai_client(client) is True
    assert calls == [client]


def test_openai_instance_instrumentation_requires_configuration(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.setattr(
        observability.logfire,
        "instrument_openai",
        lambda value: calls.append(value),
    )

    assert observability.instrument_openai_client(SimpleNamespace()) is False
    assert calls == []


def test_openai_instance_instrumentation_failure_is_nonfatal(monkeypatch) -> None:
    client = SimpleNamespace()
    monkeypatch.setattr(observability, "_configured", True)
    monkeypatch.setattr(
        observability.logfire,
        "instrument_openai",
        lambda _value: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    assert observability.instrument_openai_client(client) is False
    assert not hasattr(client, observability._OPENAI_INSTRUMENTED_ATTR)
