import json
from contextlib import contextmanager
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.main import get_command
from typer.testing import CliRunner

import yt_searchapi.cli as cli
from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.cli import (
    _commit_resume_state,
    _expanded_transcript_reserve,
    _openai_client,
    _preflight_funding,
    _suggested_next_command,
    _transcript_reserve,
    _visible_start_error,
    app,
)
from yt_searchapi.crawler import CrawlSummary
from yt_searchapi.llm_runtime import StructuredOutputError
from yt_searchapi.prompts import CLASSIFIER_PROMPT_VERSION
from yt_searchapi.records import RunStatus
from yt_searchapi.settings import (
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)
from yt_searchapi.start_settings_ui import collect_missing_start_settings
from yt_searchapi.state import (
    BudgetState,
    CrawlProjectState,
    ProjectStateStore,
    validate_resumable_classifier_prompt,
)

CLASSIFY_PROMPT_SHA256 = (
    "4d553b0b48b4bb151321bd8aa5d9904477ecfdbec8dea4f2554cef83409c8ae6"
)


class _RecordingSessionSpan:
    calls: list[tuple[str, object]] = []

    def __init__(self, **kwargs):
        self.calls.append(("init", kwargs))

    def set_outcome(self, status, stop_reason=None):
        self.calls.append(("outcome", (status, stop_reason)))

    def close(self):
        self.calls.append(("close", None))


def _use_noninteractive_start_defaults(monkeypatch) -> None:
    defaults = {
        "topic": "topic",
        "project": "project",
        "max_credits": 4,
        "language": "en",
        "start_date": "2026-01-01",
        "max_depth": 2,
        "max_queries": 8,
        "max_search_pages": 1,
        "max_channel_pages": 1,
        "gl": "us",
        "hl": "en",
        "searchapi_timeout": 90.0,
        "searchapi_retries": DEFAULT_SEARCHAPI_RETRIES,
        "searchapi_workers": DEFAULT_SEARCHAPI_WORKERS,
        "llm_workers": DEFAULT_LLM_WORKERS,
    }

    def resolve(provided, _questions):
        return {
            **defaults,
            **{key: value for key, value in provided.items() if value is not None},
        }

    monkeypatch.setattr(cli, "collect_missing_start_settings", resolve)


def _checkpoint(project, *, status="failed") -> CrawlProjectState:
    state = CrawlProjectState(
        run_id=project.name,
        topic_query="heat pumps",
        language="en",
        start_date="2026-01-01",
        gl="us",
        hl="en",
        transcript_excerpt_chars=12_000,
        max_depth=0,
        max_queries=2,
        max_search_pages=1,
        max_channel_pages=1,
        expansion={},
        classifier_system_prompt="Classify.",
        prompt_sha256=CLASSIFY_PROMPT_SHA256,
        classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
        planned_queries=[
            {"text": "heat pumps", "kind": "seed"},
            {"text": "heat pumps interview", "kind": "interview"},
            {"text": "heat pump retrofit", "kind": "expanded"},
        ],
        budget=BudgetState(
            max_credits=4,
            transcript_capacity=1,
            discovery_spent=0,
            transcript_spent=0,
        ),
        last_status=status,
    )
    ProjectStateStore(project).save(state)
    return state


def test_checkpoint_uses_configured_parallelism_defaults(tmp_path) -> None:
    state = _checkpoint(tmp_path / "default-workers")

    assert state.searchapi_workers == DEFAULT_SEARCHAPI_WORKERS
    assert state.searchapi_retries == DEFAULT_SEARCHAPI_RETRIES
    assert state.llm_workers == DEFAULT_LLM_WORKERS


def test_help_does_not_require_credentials_or_make_queries() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "SearchAPI" in result.stdout
    assert "start" in result.stdout
    assert "resume" in result.stdout


def test_readme_documents_every_cli_option() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    root = get_command(app)
    documented_options = {
        option
        for command in root.commands.values()
        for parameter in command.params
        for option in (*parameter.opts, *parameter.secondary_opts)
        if option.startswith("--")
    }

    missing = sorted(
        option for option in documented_options if f"`{option}" not in readme
    )
    assert missing == []


def test_legacy_missing_prompt_version_is_rejected_before_credentials(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "legacy-prompt"
    state = _checkpoint(project)
    payload = state.model_dump(mode="json")
    del payload["classifier_prompt_version"]
    (project / "crawl_state.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("credentials must not run")),
    )
    result = CliRunner().invoke(app, ["resume", "--project", str(project)])

    assert result.exit_code == 2
    message = " ".join((result.stdout + result.stderr).replace("│", "").split())
    assert "uses classifier prompt version 'relevance-v1'" in message
    assert "Migrate the checkpoint" in message


def test_prompt_hash_mismatch_is_rejected_before_credentials(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "tampered-prompt"
    state = _checkpoint(project)
    payload = state.model_dump(mode="json")
    payload["prompt_sha256"] = "b" * 64
    (project / "crawl_state.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("credentials must not run")),
    )
    result = CliRunner().invoke(app, ["resume", "--project", str(project)])

    assert result.exit_code == 2
    message = " ".join((result.stdout + result.stderr).replace("│", "").split())
    assert "prompt hash does not match" in message


def test_current_prompt_version_passes_resume_validation(tmp_path) -> None:
    state = _checkpoint(tmp_path / "current-prompt")

    validate_resumable_classifier_prompt(state)


def test_start_without_flags_routes_every_setting_through_interview(
    monkeypatch,
) -> None:
    captured = {}
    question_defaults = {}

    def stop_after_capture(provided, questions):
        captured.update(provided)
        question_defaults.update(
            {question.key: question.default for question in questions}
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "collect_missing_start_settings", stop_after_capture)

    result = CliRunner().invoke(app, ["start"])

    assert result.exit_code == 1
    assert set(captured) == {
        "topic",
        "project",
        "max_credits",
        "language",
        "start_date",
        "max_depth",
        "max_queries",
        "max_search_pages",
        "max_channel_pages",
        "gl",
        "hl",
        "searchapi_timeout",
        "searchapi_retries",
        "searchapi_workers",
        "llm_workers",
    }
    assert all(value is None for value in captured.values())
    assert question_defaults["searchapi_workers"] == str(DEFAULT_SEARCHAPI_WORKERS)
    assert question_defaults["searchapi_retries"] == str(DEFAULT_SEARCHAPI_RETRIES)
    assert question_defaults["llm_workers"] == str(DEFAULT_LLM_WORKERS)


def test_fully_scripted_start_does_not_open_setting_prompts(
    monkeypatch, tmp_path
) -> None:
    class FailPrompts:
        def text(self, *_args, **_kwargs):
            raise AssertionError("no supplied setting should be asked")

    captured = {}

    def resolve(provided, questions):
        captured.update(provided)
        return collect_missing_start_settings(
            provided, questions, prompts=FailPrompts()
        )

    monkeypatch.setattr(cli, "collect_missing_start_settings", resolve)
    monkeypatch.setattr(
        cli,
        "_require_new_project",
        lambda _project: (_ for _ in ()).throw(cli.typer.BadParameter("stop")),
    )

    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "heat pumps",
            "--project",
            str(tmp_path / "scripted"),
            "--max-credits",
            "8",
            "--language",
            "de",
            "--start-date",
            "2025-01-01",
            "--max-depth",
            "1",
            "--max-queries",
            "4",
            "--max-search-pages",
            "2",
            "--max-channel-pages",
            "2",
            "--country",
            "de",
            "--interface-language",
            "de-DE",
            "--searchapi-timeout",
            "30",
            "--searchapi-retries",
            "2",
            "--searchapi-workers",
            "6",
            "--llm-workers",
            "3",
        ],
    )

    assert result.exit_code == 2
    assert captured["hl"] == "de-DE"
    assert captured["searchapi_timeout"] == 30.0
    assert captured["searchapi_retries"] == 2
    assert captured["searchapi_workers"] == 6
    assert captured["llm_workers"] == 3


def test_transcript_reserve_leaves_seed_and_detail_capacity() -> None:
    for credits in range(4, 100):
        reserve = _transcript_reserve(credits)
        assert reserve >= 1
        assert credits - reserve >= 3


def test_expanded_transcript_reserve_preserves_transferred_capacity() -> None:
    budget = SearchApiCreditBudget.restore(
        {
            "max_credits": 4,
            "transcript_capacity": 4,
            "discovery_spent": 0,
            "transcript_spent": 4,
            "pending": [],
            "completed_video_ids": ["video-1", "video-2", "video-3", "video-4"],
        }
    )

    assert _expanded_transcript_reserve(budget, 8) == 4


def test_openai_client_has_bounded_timeout_and_three_sdk_retries(
    monkeypatch,
) -> None:
    captured = {}
    sentinel = object()
    instrumented = []

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli, "OpenAI", fake_openai)
    monkeypatch.setattr(
        cli,
        "instrument_openai_client",
        lambda client: instrumented.append(client) or True,
    )

    assert _openai_client("test-key") is sentinel
    assert captured == {
        "api_key": "test-key",
        "max_retries": 3,
        "timeout": 90.0,
    }
    assert instrumented == [sentinel]


def test_structured_preparation_error_says_crawl_did_not_spend_credits() -> None:
    error = StructuredOutputError(
        "expand_topic_queries",
        "response was incomplete",
        response_id="resp_test",
    )

    message = _visible_start_error(error, preparation_completed=False)

    assert "response was incomplete" in message
    assert "resp_test" in message
    assert "crawl did not start" in message
    assert "no SearchAPI crawl credits were used" in message


def test_blank_topic_fails_before_settings_or_clients(monkeypatch, tmp_path) -> None:
    _use_noninteractive_start_defaults(monkeypatch)

    def fail(*_args, **_kwargs):
        raise AssertionError("paid client setup must not run")

    monkeypatch.setattr(cli, "Settings", fail)
    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "   ",
            "--language",
            "en",
            "--max-credits",
            "4",
            "--start-date",
            "2026-01-01",
            "--project",
            str(tmp_path / "new-project"),
        ],
    )

    assert result.exit_code == 2
    assert "topic must not be blank" in result.stderr


def test_missing_credentials_fails_before_provider_clients(
    monkeypatch, tmp_path
) -> None:
    _use_noninteractive_start_defaults(monkeypatch)

    class EmptySettings:
        searchapi_api_key = None
        openai_api_key = None

    def fail(*_args, **_kwargs):
        raise AssertionError("provider client must not be constructed")

    monkeypatch.setattr(cli, "Settings", EmptySettings)
    monkeypatch.setattr(cli, "_openai_client", fail)
    monkeypatch.setattr(cli, "SearchApiClient", fail)
    monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "heat pumps",
            "--language",
            "en",
            "--max-credits",
            "4",
            "--start-date",
            "2026-01-01",
            "--project",
            str(tmp_path / "new-project"),
        ],
    )

    assert result.exit_code == 2
    assert "Missing environment variable" in result.stderr


def test_start_opens_one_manual_session_span_with_controls(
    monkeypatch, tmp_path
) -> None:
    _use_noninteractive_start_defaults(monkeypatch)
    project = tmp_path / "start-span"
    openai_client = SimpleNamespace()
    instrumented = []
    _RecordingSessionSpan.calls = []
    monkeypatch.setattr(cli, "RunSessionSpan", _RecordingSessionSpan)
    monkeypatch.setattr(cli, "_credentials", lambda: ("search-secret", "openai-secret"))
    monkeypatch.setattr(cli, "_check_searchapi_funding", lambda *_args: 100)
    monkeypatch.setattr(cli, "OpenAI", lambda **_kwargs: openai_client)
    monkeypatch.setattr(
        cli,
        "instrument_openai_client",
        lambda client: instrumented.append(client) or True,
    )
    monkeypatch.setattr(
        cli,
        "_prepare_research",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after span")),
    )

    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "heat pumps",
            "--max-credits",
            "7",
            "--max-queries",
            "5",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 1
    initial = _RecordingSessionSpan.calls[0]
    assert initial[0] == "init"
    assert initial[1] == {
        "run_id": "start-span",
        "project": project.resolve(),
        "action": "start",
        "planned_credits": 7,
        "controls": {
            "max_depth": 2,
            "max_queries": 5,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    }
    assert "search-secret" not in repr(initial)
    assert "openai-secret" not in repr(initial)
    assert instrumented == [openai_client]
    assert _RecordingSessionSpan.calls[-1] == ("close", None)


def test_resume_opens_one_manual_session_span_with_added_credits(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "resume-span"
    openai_client = SimpleNamespace()
    instrumented = []
    _checkpoint(project)
    _RecordingSessionSpan.calls = []
    monkeypatch.setattr(cli, "RunSessionSpan", _RecordingSessionSpan)
    monkeypatch.setattr(cli, "_credentials", lambda: ("search-secret", "openai-secret"))
    monkeypatch.setattr(cli, "_check_searchapi_funding", lambda *_args: 100)
    monkeypatch.setattr(cli, "OpenAI", lambda **_kwargs: openai_client)
    monkeypatch.setattr(
        cli,
        "instrument_openai_client",
        lambda client: instrumented.append(client) or True,
    )
    monkeypatch.setattr(
        cli,
        "SearchApiClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after span")
        ),
    )

    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--add-credits", "3"],
    )

    assert result.exit_code == 1
    initial = _RecordingSessionSpan.calls[0]
    assert initial[0] == "init"
    assert initial[1] == {
        "run_id": "resume-span",
        "project": project.resolve(),
        "action": "resume",
        "planned_credits": 7,
        "credits_added": 3,
        "controls": {
            "max_depth": 0,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    }
    assert "search-secret" not in repr(initial)
    assert "openai-secret" not in repr(initial)
    assert instrumented == [openai_client]
    assert _RecordingSessionSpan.calls[-1] == ("close", None)


def test_start_explains_recovery_when_preparation_has_no_checkpoint(
    monkeypatch, tmp_path
) -> None:
    _use_noninteractive_start_defaults(monkeypatch)
    project = tmp_path / "interrupted-preparation"
    project.mkdir()
    (project / "run_error.jsonl").write_text("partial audit\n", encoding="utf-8")

    def fail():
        raise AssertionError("credentials must not be read")

    monkeypatch.setattr(cli, "_credentials", fail)
    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "heat pumps",
            "--language",
            "en",
            "--max-credits",
            "4",
            "--start-date",
            "2026-01-01",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 2
    assert "No valid resumable checkpoint is available" in result.stderr
    assert "Research preparation may not have finished" in result.stderr
    assert "start --project <new-path>" in result.stderr
    assert "yt-crawl resume" not in result.stderr


def test_resume_missing_checkpoint_uses_same_recovery_guidance(tmp_path) -> None:
    project = tmp_path / "interrupted-preparation"
    project.mkdir()

    result = CliRunner().invoke(app, ["resume", "--project", str(project)])

    assert result.exit_code == 2
    assert "No valid resumable checkpoint is available" in result.stderr
    assert "start --project <new-path>" in result.stderr
    assert list(project.iterdir()) == []
    cli._require_new_project(project)


def test_resume_invalid_checkpoint_does_not_create_project_log(tmp_path) -> None:
    project = tmp_path / "invalid-checkpoint"
    project.mkdir()
    checkpoint = project / "crawl_state.json"
    checkpoint.write_text("not json", encoding="utf-8")

    result = CliRunner().invoke(app, ["resume", "--project", str(project)])

    assert result.exit_code == 2
    assert "No valid resumable checkpoint is available" in result.stderr
    assert set(project.iterdir()) == {checkpoint}
    assert not (project / "crawler.log").exists()


def test_resume_rejects_control_decrease_without_project_log(tmp_path) -> None:
    project = tmp_path / "decreased-controls"
    state = _checkpoint(project).model_copy(update={"max_depth": 2})
    ProjectStateStore(project).save(state)

    result = CliRunner().invoke(
        app, ["resume", "--project", str(project), "--max-depth", "1"]
    )

    assert result.exit_code == 2
    assert "cannot decrease from 2 to 1" in result.stderr
    assert not (project / "crawler.log").exists()


def test_resume_rejects_later_start_date_before_credentials(
    tmp_path, monkeypatch
) -> None:
    project = tmp_path / "later-start-date"
    _checkpoint(project)

    monkeypatch.setattr(
        cli,
        "_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("credentials must not run")),
    )
    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--start-date", "2026-02-01"],
    )

    assert result.exit_code == 2
    message = " ".join(result.stderr.replace("│", "").split())
    assert "cannot move later from 2026-01-01 to 2026-02-01" in message
    assert ProjectStateStore(project).load().start_date == date(2026, 1, 1)
    assert not (project / "run_config.jsonl").exists()


def test_account_preflight_rejects_underfunded_plan(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def me(self):
            calls.append("me")
            return SimpleNamespace(account=SimpleNamespace(remaining_credits=4))

    monkeypatch.setattr(cli, "SearchApiClient", FakeClient)

    with pytest.raises(cli.typer.BadParameter, match="4 credits available, 5 required"):
        _preflight_funding("key", 5)

    assert calls == ["me"]


def test_account_preflight_accepts_fully_funded_plan(monkeypatch) -> None:
    constructor_kwargs = {}

    class FakeClient:
        def __init__(self, *_args, **kwargs):
            constructor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def me(self):
            return SimpleNamespace(account=SimpleNamespace(remaining_credits=12))

    monkeypatch.setattr(cli, "SearchApiClient", FakeClient)

    assert _preflight_funding("key", 12, 30.5, 6) == 12
    assert constructor_kwargs["timeout"] == 30.5
    assert constructor_kwargs["max_workers"] == 6
    assert constructor_kwargs["max_retries"] == 0


def test_funding_check_uses_visible_status(monkeypatch) -> None:
    labels = []

    class StatusConsole:
        @contextmanager
        def status(self, label):
            labels.append(label)
            yield

    monkeypatch.setattr(cli, "console", StatusConsole())
    monkeypatch.setattr(cli, "_preflight_funding", lambda *_args: 12)

    assert cli._check_searchapi_funding("key", 7) == 12
    assert len(labels) == 1
    assert "Checking SearchAPI funding" in labels[0]


def test_underfunded_start_does_not_create_project_or_log(
    monkeypatch, tmp_path
) -> None:
    _use_noninteractive_start_defaults(monkeypatch)
    project = tmp_path / "underfunded"
    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))

    def reject(*_args):
        raise cli.typer.BadParameter("account is underfunded")

    monkeypatch.setattr(cli, "_preflight_funding", reject)

    result = CliRunner().invoke(
        app,
        [
            "start",
            "--topic",
            "heat pumps",
            "--max-credits",
            "4",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 2
    assert "underfunded" in result.stderr
    assert not project.exists()


def test_budget_stop_next_command_reuses_unspent_credits(tmp_path) -> None:
    summary = CrawlSummary(
        status=RunStatus.STOPPED_BUDGET,
        stop_reason="budget",
        videos_discovered=2,
        videos_evaluated=1,
        relevant_videos=0,
        transcripts_collected=0,
        channels_expanded=0,
        queries_executed=1,
        pending_videos=1,
    )
    budget = SearchApiCreditBudget(12, 4).snapshot()

    command = _suggested_next_command(
        summary,
        tmp_path / "my project",
        budget,
        controls={
            "max_depth": 1,
            "max_queries": 3,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    assert command is not None
    assert "yt-crawl resume" in command
    assert "--add-credits 0" in command
    assert "my project" in command


def test_budget_stop_next_command_adds_credits_when_total_is_exhausted(
    tmp_path,
) -> None:
    summary = CrawlSummary(
        status=RunStatus.STOPPED_BUDGET,
        stop_reason="budget",
        videos_discovered=2,
        videos_evaluated=1,
        relevant_videos=0,
        transcripts_collected=0,
        channels_expanded=0,
        queries_executed=1,
        pending_videos=1,
    )
    budget = SearchApiCreditBudget(12, 4)
    budget.spend_discovery(8)
    for video_id in ("video-1", "video-2", "video-3", "video-4"):
        budget.reconcile_transcript(budget.mark_relevant(video_id))

    command = _suggested_next_command(
        summary,
        tmp_path / "my project",
        budget.snapshot(),
        controls={
            "max_depth": 1,
            "max_queries": 3,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    assert command is not None
    assert "--add-credits 6" in command


@pytest.mark.parametrize("width", [80, 120])
def test_final_summary_shows_logfire_control_at_common_widths(
    monkeypatch, tmp_path, width
) -> None:
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(
            file=output,
            width=width,
            color_system=None,
            force_terminal=False,
        ),
    )
    summary = CrawlSummary(
        status=RunStatus.STOPPED_BUDGET,
        stop_reason="budget",
        videos_discovered=2,
        videos_evaluated=1,
        relevant_videos=0,
        transcripts_collected=0,
        channels_expanded=0,
        queries_executed=1,
        pending_videos=1,
        transcripts_unavailable=3,
    )
    budget = SearchApiCreditBudget(4, 1)

    cli._print_summary(
        summary,
        tmp_path / "summary-project",
        budget,
        controls={
            "max_depth": 1,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    text = output.getvalue()
    assert "Wanted transcripts not found" in text
    assert "3" in text
    assert "SearchAPI unspent across pools" in text
    assert "Discovery credits: capacity" in text
    assert "Discovery credits: spent" in text
    assert "Discovery credits: remaining" in text
    assert "Transcript credits: capacity" in text
    assert "Transcript credits: spent" in text
    assert "Transcript credits: reserved" in text
    assert "Transcript credits: remaining" in text
    assert "View logs in Logfire" in text
    assert "https://logfire.pydantic.dev/" in text


def test_completed_next_command_does_not_expand_beyond_planned_queries(
    tmp_path,
) -> None:
    project = tmp_path / "complete"
    state = _checkpoint(project, status="completed").model_copy(
        update={
            "planned_queries": [
                {"text": "heat pumps", "kind": "seed"},
                {"text": "heat pumps interview", "kind": "interview"},
            ]
        }
    )
    ProjectStateStore(project).save(state)
    summary = CrawlSummary(
        status=RunStatus.COMPLETED,
        stop_reason="frontier_exhausted",
        videos_discovered=2,
        videos_evaluated=2,
        relevant_videos=1,
        transcripts_collected=1,
        channels_expanded=0,
        queries_executed=2,
        pending_videos=0,
    )

    command = _suggested_next_command(
        summary,
        project,
        SearchApiCreditBudget(12, 4).snapshot(),
        controls={
            "max_depth": 1,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    assert "--add-credits 0" in command
    assert "--max-search-pages 2" in command
    assert "--max-queries" not in command


def test_completed_next_command_expands_an_eligible_planned_query(tmp_path) -> None:
    project = tmp_path / "complete-query"
    state = _checkpoint(project, status="completed").model_copy(
        update={
            "planned_queries": [
                {"text": "heat pumps", "kind": "seed"},
                {"text": "heat pumps interview", "kind": "interview"},
                {"text": "heat pumps costs", "kind": "expanded"},
            ]
        }
    )
    ProjectStateStore(project).save(state)
    summary = CrawlSummary(
        status=RunStatus.COMPLETED,
        stop_reason="frontier_exhausted",
        videos_discovered=2,
        videos_evaluated=2,
        relevant_videos=1,
        transcripts_collected=1,
        channels_expanded=0,
        queries_executed=1,
        pending_videos=0,
    )

    command = _suggested_next_command(
        summary,
        project,
        SearchApiCreditBudget(12, 4).snapshot(),
        controls={
            "max_depth": 1,
            "max_queries": 1,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    assert "--add-credits 0" in command
    assert "--max-queries 2" in command


def test_completed_next_command_adds_credits_only_after_aggregate_exhaustion(
    tmp_path,
) -> None:
    project = tmp_path / "complete-no-credits"
    ProjectStateStore(project).save(_checkpoint(project, status="completed"))
    summary = CrawlSummary(
        status=RunStatus.COMPLETED,
        stop_reason="frontier_exhausted",
        videos_discovered=2,
        videos_evaluated=2,
        relevant_videos=1,
        transcripts_collected=1,
        channels_expanded=0,
        queries_executed=2,
        pending_videos=0,
    )
    budget = SearchApiCreditBudget(12, 4)
    budget.spend_discovery(8)
    for video_id in ("video-1", "video-2", "video-3", "video-4"):
        budget.reconcile_transcript(budget.mark_relevant(video_id))

    command = _suggested_next_command(
        summary,
        project,
        budget.snapshot(),
        controls={
            "max_depth": 1,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
    )

    assert "--add-credits 6" in command
    assert "--max-queries 3" in command


def test_completed_next_command_returns_none_when_every_scope_is_at_its_limit(
    tmp_path,
) -> None:
    summary = CrawlSummary(
        status=RunStatus.COMPLETED,
        stop_reason="frontier_exhausted",
        videos_discovered=0,
        videos_evaluated=0,
        relevant_videos=0,
        transcripts_collected=0,
        channels_expanded=0,
        queries_executed=0,
        pending_videos=0,
    )

    command = _suggested_next_command(
        summary,
        tmp_path / "complete-limits",
        SearchApiCreditBudget(12, 4).snapshot(),
        controls={
            "max_depth": 5,
            "max_queries": 18,
            "max_search_pages": 10,
            "max_channel_pages": 10,
        },
    )

    assert command is None


def test_completed_project_rejects_credits_only_before_funding(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "complete"
    _checkpoint(project, status="completed")

    def fail(*_args, **_kwargs):
        raise AssertionError("funding and provider setup must not run")

    monkeypatch.setattr(cli, "_credentials", fail)
    monkeypatch.setattr(cli, "_preflight_funding", fail)
    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--add-credits", "8"],
    )

    assert result.exit_code == 2
    assert "Project complete under current controls" in result.stderr
    assert "credits alone do not widen crawl" in result.stderr
    assert ProjectStateStore(project).load().budget.max_credits == 4
    assert not (project / "run_config.jsonl").exists()


def test_completed_project_can_move_start_date_earlier_and_reopen_candidates(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "earlier-start-date"
    state = _checkpoint(project, status="completed").model_copy(
        update={
            "discovered_videos": {
                "newly-eligible": {},
                "still-too-old": {},
                "other-rejection": {},
            },
            "evaluated_video_ids": [
                "newly-eligible",
                "still-too-old",
                "other-rejection",
            ],
            "finalized_video_ids": [
                "newly-eligible",
                "still-too-old",
                "other-rejection",
            ],
            "dispositioned_video_ids": [
                "newly-eligible",
                "still-too-old",
                "other-rejection",
            ],
            "terminal_video_ids": [
                "newly-eligible",
                "still-too-old",
                "other-rejection",
            ],
        }
    )
    ProjectStateStore(project).save(state)
    candidates = [
        {"video_id": "newly-eligible", "published_at": "2025-06-01T00:00:00Z"},
        {"video_id": "still-too-old", "published_at": "2023-06-01T00:00:00Z"},
        {"video_id": "other-rejection", "published_at": "2025-06-01T00:00:00Z"},
        {"video_id": "newly-eligible", "published_at": None},
    ]
    decisions = [
        {
            "video_id": "newly-eligible",
            "reason": "published_before_start_date: '2025-06-01'",
        },
        {
            "video_id": "still-too-old",
            "reason": "published_before_start_date: '2023-06-01'",
        },
        {
            "video_id": "other-rejection",
            "reason": "published_before_start_date: '2025-06-01'",
        },
        {
            "video_id": "other-rejection",
            "reason": "requested_language_transcript_not_available",
        },
    ]
    (project / "video_candidate.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in candidates),
        encoding="utf-8",
    )
    (project / "relevance_decision.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in decisions),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))
    monkeypatch.setattr(cli, "_preflight_funding", lambda *_args: 100)
    monkeypatch.setattr(
        cli,
        "_openai_client",
        lambda _api_key: (_ for _ in ()).throw(RuntimeError("stop after commit")),
    )
    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--start-date", "2025-01-01"],
    )

    assert result.exit_code == 1
    committed = ProjectStateStore(project).load()
    assert committed.start_date == date(2025, 1, 1)
    assert "newly-eligible" not in committed.finalized_video_ids
    assert "newly-eligible" not in committed.dispositioned_video_ids
    assert "newly-eligible" not in committed.terminal_video_ids
    assert "newly-eligible" in committed.evaluated_video_ids
    assert set(committed.terminal_video_ids) == {"still-too-old", "other-rejection"}
    audit = (project / "run_config.jsonl").read_text(encoding="utf-8")
    assert '"start_date":"2025-01-01"' in audit


def test_stale_date_rejection_reopens_at_saved_boundary(tmp_path) -> None:
    project = tmp_path / "stale-date-rejection"
    state = _checkpoint(project, status="completed").model_copy(
        update={
            "discovered_videos": {"stale": {}},
            "finalized_video_ids": ["stale"],
            "dispositioned_video_ids": ["stale"],
            "terminal_video_ids": ["stale"],
        }
    )
    ProjectStateStore(project).save(state)
    (project / "video_candidate.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "video_id": "stale",
                        "published_at": "2026-02-01T00:00:00Z",
                    }
                ),
                json.dumps({"video_id": "stale", "published_at": None}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "relevance_decision.jsonl").write_text(
        json.dumps(
            {
                "video_id": "stale",
                "reason": "published_before_start_date: 'Feb 1, 2026'",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reopened = cli._start_date_reopened_video_ids(project, state, state.start_date)

    assert reopened == {"stale"}
    cli._require_completed_project_expansion(
        state,
        {
            "max_depth": state.max_depth,
            "max_queries": state.max_queries,
            "max_search_pages": state.max_search_pages,
            "max_channel_pages": state.max_channel_pages,
        },
        start_date=state.start_date,
        reopened_video_ids=reopened,
    )


def test_completed_project_rejects_max_queries_beyond_prepared_plan_before_funding(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "complete"
    state = _checkpoint(project, status="completed").model_copy(
        update={"max_queries": 3}
    )
    ProjectStateStore(project).save(state)

    def fail(*_args, **_kwargs):
        raise AssertionError("funding and provider setup must not run")

    monkeypatch.setattr(cli, "_credentials", fail)
    monkeypatch.setattr(cli, "_preflight_funding", fail)
    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--project",
            str(project),
            "--add-credits",
            "8",
            "--max-queries",
            "4",
        ],
    )

    assert result.exit_code == 2
    error = " ".join(result.stderr.split()).casefold()
    assert "prepared" in error
    assert "plan has only 3 variants" in error
    assert "increasing --max-queries" in error
    assert "beyond 3 does not" in error
    saved = ProjectStateStore(project).load()
    assert saved.max_queries == 3
    assert saved.budget.max_credits == 4
    assert not (project / "run_config.jsonl").exists()


def test_resume_funding_excludes_pending_reservations(monkeypatch, tmp_path) -> None:
    project = tmp_path / "pending-transcript"
    state = _checkpoint(project)
    state.budget.pending = [
        {"reservation_id": "reservation-1", "video_id": "video-1", "credits": 1}
    ]
    ProjectStateStore(project).save(state)
    allowances = []

    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))

    def capture_allowance(_api_key, allowance, *_runtime):
        allowances.append(allowance)
        raise cli.typer.BadParameter("stop after funding check")

    monkeypatch.setattr(cli, "_preflight_funding", capture_allowance)

    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--add-credits", "4"],
    )

    assert result.exit_code == 2
    assert allowances == [7]
    assert not (project / "crawler.log").exists()


@pytest.mark.parametrize(("credits_added", "expected_grant"), [(0, 4), (4, 8)])
def test_resume_records_a_fully_transferred_grant(
    monkeypatch, tmp_path, credits_added, expected_grant
) -> None:
    project = tmp_path / f"fully-transferred-{credits_added}"
    state = _checkpoint(project).model_copy(deep=True)
    state.budget = BudgetState(
        max_credits=4,
        transcript_capacity=4,
        discovery_spent=0,
        transcript_spent=4,
        completed_video_ids=["video-1", "video-2", "video-3", "video-4"],
    )
    ProjectStateStore(project).save(state)
    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))
    monkeypatch.setattr(cli, "_preflight_funding", lambda *_args: 100)

    def fail_after_record(_api_key):
        raise RuntimeError("stop after resume record")

    monkeypatch.setattr(cli, "_openai_client", fail_after_record)

    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--add-credits", str(credits_added)],
    )

    assert result.exit_code == 1
    audit = (project / "run_config.jsonl").read_text(encoding="utf-8")
    assert f'"max_searchapi_credits":{expected_grant}' in audit
    assert '"transcript_reserve_credits":4' in audit
    assert f'"prompt_version":"{CLASSIFIER_PROMPT_VERSION}:' in audit


def test_resume_commits_grant_and_controls_before_provider_setup(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "complete"
    _checkpoint(project, status="completed")

    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))
    funding_workers = []

    def funded(*args):
        funding_workers.append(args[-1])
        return 100

    monkeypatch.setattr(cli, "_preflight_funding", funded)
    setup_attempts = 0

    def fail_after_commit(_api_key):
        nonlocal setup_attempts
        setup_attempts += 1
        committed = ProjectStateStore(project).load()
        assert committed.budget.max_credits == 8
        assert committed.max_queries == 3
        assert committed.searchapi_retries == 1
        assert committed.searchapi_workers == DEFAULT_SEARCHAPI_WORKERS - 1
        assert committed.llm_workers == DEFAULT_LLM_WORKERS + 1
        assert committed.last_status == "prepared"
        audit = (project / "run_config.jsonl").read_text(encoding="utf-8")
        assert '"session_action":"resume"' in audit
        assert '"max_searchapi_credits":8' in audit
        assert '"max_queries":3' in audit
        assert '"searchapi_retries":1' in audit
        assert f'"searchapi_workers":{DEFAULT_SEARCHAPI_WORKERS - 1}' in audit
        assert f'"llm_workers":{DEFAULT_LLM_WORKERS + 1}' in audit
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(cli, "_openai_client", fail_after_commit)
    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--project",
            str(project),
            "--add-credits",
            "4",
            "--max-queries",
            "3",
            "--searchapi-retries",
            "1",
            "--searchapi-workers",
            str(DEFAULT_SEARCHAPI_WORKERS - 1),
            "--llm-workers",
            str(DEFAULT_LLM_WORKERS + 1),
        ],
    )

    assert result.exit_code == 1
    assert "RESUME EXISTING PROJECT" in result.stdout
    assert "View logs in Logfire" in result.stdout
    assert "https://logfire.pydantic.dev/" in result.stdout
    assert result.stdout.count("Run failed:") == 1
    assert funding_workers == [DEFAULT_SEARCHAPI_WORKERS - 1]
    error_audit = (project / "run_error.jsonl").read_text(encoding="utf-8")
    assert '"stage":"resume_provider_setup"' in error_audit
    assert "provider setup failed" in error_audit

    # The committed configuration is retryable without inventing another scope
    # increase or adding the same credits a second time.
    retry = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), "--add-credits", "0"],
    )
    assert retry.exit_code == 1
    assert setup_attempts == 2
    assert funding_workers == [
        DEFAULT_SEARCHAPI_WORKERS - 1,
        DEFAULT_SEARCHAPI_WORKERS - 1,
    ]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--searchapi-workers", "0"),
        ("--searchapi-workers", "33"),
        ("--llm-workers", "0"),
        ("--llm-workers", "33"),
    ],
)
def test_resume_parallelism_overrides_use_start_bounds(tmp_path, option, value) -> None:
    project = tmp_path / "invalid-parallelism"
    _checkpoint(project)

    result = CliRunner().invoke(
        app,
        ["resume", "--project", str(project), option, value],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.stderr
    assert not (project / "run_config.jsonl").exists()


def test_resume_dashboard_start_failure_is_durably_recorded(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "dashboard-start-failure"
    _checkpoint(project)
    stops = []

    class FailingDashboard:
        def __init__(self, **_kwargs):
            self.is_started = False

        def start(self):
            self.is_started = True
            raise RuntimeError("terminal display unavailable")

        def finish(self, *_args):
            pass

        def stop(self):
            stops.append("stopped")
            self.is_started = False

    monkeypatch.setattr(cli, "_credentials", lambda: ("search-key", "openai-key"))
    monkeypatch.setattr(cli, "_preflight_funding", lambda *_args: 100)
    monkeypatch.setattr(cli, "RunDashboard", FailingDashboard)

    result = CliRunner().invoke(app, ["resume", "--project", str(project)])

    assert result.exit_code == 1
    assert result.stdout.count("Run failed:") == 1
    assert "terminal display unavailable" in result.stdout
    assert "View logs in Logfire" in result.stdout
    assert "https://logfire.pydantic.dev/" in result.stdout
    assert stops == ["stopped"]
    assert '"stage":"resume_dashboard_start"' in (
        project / "run_error.jsonl"
    ).read_text(encoding="utf-8")
    assert '"status":"failed"' in (project / "run_status.jsonl").read_text(
        encoding="utf-8"
    )


def test_plan_summaries_are_compact_and_show_only_resume_changes(tmp_path) -> None:
    state = _checkpoint(tmp_path / "plan")
    controls = {
        "max_depth": 2,
        "max_queries": 3,
        "max_search_pages": 1,
        "max_channel_pages": 1,
    }

    assert cli._start_plan_summary(
        grant=12,
        max_depth=2,
        max_queries=8,
        max_search_pages=1,
        max_channel_pages=1,
    ) == (
        "Start plan · grant 12 · queries 8 · search pages 1 · channel pages 1 · depth 2"
    )
    assert (
        cli._resume_plan_summary(
            credits_added=30,
            new_grant=42,
            previous_state=state,
            controls=controls,
        )
        == "Resume plan · grant +30 → 42 · queries 2 → 3 · depth 0 → 2"
    )
    assert (
        cli._resume_plan_summary(
            credits_added=0,
            new_grant=4,
            previous_state=state,
            controls={
                "max_depth": 0,
                "max_queries": 2,
                "max_search_pages": 1,
                "max_channel_pages": 1,
            },
        )
        == "Resume plan · grant +0 → 4 · frontier unchanged"
    )
    assert cli._resume_plan_summary(
        credits_added=0,
        new_grant=4,
        previous_state=state,
        controls={
            "max_depth": 0,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
        searchapi_workers=DEFAULT_SEARCHAPI_WORKERS - 1,
        searchapi_retries=1,
        llm_workers=DEFAULT_LLM_WORKERS + 1,
    ) == (
        "Resume plan · grant +0 → 4 · frontier unchanged · "
        f"SearchAPI retries {DEFAULT_SEARCHAPI_RETRIES} → 1 · "
        f"SearchAPI workers {DEFAULT_SEARCHAPI_WORKERS} → "
        f"{DEFAULT_SEARCHAPI_WORKERS - 1} · OpenAI workers "
        f"{DEFAULT_LLM_WORKERS} → {DEFAULT_LLM_WORKERS + 1}"
    )
    assert cli._resume_plan_summary(
        credits_added=0,
        new_grant=4,
        previous_state=state,
        controls={
            "max_depth": 0,
            "max_queries": 2,
            "max_search_pages": 1,
            "max_channel_pages": 1,
        },
        start_date=date(2025, 1, 1),
    ) == ("Resume plan · grant +0 → 4 · start date 2026-01-01 → 2025-01-01")


def test_commit_resume_state_copies_instead_of_mutating_loaded_state(tmp_path) -> None:
    project = tmp_path / "copy"
    original = _checkpoint(project)
    budget = SearchApiCreditBudget.restore(original.budget.model_dump())
    budget.expand(8, 3)

    committed = _commit_resume_state(
        ProjectStateStore(project),
        original,
        budget,
        {
            "max_depth": 1,
            "max_queries": 3,
            "max_search_pages": 2,
            "max_channel_pages": 2,
        },
    )

    assert original.budget.max_credits == 4
    assert original.max_depth == 0
    assert committed.budget.max_credits == 8
    assert committed.last_status == "prepared"
    assert ProjectStateStore(project).load() == committed
