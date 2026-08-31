"""Typer CLI for interviewed, budget-safe YouTube research runs."""

from __future__ import annotations

import json
import math
import os
import shlex
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.classifier import RelevanceClassifier
from yt_searchapi.client import SearchApiClient
from yt_searchapi.crawler import CrawlConfig, CrawlSummary, ResearchCrawler
from yt_searchapi.interview import (
    INTERVIEW_QUESTIONS,
    InterviewPlanner,
    InterviewSuggestions,
)
from yt_searchapi.interview_ui import (
    INTERVIEW_CANCELLED_MESSAGE,
    ConfirmedInterview,
    InterviewCancelled,
    TerminalInterview,
)
from yt_searchapi.llm_runtime import AuditedOpenAIClient, StructuredOutputError
from yt_searchapi.observability import (
    RunSessionSpan,
    configure_observability,
    instrument_openai_client,
    print_logfire_link,
)
from yt_searchapi.prompts import (
    CLASSIFIER_PROMPT_VERSION,
    DEFAULT_LLM_MODEL,
    CompiledClassifierPrompt,
    TopicExpander,
    TopicExpansion,
    compile_classifier_prompt,
)
from yt_searchapi.records import (
    InterviewAnswerItem,
    InterviewAnswerRecord,
    InterviewExampleKind,
    RunConfigRecord,
    RunErrorRecord,
    RunStatus,
    RunStatusRecord,
)
from yt_searchapi.run_tui import RunDashboard
from yt_searchapi.settings import (
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
    Settings,
)
from yt_searchapi.start_settings_ui import (
    BASE_START_SETTING_QUESTIONS,
    collect_missing_start_settings,
    float_setting,
    integer_setting,
)
from yt_searchapi.state import (
    BudgetState,
    CheckpointPromptIntegrityError,
    CrawlProjectState,
    ProjectStateStore,
    checkpoint_recovery_message,
    validate_pending_transcript_decisions,
    validate_resumable_classifier_prompt,
)
from yt_searchapi.storage import JsonlRunWriter

app = typer.Typer(
    name="yt-crawl",
    no_args_is_help=True,
    help=(
        "Find topic-relevant YouTube videos with SearchAPI, then collect their "
        "target-language transcripts within a resumable SearchAPI credit grant."
    ),
)
console = Console()


@app.command()
def start(
    topic: Annotated[
        str | None,
        typer.Option("--topic", help="Research topic or question."),
    ] = None,
    max_credits: Annotated[
        int | None,
        typer.Option(
            "--max-credits",
            min=4,
            help="Initial hard SearchAPI credit grant for this project (minimum 4).",
        ),
    ] = None,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            help="New project directory. It must not already contain files.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            help="Video language code, for example en or de.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help="Earliest publication date (YYYY-MM-DD).",
        ),
    ] = None,
    max_depth: Annotated[
        int | None,
        typer.Option("--max-depth", min=0, max=5, help="Related/channel graph depth."),
    ] = None,
    max_queries: Annotated[
        int | None,
        typer.Option("--max-queries", min=2, max=18, help="Maximum query variants."),
    ] = None,
    max_search_pages: Annotated[
        int | None,
        typer.Option(
            "--max-search-pages",
            min=1,
            max=10,
            help="Maximum pages fetched for each planned search query.",
        ),
    ] = None,
    max_channel_pages: Annotated[
        int | None,
        typer.Option(
            "--max-channel-pages",
            min=1,
            max=10,
            help="Maximum pages fetched for each discovered channel.",
        ),
    ] = None,
    gl: Annotated[
        str | None, typer.Option("--country", help="SearchAPI YouTube gl code.")
    ] = None,
    hl: Annotated[
        str | None,
        typer.Option(
            "--interface-language",
            help="SearchAPI YouTube hl code; this is not the content-language gate.",
        ),
    ] = None,
    searchapi_timeout: Annotated[
        float | None,
        typer.Option(
            "--searchapi-timeout",
            min=0.1,
            help="Per-request SearchAPI timeout in seconds.",
        ),
    ] = None,
    searchapi_retries: Annotated[
        int | None,
        typer.Option(
            "--searchapi-retries",
            min=0,
            max=5,
            help="Additional budgeted attempts for transient SearchAPI failures.",
        ),
    ] = None,
    searchapi_workers: Annotated[
        int | None,
        typer.Option(
            "--searchapi-workers",
            min=1,
            max=32,
            help="Maximum concurrent SearchAPI requests.",
        ),
    ] = None,
    llm_workers: Annotated[
        int | None,
        typer.Option(
            "--llm-workers",
            min=1,
            max=32,
            help="Maximum concurrent OpenAI classification requests.",
        ),
    ] = None,
) -> None:
    """Create a new project, interview once, and begin its first crawl session."""

    runtime_questions = (
        float_setting(
            "searchapi_timeout",
            "How long may one SearchAPI request wait?",
            "Enter seconds before a stalled request is stopped.",
            default=90,
        ),
        integer_setting(
            "searchapi_retries",
            "How many times may a transient SearchAPI failure be retried?",
            "Choose between 0 and 5. Every dispatched retry uses another credit.",
            minimum=0,
            maximum=5,
            default=DEFAULT_SEARCHAPI_RETRIES,
        ),
        integer_setting(
            "searchapi_workers",
            "How many SearchAPI requests may run at once?",
            "Choose between 1 and 32.",
            minimum=1,
            maximum=32,
            default=DEFAULT_SEARCHAPI_WORKERS,
        ),
        integer_setting(
            "llm_workers",
            "How many video classifications may run at once?",
            "Choose between 1 and 32.",
            minimum=1,
            maximum=32,
            default=DEFAULT_LLM_WORKERS,
        ),
    )
    try:
        resolved = collect_missing_start_settings(
            {
                "topic": topic,
                "project": project,
                "max_credits": max_credits,
                "language": language,
                "start_date": start_date,
                "max_depth": max_depth,
                "max_queries": max_queries,
                "max_search_pages": max_search_pages,
                "max_channel_pages": max_channel_pages,
                "gl": gl,
                "hl": hl,
                "searchapi_timeout": searchapi_timeout,
                "searchapi_retries": searchapi_retries,
                "searchapi_workers": searchapi_workers,
                "llm_workers": llm_workers,
            },
            (*BASE_START_SETTING_QUESTIONS, *runtime_questions),
        )
    except KeyboardInterrupt:
        raise typer.Abort() from None

    topic = str(resolved["topic"]).strip()
    if not topic:
        raise typer.BadParameter("--topic must not be blank")
    language_value = str(resolved["language"]).strip()
    start_date_value = str(resolved["start_date"]).strip()
    try:
        date.fromisoformat(start_date_value)
    except ValueError as exc:
        raise typer.BadParameter("--start-date must use ISO format YYYY-MM-DD") from exc
    max_credits = int(resolved["max_credits"])
    max_depth = int(resolved["max_depth"])
    max_queries = int(resolved["max_queries"])
    max_search_pages = int(resolved["max_search_pages"])
    max_channel_pages = int(resolved["max_channel_pages"])
    gl = str(resolved["gl"]).strip().lower()
    hl = str(resolved["hl"]).strip()
    searchapi_timeout = float(resolved["searchapi_timeout"])
    searchapi_retries = int(resolved["searchapi_retries"])
    searchapi_workers = int(resolved["searchapi_workers"])
    llm_workers = int(resolved["llm_workers"])
    project = Path(resolved["project"]).expanduser().resolve()
    _require_new_project(project)
    configure_observability()
    searchapi_key, openai_key = _credentials()
    account_credits = _check_searchapi_funding(
        searchapi_key,
        max_credits,
        searchapi_timeout,
        searchapi_workers,
    )
    run_id = project.name
    writer = JsonlRunWriter(project.parent, run_id)
    transcript_reserve = _transcript_reserve(max_credits)
    search_budget = SearchApiCreditBudget(max_credits, transcript_reserve)
    writer.append(RunStatusRecord(run_id=run_id, status=RunStatus.STARTED))
    state_store = ProjectStateStore(project)
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=search_budget,
        state_store=state_store,
        plan_summary=_start_plan_summary(
            grant=max_credits,
            max_depth=max_depth,
            max_queries=max_queries,
            max_search_pages=max_search_pages,
            max_channel_pages=max_channel_pages,
        ),
        searchapi_concurrency=searchapi_workers,
        openai_concurrency=llm_workers,
        console=console,
    )
    console.print(
        Panel.fit(
            "[bold green]START NEW PROJECT[/bold green]\n"
            f"Project: [bold]{project}[/bold]\n"
            f"New SearchAPI grant: {max_credits} credits\n"
            f"Funding check: [bold green]passed[/bold green] "
            f"({account_credits} account credits available)\n"
            f"Transcript reserve: {transcript_reserve} credits\n"
            "Controls: "
            f"depth={max_depth}, queries={max_queries}, "
            f"search pages={max_search_pages}, channel pages={max_channel_pages}, "
            f"SearchAPI retries={searchapi_retries}",
            title="Interview",
        )
    )
    print_logfire_link(console)
    session_span = RunSessionSpan(
        run_id=run_id,
        project=project,
        action="start",
        planned_credits=max_credits,
        controls={
            "max_depth": max_depth,
            "max_queries": max_queries,
            "max_search_pages": max_search_pages,
            "max_channel_pages": max_channel_pages,
        },
    )
    preparation_started = False
    preparation_completed = False
    try:
        # Transient SDK retries stay inside one audited logical OpenAI call.
        openai_client = _openai_client(openai_key)
        audited_openai = AuditedOpenAIClient(
            openai_client,
            writer,
            on_event=dashboard.on_event,
            max_concurrency=llm_workers,
        )
        preparation_started = True
        expansion, classifier_prompt, confirmed = _prepare_research(
            writer=writer,
            client=audited_openai,
            topic=topic,
            language_hint=language_value,
            start_date_hint=start_date_value,
            max_credits=max_credits,
            transcript_reserve=transcript_reserve,
            account_remaining_credits=account_credits,
            max_depth=max_depth,
            max_queries=max_queries,
            max_search_pages=max_search_pages,
            max_channel_pages=max_channel_pages,
            gl=gl,
            hl=hl,
            searchapi_timeout_seconds=searchapi_timeout,
            searchapi_retries=searchapi_retries,
            searchapi_workers=searchapi_workers,
            llm_workers=llm_workers,
            dashboard=dashboard,
        )
        preparation_completed = True
        raw_dir = project / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        crawler_config = CrawlConfig(
            topic_query=topic,
            language=confirmed.language_code,
            start_date=confirmed.publication_start_date,
            max_depth=max_depth,
            max_queries=max_queries,
            max_search_pages=max_search_pages,
            max_channel_pages=max_channel_pages,
            gl=gl,
            hl=hl,
            searchapi_timeout_seconds=searchapi_timeout,
            searchapi_retries=searchapi_retries,
            searchapi_workers=searchapi_workers,
            llm_workers=llm_workers,
        )
        dashboard.update("crawler", "Initializing the saved frontier")
        with SearchApiClient(
            searchapi_key,
            cache_dir=project / ".cache" / "searchapi",
            cache_ttl=None,
            jsonl_prefix=raw_dir,
            max_retries=0,
            timeout=searchapi_timeout,
            max_workers=searchapi_workers,
        ) as searchapi:
            summary = ResearchCrawler(
                config=crawler_config,
                expansion=expansion,
                classifier_prompt=classifier_prompt,
                searchapi=searchapi,
                classifier=RelevanceClassifier(audited_openai),
                llm_client=audited_openai,
                search_budget=search_budget,
                writer=writer,
                on_progress=dashboard.update,
                on_api_event=dashboard.on_event,
                on_crawl_progress=dashboard.on_crawl_progress,
                state_store=state_store,
            ).run()
        dashboard.finish(summary.status.value, summary.stop_reason)
        session_span.set_outcome(
            summary.status.value,
            (
                "crawler_failed"
                if summary.status is RunStatus.FAILED
                else summary.stop_reason
            ),
        )
    except (typer.Abort, KeyboardInterrupt) as exc:
        session_span.set_outcome("cancelled", type(exc).__name__)
        raise
    except Exception as exc:
        session_span.set_outcome("failed", type(exc).__name__)
        logger.error(
            "Start session failed project={} error_type={}",
            project,
            type(exc).__name__,
        )
        if not preparation_started or preparation_completed:
            _record_cli_failure(writer, "start_session", exc)
        visible_error = _visible_start_error(
            exc,
            preparation_completed=preparation_completed,
        )
        if dashboard.is_started:
            dashboard.finish("failed", visible_error)
        console.print(f"[bold red]Run failed:[/bold red] {visible_error}")
        print_logfire_link(console)
        raise typer.Exit(code=1) from None
    finally:
        dashboard.stop()
        session_span.close()
    _print_summary(
        summary,
        writer.run_dir,
        search_budget,
        controls={
            "max_depth": max_depth,
            "max_queries": max_queries,
            "max_search_pages": max_search_pages,
            "max_channel_pages": max_channel_pages,
        },
    )
    if summary.status is RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command()
def resume(
    project: Annotated[
        Path,
        typer.Option(
            "--project",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Existing project directory containing crawl_state.json.",
        ),
    ],
    add_credits: Annotated[
        int,
        typer.Option(
            "--add-credits",
            min=0,
            help=(
                "Add N credits to the saved lifetime SearchAPI grant; 0 reuses "
                "the existing unspent allowance."
            ),
        ),
    ] = 0,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help=(
                "Move the earliest publication date earlier. A resumed project "
                "cannot move this date later. Use YYYY-MM-DD."
            ),
        ),
    ] = None,
    max_depth: Annotated[
        int | None,
        typer.Option("--max-depth", min=0, max=5, help="Increase graph depth."),
    ] = None,
    max_queries: Annotated[
        int | None,
        typer.Option("--max-queries", min=2, max=18, help="Increase query count."),
    ] = None,
    max_search_pages: Annotated[
        int | None,
        typer.Option(
            "--max-search-pages",
            min=1,
            max=10,
            help="Increase pages allowed for each search query.",
        ),
    ] = None,
    max_channel_pages: Annotated[
        int | None,
        typer.Option(
            "--max-channel-pages",
            min=1,
            max=10,
            help="Increase pages allowed for each discovered channel.",
        ),
    ] = None,
    searchapi_retries: Annotated[
        int | None,
        typer.Option(
            "--searchapi-retries",
            min=0,
            max=5,
            help="Set additional budgeted transient retries for this project.",
        ),
    ] = None,
    searchapi_workers: Annotated[
        int | None,
        typer.Option(
            "--searchapi-workers",
            min=1,
            max=32,
            help="Set the maximum concurrent SearchAPI requests for this project.",
        ),
    ] = None,
    llm_workers: Annotated[
        int | None,
        typer.Option(
            "--llm-workers",
            min=1,
            max=32,
            help=(
                "Set the maximum concurrent OpenAI classification requests "
                "for this project."
            ),
        ),
    ] = None,
) -> None:
    """Continue an existing project without repeating preparation or finished work."""

    project = project.expanduser().resolve()
    state_store = ProjectStateStore(project)
    try:
        state = state_store.load()
    except CheckpointPromptIntegrityError as exc:
        raise typer.BadParameter(str(exc), param_hint="--project") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(
            checkpoint_recovery_message(project), param_hint="--project"
        ) from exc
    try:
        validate_resumable_classifier_prompt(state)
    except CheckpointPromptIntegrityError as exc:
        raise typer.BadParameter(str(exc), param_hint="--project") from exc
    try:
        validate_pending_transcript_decisions(state)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--project") from exc
    try:
        effective_start_date = (
            state.start_date if start_date is None else date.fromisoformat(start_date)
        )
        if start_date is not None and effective_start_date.isoformat() != start_date:
            raise ValueError("start date is not canonical ISO format")
    except ValueError as exc:
        raise typer.BadParameter(
            "--start-date must use ISO format YYYY-MM-DD",
            param_hint="--start-date",
        ) from exc
    if effective_start_date > state.start_date:
        raise typer.BadParameter(
            f"--start-date cannot move later from {state.start_date.isoformat()} "
            f"to {effective_start_date.isoformat()}; it may only move earlier.",
            param_hint="--start-date",
        )
    reopened_video_ids = _start_date_reopened_video_ids(
        project,
        state,
        effective_start_date,
    )
    controls = _expanded_controls(
        state,
        max_depth=max_depth,
        max_queries=max_queries,
        max_search_pages=max_search_pages,
        max_channel_pages=max_channel_pages,
    )
    _require_completed_project_expansion(
        state,
        controls,
        start_date=effective_start_date,
        reopened_video_ids=reopened_video_ids,
    )
    effective_searchapi_workers = (
        state.searchapi_workers if searchapi_workers is None else searchapi_workers
    )
    effective_searchapi_retries = (
        state.searchapi_retries if searchapi_retries is None else searchapi_retries
    )
    effective_llm_workers = state.llm_workers if llm_workers is None else llm_workers
    if state.run_id != project.name:
        raise typer.BadParameter(
            "Project path does not match the run ID in crawl_state.json",
            param_hint="--project",
        )
    configure_observability()
    searchapi_key, openai_key = _credentials()
    budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    previous_budget = budget.snapshot()
    new_grant = previous_budget.max_credits + add_credits
    spent = previous_budget.discovery_spent + previous_budget.transcript_spent
    unused_before = previous_budget.total_remaining
    invocation_allowance = new_grant - previous_budget.total_committed
    account_credits = _check_searchapi_funding(
        searchapi_key,
        invocation_allowance,
        state.searchapi_timeout_seconds,
        effective_searchapi_workers,
    )
    if add_credits:
        budget.expand(
            new_grant,
            _expanded_transcript_reserve(budget, new_grant),
        )

    writer = JsonlRunWriter(project.parent, state.run_id)
    resume_record = RunConfigRecord(
        run_id=state.run_id,
        topic_query=state.topic_query,
        expanded_queries=tuple(item["text"] for item in state.planned_queries),
        language=state.language,
        start_date=effective_start_date,
        max_searchapi_credits=new_grant,
        transcript_reserve_credits=budget.snapshot().transcript_capacity,
        session_action="resume",
        credits_added=add_credits,
        account_remaining_credits=account_credits,
        **controls,
        gl=state.gl,
        hl=state.hl,
        searchapi_timeout_seconds=state.searchapi_timeout_seconds,
        searchapi_retries=effective_searchapi_retries,
        searchapi_workers=effective_searchapi_workers,
        llm_workers=effective_llm_workers,
        model=DEFAULT_LLM_MODEL,
        prompt_version=(f"{state.classifier_prompt_version}:{state.prompt_sha256}"),
        topic_expansion=state.expansion,
        classifier_system_prompt=state.classifier_system_prompt,
        prompt_sha256=state.prompt_sha256,
    )
    changed_controls = [
        f"{name.removeprefix('max_')} {getattr(state, name)} → {value}"
        for name, value in controls.items()
        if value != getattr(state, name)
    ]
    if effective_start_date != state.start_date:
        changed_controls.append(
            f"start_date {state.start_date.isoformat()} → "
            f"{effective_start_date.isoformat()}"
        )
    if effective_searchapi_retries != state.searchapi_retries:
        changed_controls.append(
            f"searchapi_retries {state.searchapi_retries} → "
            f"{effective_searchapi_retries}"
        )
    plan_summary = _resume_plan_summary(
        credits_added=add_credits,
        new_grant=new_grant,
        previous_state=state,
        controls=controls,
        start_date=effective_start_date,
        searchapi_retries=effective_searchapi_retries,
        searchapi_workers=effective_searchapi_workers,
        llm_workers=effective_llm_workers,
    )
    session_span = RunSessionSpan(
        run_id=state.run_id,
        project=project,
        action="resume",
        planned_credits=invocation_allowance,
        credits_added=add_credits,
        controls=controls,
    )
    dashboard: RunDashboard | None = None
    failure_stage = "resume_checkpoint_commit"
    try:
        # The mutable checkpoint is authoritative for dispatch. Commit the funded
        # grant, monotonic scope, and worker overrides before the audit row or any
        # provider setup, so a setup failure can be retried without losing the
        # approved configuration.
        state = _commit_resume_state(
            state_store,
            state,
            budget,
            controls,
            start_date=effective_start_date,
            reopened_video_ids=reopened_video_ids,
            searchapi_retries=effective_searchapi_retries,
            searchapi_workers=effective_searchapi_workers,
            llm_workers=effective_llm_workers,
        )
        writer.append(resume_record)
        logger.info(
            "Resume funded project={} previous_grant={} added={} new_grant={} "
            "already_spent={} unused_before={} account_credits={} controls={} "
            "searchapi_retries={} searchapi_workers={} llm_workers={} "
            "reopened_videos={} changed={}",
            project,
            previous_budget.max_credits,
            add_credits,
            new_grant,
            spent,
            unused_before,
            account_credits,
            _format_controls(controls),
            state.searchapi_retries,
            state.searchapi_workers,
            state.llm_workers,
            len(reopened_video_ids),
            changed_controls or ["none"],
        )
        failure_stage = "resume_dashboard_start"
        dashboard = RunDashboard(
            mode="RESUME EXISTING PROJECT",
            project_dir=project,
            budget=budget,
            state_store=state_store,
            plan_summary=plan_summary,
            searchapi_concurrency=state.searchapi_workers,
            openai_concurrency=state.llm_workers,
            console=console,
        )
        dashboard.start()
        failure_stage = "resume_dashboard_restore"
        dashboard.update(
            "preparation", "Funding check passed; restoring the saved frontier"
        )
        failure_stage = "resume_provider_setup"
        openai_client = _openai_client(openai_key)
        audited_openai = AuditedOpenAIClient(
            openai_client,
            writer,
            on_event=dashboard.on_event,
            max_concurrency=state.llm_workers,
        )
        config = CrawlConfig(
            topic_query=state.topic_query,
            language=state.language,
            start_date=state.start_date,
            transcript_excerpt_chars=state.transcript_excerpt_chars,
            gl=state.gl,
            hl=state.hl,
            searchapi_timeout_seconds=state.searchapi_timeout_seconds,
            searchapi_retries=state.searchapi_retries,
            searchapi_workers=state.searchapi_workers,
            llm_workers=state.llm_workers,
            **controls,
        )
        expansion = TopicExpansion.model_validate(state.expansion)
        classifier_prompt = CompiledClassifierPrompt(
            system_prompt=state.classifier_system_prompt,
            prompt_sha256=state.prompt_sha256,
        )
        failure_stage = "resume_crawler_setup"
        with SearchApiClient(
            searchapi_key,
            cache_dir=project / ".cache" / "searchapi",
            cache_ttl=None,
            jsonl_prefix=project / "raw",
            max_retries=0,
            timeout=state.searchapi_timeout_seconds,
            max_workers=state.searchapi_workers,
        ) as searchapi:
            crawler = ResearchCrawler(
                config=config,
                expansion=expansion,
                classifier_prompt=classifier_prompt,
                searchapi=searchapi,
                classifier=RelevanceClassifier(audited_openai),
                llm_client=audited_openai,
                search_budget=budget,
                writer=writer,
                on_progress=dashboard.update,
                on_api_event=dashboard.on_event,
                on_crawl_progress=dashboard.on_crawl_progress,
                state_store=state_store,
                resume_state=state,
            )
            failure_stage = "resume_crawl"
            summary = crawler.run()
        failure_stage = "resume_dashboard_finish"
        dashboard.finish(summary.status.value, summary.stop_reason)
        session_span.set_outcome(
            summary.status.value,
            (
                "crawler_failed"
                if summary.status is RunStatus.FAILED
                else summary.stop_reason
            ),
        )
    except (typer.Abort, KeyboardInterrupt) as exc:
        session_span.set_outcome("cancelled", type(exc).__name__)
        raise
    except Exception as exc:
        session_span.set_outcome("failed", type(exc).__name__)
        logger.error(
            "Resume session failed project={} error_type={}",
            project,
            type(exc).__name__,
        )
        _record_cli_failure(writer, failure_stage, exc)
        if dashboard is not None and dashboard.is_started:
            try:
                dashboard.finish("failed", str(exc).strip() or type(exc).__name__)
            except Exception as render_exc:
                logger.error(
                    "Could not render resume failure state error_type={}",
                    type(render_exc).__name__,
                )
        console.print(f"[bold red]Run failed:[/bold red] {exc}")
        print_logfire_link(console)
        raise typer.Exit(code=1) from None
    finally:
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception as stop_exc:
                logger.error(
                    "Could not stop resume dashboard error_type={}",
                    type(stop_exc).__name__,
                )
        session_span.close()
    _print_summary(summary, project, budget, controls=controls)
    if summary.status is RunStatus.FAILED:
        raise typer.Exit(code=1)


def _prepare_research(
    *,
    writer: JsonlRunWriter,
    client: AuditedOpenAIClient,
    topic: str,
    language_hint: str,
    start_date_hint: str,
    max_credits: int,
    transcript_reserve: int,
    account_remaining_credits: int,
    max_depth: int,
    max_queries: int,
    max_search_pages: int,
    max_channel_pages: int,
    gl: str = "us",
    hl: str | None = None,
    searchapi_timeout_seconds: float = 90.0,
    searchapi_retries: int = DEFAULT_SEARCHAPI_RETRIES,
    searchapi_workers: int = DEFAULT_SEARCHAPI_WORKERS,
    llm_workers: int = DEFAULT_LLM_WORKERS,
    dashboard: RunDashboard | None = None,
):
    """Run the paid preparation stage with an append-only local audit trail."""

    try:
        planner = InterviewPlanner(client)

        def load_suggestions(selected_language: str) -> InterviewSuggestions:
            with console.status(
                "[bold blue]Preparing example suggestions…[/bold blue]"
            ):
                with client.call_context(
                    "discovery",
                    "generate_interview_examples",
                ):
                    return planner.suggest_examples(topic, selected_language)

        interview = TerminalInterview(
            suggestion_loader=load_suggestions,
            language_prefill=language_hint,
            start_date_prefill=start_date_hint,
            ask_language=False,
            ask_start_date=False,
            console=console,
        )
        confirmed = interview.run()
        suggestions = interview.suggestions
        assert suggestions is not None
        brief = confirmed.to_topic_brief(topic_query=topic)
        _record_confirmed_interview(writer, confirmed, suggestions)
        if dashboard is not None:
            dashboard.start()
            dashboard.update(
                "preparation", "Compiling the research scope and query plan"
            )
            with client.call_context("discovery", "expand_topic_queries"):
                expansion = TopicExpander(client).expand(brief)
        else:
            with console.status(
                "[bold blue]Compiling the research scope and query plan…[/bold blue]"
            ):
                with client.call_context("discovery", "expand_topic_queries"):
                    expansion = TopicExpander(client).expand(brief)
        classifier_prompt = compile_classifier_prompt(brief, expansion)
        writer.append(
            RunConfigRecord(
                run_id=writer.run_id,
                topic_query=topic,
                expanded_queries=_planned_query_strings(expansion),
                language=confirmed.language_code,
                start_date=confirmed.publication_start_date,
                max_searchapi_credits=max_credits,
                transcript_reserve_credits=transcript_reserve,
                account_remaining_credits=account_remaining_credits,
                max_depth=max_depth,
                max_queries=max_queries,
                max_search_pages=max_search_pages,
                max_channel_pages=max_channel_pages,
                gl=gl,
                hl=hl or confirmed.language_code,
                searchapi_timeout_seconds=searchapi_timeout_seconds,
                searchapi_retries=searchapi_retries,
                searchapi_workers=searchapi_workers,
                llm_workers=llm_workers,
                model=DEFAULT_LLM_MODEL,
                prompt_version=(
                    f"{CLASSIFIER_PROMPT_VERSION}:{classifier_prompt.prompt_sha256}"
                ),
                topic_expansion=expansion.model_dump(mode="json"),
                classifier_system_prompt=classifier_prompt.system_prompt,
                prompt_sha256=classifier_prompt.prompt_sha256,
            )
        )
        return expansion, classifier_prompt, confirmed
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, (InterviewCancelled, KeyboardInterrupt)):
            message = INTERVIEW_CANCELLED_MESSAGE
        else:
            message = str(exc).strip() or "research preparation interrupted"
        writer.append(
            RunErrorRecord(
                run_id=writer.run_id,
                stage="research_preparation",
                message=message,
                exception_type=type(exc).__name__,
            )
        )
        writer.append(
            RunStatusRecord(
                run_id=writer.run_id,
                status=RunStatus.FAILED,
                reason=message,
            )
        )
        if isinstance(exc, (InterviewCancelled, KeyboardInterrupt)):
            console.print(f"[yellow]{message}[/yellow]")
            raise typer.Abort() from None
        raise


def _record_confirmed_interview(
    writer: JsonlRunWriter,
    confirmed: ConfirmedInterview,
    suggestions: InterviewSuggestions,
) -> None:
    """Append only the reviewed values, with structured provenance."""

    for question in INTERVIEW_QUESTIONS:
        if question.question_id == "positive_examples":
            kind = InterviewExampleKind.POSITIVE
        elif question.question_id == "negative_examples":
            kind = InterviewExampleKind.NEGATIVE
        else:
            kind = InterviewExampleKind.CONTEXT
        generated = tuple(
            example.model_dump(mode="json")
            for example in suggestions.examples
            if question.shows_suggestions == example.label
        )
        items = confirmed.items_for(question.question_id)
        writer.append(
            InterviewAnswerRecord(
                run_id=writer.run_id,
                question_id=question.question_id,
                question_text=question.prompt,
                answer=json.dumps(
                    tuple(item.text for item in items), ensure_ascii=False
                ),
                answer_items=tuple(
                    InterviewAnswerItem.model_validate(item.model_dump(mode="json"))
                    for item in items
                ),
                example_kind=kind,
                generated_examples=generated,
                generated_example=(
                    json.dumps(generated, ensure_ascii=False) if generated else None
                ),
            )
        )


def _record_cli_failure(
    writer: JsonlRunWriter, stage: str, exception: Exception
) -> None:
    message = str(exception).strip() or type(exception).__name__
    writer.append(
        RunErrorRecord(
            run_id=writer.run_id,
            stage=stage,
            message=message,
            exception_type=type(exception).__name__,
        )
    )
    writer.append(
        RunStatusRecord(
            run_id=writer.run_id,
            status=RunStatus.FAILED,
            reason=message,
        )
    )


def _visible_start_error(
    exception: Exception,
    *,
    preparation_completed: bool,
) -> str:
    message = str(exception).strip() or type(exception).__name__
    if not preparation_completed and isinstance(exception, StructuredOutputError):
        return (
            f"{message}\nThe crawl did not start; no SearchAPI crawl credits were used."
        )
    return message


def _transcript_reserve(max_credits: int) -> int:
    # Keep two seed searches plus at least one video-detail call possible.
    return min(max(1, math.ceil(max_credits / 3)), max_credits - 3)


def _expanded_transcript_reserve(budget: SearchApiCreditBudget, new_grant: int) -> int:
    snapshot = budget.snapshot()
    # Capacity transferred from discovery is one-way.  A later grant may add
    # fresh discovery headroom, but must never reopen capacity already made
    # available for transcripts.
    minimum = max(
        snapshot.transcript_capacity,
        snapshot.transcript_spent + snapshot.transcript_reserved,
    )
    added = new_grant - snapshot.max_credits
    discovery_headroom = 1 if added > 0 else 0
    maximum = min(
        new_grant - snapshot.discovery_spent - discovery_headroom,
        new_grant - 1,
    )
    return min(max(_transcript_reserve(new_grant), minimum), maximum)


def _credentials() -> tuple[str, str]:
    settings = Settings()
    searchapi_key = settings.searchapi_api_key or os.getenv("SEARCHAPI_API_KEY")
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not searchapi_key or not openai_key:
        missing = [
            name
            for name, value in (
                ("SEARCHAPI_API_KEY", searchapi_key),
                ("OPENAI_API_KEY", openai_key),
            )
            if not value
        ]
        raise typer.BadParameter(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env or export them before running."
        )
    return searchapi_key, openai_key


def _preflight_funding(
    api_key: str,
    planned_allowance: int,
    searchapi_timeout_seconds: float = 90.0,
    searchapi_workers: int = 1,
) -> int:
    """Require real-time account credits to cover every locally allowed dispatch."""

    with SearchApiClient(
        api_key,
        timeout=searchapi_timeout_seconds,
        max_retries=0,
        max_workers=searchapi_workers,
    ) as client:
        remaining = client.me().account.remaining_credits
    if remaining < planned_allowance:
        raise typer.BadParameter(
            "SearchAPI account is underfunded for this session: "
            f"{remaining} credits available, {planned_allowance} required. "
            "No crawl request was sent and no added grant was committed."
        )
    return remaining


def _check_searchapi_funding(
    api_key: str,
    planned_allowance: int,
    searchapi_timeout_seconds: float = 90.0,
    searchapi_workers: int = 1,
) -> int:
    """Keep the live account check visible before any run dashboard exists."""

    with console.status("[bold cyan]Checking SearchAPI funding…[/bold cyan]"):
        return _preflight_funding(
            api_key,
            planned_allowance,
            searchapi_timeout_seconds,
            searchapi_workers,
        )


def _start_plan_summary(
    *,
    grant: int,
    max_depth: int,
    max_queries: int,
    max_search_pages: int,
    max_channel_pages: int,
) -> str:
    return (
        f"Start plan · grant {grant} · queries {max_queries} · "
        f"search pages {max_search_pages} · channel pages {max_channel_pages} · "
        f"depth {max_depth}"
    )


def _resume_plan_summary(
    *,
    credits_added: int,
    new_grant: int,
    previous_state: CrawlProjectState,
    controls: dict[str, int],
    start_date: date | None = None,
    searchapi_retries: int | None = None,
    searchapi_workers: int | None = None,
    llm_workers: int | None = None,
) -> str:
    labels = {
        "max_queries": "queries",
        "max_search_pages": "search pages",
        "max_channel_pages": "channel pages",
        "max_depth": "depth",
    }
    changes = [
        f"{labels[name]} {getattr(previous_state, name)} → {controls[name]}"
        for name in labels
        if controls[name] != getattr(previous_state, name)
    ]
    effective_start_date = (
        previous_state.start_date if start_date is None else start_date
    )
    if effective_start_date != previous_state.start_date:
        changes.append(
            "start date "
            f"{previous_state.start_date.isoformat()} → "
            f"{effective_start_date.isoformat()}"
        )
    frontier = " · ".join(changes) if changes else "frontier unchanged"
    effective_searchapi_workers = (
        previous_state.searchapi_workers
        if searchapi_workers is None
        else searchapi_workers
    )
    effective_searchapi_retries = (
        previous_state.searchapi_retries
        if searchapi_retries is None
        else searchapi_retries
    )
    effective_llm_workers = (
        previous_state.llm_workers if llm_workers is None else llm_workers
    )
    parallelism: list[str] = []
    if effective_searchapi_retries != previous_state.searchapi_retries:
        parallelism.append(
            "SearchAPI retries "
            f"{previous_state.searchapi_retries} → {effective_searchapi_retries}"
        )
    if effective_searchapi_workers != previous_state.searchapi_workers:
        parallelism.append(
            "SearchAPI workers "
            f"{previous_state.searchapi_workers} → {effective_searchapi_workers}"
        )
    if effective_llm_workers != previous_state.llm_workers:
        parallelism.append(
            f"OpenAI workers {previous_state.llm_workers} → {effective_llm_workers}"
        )
    detail = " · ".join((frontier, *parallelism))
    return f"Resume plan · grant +{credits_added} → {new_grant} · {detail}"


def _require_new_project(project: Path) -> None:
    if project.exists() and (not project.is_dir() or any(project.iterdir())):
        if project.is_dir():
            try:
                ProjectStateStore(project).load()
            except (FileNotFoundError, ValueError):
                raise typer.BadParameter(
                    checkpoint_recovery_message(project),
                    param_hint="--project",
                ) from None
        raise typer.BadParameter(
            f"--project must be a new or empty directory; {project} is not empty. "
            "Use `yt-crawl resume --project ...` for an existing project.",
            param_hint="--project",
        )


def _require_completed_project_expansion(
    state: CrawlProjectState,
    controls: dict[str, int],
    *,
    start_date: date | None = None,
    reopened_video_ids: set[str] | None = None,
) -> None:
    """Prevent a no-op credit grant after all current scope has completed."""

    plan_length = len(state.planned_queries)
    query_scope_increased = min(controls["max_queries"], plan_length) > min(
        int(state.max_queries), plan_length
    )
    controls_increased = query_scope_increased or any(
        value > int(getattr(state, name))
        for name, value in controls.items()
        if name != "max_queries"
    )
    date_scope_increased = start_date is not None and start_date < state.start_date
    date_rejections_reopened = bool(reopened_video_ids)
    if (
        state.last_status == RunStatus.COMPLETED.value
        and not controls_increased
        and not date_scope_increased
        and not date_rejections_reopened
    ):
        query_limit_note = ""
        if controls["max_queries"] > int(state.max_queries):
            query_limit_note = (
                f" The prepared plan has only {plan_length} variants, so increasing "
                f"--max-queries beyond {plan_length} does not widen the crawl."
            )
        raise typer.BadParameter(
            "Project complete under current controls; increase at least one scope "
            "control because credits alone do not widen crawl. Alternatively, "
            "move --start-date earlier."
            f"{query_limit_note}"
        )


def _commit_resume_state(
    state_store: ProjectStateStore,
    state: CrawlProjectState,
    budget: SearchApiCreditBudget,
    controls: dict[str, int],
    *,
    start_date: date | None = None,
    reopened_video_ids: set[str] | None = None,
    searchapi_retries: int | None = None,
    searchapi_workers: int | None = None,
    llm_workers: int | None = None,
) -> CrawlProjectState:
    """Atomically commit a funded resume grant before work can be dispatched."""

    reopened = set() if reopened_video_ids is None else reopened_video_ids
    updated = state.model_copy(
        deep=True,
        update={
            **controls,
            "start_date": state.start_date if start_date is None else start_date,
            "finalized_video_ids": sorted(set(state.finalized_video_ids) - reopened),
            "dispositioned_video_ids": sorted(
                set(state.dispositioned_video_ids) - reopened
            ),
            "terminal_video_ids": sorted(set(state.terminal_video_ids) - reopened),
            "queued_video_ids": sorted(set(state.queued_video_ids) - reopened),
            "searchapi_retries": (
                state.searchapi_retries
                if searchapi_retries is None
                else searchapi_retries
            ),
            "searchapi_workers": (
                state.searchapi_workers
                if searchapi_workers is None
                else searchapi_workers
            ),
            "llm_workers": state.llm_workers if llm_workers is None else llm_workers,
            "budget": BudgetState.model_validate(budget.export_state()),
            "last_status": "prepared",
        },
    )
    state_store.save(updated)
    return updated


def _start_date_reopened_video_ids(
    project: Path,
    state: CrawlProjectState,
    start_date: date,
) -> set[str]:
    """Return stale date rejections admitted by the effective boundary."""

    candidate_path = project / "video_candidate.jsonl"
    decision_path = project / "relevance_decision.jsonl"
    if not candidate_path.exists() or not decision_path.exists():
        return set()

    try:
        latest_candidates = _latest_dated_candidate_records(candidate_path)
        latest_decisions = _latest_jsonl_records(decision_path, "video_id")
        reopened: set[str] = set()
        for video_id, decision in latest_decisions.items():
            if video_id not in state.discovered_videos:
                continue
            reason = str(decision.get("reason", ""))
            if not reason.startswith("published_before_start_date"):
                continue
            published_at = latest_candidates.get(video_id, {}).get("published_at")
            if not isinstance(published_at, str):
                continue
            published_date = date.fromisoformat(published_at[:10])
            if published_date >= start_date:
                reopened.add(video_id)
        return reopened
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(
            "Cannot safely change --start-date because the existing candidate "
            "or decision audit stream is invalid.",
            param_hint="--start-date",
        ) from exc


def _latest_jsonl_records(path: Path, key: str) -> dict[str, dict[str, object]]:
    """Read the last JSON object for each key from an append-only audit stream."""

    latest: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get(key), str):
                raise ValueError(f"invalid {path.name} record")
            latest[str(payload[key])] = payload
    return latest


def _latest_dated_candidate_records(path: Path) -> dict[str, dict[str, object]]:
    """Read each video's latest candidate record containing an exact date."""

    latest: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            video_id = payload.get("video_id") if isinstance(payload, dict) else None
            published_at = (
                payload.get("published_at") if isinstance(payload, dict) else None
            )
            if not isinstance(video_id, str):
                raise ValueError(f"invalid {path.name} record")
            if isinstance(published_at, str):
                latest[video_id] = payload
    return latest


def _expanded_controls(
    state: CrawlProjectState,
    *,
    max_depth: int | None,
    max_queries: int | None,
    max_search_pages: int | None,
    max_channel_pages: int | None,
) -> dict[str, int]:
    requested = {
        "max_depth": max_depth,
        "max_queries": max_queries,
        "max_search_pages": max_search_pages,
        "max_channel_pages": max_channel_pages,
    }
    controls: dict[str, int] = {}
    for name, value in requested.items():
        previous = int(getattr(state, name))
        chosen = previous if value is None else value
        if chosen < previous:
            option = name.replace("_", "-")
            raise typer.BadParameter(
                f"--{option} cannot decrease from {previous} to {chosen}; "
                "resume controls are monotonic."
            )
        controls[name] = chosen
    return controls


def _format_controls(controls: dict[str, int]) -> str:
    return ", ".join(
        f"{name.removeprefix('max_')}={value}" for name, value in controls.items()
    )


def _openai_client(api_key: str) -> OpenAI:
    """Create a bounded OpenAI client with three transient retries."""

    client = OpenAI(api_key=api_key, max_retries=3, timeout=90.0)
    instrument_openai_client(client)
    return client


def _planned_query_strings(expansion) -> tuple[str, ...]:
    values = list(expansion.search_queries)
    values.extend(expansion.channel_discovery_queries)
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _print_summary(
    summary: CrawlSummary,
    run_dir: Path,
    search_budget: SearchApiCreditBudget,
    *,
    controls: dict[str, int],
) -> None:
    search = search_budget.snapshot()
    table = Table(title=f"Project session {summary.status.value}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, value in (
        ("Relevant videos", summary.relevant_videos),
        ("Transcripts", summary.transcripts_collected),
        ("Wanted transcripts not found", summary.transcripts_unavailable),
        ("Videos evaluated", summary.videos_evaluated),
        ("Videos discovered", summary.videos_discovered),
        ("SearchAPI lifetime grant", search.max_credits),
        ("SearchAPI credits spent", search.discovery_spent + search.transcript_spent),
        ("SearchAPI unspent across pools", search.total_remaining),
        ("Discovery credits: capacity", search.discovery_capacity),
        ("Discovery credits: spent", search.discovery_spent),
        ("Discovery credits: remaining", search.discovery_remaining),
        ("Transcript credits: capacity", search.transcript_capacity),
        ("Transcript credits: spent", search.transcript_spent),
        ("Transcript credits: reserved", search.transcript_reserved),
        ("Transcript credits: remaining", search.transcript_remaining),
    ):
        table.add_row(label, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)
    console.print(f"[bold]Project data:[/bold] {run_dir.resolve()}")
    console.print(
        f"[bold]Raw SearchAPI responses:[/bold] {(run_dir / 'raw').resolve()}"
    )
    print_logfire_link(console)
    console.print(
        f"[bold]Stop reason:[/bold] {summary.stop_reason}\n"
        f"[bold]Project data:[/bold] {shlex.quote(str(run_dir.resolve()))}"
    )
    next_command = _suggested_next_command(summary, run_dir, search, controls=controls)
    if next_command:
        if summary.status is RunStatus.COMPLETED:
            label = (
                "Expand scope with unspent SearchAPI credits"
                if search.total_remaining
                else "Expand scope and add SearchAPI credits"
            )
        else:
            label = (
                "Resume the current scope with unspent SearchAPI credits"
                if search.total_remaining
                else "Resume the current scope and add SearchAPI credits"
            )
        console.print(f"[bold cyan]{label}:[/bold cyan] {next_command}")


def _suggested_next_command(
    summary: CrawlSummary,
    run_dir: Path,
    budget,
    *,
    controls: dict[str, int],
) -> str | None:
    parts = [
        "uv run yt-crawl resume",
        "--project",
        shlex.quote(str(run_dir.resolve())),
    ]
    if summary.status is RunStatus.COMPLETED:
        add_credits = (
            max(4, min(25, max(1, budget.max_credits // 2)))
            if budget.total_remaining == 0
            else 0
        )
        planned_query_limit = 18
        try:
            state = ProjectStateStore(run_dir).load()
            planned_query_limit = min(18, len(state.planned_queries))
        except (FileNotFoundError, ValueError):
            pass
        for name, limit in (
            ("max_queries", planned_query_limit),
            ("max_search_pages", 10),
            ("max_channel_pages", 10),
            ("max_depth", 5),
        ):
            current = controls[name]
            if current < limit:
                parts.extend(("--add-credits", str(add_credits)))
                parts.extend((f"--{name.replace('_', '-')}", str(current + 1)))
                return " ".join(parts)
        return None
    add_credits = 0
    if budget.total_remaining == 0:
        add_credits = max(4, min(25, max(1, budget.max_credits // 2)))
    parts.extend(("--add-credits", str(add_credits)))
    return " ".join(parts)


if __name__ == "__main__":
    app()
