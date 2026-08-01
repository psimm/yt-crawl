"""Resume-aware Rich dashboard for an active crawler session."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Literal

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.llm_runtime import (
    GPT56_LUNA_MODEL,
    LlmTokenMetrics,
    estimate_gpt56_luna_standard_cost,
)
from yt_searchapi.observability import logfire_link
from yt_searchapi.runtime_events import RuntimeEvent
from yt_searchapi.state import CrawlProjectState, ProjectStateStore

RunMode = Literal["START NEW PROJECT", "RESUME EXISTING PROJECT"]


@dataclass(frozen=True, slots=True)
class ApiTotals:
    """Cumulative provider totals recovered from the append-only audit."""

    llm_input_tokens: int = 0
    llm_cached_input_tokens: int = 0
    llm_cache_write_tokens: int = 0
    llm_output_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    llm_calls: int = 0
    searchapi_calls: int = 0
    cache_hits: int = 0
    errors: int = 0

    @property
    def llm_total_tokens(self) -> int:
        return self.llm_input_tokens + self.llm_output_tokens


@dataclass(frozen=True, slots=True)
class CrawlTotals:
    discovered: int = 0
    evaluated: int = 0
    relevant: int = 0
    transcripts: int = 0
    pending: int = 0
    queries_done: int = 0
    queries_started: int = 0
    queries_planned: int = 0
    channels_done: int = 0
    channels_discovered: int = 0


def load_api_totals(project_dir: str | Path) -> ApiTotals:
    """Aggregate complete API audit rows without changing project data."""

    path = Path(project_dir) / "api_call.jsonl"
    if not path.is_file():
        return ApiTotals()
    llm_input = llm_cached = llm_writes = llm_output = 0
    llm_cost = 0.0
    llm_calls = searchapi_calls = cache_hits = errors = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            provider = row.get("provider")
            status = row.get("status")
            if provider == "openai":
                llm_calls += 1
                input_tokens = int(row.get("llm_input_tokens", 0) or 0)
                cached_tokens = int(row.get("llm_cached_input_tokens", 0) or 0)
                write_tokens = int(row.get("llm_cache_write_tokens", 0) or 0)
                output_tokens = int(row.get("llm_output_tokens", 0) or 0)
                llm_input += input_tokens
                llm_cached += cached_tokens
                llm_writes += write_tokens
                llm_output += output_tokens
                recorded_cost = row.get("llm_estimated_cost_usd")
                if recorded_cost is None:
                    recorded_cost = estimate_gpt56_luna_standard_cost(
                        LlmTokenMetrics(
                            input_tokens=input_tokens,
                            cached_input_tokens=cached_tokens,
                            cache_write_tokens=write_tokens,
                            output_tokens=output_tokens,
                        ),
                        model=str(row.get("llm_model") or GPT56_LUNA_MODEL),
                    )
                llm_cost += float(recorded_cost or 0)
            elif provider == "searchapi":
                searchapi_calls += 1
                cache_hits += int(status == "cache_hit")
            errors += int(status == "error")
    return ApiTotals(
        llm_input_tokens=llm_input,
        llm_cached_input_tokens=llm_cached,
        llm_cache_write_tokens=llm_writes,
        llm_output_tokens=llm_output,
        llm_estimated_cost_usd=llm_cost,
        llm_calls=llm_calls,
        searchapi_calls=searchapi_calls,
        cache_hits=cache_hits,
        errors=errors,
    )


class RunDashboard:
    """Live view backed by durable state and audit files.

    Runtime events are used only for in-flight counts and the current task. All
    cumulative values are reloaded from the crawler's existing persisted files.
    """

    def __init__(
        self,
        *,
        mode: RunMode,
        project_dir: str | Path,
        budget: SearchApiCreditBudget,
        state_store: ProjectStateStore,
        plan_summary: str | None = None,
        searchapi_concurrency: int = 1,
        openai_concurrency: int = 1,
        console: Console | None = None,
    ) -> None:
        if searchapi_concurrency < 1:
            raise ValueError("searchapi_concurrency must be at least 1")
        if openai_concurrency < 1:
            raise ValueError("openai_concurrency must be at least 1")
        self.mode = mode
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.budget = budget
        self.state_store = state_store
        self._plan_summary = plan_summary.strip() if plan_summary else None
        self.searchapi_concurrency = searchapi_concurrency
        self.openai_concurrency = openai_concurrency
        self.console = console or Console()
        self._started_at = monotonic()
        self._status = "PREPARING" if mode == "START NEW PROJECT" else "RESTORING"
        self._activity = (
            "Preparing the query plan"
            if mode == "START NEW PROJECT"
            else "Restoring the saved frontier"
        )
        # SearchAPI events are emitted when a batch member is accepted for
        # dispatch. A batch may be larger than the client's worker pool, so
        # this stores outstanding work; render() separates active slots from
        # queued work. OpenAI events are emitted after its semaphore is
        # acquired and therefore are already active calls.
        self._active = {"searchapi": 0, "openai": 0}
        self._active_operations: dict[tuple[str, str], int] = {}
        self._last_state: CrawlProjectState | None = None
        self._last_api_totals = ApiTotals()
        self._lock = threading.RLock()
        self._live: Live | None = None

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._live is not None

    def __enter__(self) -> RunDashboard:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def start(self) -> None:
        with self._lock:
            if self._live is not None:
                return
            self._started_at = monotonic()
            self._reload_persisted()
            live = Live(
                console=self.console,
                get_renderable=self.render,
                refresh_per_second=4,
                transient=False,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live = live
        try:
            live.start(refresh=True)
        except BaseException:
            with self._lock:
                if self._live is live:
                    self._live = None
            raise

    def stop(self) -> None:
        with self._lock:
            live, self._live = self._live, None
        if live is not None:
            live.stop()

    def update(self, stage: str, message: str) -> None:
        """Update current crawler work while persisted counts catch up."""

        with self._lock:
            self._status = "RUNNING" if stage != "preparation" else "PREPARING"
            self._activity = message
            self._reload_persisted()
        self._refresh()

    def finish(self, status: str, message: str) -> None:
        with self._lock:
            self._status = status.replace("_", " ").upper()
            self._activity = message
            self._reload_persisted()
        self._refresh()

    def set_plan_summary(self, summary: str | None) -> None:
        """Set the compact run-plan line shown for the rest of this session."""

        with self._lock:
            self._plan_summary = (
                summary.strip() if summary and summary.strip() else None
            )
        self._refresh()

    def on_event(self, event: RuntimeEvent) -> None:
        """Track provider concurrency; cumulative data still comes from JSONL."""

        with self._lock:
            operation_key = (event.provider, event.operation)
            if event.phase == "started":
                if not event.cache_hit:
                    self._active[event.provider] += 1
                    self._active_operations[operation_key] = (
                        self._active_operations.get(operation_key, 0) + 1
                    )
                    self._activity = _active_work(
                        self._active_operations,
                        searchapi_concurrency=self.searchapi_concurrency,
                    )
                else:
                    self._activity = f"SearchAPI cache: {_humanize(event.operation)}"
            else:
                if not event.cache_hit:
                    self._active[event.provider] = max(
                        0, self._active[event.provider] - 1
                    )
                    remaining = self._active_operations.get(operation_key, 0) - 1
                    if remaining > 0:
                        self._active_operations[operation_key] = remaining
                    else:
                        self._active_operations.pop(operation_key, None)
                if self._active_operations:
                    self._activity = _active_work(
                        self._active_operations,
                        searchapi_concurrency=self.searchapi_concurrency,
                    )
                else:
                    outcome = "failed" if event.status == "error" else "finished"
                    self._activity = (
                        f"{_provider_name(event)} {outcome}: "
                        f"{_humanize(event.operation)}"
                    )

    def render(self) -> RenderableType:
        with self._lock:
            self._reload_persisted()
            crawl = _crawl_totals(self._last_state)
            api = self._last_api_totals
            budget = self.budget.snapshot()
            outstanding_searchapi = self._active["searchapi"]
            active_searchapi = min(outstanding_searchapi, self.searchapi_concurrency)
            queued_searchapi = max(
                0, outstanding_searchapi - self.searchapi_concurrency
            )
            active_openai = self._active["openai"]
            elapsed = _format_elapsed(monotonic() - self._started_at)
            narrow = self.console.size.width < 100

            heading = Table.grid(expand=True)
            heading.add_column(ratio=3)
            heading.add_column(justify="right", ratio=1)
            heading.add_row(
                Text(str(self.project_dir), overflow="ellipsis"), self._status
            )
            heading.add_row("Session", elapsed)
            header = Panel(
                heading,
                title=Text(self.mode, style="bold white"),
                border_style="cyan" if self.mode.startswith("RESUME") else "green",
                box=box.ROUNDED,
            )

            activity = Panel(
                Text(self._activity, overflow="fold"),
                title="Current task",
                border_style="bright_blue",
                box=box.ROUNDED,
            )

            metrics = Table.grid(expand=True, padding=(0, 1))
            if narrow:
                metrics.add_column()
                metrics.add_row(
                    "Videos  "
                    f"discovered {crawl.discovered:,}  |  "
                    f"evaluated {crawl.evaluated:,}"
                )
                metrics.add_row(
                    "Results  "
                    f"relevant {crawl.relevant:,}  |  "
                    f"transcripts {crawl.transcripts:,}  "
                    f"|  pending {crawl.pending:,}"
                )
                metrics.add_row(
                    "Progress  "
                    f"queries {crawl.queries_done:,}/{crawl.queries_planned:,} "
                    f"({crawl.queries_started:,} started)  |  "
                    f"channels {crawl.channels_done:,}/{crawl.channels_discovered:,}"
                )
            else:
                metrics.add_column(ratio=1)
                metrics.add_column(ratio=1)
                metrics.add_column(ratio=1)
                metrics.add_row(
                    f"[bold]{crawl.discovered:,}[/] discovered\n"
                    f"{crawl.evaluated:,} evaluated",
                    f"[bold]{crawl.relevant:,}[/] relevant\n"
                    f"{crawl.transcripts:,} transcripts · "
                    f"{crawl.pending:,} pending",
                    f"[bold]{crawl.queries_done:,}/"
                    f"{crawl.queries_planned:,}[/] queries\n"
                    f"{crawl.channels_done:,}/"
                    f"{crawl.channels_discovered:,} channels",
                )
            metrics_panel = Panel(
                metrics, title="Crawl", border_style="blue", box=box.ROUNDED
            )

            spent = budget.discovery_spent + budget.transcript_spent
            credit_text = Text()
            credit_text.append(
                "SearchAPI  "
                f"grant {budget.max_credits:,}  |  spent {spent:,}  |  "
                f"unspent across pools {budget.total_remaining:,}\n"
            )
            credit_text.append(
                "Discovery  "
                f"capacity {budget.discovery_capacity:,}  |  "
                f"spent {budget.discovery_spent:,}  |  "
                f"remaining {budget.discovery_remaining:,}\n"
            )
            credit_text.append(
                "Transcript  "
                f"capacity {budget.transcript_capacity:,}  |  "
                f"spent {budget.transcript_spent:,}  |  "
                f"reserved {budget.transcript_reserved:,}  |  "
                f"remaining {budget.transcript_remaining:,}\n"
            )
            credit_text.append_text(
                _credit_bar(
                    budget.total_committed,
                    budget.max_credits,
                    width=24 if narrow else 40,
                )
            )
            credits_panel = Panel(
                credit_text, title="Credits", border_style="magenta", box=box.ROUNDED
            )

            usage = Table.grid(expand=True, padding=(0, 1))
            usage_rows = (
                "LLM tokens  "
                f"input {api.llm_input_tokens:,}  |  "
                f"cached {api.llm_cached_input_tokens:,}  |  "
                f"writes {api.llm_cache_write_tokens:,}  |  "
                f"output {api.llm_output_tokens:,}",
                "LLM total  "
                f"tokens {api.llm_total_tokens:,}  |  calls {api.llm_calls:,}  |  "
                f"Standard cost ${api.llm_estimated_cost_usd:.6f}",
                "Requests  "
                f"SearchAPI {active_searchapi}/{self.searchapi_concurrency}"
                + (f" · {queued_searchapi} queued" if queued_searchapi else "")
                + "  |  "
                f"OpenAI {active_openai}/{self.openai_concurrency}  |  "
                f"total {active_searchapi + active_openai}",
                "Audit  "
                f"SearchAPI cache hits {api.cache_hits:,}  |  "
                f"errors {api.errors:,}",
            )
            usage.add_column()
            for row in usage_rows:
                usage.add_row(row)
            usage_panel = Panel(
                usage, title="Runtime", border_style="cyan", box=box.ROUNDED
            )

            footer = Table.grid(expand=True)
            footer.add_column(no_wrap=False, overflow="fold")
            footer.add_row(logfire_link())
            renderables: list[RenderableType] = [header]
            if self._plan_summary:
                plan = Table.grid(expand=True, padding=(0, 1))
                plan.add_column(width=4, no_wrap=True)
                plan.add_column(no_wrap=False, overflow="fold")
                plan.add_row(Text("Plan", style="bold cyan"), Text(self._plan_summary))
                renderables.append(plan)
            renderables.extend(
                [activity, metrics_panel, credits_panel, usage_panel, footer]
            )
            return Group(*renderables)

    def _reload_persisted(self) -> None:
        try:
            self._last_state = self.state_store.load()
        except (FileNotFoundError, OSError, ValueError):
            pass
        try:
            self._last_api_totals = load_api_totals(self.project_dir)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def _refresh(self) -> None:
        with self._lock:
            live = self._live
        if live is not None:
            live.refresh()


def _crawl_totals(state: CrawlProjectState | None) -> CrawlTotals:
    if state is None:
        return CrawlTotals()
    pending = 0
    for video_id, video in state.discovered_videos.items():
        if video_id in state.terminal_video_ids:
            continue
        if int(video.get("depth", 0)) <= state.max_depth:
            pending += 1
    queries_planned = min(state.max_queries, len(state.planned_queries))
    queries_started = sum(
        item.pages_completed > 0 for item in state.query_progress.values()
    )
    queries_done = sum(
        item.exhausted or item.pages_completed >= state.max_search_pages
        for item in state.query_progress.values()
    )
    return CrawlTotals(
        discovered=len(state.discovered_videos),
        evaluated=len(state.evaluated_video_ids),
        relevant=len(state.relevant_ids),
        transcripts=len(state.transcript_ids),
        pending=pending,
        queries_done=min(queries_done, queries_planned),
        queries_started=min(queries_started, queries_planned),
        queries_planned=queries_planned,
        channels_done=sum(
            item.exhausted or item.pages_completed >= state.max_channel_pages
            for item in state.channel_progress.values()
        ),
        channels_discovered=len(state.discovered_channels),
    )


def _credit_bar(committed: int, grant: int, *, width: int) -> Text:
    fraction = min(1.0, max(0.0, committed / grant)) if grant else 0.0
    filled = round(width * fraction)
    text = Text()
    text.append("━" * filled, style="bold magenta")
    text.append("─" * (width - filled), style="dim")
    text.append(f"  {fraction:.0%}", style="dim")
    return text


def _provider_name(event: RuntimeEvent) -> str:
    return "SearchAPI" if event.provider == "searchapi" else "OpenAI"


def _active_work(
    active_operations: dict[tuple[str, str], int], *, searchapi_concurrency: int
) -> str:
    searchapi_outstanding = sum(
        count
        for (provider, _operation), count in active_operations.items()
        if provider == "searchapi"
    )
    openai_active = sum(
        count
        for (provider, _operation), count in active_operations.items()
        if provider == "openai"
    )
    searchapi_active = min(searchapi_outstanding, searchapi_concurrency)
    searchapi_queued = max(0, searchapi_outstanding - searchapi_concurrency)
    total = searchapi_active + openai_active
    work = []
    for (provider, operation), count in active_operations.items():
        provider_name = "SearchAPI" if provider == "searchapi" else "OpenAI"
        quantity = f" ×{count}" if count > 1 else ""
        work.append(f"{provider_name}: {_humanize(operation)}{quantity}")
    queue = f" · {searchapi_queued} queued" if searchapi_queued else ""
    prefix = f"{total} requests active{queue} — " if total > 1 else ""
    return prefix + " · ".join(work)


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _format_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


__all__ = ["ApiTotals", "RunDashboard", "load_api_totals"]
