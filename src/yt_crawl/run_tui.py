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

from yt_crawl.budget import SearchApiCreditBudget
from yt_crawl.observability import logfire_link
from yt_crawl.runtime_events import CrawlProgressSnapshot, RuntimeEvent
from yt_crawl.state import CrawlProjectState, ProjectStateStore

RunMode = Literal["START NEW PROJECT", "RESUME EXISTING PROJECT", "SHOW PROJECT"]


@dataclass(frozen=True, slots=True)
class FrontierSettings:
    language: str | None = None
    start_date: str | None = None
    max_depth: int | None = None
    max_queries: int | None = None
    max_search_pages: int | None = None
    max_channel_pages: int | None = None

    @classmethod
    def from_state(cls, state: CrawlProjectState) -> FrontierSettings:
        return cls(
            language=state.language,
            start_date=state.start_date.isoformat(),
            max_depth=state.max_depth,
            max_queries=state.max_queries,
            max_search_pages=state.max_search_pages,
            max_channel_pages=state.max_channel_pages,
        )


@dataclass(frozen=True, slots=True)
class ApiTotals:
    """Cumulative provider totals recovered from the append-only audit."""

    llm_calls: int = 0
    searchapi_calls: int = 0
    cache_hits: int = 0
    errors: int = 0


def load_api_totals(project_dir: str | Path) -> ApiTotals:
    """Aggregate complete API audit rows without changing project data."""

    path = Path(project_dir) / "api_call.jsonl"
    if not path.is_file():
        return ApiTotals()
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
            elif provider == "searchapi":
                searchapi_calls += 1
                cache_hits += int(status == "cache_hit")
            errors += int(status == "error")
    return ApiTotals(
        llm_calls=llm_calls,
        searchapi_calls=searchapi_calls,
        cache_hits=cache_hits,
        errors=errors,
    )


def _updated_api_totals(totals: ApiTotals, event: RuntimeEvent) -> ApiTotals:
    """Apply one completed provider event to cumulative presentation totals."""

    if event.phase != "finished":
        return totals
    is_openai = event.provider == "openai"
    is_searchapi = event.provider == "searchapi"
    return ApiTotals(
        llm_calls=totals.llm_calls + int(is_openai),
        searchapi_calls=totals.searchapi_calls + int(is_searchapi),
        cache_hits=totals.cache_hits + int(is_searchapi and event.cache_hit),
        errors=totals.errors + int(event.status == "error"),
    )


class RunDashboard:
    """Live view hydrated from durable files, then updated from runtime events.

    Startup recovers cumulative values from the checkpoint and append-only audit.
    Rendering thereafter consumes only small immutable in-memory snapshots.
    """

    def __init__(
        self,
        *,
        mode: RunMode,
        project_dir: str | Path,
        budget: SearchApiCreditBudget,
        state_store: ProjectStateStore,
        plan_summary: str | None = None,
        frontier: FrontierSettings | None = None,
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
        self._frontier = frontier
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
        self._crawl = CrawlProgressSnapshot()
        self._last_api_totals = ApiTotals()
        self._lock = threading.RLock()
        self._live: Live | None = None
        self._hydrate_persisted()

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
            live = Live(
                console=self.console,
                get_renderable=self.render,
                refresh_per_second=2,
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
        """Update the current task without filesystem work or forced rendering."""

        with self._lock:
            self._status = "RUNNING" if stage != "preparation" else "PREPARING"
            self._activity = message

    def finish(self, status: str, message: str) -> None:
        with self._lock:
            self._status = status.replace("_", " ").upper()
            self._activity = message

    def set_plan_summary(self, summary: str | None) -> None:
        """Set the compact run-plan line shown for the rest of this session."""

        with self._lock:
            self._plan_summary = (
                summary.strip() if summary and summary.strip() else None
            )

    def on_crawl_progress(self, snapshot: CrawlProgressSnapshot) -> None:
        """Accept an already-aggregated crawler snapshot without filesystem work."""

        with self._lock:
            self._crawl = snapshot

    def on_event(self, event: RuntimeEvent) -> None:
        """Track provider concurrency and increment cumulative API totals."""

        with self._lock:
            if event.phase == "finished":
                self._last_api_totals = _updated_api_totals(
                    self._last_api_totals, event
                )
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
            crawl = self._crawl
            api = self._last_api_totals
            outstanding_searchapi = self._active["searchapi"]
            active_openai = self._active["openai"]
            status = self._status
            activity_text = self._activity
            plan_summary = self._plan_summary
            frontier = self._frontier

        budget = self.budget.snapshot()
        active_searchapi = min(outstanding_searchapi, self.searchapi_concurrency)
        queued_searchapi = max(0, outstanding_searchapi - self.searchapi_concurrency)
        elapsed = _format_elapsed(monotonic() - self._started_at)
        narrow = self.console.size.width < 100

        heading = Table.grid(expand=True)
        heading.add_column(ratio=3)
        heading.add_column(justify="right", ratio=1)
        show = self.mode == "SHOW PROJECT"
        heading.add_row(
            Text(str(self.project_dir), overflow="ellipsis"),
            "" if show else status,
        )
        heading.add_row("Session", "SNAPSHOT" if show else elapsed)
        header = Panel(
            heading,
            title=Text(self.mode, style="bold white"),
            border_style=(
                "cyan"
                if self.mode.startswith(("RESUME", "SHOW"))
                else "green"
            ),
            box=box.ROUNDED,
        )

        activity = Panel(
            Text(activity_text, overflow="fold"),
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
                f"channels {crawl.channels_exhausted:,} exhausted · "
                f"{crawl.channels_page_capped:,} page cap · "
                f"{crawl.channels_discovered:,} discovered"
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
                f"{crawl.channels_discovered:,} channels discovered\n"
                f"{crawl.channels_exhausted:,} exhausted · "
                f"{crawl.channels_page_capped:,} page cap",
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
            f"LLM  calls {api.llm_calls:,}",
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
        if plan_summary:
            plan = Table.grid(expand=True, padding=(0, 1))
            plan.add_column(width=4, no_wrap=True)
            plan.add_column(no_wrap=False, overflow="fold")
            plan.add_row(Text("Plan", style="bold cyan"), Text(plan_summary))
            renderables.append(plan)
        frontier_text = _frontier_text(frontier)
        if frontier_text:
            frontier_grid = Table.grid(expand=True, padding=(0, 1))
            frontier_grid.add_column(width=8, no_wrap=True)
            frontier_grid.add_column(no_wrap=False, overflow="fold")
            frontier_grid.add_row(
                Text("Frontier", style="bold cyan"), Text(frontier_text)
            )
            renderables.append(frontier_grid)
        renderables.extend(
            [activity, metrics_panel, credits_panel, usage_panel, footer]
        )
        return Group(*renderables)

    def _hydrate_persisted(self) -> None:
        """Load durable totals once before live event-driven updates begin."""

        try:
            state = self.state_store.load()
            self._crawl = _crawl_totals(state)
            if self._frontier is None:
                self._frontier = FrontierSettings.from_state(state)
        except (FileNotFoundError, OSError, ValueError):
            pass
        try:
            self._last_api_totals = load_api_totals(self.project_dir)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass


def _crawl_totals(state: CrawlProjectState | None) -> CrawlProgressSnapshot:
    if state is None:
        return CrawlProgressSnapshot()
    terminal_video_ids = set(state.terminal_video_ids)
    pending = sum(
        video_id not in terminal_video_ids
        and int(video.get("depth", 0)) <= state.max_depth
        for video_id, video in state.discovered_videos.items()
    )
    queries_planned = min(state.max_queries, len(state.planned_queries))
    queries_started = sum(
        item.pages_completed > 0 for item in state.query_progress.values()
    )
    queries_done = sum(
        item.exhausted or item.pages_completed >= state.max_search_pages
        for item in state.query_progress.values()
    )
    channels_exhausted = sum(
        item.exhausted for item in state.channel_progress.values()
    )
    channels_page_capped = sum(
        not item.exhausted and item.pages_completed >= state.max_channel_pages
        for item in state.channel_progress.values()
    )
    return CrawlProgressSnapshot(
        discovered=len(state.discovered_videos),
        evaluated=len(state.evaluated_video_ids),
        relevant=len(state.relevant_ids),
        transcripts=len(state.transcript_ids),
        pending=pending,
        queries_done=min(queries_done, queries_planned),
        queries_started=min(queries_started, queries_planned),
        queries_planned=queries_planned,
        channels_done=channels_exhausted + channels_page_capped,
        channels_discovered=len(state.discovered_channels),
        channels_exhausted=channels_exhausted,
        channels_page_capped=channels_page_capped,
    )


def _frontier_text(frontier: FrontierSettings | None) -> str | None:
    if frontier is None:
        return None
    scope = [
        f"language {frontier.language}" if frontier.language else None,
        f"start {frontier.start_date}" if frontier.start_date else None,
        f"depth {frontier.max_depth}" if frontier.max_depth is not None else None,
        f"queries {frontier.max_queries}" if frontier.max_queries is not None else None,
        (
            f"search pages {frontier.max_search_pages}"
            if frontier.max_search_pages is not None
            else None
        ),
        (
            f"channel pages {frontier.max_channel_pages}"
            if frontier.max_channel_pages is not None
            else None
        ),
    ]
    parts = [item for item in scope if item]
    return " · ".join(parts) if parts else None


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
