from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Event, Thread

import pytest
from rich.console import Console

import yt_searchapi.run_tui as run_tui
from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.records import ApiCallRecord
from yt_searchapi.run_tui import RunDashboard, load_api_totals
from yt_searchapi.runtime_events import CrawlProgressSnapshot, RuntimeEvent
from yt_searchapi.state import (
    BudgetState,
    CrawlProjectState,
    PageProgress,
    ProjectStateStore,
)
from yt_searchapi.storage import JsonlRunWriter


def _saved_project(tmp_path):
    project = tmp_path / "saved-project"
    state = CrawlProjectState(
        run_id=project.name,
        topic_query="heat pumps",
        language="de",
        start_date="2025-01-01",
        gl="de",
        hl="de",
        transcript_excerpt_chars=12_000,
        max_depth=1,
        max_queries=3,
        max_search_pages=2,
        max_channel_pages=1,
        expansion={},
        classifier_system_prompt="Classify.",
        prompt_sha256="4d553b0b48b4bb151321bd8aa5d9904477ecfdbec8dea4f2554cef83409c8ae6",
        planned_queries=[
            {"text": "query one", "kind": "seed"},
            {"text": "query two", "kind": "interview"},
            {"text": "query three", "kind": "expanded"},
        ],
        budget=BudgetState(
            max_credits=10,
            transcript_capacity=3,
            discovery_spent=2,
            transcript_spent=1,
        ),
        discovered_videos={
            "video-1": {"video_id": "video-1", "depth": 0},
            "video-2": {"video_id": "video-2", "depth": 1},
            "video-3": {"video_id": "video-3", "depth": 2},
        },
        discovered_channels={"channel-1": {"channel_id": "channel-1", "depth": 1}},
        evaluated_video_ids=["video-1", "video-2"],
        terminal_video_ids=["video-1"],
        relevant_ids=["video-1"],
        transcript_ids=["video-1"],
        query_progress={
            "query one": PageProgress(pages_completed=2, exhausted=True),
            "query two": PageProgress(pages_completed=1, exhausted=False),
        },
        channel_progress={"channel-1": PageProgress(pages_completed=1, exhausted=True)},
        last_status="failed",
    )
    ProjectStateStore(project).save(state)
    writer = JsonlRunWriter(project.parent, project.name)
    writer.append(
        ApiCallRecord(
            run_id=project.name,
            provider="openai",
            operation="topic expansion",
            status="success",
            llm_input_tokens=1_200,
            llm_cached_input_tokens=400,
            llm_cache_write_tokens=200,
            llm_output_tokens=300,
            llm_model="gpt-5.6-luna",
        )
    )
    writer.append(
        ApiCallRecord(
            run_id=project.name,
            provider="searchapi",
            operation="youtube_search",
            status="cache_hit",
        )
    )
    writer.append(
        ApiCallRecord(
            run_id=project.name,
            provider="searchapi",
            operation="youtube_video",
            status="error",
            searchapi_credits=1,
            error="timeout",
        )
    )
    budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    return project, budget


def _render(dashboard: RunDashboard, width: int) -> str:
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        color_system=None,
        force_terminal=False,
        record=True,
    )
    dashboard.console = console
    console.print(dashboard.render())
    return output.getvalue()


def test_api_totals_are_cumulative(tmp_path) -> None:
    project, _budget = _saved_project(tmp_path)

    totals = load_api_totals(project)

    assert totals.llm_input_tokens == 1_200
    assert totals.llm_cached_input_tokens == 400
    assert totals.llm_cache_write_tokens == 200
    assert totals.llm_output_tokens == 300
    assert totals.llm_total_tokens == 1_500
    assert totals.llm_estimated_cost_usd == pytest.approx(0.000538)
    assert totals.llm_calls == 1
    assert totals.searchapi_calls == 2
    assert totals.cache_hits == 1
    assert totals.errors == 1


@pytest.mark.parametrize("width", [80, 120])
def test_resume_dashboard_recovers_metrics_and_renders_at_common_widths(
    tmp_path, width
) -> None:
    project, budget = _saved_project(tmp_path)
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=ProjectStateStore(project),
        plan_summary="Continuing 3 queries with 10 added credits",
    )

    text = _render(dashboard, width)

    assert "RESUME EXISTING PROJECT" in text
    assert ("discovered 3" if width == 80 else "3 discovered") in text
    assert ("evaluated 2" if width == 80 else "2 evaluated") in text
    assert ("relevant 1" if width == 80 else "1 relevant") in text
    assert ("transcripts 1" if width == 80 else "1 transcripts") in text
    assert ("pending 1" if width == 80 else "1 pending") in text
    assert "1/3" in text
    assert "1/1" in text
    assert "input 1,200" in text
    assert "cached 400" in text
    assert "writes 200" in text
    assert "output 300" in text
    assert "tokens 1,500" in text
    assert "Standard cost $0.000538" in text
    assert "SearchAPI 0/1" in text
    assert "OpenAI 0/1" in text
    assert "cache hits 1" in text
    assert "errors 1" in text
    assert "unspent across pools 7" in text
    assert "Discovery" in text
    assert "capacity 7" in text
    assert "Transcript" in text
    assert "capacity 3" in text
    assert "reserved 0" in text
    assert "View logs in Logfire" in text
    assert "https://logfire.pydantic.dev/" in text
    assert "Plan" in text
    assert "Continuing 3 queries with 10 added credits" in text
    assert "crawler.\nlog" not in text


@pytest.mark.parametrize("width", [80, 120, 121, 140, 180, 220])
def test_runtime_rows_stay_intact_at_common_and_wide_widths(tmp_path, width) -> None:
    project, budget = _saved_project(tmp_path)
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=ProjectStateStore(project),
    )

    text = _render(dashboard, width)

    assert (
        "LLM tokens  input 1,200  |  cached 400  |  writes 200  |  output 300" in text
    )
    assert "LLM total  tokens 1,500  |  calls 1  |  Standard cost $0.000538" in text
    assert "Requests  SearchAPI 0/1  |  OpenAI 0/1  |  total 0" in text
    assert "Audit  SearchAPI cache hits 1  |  errors 1" in text


def test_plan_summary_can_be_set_or_cleared(tmp_path) -> None:
    project, budget = _saved_project(tmp_path)
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=ProjectStateStore(project),
    )

    assert "Plan" not in _render(dashboard, 80)
    dashboard.set_plan_summary("2 previous queries done; 3 now available")
    assert "2 previous queries done; 3 now available" in _render(dashboard, 80)
    dashboard.set_plan_summary(None)
    assert "Plan" not in _render(dashboard, 80)


def test_long_resume_plan_wraps_without_hiding_changes_at_80_columns(tmp_path) -> None:
    project, budget = _saved_project(tmp_path)
    summary = (
        "Resume plan · grant +30 → 42 · queries 3 → 8 · "
        "search pages 1 → 2 · channel pages 1 → 4 · depth 1 → 2"
    )
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=ProjectStateStore(project),
        plan_summary=summary,
    )

    text = _render(dashboard, 80)

    assert "Plan Resume plan" in text
    assert "grant +30 → 42" in text
    assert "queries 3 → 8" in text
    assert "search pages 1 → 2" in text
    assert "channel\n     pages 1 → 4" in text
    assert "depth 1 → 2" in text
    assert "…" not in "\n".join(
        line for line in text.splitlines() if "Plan" in line or line.startswith("     ")
    )


def test_channel_done_requires_exhaustion_or_page_limit(tmp_path) -> None:
    project, budget = _saved_project(tmp_path)
    state_store = ProjectStateStore(project)
    state = state_store.load()
    state.max_channel_pages = 2
    state.discovered_channels.update(
        {
            "channel-2": {"channel_id": "channel-2", "depth": 1},
            "channel-3": {"channel_id": "channel-3", "depth": 1},
        }
    )
    state.channel_progress = {
        "channel-1": PageProgress(pages_completed=1, exhausted=False),
        "channel-2": PageProgress(pages_completed=1, exhausted=True),
        "channel-3": PageProgress(pages_completed=2, exhausted=False),
    }
    state_store.save(state)
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=state_store,
    )

    text = _render(dashboard, 120)

    assert "2/3 channels" in text


def test_start_and_resume_headers_are_explicit(tmp_path) -> None:
    project = tmp_path / "new-project"
    budget = SearchApiCreditBudget(4, 1)
    common = {
        "project_dir": project,
        "budget": budget,
        "state_store": ProjectStateStore(project),
    }

    start_text = _render(RunDashboard(mode="START NEW PROJECT", **common), 80)
    resume_text = _render(RunDashboard(mode="RESUME EXISTING PROJECT", **common), 80)

    assert "START NEW PROJECT" in start_text
    assert "RESUME EXISTING PROJECT" in resume_text
    assert "START NEW PROJECT" not in resume_text


def test_runtime_events_keep_active_counts_balanced(tmp_path) -> None:
    project = tmp_path / "events"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
    )

    dashboard.on_event(
        RuntimeEvent(provider="openai", operation="classify", phase="started")
    )
    active = _render(dashboard, 80)
    dashboard.on_event(
        RuntimeEvent(
            provider="openai",
            operation="classify",
            phase="finished",
            status="error",
            error="timeout",
        )
    )
    finished = _render(dashboard, 80)

    assert "OpenAI 1/1" in active
    assert "total 1" in active
    assert "OpenAI 0/1" in finished
    assert "total 0" in finished
    assert dashboard._active_operations == {}


def test_completed_events_update_api_totals_without_rescanning_audit(tmp_path) -> None:
    project = tmp_path / "incremental-api"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
    )

    dashboard.on_event(
        RuntimeEvent(
            provider="openai",
            operation="classify",
            phase="finished",
            status="success",
            input_tokens=1_000,
            cached_input_tokens=400,
            cache_write_tokens=100,
            output_tokens=200,
            estimated_cost_usd=0.0004,
        )
    )
    dashboard.on_event(
        RuntimeEvent(
            provider="searchapi",
            operation="youtube_video",
            phase="finished",
            status="cache_hit",
            cache_hit=True,
        )
    )

    text = _render(dashboard, 120)

    assert "input 1,000" in text
    assert "cached 400" in text
    assert "writes 100" in text
    assert "output 200" in text
    assert "tokens 1,200" in text
    assert "calls 1" in text
    assert "Standard cost $0.000400" in text
    assert "cache hits 1" in text


def test_crawl_snapshot_updates_rendered_totals_without_loading_state(tmp_path) -> None:
    project = tmp_path / "incremental-crawl"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
    )
    dashboard.on_crawl_progress(
        CrawlProgressSnapshot(
            discovered=12,
            evaluated=10,
            relevant=4,
            transcripts=5,
            pending=2,
            queries_done=3,
            queries_started=4,
            queries_planned=5,
            channels_done=6,
            channels_discovered=9,
        )
    )

    text = _render(dashboard, 120)

    assert "12 discovered" in text
    assert "10 evaluated" in text
    assert "4 relevant" in text
    assert "5 transcripts · 2 pending" in text
    assert "3/5 queries" in text
    assert "6/9 channels" in text


def test_finished_event_keeps_remaining_parallel_work_visible(tmp_path) -> None:
    project = tmp_path / "overlapping-events"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(8, 2),
        state_store=ProjectStateStore(project),
        searchapi_concurrency=4,
        openai_concurrency=3,
    )

    dashboard.on_event(
        RuntimeEvent(provider="searchapi", operation="youtube_search", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(provider="searchapi", operation="youtube_video", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(provider="openai", operation="classify", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(
            provider="searchapi",
            operation="youtube_search",
            phase="finished",
            status="success",
        )
    )

    overlapping = _render(dashboard, 80)
    assert "2 requests active" in overlapping
    assert "SearchAPI: youtube video" in overlapping
    assert "OpenAI: classify" in overlapping
    assert "SearchAPI finished: youtube search" not in overlapping
    assert "SearchAPI 1/4" in overlapping
    assert "OpenAI 1/3" in overlapping

    dashboard.on_event(
        RuntimeEvent(
            provider="searchapi",
            operation="youtube_video",
            phase="finished",
            status="success",
        )
    )
    one_remaining = _render(dashboard, 80)
    assert "OpenAI: classify" in one_remaining
    assert "SearchAPI finished: youtube video" not in one_remaining

    dashboard.on_event(
        RuntimeEvent(
            provider="openai",
            operation="classify",
            phase="finished",
            status="success",
        )
    )
    complete = _render(dashboard, 80)
    assert "OpenAI finished: classify" in complete
    assert "total 0" in complete


def test_runtime_shows_configured_provider_concurrency(tmp_path) -> None:
    project = tmp_path / "concurrent-events"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
        searchapi_concurrency=3,
        openai_concurrency=4,
    )

    dashboard.on_event(
        RuntimeEvent(provider="searchapi", operation="search", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(provider="searchapi", operation="search", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(provider="openai", operation="classify", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(provider="openai", operation="classify", phase="started")
    )

    text = _render(dashboard, 80)
    assert "SearchAPI 2/3" in text
    assert "OpenAI 2/4" in text


def test_runtime_caps_searchapi_active_count_and_shows_queued_work(tmp_path) -> None:
    project = tmp_path / "queued-searchapi"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
        searchapi_concurrency=2,
        openai_concurrency=4,
    )

    for _ in range(4):
        dashboard.on_event(
            RuntimeEvent(
                provider="searchapi", operation="youtube_video", phase="started"
            )
        )

    text = _render(dashboard, 80)

    assert "SearchAPI 2/2 · 2 queued" in text
    assert "2 requests active · 2 queued" in text
    assert "SearchAPI 4/2" not in text

    for _ in range(4):
        dashboard.on_event(
            RuntimeEvent(
                provider="searchapi",
                operation="youtube_video",
                phase="finished",
                status="success",
            )
        )

    assert "SearchAPI 0/2" in _render(dashboard, 80)
    assert dashboard._active_operations == {}


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"searchapi_concurrency": 0}, "searchapi_concurrency"),
        ({"openai_concurrency": 0}, "openai_concurrency"),
    ],
)
def test_runtime_rejects_invalid_provider_concurrency(
    tmp_path, argument, message
) -> None:
    with pytest.raises(ValueError, match=message):
        RunDashboard(
            mode="START NEW PROJECT",
            project_dir=tmp_path / "invalid-concurrency",
            budget=SearchApiCreditBudget(4, 1),
            state_store=ProjectStateStore(tmp_path / "invalid-concurrency"),
            **argument,
        )


def test_live_methods_run_only_after_releasing_dashboard_lock(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "lock-order"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(4, 1),
        state_store=ProjectStateStore(project),
    )
    instances = []

    class ProbeLive:
        def __init__(self, *, get_renderable, **_kwargs) -> None:
            self.get_renderable = get_renderable
            self.calls: list[str] = []
            instances.append(self)

        def _render_on_another_thread(self) -> None:
            errors: list[BaseException] = []

            def render() -> None:
                try:
                    self.get_renderable()
                except BaseException as exc:  # pragma: no cover - assertion detail
                    errors.append(exc)

            thread = Thread(target=render, daemon=True)
            thread.start()
            thread.join(timeout=1)
            assert not thread.is_alive(), (
                "Live method waited for a render thread blocked on the dashboard lock"
            )
            assert not errors

        def start(self, *, refresh: bool) -> None:
            self.calls.append("start")
            if refresh:
                self._render_on_another_thread()

        def refresh(self) -> None:
            self.calls.append("refresh")
            self._render_on_another_thread()

        def stop(self) -> None:
            self.calls.append("stop")
            self._render_on_another_thread()

    monkeypatch.setattr(run_tui, "Live", ProbeLive)

    dashboard.start()
    probe = instances[0]
    dashboard.update("crawler", "Running")
    dashboard.set_plan_summary("Grant 4")
    dashboard.finish("completed", "Done")
    refresh_count = probe.calls.count("refresh")
    dashboard.on_event(
        RuntimeEvent(provider="openai", operation="classify", phase="started")
    )
    dashboard.on_event(
        RuntimeEvent(
            provider="openai",
            operation="classify",
            phase="finished",
            status="success",
        )
    )
    assert probe.calls.count("refresh") == refresh_count
    dashboard.stop()

    assert probe.calls == ["start", "stop"]


def test_repeated_renders_do_not_reload_checkpoint_or_api_audit(
    monkeypatch, tmp_path
) -> None:
    project, budget = _saved_project(tmp_path)
    state_store = ProjectStateStore(project)
    state_loads = 0
    api_loads = 0
    original_state_load = state_store.load
    original_api_load = run_tui.load_api_totals

    def tracked_state_load():
        nonlocal state_loads
        state_loads += 1
        return original_state_load()

    def tracked_api_load(project_dir):
        nonlocal api_loads
        api_loads += 1
        return original_api_load(project_dir)

    monkeypatch.setattr(state_store, "load", tracked_state_load)
    monkeypatch.setattr(run_tui, "load_api_totals", tracked_api_load)
    dashboard = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=project,
        budget=budget,
        state_store=state_store,
    )

    assert (state_loads, api_loads) == (1, 1)
    for _ in range(20):
        dashboard.render()
    assert (state_loads, api_loads) == (1, 1)


def test_concurrent_finish_callbacks_return_while_dashboard_renders(tmp_path) -> None:
    project = tmp_path / "callback-stress"
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(8, 2),
        state_store=ProjectStateStore(project),
        searchapi_concurrency=16,
        openai_concurrency=16,
    )
    request_count = 100
    started = [
        RuntimeEvent(provider="searchapi", operation="youtube_video", phase="started")
        for _index in range(request_count)
    ] + [
        RuntimeEvent(provider="openai", operation="classify", phase="started")
        for _index in range(request_count)
    ]
    finished = [
        RuntimeEvent(
            provider=event.provider,
            operation=event.operation,
            phase="finished",
            status="success",
        )
        for event in started
    ]

    with ThreadPoolExecutor(max_workers=16) as executor:
        start_futures = [
            executor.submit(dashboard.on_event, event) for event in started
        ]
        for future in start_futures:
            future.result(timeout=2)

        stop_rendering = Event()

        def render_until_stopped() -> None:
            while not stop_rendering.is_set():
                dashboard.render()
                stop_rendering.wait(0.001)

        render_thread = Thread(target=render_until_stopped, daemon=True)
        render_thread.start()
        try:
            finish_futures = [
                executor.submit(dashboard.on_event, event) for event in finished
            ]
            for future in finish_futures:
                future.result(timeout=2)
        finally:
            stop_rendering.set()
            render_thread.join(timeout=2)

    assert not render_thread.is_alive()
    assert dashboard._active_operations == {}
    text = _render(dashboard, 80)
    assert "SearchAPI 0/16" in text
    assert "OpenAI 0/16" in text
    assert "total 0" in text
