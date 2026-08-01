import json
from collections import Counter
from io import StringIO
from pathlib import Path

from rich.console import Console

from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.dashboard import build_dashboard_context, build_dashboard_data
from yt_searchapi.records import validate_run_record
from yt_searchapi.run_tui import RunDashboard
from yt_searchapi.state import ProjectStateStore

MOCK_RUN_DIR = Path(__file__).parents[1] / "examples" / "mock-run"


def test_mock_run_is_complete_and_every_jsonl_line_validates() -> None:
    expected_types = {
        "api_call",
        "budget_event",
        "channel",
        "discovery_edge",
        "interview_answer",
        "query",
        "relevance_decision",
        "run_config",
        "run_error",
        "run_metric",
        "run_status",
        "transcript",
        "video_candidate",
    }
    records = []

    for path in sorted(MOCK_RUN_DIR.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, f"{path.name} must not be empty"
        for line_number, line in enumerate(lines, start=1):
            decoded = json.loads(line)
            parsed = validate_run_record(decoded)
            assert parsed.record_type == path.stem, (
                f"{path.name}:{line_number} contains {parsed.record_type!r}"
            )
            records.append(parsed)

    counts = Counter(record.record_type for record in records)
    assert set(counts) == expected_types
    assert counts["video_candidate"] == 7
    assert counts["transcript"] == 4
    assert {record.run_id for record in records} == {"mock-library-heat-2026-07-31"}
    budget_events = [
        record for record in records if record.record_type == "budget_event"
    ]
    assert {event.budget_kind.value for event in budget_events} == {
        "search_api_credits"
    }

    transcripts = [record for record in records if record.record_type == "transcript"]
    assert sum(record.is_available for record in transcripts) == 3

    decisions = [
        record for record in records if record.record_type == "relevance_decision"
    ]
    final_by_video = {}
    for decision in sorted(decisions, key=lambda item: item.recorded_at):
        final_by_video[decision.video_id] = decision.label.value
    assert Counter(final_by_video.values()) == {"relevant": 3, "irrelevant": 4}

    state = ProjectStateStore(MOCK_RUN_DIR).load()
    assert state.last_status == "completed"
    assert state.budget.discovery_spent + state.budget.transcript_spent == 11
    candidates = {
        record.video_id for record in records if record.record_type == "video_candidate"
    }
    channels = {
        record.channel_id for record in records if record.record_type == "channel"
    }
    assert set(state.discovered_videos) == candidates
    assert set(state.discovered_channels) == channels
    assert set(state.evaluated_video_ids) == candidates
    assert set(state.terminal_video_ids) == candidates
    assert set(state.relevant_ids) == {
        video_id for video_id, label in final_by_video.items() if label == "relevant"
    }
    assert set(state.transcript_ids) == {record.video_id for record in transcripts}

    budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    tui = RunDashboard(
        mode="RESUME EXISTING PROJECT",
        project_dir=MOCK_RUN_DIR,
        budget=budget,
        state_store=ProjectStateStore(MOCK_RUN_DIR),
    )
    tui_output = StringIO()
    tui.console = Console(
        file=tui_output,
        width=120,
        color_system=None,
        force_terminal=False,
    )
    tui.console.print(tui.render())
    tui_text = tui_output.getvalue()
    assert "7 discovered" in tui_text
    assert "3 relevant" in tui_text
    assert "4 transcripts" in tui_text
    assert "0 pending" in tui_text
    assert "2/2 queries" in tui_text
    assert "1/4 channels" in tui_text

    dashboard = build_dashboard_context(
        build_dashboard_data(MOCK_RUN_DIR), MOCK_RUN_DIR
    )
    assert dashboard["run"]["sessions"] == 2
    assert dashboard["budgets"][0] == {
        "label": "SearchAPI total",
        "used": 11,
        "limit": 50,
        "remaining": 39,
        "reserved": 0,
        "percent": 22,
        "unit": "credits",
        "tone": "blue",
    }
    assert dashboard["metrics"]["queries"] == 6
    assert dashboard["continuation"]["executed_queries"] == 2
    assert dashboard["continuation"]["deferred_queries"] == 4
    assert dashboard["continuation"]["unfinished_queries"] == 0
    assert "--max-queries 3" in dashboard["continuation"]["resume_command"]
