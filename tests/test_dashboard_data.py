import hashlib
from datetime import UTC, date, datetime

from yt_searchapi.dashboard import (
    _continuation_reason,
    _suggested_resume_command,
    build_dashboard_context,
    build_dashboard_data,
    render_dashboard,
)
from yt_searchapi.records import (
    ApiCallRecord,
    BudgetAction,
    BudgetEventRecord,
    BudgetKind,
    DiscoveryEdgeRecord,
    QueryRecord,
    RelevanceDecisionRecord,
    RelevanceLabel,
    RunConfigRecord,
    RunStatus,
    RunStatusRecord,
    TranscriptRecord,
    TranscriptSegment,
    VideoCandidateRecord,
)
from yt_searchapi.state import (
    BudgetState,
    CrawlProjectState,
    PageProgress,
    ProjectStateStore,
)
from yt_searchapi.storage import JsonlRunWriter


def _populated_run(tmp_path):
    writer = JsonlRunWriter(tmp_path, "run-dashboard")
    recorded_at = datetime(2026, 7, 31, tzinfo=UTC)
    writer.append(
        RunConfigRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            topic_query="urban heat <adaptation>",
            expanded_queries=("urban heat interview",),
            language="en",
            start_date=date(2022, 1, 1),
            max_searchapi_credits=20,
            transcript_reserve_credits=12,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
            prompt_sha256="a" * 64,
        )
    )
    writer.append(
        RunStatusRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            status=RunStatus.STARTED,
        )
    )
    writer.append(
        QueryRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            query="urban heat adaptation",
            query_kind="seed",
            status="planned",
        )
    )
    writer.append(
        QueryRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            query="urban heat adaptation",
            query_kind="seed",
            status="executed",
            source_request_id="search-1",
        )
    )
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-1",
            title="Initial title",
            url="https://youtube.test/watch?v=video-1",
            discovered_via="search",
            discovered_from_id="urban heat adaptation",
            discovery_query="urban heat adaptation",
        )
    )
    # The detail snapshot has richer fields but deliberately no URL. The
    # dashboard should keep the latest non-null value for every field.
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-1",
            title="Detailed title",
            description="Detailed description",
            channel_id="channel-1",
            channel_title="Climate practitioners",
            published_at=recorded_at,
            duration_seconds=125,
            discovered_via="search",
            discovered_from_id="urban heat adaptation",
            discovery_query="urban heat adaptation",
        )
    )
    writer.append(
        DiscoveryEdgeRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-1",
            source_type="search",
            source_ref="urban heat adaptation",
            depth=0,
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="metadata-decision",
            video_id="video-1",
            label=RelevanceLabel.NEEDS_TRANSCRIPT,
            decision_point="video_metadata",
            primary_reason="topic_match",
            requested_language="en",
            detected_language=None,
            language_matches=False,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=100,
            llm_output_tokens=20,
            transcript_reserved=True,
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="transcript-decision",
            video_id="video-1",
            label=RelevanceLabel.IRRELEVANT,
            decision_point="transcript",
            primary_reason="off_topic",
            requested_language="en",
            detected_language="en",
            language_matches=True,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=200,
            llm_output_tokens=30,
            transcript_reserved=True,
        )
    )
    writer.append(
        TranscriptRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-1",
            requested_language="en",
            language="en",
            transcript_type="manual",
            is_available=True,
            segments=(
                TranscriptSegment(
                    text="First segment.", start_seconds=0, duration_seconds=2.5
                ),
                TranscriptSegment(
                    text="Second segment.", start_seconds=2.5, duration_seconds=3
                ),
            ),
            searchapi_credits=1,
        )
    )
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-2",
            title="Accepted cooling center interview",
            channel_id="channel-2",
            channel_title="City library",
            published_at=recorded_at,
            duration_seconds=300,
            discovered_via="search",
            discovered_from_id="urban heat adaptation",
            discovery_query="urban heat adaptation",
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="accepted-transcript-decision",
            video_id="video-2",
            label=RelevanceLabel.RELEVANT,
            decision_point="transcript",
            primary_reason="topic_match",
            requested_language="en",
            detected_language="en",
            language_matches=True,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=150,
            llm_output_tokens=20,
            transcript_reserved=True,
        )
    )
    writer.append(
        TranscriptRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-2",
            requested_language="en",
            language="en",
            transcript_type="manual",
            is_available=True,
            segments=(
                TranscriptSegment(
                    text="Libraries provide cooling rooms.",
                    start_seconds=0,
                    duration_seconds=4,
                ),
            ),
            searchapi_credits=1,
        )
    )
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-pending",
            title="Pending transcript review",
            discovered_via="related",
            discovered_from_id="video-2",
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="pending-metadata-decision",
            video_id="video-pending",
            label=RelevanceLabel.NEEDS_TRANSCRIPT,
            decision_point="video_metadata",
            primary_reason="topic_match",
            requested_language="en",
            detected_language=None,
            language_matches=False,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=100,
            llm_output_tokens=20,
            transcript_reserved=True,
        )
    )
    writer.append(
        ApiCallRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            provider="searchapi",
            operation="youtube_transcripts",
            status="success",
            searchapi_credits=1,
            latency_seconds=0.25,
        )
    )
    writer.append(
        ApiCallRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            provider="openai",
            operation="classify_video_transcript",
            status="success",
            llm_input_tokens=200,
            llm_output_tokens=30,
            latency_seconds=0.5,
        )
    )
    writer.append(
        BudgetEventRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            budget_kind=BudgetKind.SEARCH_API_CREDITS,
            pool="transcript",
            action=BudgetAction.RECONCILED,
            amount=1,
            remaining=11,
            purpose="transcript:video-1",
        )
    )
    writer.append(
        RunStatusRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            status=RunStatus.COMPLETED,
            reason="frontier_exhausted",
        )
    )
    return writer.run_dir


def test_duckdb_normalizes_latest_video_decision_and_transcript_segments(
    tmp_path,
) -> None:
    data = build_dashboard_data(_populated_run(tmp_path))

    assert data.run_id == "run-dashboard"
    assert data.status == "completed"
    assert data.config["expanded_queries"] == ["urban heat interview"]

    assert len(data.videos) == 3
    video = next(item for item in data.videos if item["video_id"] == "video-1")
    assert video["title"] == "Detailed title"
    assert video["url"] == "https://youtube.test/watch?v=video-1"
    assert video["final_label"] == "irrelevant"
    assert video["decision_point"] == "transcript"
    assert video["duration_display"] == "2:05"
    assert video["transcript_segments"] == 2

    assert len(data.transcript_segments) == 3
    assert [segment["text"] for segment in data.transcript_segments[:2]] == [
        "First segment.",
        "Second segment.",
    ]
    assert data.transcripts[0]["text"] == "First segment. Second segment."
    assert data.transcripts[0]["transcript_duration_seconds"] == 5.5

    summary = {metric["key"]: metric["value"] for metric in data.summary}
    assert summary == {
        "videos_discovered": 3,
        "videos_evaluated": 2,
        "relevant_videos": 1,
        "transcripts_collected": 2,
        "searchapi_credits": 1,
        "llm_tokens": 230,
    }


def test_presentation_context_normalizes_queries_decisions_and_coverage(
    tmp_path,
) -> None:
    run_dir = _populated_run(tmp_path)
    context = build_dashboard_context(build_dashboard_data(run_dir), run_dir)

    assert context["metrics"]["candidates"] == 3
    assert context["metrics"]["relevant"] == 1
    assert context["metrics"]["irrelevant"] == 1
    assert [row["id"] for row in context["relevant_videos"]] == ["video-2"]
    assert [row["id"] for row in context["irrelevant_videos"]] == ["video-1"]
    assert [row["id"] for row in context["other_videos"]] == ["video-pending"]
    assert all(
        row["id"] != "video-pending"
        for row in context["relevant_videos"] + context["irrelevant_videos"]
    )

    assert context["queries"] == [
        {
            "query": "urban heat adaptation",
            "kind": "seed",
            "status": "executed",
            "results": 2,
            "relevant": 1,
        }
    ]
    assert context["funnel"][1]["count"] == 2
    assert context["transcript"]["eligible"] == 1
    assert context["transcript"]["available"] == 1
    assert context["transcript"]["unavailable"] == 0
    assert context["transcript"]["coverage_percent"] == 100
    assert context["transcript"]["segments"] == 1
    assert context["transcript"]["words"] == 4


def test_dashboard_uses_latest_resume_decision_and_budget_deferral(tmp_path) -> None:
    run_dir = _populated_run(tmp_path)
    writer = JsonlRunWriter(run_dir.parent, run_dir.name)
    recorded_at = datetime(2026, 7, 31, 12, tzinfo=UTC)
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-resumed",
            title="Budget-deferred interview",
            discovered_via="search",
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="budget-deferred-decision",
            video_id="video-resumed",
            label=RelevanceLabel.DEFERRED_BUDGET,
            decision_point="search_result",
            primary_reason="discovery requires 1 credits but only 0 remain",
            requested_language="en",
            language_matches=None,
            model="not-classified",
            prompt_version="disposition-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=0,
            llm_output_tokens=0,
            transcript_reserved=False,
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="resumed-decision",
            video_id="video-resumed",
            label=RelevanceLabel.IRRELEVANT,
            decision_point="video_metadata",
            primary_reason="wrong_language",
            requested_language="en",
            detected_language="de",
            language_matches=False,
            model="deterministic",
            prompt_version="deterministic-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=0,
            llm_output_tokens=0,
            transcript_reserved=False,
        )
    )

    data = build_dashboard_data(run_dir)
    context = build_dashboard_context(data, run_dir)

    resumed = next(
        video for video in data.videos if video["video_id"] == "video-resumed"
    )
    assert resumed["final_label"] == "irrelevant"
    assert [
        decision["label"]
        for decision in data.decisions
        if decision["video_id"] == "video-resumed"
    ] == ["deferred_budget", "irrelevant"]
    assert [video["id"] for video in context["irrelevant_videos"]] == [
        "video-resumed",
        "video-1",
    ]
    assert all(video["id"] != "video-resumed" for video in context["other_videos"])

    deferred_run_dir = _deferred_budget_run(tmp_path)
    deferred_context = build_dashboard_context(
        build_dashboard_data(deferred_run_dir), deferred_run_dir
    )
    assert deferred_context["metrics"]["irrelevant"] == 0
    assert (
        deferred_context["other_videos"][0]["decision_display"]
        == "Deferred " + chr(0x2014) + " budget"
    )
    html = render_dashboard(deferred_run_dir).read_text(encoding="utf-8")
    assert "Deferred \u2014 budget" in html


def _deferred_budget_run(tmp_path):
    writer = JsonlRunWriter(tmp_path, "run-deferred-budget")
    recorded_at = datetime(2026, 7, 31, 13, tzinfo=UTC)
    writer.append(
        VideoCandidateRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            video_id="video-deferred",
            title="Deferred candidate",
            discovered_via="search",
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=writer.run_id,
            recorded_at=recorded_at,
            decision_id="deferred-decision",
            video_id="video-deferred",
            label=RelevanceLabel.DEFERRED_BUDGET,
            decision_point="search_result",
            primary_reason="discovery requires 1 credits but only 0 remain",
            requested_language="en",
            language_matches=None,
            model="not-classified",
            prompt_version="disposition-v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=0,
            llm_output_tokens=0,
            transcript_reserved=False,
        )
    )
    return writer.run_dir


def test_missing_optional_jsonl_files_produce_empty_dashboard(tmp_path) -> None:
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    data = build_dashboard_data(run_dir)

    assert data.run_id == "empty-run"
    assert data.status == "unknown"
    assert data.videos == ()
    assert data.transcripts == ()
    assert all(metric["value"] == 0 for metric in data.summary)

    output = render_dashboard(run_dir)
    assert "empty-run" in output.read_text(encoding="utf-8")


def test_render_dashboard_is_self_contained_and_escapes_record_content(
    tmp_path,
) -> None:
    run_dir = _populated_run(tmp_path)

    output = render_dashboard(run_dir)
    html = output.read_text(encoding="utf-8")

    assert output == run_dir / "dashboard.html"
    assert "urban heat &lt;adaptation&gt;" in html
    for section in (
        "SearchAPI credit budget",
        "Project cannot be resumed",
        "Discovery funnel",
        "Transcript coverage",
        "Query plan",
        "Relevance decisions",
        "Transcript inventory",
        "Errors and warnings",
    ):
        assert section in html
    assert "Accepted cooling center interview" in html
    assert "Pending transcript review" in html
    assert "needs transcript" in html.casefold()
    assert "planned variants" in html
    assert '<th scope="col" class="number">Credits</th>' not in html
    assert "No valid resumable checkpoint is available" in html
    assert "LLM discovery" not in html
    assert "http://" not in html
    assert "https://youtube.test/watch?v=video-1" in html
    assert "<link " not in html
    assert "<script src=" not in html
    assert "@import" not in html
    assert "cdn." not in html.casefold()


def test_latest_resume_config_and_active_session_timing_are_current(tmp_path) -> None:
    run_dir = _populated_run(tmp_path)
    writer = JsonlRunWriter(run_dir.parent, run_dir.name)
    resumed_at = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)
    writer.append(
        RunConfigRecord(
            run_id=writer.run_id,
            recorded_at=resumed_at,
            topic_query="urban heat <adaptation>",
            expanded_queries=("urban heat interview",),
            language="en",
            start_date=date(2022, 1, 1),
            max_searchapi_credits=35,
            transcript_reserve_credits=15,
            session_action="resume",
            credits_added=15,
            account_remaining_credits=80,
            max_depth=3,
            max_queries=10,
            max_search_pages=2,
            max_channel_pages=2,
            model="gpt-5.6-luna",
            prompt_version="relevance-v1",
        )
    )
    writer.append(
        RunStatusRecord(
            run_id=writer.run_id,
            recorded_at=resumed_at,
            status=RunStatus.STARTED,
        )
    )

    data = build_dashboard_data(run_dir)
    context = build_dashboard_context(data, run_dir)

    assert data.status == "started"
    assert data.completed_at is None
    assert data.started_at == "2026-07-31T10:00:00Z"
    assert data.config["max_searchapi_credits"] == 35
    assert data.config["max_queries"] == 10
    assert [session["action"] for session in data.sessions] == ["start", "resume"]
    assert context["run"]["sessions"] == 2
    assert context["budgets"][0]["label"] == "SearchAPI total"
    assert context["budgets"][0]["limit"] == 35
    assert context["continuation"]["resume_command"] is None
    assert (
        "No valid resumable checkpoint" in context["continuation"]["checkpoint_message"]
    )


def test_checkpoint_separates_planned_executed_deferred_and_unfinished_queries(
    tmp_path,
) -> None:
    run_dir = _populated_run(tmp_path)
    state = CrawlProjectState(
        run_id=run_dir.name,
        topic_query="urban heat adaptation",
        language="en",
        start_date=date(2022, 1, 1),
        gl="us",
        hl="en",
        transcript_excerpt_chars=12_000,
        max_depth=1,
        max_queries=2,
        max_search_pages=2,
        max_channel_pages=1,
        expansion={},
        classifier_system_prompt="Classify.",
        prompt_sha256="4d553b0b48b4bb151321bd8aa5d9904477ecfdbec8dea4f2554cef83409c8ae6",
        planned_queries=[
            {"text": "query one", "kind": "seed"},
            {"text": "query two", "kind": "interview"},
            {"text": "query three", "kind": "expanded"},
            {"text": "query four", "kind": "channel"},
        ],
        budget=BudgetState(
            max_credits=20,
            transcript_capacity=12,
            discovery_spent=1,
            transcript_spent=1,
        ),
        query_progress={
            "query one": PageProgress(pages_completed=1, exhausted=True),
            "query two": PageProgress(pages_completed=1, exhausted=False),
        },
        last_status="completed",
    )
    ProjectStateStore(run_dir).save(state)

    context = build_dashboard_context(build_dashboard_data(run_dir), run_dir)

    assert context["metrics"]["queries"] == 4
    assert context["continuation"]["executed_queries"] == 2
    assert context["continuation"]["deferred_queries"] == 2
    assert context["continuation"]["unfinished_queries"] == 1
    assert context["continuation"]["controls"]["max_search_pages"] == 2
    assert "--max-queries 3" in context["continuation"]["resume_command"]
    assert (
        "widens scope using the remaining SearchAPI credits"
        in context["continuation"]["recommendation_reason"]
    )

    html = render_dashboard(run_dir).read_text(encoding="utf-8")
    assert "Expand this completed project" in html
    assert '4</p><p class="stat__hint">planned variants' in html
    assert "Executed queries: 2" in html
    assert "Held by max_queries: 2" in html
    assert "Unfinished within current limits: 1" in html


def test_active_checkpoint_prompt_provenance_overrides_historical_config(
    tmp_path,
) -> None:
    run_dir = _populated_run(tmp_path)
    prompt = "Active binary classifier prompt."
    active_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    state = CrawlProjectState(
        run_id=run_dir.name,
        topic_query="urban heat adaptation",
        language="en",
        start_date=date(2022, 1, 1),
        gl="us",
        hl="en",
        transcript_excerpt_chars=12_000,
        max_depth=1,
        max_queries=2,
        max_search_pages=2,
        max_channel_pages=1,
        expansion={},
        classifier_system_prompt=prompt,
        prompt_sha256=active_hash,
        classifier_prompt_version="relevance-v2",
        planned_queries=[
            {"text": "query one", "kind": "seed"},
            {"text": "query two", "kind": "interview"},
        ],
        budget=BudgetState(
            max_credits=20,
            transcript_capacity=12,
            discovery_spent=1,
            transcript_spent=1,
        ),
        last_status="completed",
    )
    ProjectStateStore(run_dir).save(state)

    data = build_dashboard_data(run_dir)
    historical_sessions = list(data.sessions)
    context = build_dashboard_context(data, run_dir)

    assert data.config["prompt_version"] == "relevance-v1"
    assert data.config["prompt_sha256"] == "a" * 64
    assert context["run"]["prompt_version"] == "relevance-v2"
    assert context["run"]["prompt_sha256"] == active_hash
    assert context["continuation"]["sessions"] == historical_sessions

    html = render_dashboard(run_dir).read_text(encoding="utf-8")
    assert "prompt relevance-v2" in html
    assert active_hash in html


def test_dashboard_recommendations_match_frontier_and_aggregate_credit_state(
    tmp_path,
) -> None:
    controls = {
        "max_depth": 1,
        "max_queries": 2,
        "max_search_pages": 1,
        "max_channel_pages": 1,
    }
    completed = _suggested_resume_command(
        tmp_path / "complete",
        status="completed",
        controls=controls,
        available_credits=7,
        lifetime_grant=12,
        planned_queries=4,
        deferred_queries=2,
    )
    stopped = _suggested_resume_command(
        tmp_path / "stopped",
        status="stopped_budget",
        controls=controls,
        available_credits=7,
        lifetime_grant=12,
        planned_queries=4,
        deferred_queries=2,
    )
    failed = _suggested_resume_command(
        tmp_path / "failed",
        status="failed",
        controls=controls,
        available_credits=0,
        lifetime_grant=12,
        planned_queries=4,
        deferred_queries=2,
    )

    assert "--add-credits 0" in completed
    assert "--max-queries 3" in completed
    assert "--add-credits 0" in stopped
    assert "--max-" not in stopped
    assert "--add-credits 6" in failed
    assert "--max-" not in failed
    assert "before widening it" in _continuation_reason(
        status="stopped_budget", available_credits=7, has_command=True
    )


def test_dashboard_completed_limits_explain_when_no_scope_can_expand(tmp_path) -> None:
    command = _suggested_resume_command(
        tmp_path / "complete-limits",
        status="completed",
        controls={
            "max_depth": 5,
            "max_queries": 3,
            "max_search_pages": 10,
            "max_channel_pages": 10,
        },
        available_credits=7,
        lifetime_grant=12,
        planned_queries=3,
        deferred_queries=0,
    )

    assert command is None
    assert "adding credits alone cannot create more work" in _continuation_reason(
        status="completed", available_credits=7, has_command=False
    )


def test_custom_template_receives_dashboard_and_top_level_context(tmp_path) -> None:
    run_dir = _populated_run(tmp_path)
    template = tmp_path / "template.html"
    template.write_text(
        "<p>{{ dashboard.run_id }}:{{ status }}:{{ videos|length }}</p>",
        encoding="utf-8",
    )

    output = render_dashboard(
        run_dir,
        tmp_path / "custom.html",
        template_path=template,
    )

    assert output.read_text(encoding="utf-8") == ("<p>run-dashboard:completed:3</p>")
