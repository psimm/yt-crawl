from datetime import UTC, date, datetime

from yt_searchapi.analysis import DuckDBAnalytics
from yt_searchapi.records import (
    DecisionPoint,
    DiscoverySource,
    QueryKind,
    QueryRecord,
    QueryStatus,
    RelevanceDecisionRecord,
    RelevanceLabel,
    RunConfigRecord,
    RunStatus,
    RunStatusRecord,
    TranscriptRecord,
    TranscriptSegment,
    VideoCandidateRecord,
)
from yt_searchapi.storage import JsonlRunWriter


def test_analysis_exposes_topics_metadata_and_live_filters(tmp_path):
    run_id = "analysis-fixture"
    writer = JsonlRunWriter(tmp_path, run_id)
    timestamp = datetime(2026, 8, 1, tzinfo=UTC)

    writer.append(
        RunConfigRecord(
            run_id=run_id,
            topic_query="personal finance",
            expanded_queries=("ETF Sparplan Rente",),
            language="de",
            start_date=date(2025, 1, 1),
            max_searchapi_credits=10,
            transcript_reserve_credits=5,
            model="fixture",
            prompt_version="fixture-v1",
        )
    )
    writer.append(
        RunStatusRecord(
            run_id=run_id, status=RunStatus.COMPLETED, recorded_at=timestamp
        )
    )
    writer.append(
        QueryRecord(
            run_id=run_id,
            query="ETF Sparplan Rente",
            query_kind=QueryKind.EXPANDED,
            status=QueryStatus.EXECUTED,
            recorded_at=timestamp,
        )
    )
    writer.append(
        VideoCandidateRecord(
            run_id=run_id,
            video_id="fixture-video",
            title="ETF Sparplan für die Rente",
            url="https://www.youtube.com/watch?v=fixture-video",
            description="Wie du fürs Alter investierst und dein Depot aufbaust.",
            channel_id="fixture-channel",
            channel_title="Fixture Finance",
            published_at=datetime(2026, 2, 1, tzinfo=UTC),
            discovered_via=DiscoverySource.SEARCH,
            discovery_query="ETF Sparplan Rente",
            views=1234,
            likes=56,
            keywords=("ETF", "Sparplan", "Rente"),
            recorded_at=timestamp,
        )
    )
    writer.append(
        RelevanceDecisionRecord(
            run_id=run_id,
            decision_id="fixture-decision",
            video_id="fixture-video",
            label=RelevanceLabel.RELEVANT,
            decision_point=DecisionPoint.TRANSCRIPT,
            primary_reason="topic_match",
            requested_language="de",
            detected_language="de",
            language_matches=True,
            model="fixture",
            prompt_version="fixture-v1",
            prompt_sha256="0" * 64,
            llm_input_tokens=0,
            llm_output_tokens=0,
            transcript_reserved=True,
            recorded_at=timestamp,
        )
    )
    writer.append(
        TranscriptRecord(
            run_id=run_id,
            video_id="fixture-video",
            requested_language="de",
            language="de",
            transcript_type="manual",
            is_available=True,
            segments=(
                TranscriptSegment(
                    text="Ein ETF Sparplan hilft bei der Altersvorsorge.",
                    start_seconds=0,
                    duration_seconds=5,
                ),
            ),
            searchapi_credits=1,
            recorded_at=timestamp,
        )
    )

    analytics = DuckDBAnalytics(writer.run_dir)
    snapshot = analytics.snapshot()

    assert snapshot["run"]["firstPublished"].startswith("2026-02-01")
    assert snapshot["summary"]["candidates"] == 1
    assert snapshot["summary"]["total_views"] == 1234
    assert snapshot["summary"]["total_likes"] == 56
    assert snapshot["summary"]["transcripts_available"] == 1
    assert (
        next(
            topic
            for topic in snapshot["topics"]
            if topic["topic_id"] == "investing_etfs"
        )["videos"]
        == 1
    )
    assert (
        analytics.videos({"topic": ["investing_etfs"], "limit": ["10"]})["total"] == 1
    )
    assert snapshot["queries"] == [
        {
            "query": "ETF Sparplan Rente",
            "kind": "expanded",
            "status": "executed",
            "videos": 1,
            "accepted": 1,
        }
    ]
