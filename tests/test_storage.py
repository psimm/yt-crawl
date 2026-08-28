import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from yt_searchapi.records import (
    BudgetAction,
    BudgetEventRecord,
    BudgetKind,
    DiscoveryEdgeRecord,
    InterviewAnswerRecord,
    InterviewExampleKind,
    QueryRecord,
    RelevanceDecisionRecord,
    RelevanceLabel,
    RunConfigRecord,
    RunMetricRecord,
    TranscriptRecord,
    TranscriptSegment,
    validate_run_record,
)
from yt_searchapi.settings import (
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)
from yt_searchapi.storage import JsonlRunWriter, RunIdMismatchError


def config_record(run_id: str = "run-001") -> RunConfigRecord:
    return RunConfigRecord(
        run_id=run_id,
        topic_query="urban heat adaptation",
        expanded_queries=("urban heat adaptation interview",),
        language="en",
        start_date=date(2020, 1, 1),
        max_searchapi_credits=100,
        transcript_reserve_credits=60,
        model="gpt-5.6-luna",
        prompt_version="v1",
    )


def test_resume_config_can_record_a_fully_transferred_grant() -> None:
    payload = config_record().model_dump()
    payload.update(
        max_searchapi_credits=4,
        transcript_reserve_credits=4,
        session_action="resume",
    )

    record = RunConfigRecord(**payload)

    assert record.transcript_reserve_credits == record.max_searchapi_credits


def test_run_config_uses_configured_parallelism_defaults() -> None:
    record = config_record()

    assert record.searchapi_workers == DEFAULT_SEARCHAPI_WORKERS
    assert record.searchapi_retries == DEFAULT_SEARCHAPI_RETRIES
    assert record.llm_workers == DEFAULT_LLM_WORKERS


def test_start_config_requires_a_discovery_credit() -> None:
    payload = config_record().model_dump()
    payload.update(max_searchapi_credits=4, transcript_reserve_credits=4)

    with pytest.raises(ValidationError, match="new project needs at least one"):
        RunConfigRecord(**payload)


def test_legacy_llm_budget_records_remain_readable_but_are_not_rewritten() -> None:
    config_payload = config_record().model_dump(mode="json")
    config_payload["llm_discovery_tokens"] = 10_000
    config_payload["llm_transcript_tokens"] = 30_000

    parsed_config = validate_run_record(config_payload)

    assert isinstance(parsed_config, RunConfigRecord)
    assert "llm_discovery_tokens" not in parsed_config.model_dump()
    assert "llm_transcript_tokens" not in parsed_config.model_dump()

    legacy_event = BudgetEventRecord(
        run_id="run-001",
        budget_kind=BudgetKind.LEGACY_LLM_TOKENS,
        pool="discovery",
        action=BudgetAction.SPENT,
        amount=100,
        remaining=900,
        purpose="legacy",
    )
    parsed_event = validate_run_record(legacy_event.model_dump(mode="json"))
    assert isinstance(parsed_event, BudgetEventRecord)
    assert parsed_event.budget_kind is BudgetKind.LEGACY_LLM_TOKENS


def test_writer_appends_without_overwriting_and_splits_record_types(tmp_path) -> None:
    writer = JsonlRunWriter(tmp_path, "run-001")
    config_path = writer.append(config_record())
    writer.append(config_record())
    answer_path = writer.append(
        InterviewAnswerRecord(
            run_id="run-001",
            question_id="include-1",
            question_text="Should practitioner interviews be included?",
            answer="Yes, when they describe implemented interventions.",
            example_kind=InterviewExampleKind.POSITIVE,
        )
    )

    assert config_path.name == "run_config.jsonl"
    assert answer_path.name == "interview_answer.jsonl"
    assert len(config_path.read_text(encoding="utf-8").splitlines()) == 2
    assert {path.name for path in writer.jsonl_paths()} == {
        "interview_answer.jsonl",
        "run_config.jsonl",
    }


def test_query_plans_and_discovery_edges_preserve_unexecuted_lineage(tmp_path) -> None:
    writer = JsonlRunWriter(tmp_path, "run-001")
    writer.append(
        QueryRecord(
            run_id="run-001",
            query="urban heat adaptation interview",
            query_kind="interview",
            status="deferred",
        )
    )
    writer.append(
        DiscoveryEdgeRecord(
            run_id="run-001",
            video_id="video-1",
            source_type="search",
            source_ref="urban heat adaptation",
            depth=0,
        )
    )

    assert (writer.run_dir / "query.jsonl").exists()
    edge = json.loads(
        (writer.run_dir / "discovery_edge.jsonl").read_text(encoding="utf-8")
    )
    assert edge["video_id"] == "video-1"
    assert edge["depth"] == 0


def test_discovery_edge_requires_exactly_one_target() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        DiscoveryEdgeRecord(
            run_id="run-001",
            video_id="video-1",
            channel_id="channel-1",
            source_type="related",
            source_ref="video-0",
            depth=1,
        )


def test_every_line_is_json_and_validates_as_a_discriminated_record(tmp_path) -> None:
    writer = JsonlRunWriter(tmp_path, "run-001")
    writer.append(config_record())

    line = (
        (writer.run_dir / "run_config.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    decoded = json.loads(line)
    parsed = validate_run_record(decoded)

    assert isinstance(parsed, RunConfigRecord)
    assert parsed.recorded_at.tzinfo is not None
    assert parsed.schema_version == "1"


def test_writer_rejects_cross_run_records_and_unsafe_ids(tmp_path) -> None:
    writer = JsonlRunWriter(tmp_path, "run-001")

    with pytest.raises(RunIdMismatchError):
        writer.append(config_record("run-002"))
    with pytest.raises(ValueError):
        JsonlRunWriter(tmp_path, "../escape")
    with pytest.raises(ValueError):
        writer.path_for("../../escape")


def test_record_schemas_forbid_unknown_fields() -> None:
    payload = config_record().model_dump()
    payload["api_key"] = "must-not-be-persisted"

    with pytest.raises(ValidationError):
        RunConfigRecord.model_validate(payload)


def test_legacy_relevance_decision_is_compacted_during_validation() -> None:
    compact = validate_run_record(
        {
            "schema_version": "1",
            "run_id": "run-001",
            "record_type": "relevance_decision",
            "decision_id": "decision-1",
            "video_id": "video-1",
            "label": "irrelevant",
            "decision_point": "video_metadata",
            "reason": "irrelevant: outside the research scope",
            "confidence": 0.99,
            "requested_language": "en",
            "detected_language": "en",
            "language_matches": True,
            "published_after_start_date": True,
            "criteria": [],
            "positive_example_ids": [],
            "negative_example_ids": [],
            "model": "gpt-5.6-luna",
            "prompt_version": "relevance-v3",
            "prompt_sha256": "a" * 64,
            "llm_input_tokens": 100,
            "llm_output_tokens": 20,
            "transcript_reserved": False,
        }
    )

    assert compact.primary_reason == "irrelevant: outside the research scope"
    assert "confidence" not in compact.model_dump()
    assert "criteria" not in compact.model_dump()


def test_relevant_decision_requires_transcript_reservation() -> None:
    with pytest.raises(ValidationError, match="reserve transcript credits"):
        RelevanceDecisionRecord(
            run_id="run-001",
            decision_id="decision-1",
            video_id="video-1",
            label=RelevanceLabel.RELEVANT,
            decision_point="video_metadata",
            primary_reason="topic_match",
            requested_language="en",
            detected_language="en",
            language_matches=True,
            model="gpt-5.6-luna",
            prompt_version="v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=200,
            llm_output_tokens=30,
            transcript_reserved=False,
        )


def test_needs_transcript_lifecycle_record_requires_reservation() -> None:
    with pytest.raises(ValidationError, match="reserve transcript credits"):
        RelevanceDecisionRecord(
            run_id="run-001",
            decision_id="decision-deferred",
            video_id="video-1",
            label=RelevanceLabel.NEEDS_TRANSCRIPT,
            decision_point="video_metadata",
            primary_reason="topic_match",
            requested_language="en",
            language_matches=False,
            model="gpt-5.6-luna",
            prompt_version="v1",
            prompt_sha256="a" * 64,
            llm_input_tokens=200,
            llm_output_tokens=30,
            transcript_reserved=False,
        )


def test_transcript_availability_is_internally_consistent() -> None:
    transcript = TranscriptRecord(
        run_id="run-001",
        video_id="video-1",
        requested_language="en",
        language="en",
        transcript_type="auto",
        is_available=True,
        segments=(
            TranscriptSegment(
                text="First segment.", start_seconds=0, duration_seconds=1
            ),
            TranscriptSegment(
                text="Second segment.", start_seconds=1, duration_seconds=2
            ),
        ),
        searchapi_credits=1,
    )
    assert transcript.text == "First segment. Second segment."

    with pytest.raises(ValidationError):
        TranscriptRecord(
            run_id="run-001",
            video_id="video-2",
            requested_language="en",
            is_available=False,
            searchapi_credits=1,
        )


def test_threaded_appends_produce_complete_lines(tmp_path) -> None:
    writer = JsonlRunWriter(tmp_path, "run-001")

    def append_metric(index: int) -> None:
        writer.append(
            RunMetricRecord(
                run_id="run-001",
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
                name="candidate_seen",
                value=index,
                unit="count",
            )
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(append_metric, range(100)))

    lines = (
        (writer.run_dir / "run_metric.jsonl").read_text(encoding="utf-8").splitlines()
    )
    decoded = [json.loads(line) for line in lines]
    assert len(decoded) == 100
    assert {item["value"] for item in decoded} == set(range(100))
