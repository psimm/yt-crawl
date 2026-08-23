# ruff: noqa: E501
"""Offline DuckDB-backed dashboard generation for one crawler run.

The crawler deliberately persists small, append-only JSONL streams rather than a
mutable database.  This module treats those streams as the source of truth,
normalizes them in an in-memory DuckDB database, and writes a self-contained HTML
snapshot.  It never contacts SearchAPI, OpenAI, a CDN, or any other network
service.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb
from jinja2 import Environment, StrictUndefined, select_autoescape

from yt_searchapi.state import ProjectStateStore, checkpoint_recovery_message

_JSONL_RECORD_TYPES = (
    "run_config",
    "run_status",
    "interview_answer",
    "query",
    "discovery_edge",
    "channel",
    "video_candidate",
    "relevance_decision",
    "transcript",
    "api_call",
    "budget_event",
    "run_metric",
    "run_error",
)


@dataclass(frozen=True, slots=True)
class DashboardData:
    """Serializable, presentation-ready data for a single crawler run."""

    run_id: str
    generated_at: str
    status: str
    status_reason: str | None
    started_at: str | None
    completed_at: str | None
    config: dict[str, Any]
    summary: tuple[dict[str, Any], ...]
    run_metrics: tuple[dict[str, Any], ...]
    discovery_sources: tuple[dict[str, Any], ...]
    videos: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    transcripts: tuple[dict[str, Any], ...]
    transcript_segments: tuple[dict[str, Any], ...]
    queries: tuple[dict[str, Any], ...]
    channels: tuple[dict[str, Any], ...]
    interview_examples: tuple[dict[str, Any], ...]
    api_operations: tuple[dict[str, Any], ...]
    budget_pools: tuple[dict[str, Any], ...]
    sessions: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]

    def template_context(self) -> dict[str, Any]:
        """Return plain Python containers suitable for Jinja templates."""

        return asdict(self)


def build_dashboard_data(run_dir: str | Path) -> DashboardData:
    """Load and transform one run directory without making network requests.

    Missing record files are treated as empty tables.  A directory containing
    records from more than one run is rejected because combining their budgets
    and decisions would produce misleading metrics.
    """

    directory = _validated_run_dir(run_dir)
    connection = duckdb.connect(database=":memory:")
    try:
        _ingest_jsonl(connection, directory)
        _create_normalized_views(connection)
        return _query_dashboard_data(connection, directory)
    finally:
        connection.close()


def render_dashboard(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    template_path: str | Path | None = None,
) -> Path:
    """Render a self-contained HTML dashboard and return its absolute path.

    When ``template_path`` is omitted, a built-in offline template is used.
    Custom templates receive every :class:`DashboardData` field as a top-level
    variable and also receive the complete mapping as ``dashboard``.
    """

    directory = _validated_run_dir(run_dir)
    data = build_dashboard_data(directory)
    destination = (
        directory / "dashboard.html"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    raw_context = data.template_context()
    if template_path is not None:
        source = Path(template_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"dashboard template does not exist: {source}")
        template_text = source.read_text(encoding="utf-8")
        template_context = {
            "dashboard": raw_context,
            **raw_context,
        }
    else:
        template_text = (
            files("yt_searchapi")
            .joinpath("templates", "dashboard.html.j2")
            .read_text(encoding="utf-8")
        )
        template_context = {
            "dashboard": build_dashboard_context(data, directory),
        }

    environment = Environment(
        autoescape=select_autoescape(default_for_string=True, default=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = environment.from_string(template_text).render(**template_context)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def build_dashboard_context(
    data: DashboardData,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Adapt normalized run data to the polished template's nested shape."""

    directory = _validated_run_dir(run_dir)
    relevant_videos = [video for video in data.videos if _is_final_relevant(video)]
    irrelevant_videos = [
        video for video in data.videos if video.get("final_label") == "irrelevant"
    ]
    other_videos = [
        video
        for video in data.videos
        if video.get("final_label") not in {"relevant", "irrelevant", None}
    ]
    presentation_videos = {
        str(video["video_id"]): _presentation_video(video)
        for video in (*relevant_videos, *irrelevant_videos, *other_videos)
    }
    relevant_ids = {str(video["video_id"]) for video in relevant_videos}
    transcript_by_video = {
        str(transcript["video_id"]): transcript for transcript in data.transcripts
    }

    available_relevant = sum(
        transcript_by_video.get(video_id, {}).get("is_available") is True
        for video_id in relevant_ids
    )
    eligible_transcripts = len(relevant_ids)
    transcript_items = [
        _presentation_transcript_item(
            video_id,
            presentation_videos[video_id]["title"],
            transcript_by_video.get(video_id),
        )
        for video_id in sorted(
            relevant_ids,
            key=lambda item: presentation_videos[item]["title"].casefold(),
        )
    ]

    query_rows = _presentation_queries(data, relevant_ids)
    provenance = _presentation_provenance(data.discovery_sources)
    checkpoint = _checkpoint_summary(directory)
    budgets = _presentation_budgets(data, checkpoint)
    checkpoint_status = str(checkpoint["last_status"]) if checkpoint else data.status
    active_prompt_version = (
        checkpoint.get("classifier_prompt_version") if checkpoint else None
    ) or data.config.get("prompt_version")
    active_prompt_sha256 = (
        checkpoint.get("prompt_sha256") if checkpoint else None
    ) or data.config.get("prompt_sha256")
    display_status = (
        "ready_to_resume" if checkpoint_status == "prepared" else checkpoint_status
    )
    evaluated = len(relevant_videos) + len(irrelevant_videos)
    candidates = len(data.videos)
    available_total = sum(
        transcript.get("is_available") is True for transcript in data.transcripts
    )
    funnel_counts = (
        ("Discovered", candidates, "unique candidate videos"),
        ("Evaluated", evaluated, "with a final decision"),
        ("Relevant", len(relevant_videos), "accepted after full review"),
        ("Transcribed", available_relevant, "available for accepted videos"),
    )
    funnel = [
        {
            "label": label,
            "count": count,
            "percent": _bounded_percent(count, candidates),
            "hint": hint,
        }
        for label, count, hint in funnel_counts
    ]

    cache_hits = sum(int(row.get("cache_hits") or 0) for row in data.api_operations)
    pending_videos = _latest_metric(data.run_metrics, "pending_videos")
    if checkpoint:
        pending_videos = checkpoint["pending_videos"]
    controls = (
        dict(checkpoint["controls"])
        if checkpoint
        else {
            name: int(data.config.get(name) or 0)
            for name in (
                "max_depth",
                "max_queries",
                "max_search_pages",
                "max_channel_pages",
            )
        }
    )
    query_counts = _presentation_query_counts(query_rows, checkpoint)
    resume_command = (
        _suggested_resume_command(
            directory,
            status=checkpoint_status,
            controls=controls,
            available_credits=int(budgets[0]["remaining"]) if budgets else 0,
            lifetime_grant=int(budgets[0]["limit"]) if budgets else 0,
            planned_queries=query_counts["planned"],
            deferred_queries=query_counts["deferred"],
        )
        if checkpoint
        else None
    )
    continuation_reason = _continuation_reason(
        status=checkpoint_status,
        available_credits=int(budgets[0]["remaining"]) if budgets else 0,
        has_command=resume_command is not None,
    )

    return {
        "generated_at": data.generated_at,
        "run": {
            "id": data.run_id,
            "status": display_status,
            "topic_query": data.config.get("topic_query"),
            "language": data.config.get("language"),
            "start_date": data.config.get("start_date"),
            "started_at": data.started_at,
            "completed_at": data.completed_at,
            "duration": _duration_between(data.started_at, data.completed_at),
            "model": data.config.get("model"),
            "prompt_version": active_prompt_version,
            "prompt_sha256": active_prompt_sha256,
            "status_reason": (
                "Resume configuration committed; crawler setup has not started."
                if checkpoint_status == "prepared"
                else data.status_reason
            ),
            "sessions": len(data.sessions),
        },
        "metrics": {
            "queries": query_counts["planned"],
            "channels": len(data.channels),
            "candidates": candidates,
            "relevant": len(relevant_videos),
            "irrelevant": len(irrelevant_videos),
            "transcripts": available_total,
            "cache_hits": cache_hits,
            "pending": int(pending_videos or 0),
            "errors": len(data.errors),
        },
        "budgets": budgets,
        "continuation": {
            "action": (
                "recover"
                if not checkpoint
                else ("expand" if checkpoint_status == "completed" else "resume")
            ),
            "controls": controls,
            "resume_command": resume_command,
            "recommendation_reason": continuation_reason,
            "sessions": list(data.sessions),
            "deferred_videos": (
                checkpoint["deferred_videos"] if checkpoint else len(other_videos)
            ),
            "executed_queries": query_counts["executed"],
            "deferred_queries": query_counts["deferred"],
            "unfinished_queries": query_counts["unfinished"],
            "checkpoint_available": bool(checkpoint),
            "checkpoint_message": (
                None if checkpoint else checkpoint_recovery_message(directory)
            ),
        },
        "funnel": funnel,
        "provenance": provenance,
        "queries": query_rows,
        "relevant_videos": [
            presentation_videos[str(video["video_id"])] for video in relevant_videos
        ],
        "irrelevant_videos": [
            presentation_videos[str(video["video_id"])] for video in irrelevant_videos
        ],
        "other_videos": [
            presentation_videos[str(video["video_id"])] for video in other_videos
        ],
        "transcript": {
            "eligible": eligible_transcripts,
            "available": available_relevant,
            "unavailable": eligible_transcripts - available_relevant,
            "coverage_percent": _bounded_percent(
                available_relevant, eligible_transcripts
            ),
            "segments": sum(item["segments"] for item in transcript_items),
            "words": sum(item["words"] for item in transcript_items),
            "items": transcript_items,
        },
        "errors": [
            {
                "recorded_at": error.get("recorded_at"),
                "stage": error.get("stage"),
                "message": error.get("message"),
                "ref": error.get("video_id") or error.get("channel_id"),
            }
            for error in data.errors
        ],
        "paths": {
            "run_dir": _display_path(directory),
            "database": None,
            "jsonl_files": _jsonl_file_inventory(directory),
        },
    }


def _validated_run_dir(run_dir: str | Path) -> Path:
    directory = Path(run_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"run directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"run path is not a directory: {directory}")
    return directory


def _ingest_jsonl(connection: duckdb.DuckDBPyConnection, run_dir: Path) -> None:
    connection.execute(
        """
        CREATE TABLE raw_records (
            source_file VARCHAR NOT NULL,
            source_line BIGINT NOT NULL,
            payload JSON NOT NULL
        )
        """
    )
    for record_type in _JSONL_RECORD_TYPES:
        path = run_dir / f"{record_type}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        connection.execute(
            """
            INSERT INTO raw_records
            SELECT ?, row_number() OVER (), json
            FROM read_json_objects_auto(?, format = 'newline_delimited')
            """,
            [path.name, str(path)],
        )


def _create_normalized_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE VIEW records AS
        SELECT
            source_file,
            source_line,
            json_extract_string(payload, '$.record_type') AS record_type,
            json_extract_string(payload, '$.run_id') AS run_id,
            json_extract_string(payload, '$.recorded_at') AS recorded_at,
            payload
        FROM raw_records
        """
    )
    connection.execute(
        """
        CREATE VIEW video_candidates AS
        SELECT
            run_id,
            source_line,
            recorded_at,
            json_extract_string(payload, '$.video_id') AS video_id,
            json_extract_string(payload, '$.title') AS title,
            coalesce(
                json_extract_string(payload, '$.url'),
                json_extract_string(payload, '$.raw_payload.link')
            ) AS url,
            json_extract_string(payload, '$.description') AS description,
            json_extract_string(payload, '$.channel_id') AS channel_id,
            json_extract_string(payload, '$.channel_title') AS channel_title,
            json_extract_string(payload, '$.published_at') AS published_at,
            try_cast(json_extract_string(payload, '$.duration_seconds') AS BIGINT)
                AS duration_seconds,
            coalesce(
                try_cast(json_extract_string(payload, '$.views') AS BIGINT),
                try_cast(json_extract_string(payload, '$.raw_payload.views') AS BIGINT)
            ) AS views,
            coalesce(
                try_cast(json_extract_string(payload, '$.likes') AS BIGINT),
                try_cast(json_extract_string(payload, '$.raw_payload.likes') AS BIGINT)
            ) AS likes,
            coalesce(
                json_extract_string(payload, '$.category'),
                json_extract_string(payload, '$.raw_payload.category')
            ) AS category,
            coalesce(
                json_extract(payload, '$.keywords'),
                json_extract(payload, '$.raw_payload.keywords')
            ) AS keywords,
            coalesce(
                json_extract_string(payload, '$.thumbnail'),
                json_extract_string(payload, '$.raw_payload.thumbnail'),
                json_extract_string(payload, '$.raw_payload.thumbnail.static')
            ) AS thumbnail,
            coalesce(
                try_cast(json_extract_string(payload, '$.is_live_content') AS BOOLEAN),
                try_cast(
                    json_extract_string(payload, '$.raw_payload.is_live_content')
                    AS BOOLEAN
                )
            ) AS is_live_content,
            json_extract_string(payload, '$.discovered_via') AS discovered_via,
            json_extract_string(payload, '$.discovered_from_id')
                AS discovered_from_id,
            json_extract_string(payload, '$.discovery_query') AS discovery_query
        FROM records
        WHERE record_type = 'video_candidate'
        """
    )
    connection.execute(
        """
        CREATE VIEW decisions AS
        SELECT
            run_id,
            source_line,
            recorded_at,
            json_extract_string(payload, '$.decision_id') AS decision_id,
            json_extract_string(payload, '$.video_id') AS video_id,
            json_extract_string(payload, '$.label') AS label,
            json_extract_string(payload, '$.decision_point') AS decision_point,
            json_extract_string(payload, '$.reason') AS reason,
            try_cast(json_extract_string(payload, '$.confidence') AS DOUBLE)
                AS confidence,
            json_extract_string(payload, '$.requested_language')
                AS requested_language,
            json_extract_string(payload, '$.detected_language') AS detected_language,
            try_cast(json_extract_string(payload, '$.language_matches') AS BOOLEAN)
                AS language_matches,
            try_cast(
                json_extract_string(payload, '$.published_after_start_date') AS BOOLEAN
            ) AS published_after_start_date,
            json_extract(payload, '$.criteria') AS criteria,
            json_extract_string(payload, '$.model') AS model,
            json_extract_string(payload, '$.prompt_version') AS prompt_version,
            try_cast(json_extract_string(payload, '$.llm_input_tokens') AS BIGINT)
                AS llm_input_tokens,
            try_cast(json_extract_string(payload, '$.llm_output_tokens') AS BIGINT)
                AS llm_output_tokens,
            try_cast(json_extract_string(payload, '$.transcript_reserved') AS BOOLEAN)
                AS transcript_reserved
        FROM records
        WHERE record_type = 'relevance_decision'
        """
    )
    connection.execute(
        """
        CREATE VIEW transcript_rows AS
        SELECT
            run_id,
            source_line,
            recorded_at,
            json_extract_string(payload, '$.video_id') AS video_id,
            json_extract_string(payload, '$.requested_language')
                AS requested_language,
            json_extract_string(payload, '$.language') AS language,
            json_extract_string(payload, '$.transcript_type') AS transcript_type,
            try_cast(json_extract_string(payload, '$.is_available') AS BOOLEAN)
                AS is_available,
            json_extract_string(payload, '$.unavailable_reason')
                AS unavailable_reason,
            json_extract_string(payload, '$.source_request_id') AS source_request_id,
            coalesce(
                try_cast(json_extract_string(payload, '$.searchapi_credits') AS BIGINT),
                0
            ) AS searchapi_credits,
            json_extract(payload, '$.segments') AS segments
        FROM records
        WHERE record_type = 'transcript'
        """
    )
    connection.execute(
        """
        CREATE VIEW transcript_segments AS
        SELECT
            transcript.run_id,
            transcript.video_id,
            transcript.source_line AS transcript_source_line,
            try_cast(segment.key AS BIGINT) AS segment_index,
            json_extract_string(segment.value, '$.text') AS text,
            try_cast(json_extract_string(segment.value, '$.start_seconds') AS DOUBLE)
                AS start_seconds,
            try_cast(json_extract_string(segment.value, '$.duration_seconds') AS DOUBLE)
                AS duration_seconds
        FROM transcript_rows AS transcript
        CROSS JOIN LATERAL json_each(transcript.segments) AS segment
        """
    )


def _query_dashboard_data(
    connection: duckdb.DuckDBPyConnection, run_dir: Path
) -> DashboardData:
    run_ids = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT run_id FROM records WHERE run_id IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]
    if len(run_ids) > 1:
        raise ValueError(
            "dashboard run directory contains multiple run IDs: " + ", ".join(run_ids)
        )
    run_id = run_ids[0] if run_ids else run_dir.name

    config = _query_config(connection)
    status_row = _fetch_one_dict(
        connection,
        """
        SELECT
            json_extract_string(payload, '$.status') AS status,
            json_extract_string(payload, '$.reason') AS reason
        FROM records
        WHERE record_type = 'run_status'
        ORDER BY source_line DESC
        LIMIT 1
        """,
    )
    status = str(status_row.get("status") or "unknown")
    status_reason = _optional_string(status_row.get("reason"))
    timing = _fetch_one_dict(
        connection,
        """
        WITH status_rows AS (
            SELECT
                source_line,
                recorded_at,
                json_extract_string(payload, '$.status') AS status
            FROM records
            WHERE record_type = 'run_status'
        ), latest_start AS (
            SELECT max(source_line) AS source_line
            FROM status_rows
            WHERE status = 'started'
        )
        SELECT
            max(recorded_at) FILTER (
                WHERE status = 'started'
                AND source_line = (SELECT source_line FROM latest_start)
            ) AS started_at,
            max(recorded_at) FILTER (
                WHERE status IN ('completed', 'stopped_budget', 'failed')
                AND source_line > (SELECT source_line FROM latest_start)
            ) AS completed_at
        FROM status_rows
        """,
    )

    run_metrics = tuple(
        _query_dicts(
            connection,
            """
            SELECT name, metric_value AS value, unit
            FROM (
                SELECT
                    json_extract_string(payload, '$.name') AS name,
                    json_extract(payload, '$.value') AS metric_value,
                    json_extract_string(payload, '$.unit') AS unit,
                    row_number() OVER (
                        PARTITION BY json_extract_string(payload, '$.name')
                        ORDER BY source_line DESC
                    ) AS rank
                FROM records
                WHERE record_type = 'run_metric'
            )
            WHERE rank = 1
            ORDER BY name
            """,
            json_fields={"value"},
        )
    )
    decisions = tuple(_query_decisions(connection))
    transcripts, transcript_segments = _query_transcripts(connection)
    videos = tuple(_query_videos(connection))
    api_operations = tuple(_query_api_operations(connection))
    budget_pools = tuple(_query_budget_pools(connection, config))
    sessions = tuple(_query_sessions(connection))
    summary = tuple(
        _query_summary(
            connection,
            videos=videos,
            transcripts=transcripts,
            api_operations=api_operations,
            budget_pools=budget_pools,
        )
    )

    discovery_sources = tuple(
        _query_dicts(
            connection,
            """
            WITH source_edges AS (
                SELECT DISTINCT
                    coalesce(
                        json_extract_string(payload, '$.source_type'), 'unknown'
                    ) AS source,
                    json_extract_string(payload, '$.video_id') AS video_id
                FROM records
                WHERE record_type = 'discovery_edge'
                    AND json_extract_string(payload, '$.video_id') IS NOT NULL
            ), final_decisions AS (
                SELECT video_id, label, decision_point
                FROM (
                    SELECT
                        video_id,
                        label,
                        decision_point,
                        row_number() OVER (
                            PARTITION BY video_id ORDER BY source_line DESC
                        ) AS rank
                    FROM decisions
                )
                WHERE rank = 1
            ), source_videos AS (
                SELECT
                    source_edges.source,
                    source_edges.video_id,
                    coalesce(
                        final_decisions.label = 'relevant'
                        AND final_decisions.decision_point = 'transcript',
                        false
                    ) AS is_relevant
                FROM source_edges
                LEFT JOIN final_decisions USING (video_id)
            ), source_channels AS (
                SELECT
                    coalesce(
                        json_extract_string(payload, '$.source_type'), 'unknown'
                    ) AS source,
                    count(DISTINCT json_extract_string(payload, '$.channel_id'))
                        AS channels
                FROM records
                WHERE record_type = 'discovery_edge'
                    AND json_extract_string(payload, '$.channel_id') IS NOT NULL
                GROUP BY source
            ), source_edge_counts AS (
                SELECT
                    coalesce(
                        json_extract_string(payload, '$.source_type'), 'unknown'
                    ) AS source,
                    count(*) AS edges
                FROM records
                WHERE record_type = 'discovery_edge'
                GROUP BY source
            )
            SELECT
                source_videos.source,
                count(*) AS videos,
                count(*) FILTER (WHERE source_videos.is_relevant) AS relevant,
                coalesce(max(source_channels.channels), 0) AS channels,
                coalesce(max(source_edge_counts.edges), 0) AS edges
            FROM source_videos
            LEFT JOIN source_channels USING (source)
            LEFT JOIN source_edge_counts USING (source)
            GROUP BY source_videos.source
            ORDER BY edges DESC, source
            """,
        )
    )
    queries = tuple(
        _query_dicts(
            connection,
            """
            SELECT
                json_extract_string(payload, '$.query') AS query,
                json_extract_string(payload, '$.query_kind') AS query_kind,
                json_extract_string(payload, '$.status') AS status,
                json_extract_string(payload, '$.source_request_id')
                    AS source_request_id,
                recorded_at
            FROM records
            WHERE record_type = 'query'
            ORDER BY source_line
            """,
        )
    )
    channels = tuple(
        _query_dicts(
            connection,
            """
            SELECT
                channel_id,
                arg_max(title, source_line) AS title,
                arg_max(url, source_line) AS url,
                arg_max(discovered_via, source_line) AS discovered_via,
                arg_max(discovered_from_id, source_line) AS discovered_from_id
            FROM (
                SELECT
                    source_line,
                    json_extract_string(payload, '$.channel_id') AS channel_id,
                    json_extract_string(payload, '$.title') AS title,
                    json_extract_string(payload, '$.url') AS url,
                    json_extract_string(payload, '$.discovered_via')
                        AS discovered_via,
                    json_extract_string(payload, '$.discovered_from_id')
                        AS discovered_from_id
                FROM records
                WHERE record_type = 'channel'
            )
            GROUP BY channel_id
            ORDER BY coalesce(arg_max(title, source_line), channel_id)
            """,
        )
    )
    interview_examples = tuple(
        _query_dicts(
            connection,
            """
            SELECT
                json_extract_string(payload, '$.question_id') AS question_id,
                json_extract_string(payload, '$.question_text') AS question_text,
                json_extract_string(payload, '$.answer') AS answer,
                json_extract_string(payload, '$.example_kind') AS example_kind,
                json_extract_string(payload, '$.generated_example')
                    AS generated_example
            FROM records
            WHERE record_type = 'interview_answer'
            ORDER BY source_line
            """,
        )
    )
    errors = tuple(
        _query_dicts(
            connection,
            """
            SELECT
                recorded_at,
                json_extract_string(payload, '$.stage') AS stage,
                json_extract_string(payload, '$.message') AS message,
                json_extract_string(payload, '$.exception_type') AS exception_type,
                json_extract_string(payload, '$.video_id') AS video_id,
                json_extract_string(payload, '$.channel_id') AS channel_id,
                try_cast(json_extract_string(payload, '$.retryable') AS BOOLEAN)
                    AS retryable
            FROM records
            WHERE record_type = 'run_error'
            ORDER BY source_line DESC
            """,
        )
    )

    return DashboardData(
        run_id=run_id,
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        status_reason=status_reason,
        started_at=_optional_string(timing.get("started_at")),
        completed_at=_optional_string(timing.get("completed_at")),
        config=config,
        summary=summary,
        run_metrics=run_metrics,
        discovery_sources=discovery_sources,
        videos=videos,
        decisions=decisions,
        transcripts=transcripts,
        transcript_segments=transcript_segments,
        queries=queries,
        channels=channels,
        interview_examples=interview_examples,
        api_operations=api_operations,
        budget_pools=budget_pools,
        sessions=sessions,
        errors=errors,
    )


def _query_config(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    row = _fetch_one_dict(
        connection,
        """
        SELECT
            json_extract_string(payload, '$.topic_query') AS topic_query,
            json_extract(payload, '$.expanded_queries') AS expanded_queries,
            json_extract_string(payload, '$.language') AS language,
            json_extract_string(payload, '$.start_date') AS start_date,
            try_cast(
                json_extract_string(payload, '$.max_searchapi_credits') AS BIGINT
            ) AS max_searchapi_credits,
            try_cast(
                json_extract_string(payload, '$.transcript_reserve_credits') AS BIGINT
            ) AS transcript_reserve_credits,
            json_extract_string(payload, '$.session_action') AS session_action,
            try_cast(json_extract_string(payload, '$.credits_added') AS BIGINT)
                AS credits_added,
            try_cast(
                json_extract_string(payload, '$.account_remaining_credits') AS BIGINT
            ) AS account_remaining_credits,
            try_cast(json_extract_string(payload, '$.max_depth') AS BIGINT)
                AS max_depth,
            try_cast(json_extract_string(payload, '$.max_queries') AS BIGINT)
                AS max_queries,
            try_cast(json_extract_string(payload, '$.max_search_pages') AS BIGINT)
                AS max_search_pages,
            try_cast(json_extract_string(payload, '$.max_channel_pages') AS BIGINT)
                AS max_channel_pages,
            json_extract_string(payload, '$.model') AS model,
            json_extract_string(payload, '$.prompt_version') AS prompt_version,
            json_extract_string(payload, '$.prompt_sha256') AS prompt_sha256
        FROM records
        WHERE record_type = 'run_config'
        ORDER BY source_line DESC
        LIMIT 1
        """,
        json_fields={"expanded_queries"},
    )
    defaults: dict[str, Any] = {
        "topic_query": None,
        "expanded_queries": [],
        "language": None,
        "start_date": None,
        "max_searchapi_credits": None,
        "transcript_reserve_credits": None,
        "session_action": "start",
        "credits_added": 0,
        "account_remaining_credits": None,
        "max_depth": 2,
        "max_queries": 8,
        "max_search_pages": 1,
        "max_channel_pages": 1,
        "model": None,
        "prompt_version": None,
        "prompt_sha256": None,
    }
    defaults.update(row)
    if defaults["expanded_queries"] is None:
        defaults["expanded_queries"] = []
    return defaults


def _query_decisions(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = _query_dicts(
        connection,
        """
        SELECT
            decision_id,
            video_id,
            label,
            decision_point,
            reason,
            confidence,
            requested_language,
            detected_language,
            language_matches,
            published_after_start_date,
            criteria,
            model,
            prompt_version,
            coalesce(llm_input_tokens, 0) AS llm_input_tokens,
            coalesce(llm_output_tokens, 0) AS llm_output_tokens,
            transcript_reserved,
            recorded_at
        FROM decisions
        ORDER BY source_line
        """,
        json_fields={"criteria"},
    )
    for row in rows:
        row["confidence_percent"] = _percent(row.get("confidence"))
    return rows


def _query_transcripts(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    segment_rows = tuple(
        _query_dicts(
            connection,
            """
            SELECT
                video_id,
                transcript_source_line,
                segment_index,
                text,
                start_seconds,
                duration_seconds
            FROM transcript_segments
            ORDER BY transcript_source_line, segment_index
            """,
        )
    )
    transcript_rows = tuple(
        _query_dicts(
            connection,
            """
            WITH segment_summary AS (
                SELECT
                    video_id,
                    transcript_source_line,
                    count(*) AS segment_count,
                    sum(length(coalesce(text, ''))) AS transcript_characters,
                    string_agg(trim(coalesce(text, '')), ' ' ORDER BY segment_index)
                        AS text,
                    max(coalesce(start_seconds, 0) + coalesce(duration_seconds, 0))
                        AS transcript_duration_seconds
                FROM transcript_segments
                GROUP BY video_id, transcript_source_line
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY video_id ORDER BY source_line DESC
                ) AS rank
                FROM transcript_rows
            )
            SELECT
                ranked.video_id,
                ranked.requested_language,
                ranked.language,
                ranked.transcript_type,
                ranked.is_available,
                ranked.unavailable_reason,
                ranked.source_request_id,
                ranked.searchapi_credits,
                coalesce(segment_summary.segment_count, 0) AS segment_count,
                coalesce(segment_summary.transcript_characters, 0)
                    AS transcript_characters,
                coalesce(segment_summary.text, '') AS text,
                coalesce(segment_summary.transcript_duration_seconds, 0)
                    AS transcript_duration_seconds,
                ranked.recorded_at
            FROM ranked
            LEFT JOIN segment_summary
                ON segment_summary.video_id = ranked.video_id
                AND segment_summary.transcript_source_line = ranked.source_line
            WHERE ranked.rank = 1
            ORDER BY ranked.video_id
            """,
        )
    )
    return transcript_rows, segment_rows


def _query_videos(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = _query_dicts(
        connection,
        """
        WITH candidate AS (
            SELECT
                video_id,
                arg_max(title, source_line) AS title,
                arg_max(url, source_line) AS url,
                arg_max(description, source_line) AS description,
                arg_max(channel_id, source_line) AS channel_id,
                arg_max(channel_title, source_line) AS channel_title,
                arg_max(published_at, source_line) AS published_at,
                arg_max(duration_seconds, source_line) AS duration_seconds,
                arg_max(discovered_via, source_line) AS discovered_via,
                arg_max(discovered_from_id, source_line) AS discovered_from_id,
                arg_max(discovery_query, source_line) AS discovery_query
            FROM video_candidates
            GROUP BY video_id
        ), final_decision AS (
            SELECT * EXCLUDE (decision_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY video_id
                        ORDER BY source_line DESC
                    ) AS decision_rank
                FROM decisions
            )
            WHERE decision_rank = 1
        ), latest_transcript AS (
            SELECT * EXCLUDE (transcript_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY video_id ORDER BY source_line DESC
                    ) AS transcript_rank
                FROM transcript_rows
            )
            WHERE transcript_rank = 1
        ), segment_summary AS (
            SELECT
                video_id,
                transcript_source_line,
                count(*) AS segment_count,
                sum(length(coalesce(text, ''))) AS transcript_characters
            FROM transcript_segments
            GROUP BY video_id, transcript_source_line
        )
        SELECT
            candidate.video_id,
            candidate.title,
            candidate.url,
            candidate.description,
            candidate.channel_id,
            candidate.channel_title,
            candidate.published_at,
            candidate.duration_seconds,
            candidate.discovered_via,
            candidate.discovered_from_id,
            candidate.discovery_query,
            final_decision.label AS final_label,
            final_decision.decision_point,
            final_decision.reason AS decision_reason,
            final_decision.confidence,
            final_decision.detected_language,
            final_decision.language_matches,
            final_decision.published_after_start_date,
            latest_transcript.is_available AS transcript_available,
            latest_transcript.language AS transcript_language,
            latest_transcript.transcript_type,
            latest_transcript.unavailable_reason AS transcript_unavailable_reason,
            coalesce(segment_summary.segment_count, 0) AS transcript_segments,
            coalesce(segment_summary.transcript_characters, 0)
                AS transcript_characters
        FROM candidate
        LEFT JOIN final_decision USING (video_id)
        LEFT JOIN latest_transcript USING (video_id)
        LEFT JOIN segment_summary
            ON segment_summary.video_id = latest_transcript.video_id
            AND segment_summary.transcript_source_line = latest_transcript.source_line
        ORDER BY
            CASE final_decision.label WHEN 'relevant' THEN 0 ELSE 1 END,
            candidate.title,
            candidate.video_id
        """,
    )
    for row in rows:
        row["confidence_percent"] = _percent(row.get("confidence"))
        row["duration_display"] = _duration_display(row.get("duration_seconds"))
    return rows


def _query_api_operations(
    connection: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    return _query_dicts(
        connection,
        """
        SELECT
            json_extract_string(payload, '$.provider') AS provider,
            json_extract_string(payload, '$.operation') AS operation,
            count(*) AS calls,
            count(*) FILTER (
                WHERE json_extract_string(payload, '$.status') = 'success'
            ) AS successes,
            count(*) FILTER (
                WHERE json_extract_string(payload, '$.status') = 'error'
            ) AS failures,
            count(*) FILTER (
                WHERE json_extract_string(payload, '$.status') = 'cache_hit'
            ) AS cache_hits,
            sum(coalesce(try_cast(
                json_extract_string(payload, '$.searchapi_credits') AS BIGINT
            ), 0)) AS searchapi_credits,
            sum(coalesce(try_cast(
                json_extract_string(payload, '$.llm_input_tokens') AS BIGINT
            ), 0)) AS llm_input_tokens,
            sum(coalesce(try_cast(
                json_extract_string(payload, '$.llm_output_tokens') AS BIGINT
            ), 0)) AS llm_output_tokens,
            round(avg(try_cast(
                json_extract_string(payload, '$.latency_seconds') AS DOUBLE
            )), 3) AS average_latency_seconds
        FROM records
        WHERE record_type = 'api_call'
        GROUP BY provider, operation
        ORDER BY provider, operation
        """,
    )


def _query_sessions(
    connection: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """Return one preparation-complete config row per crawl session."""

    return _query_dicts(
        connection,
        """
        SELECT
            recorded_at,
            coalesce(json_extract_string(payload, '$.session_action'), 'start')
                AS action,
            coalesce(try_cast(
                json_extract_string(payload, '$.credits_added') AS BIGINT
            ), 0) AS credits_added,
            try_cast(
                json_extract_string(payload, '$.max_searchapi_credits') AS BIGINT
            ) AS lifetime_grant,
            try_cast(
                json_extract_string(payload, '$.account_remaining_credits') AS BIGINT
            ) AS account_remaining_credits,
            try_cast(json_extract_string(payload, '$.max_depth') AS BIGINT)
                AS max_depth,
            try_cast(json_extract_string(payload, '$.max_queries') AS BIGINT)
                AS max_queries,
            try_cast(json_extract_string(payload, '$.max_search_pages') AS BIGINT)
                AS max_search_pages,
            try_cast(json_extract_string(payload, '$.max_channel_pages') AS BIGINT)
                AS max_channel_pages
        FROM records
        WHERE record_type = 'run_config'
          AND coalesce(
              json_extract_string(payload, '$.prompt_version'), ''
          ) <> 'pending-interview'
        ORDER BY source_line
        """,
    )


def _query_budget_pools(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = _query_dicts(
        connection,
        """
        SELECT
            budget_kind,
            pool,
            action AS last_action,
            amount AS last_amount,
            remaining,
            purpose AS last_purpose
        FROM (
            SELECT
                json_extract_string(payload, '$.budget_kind') AS budget_kind,
                json_extract_string(payload, '$.pool') AS pool,
                json_extract_string(payload, '$.action') AS action,
                try_cast(json_extract_string(payload, '$.amount') AS BIGINT) AS amount,
                try_cast(json_extract_string(payload, '$.remaining') AS BIGINT)
                    AS remaining,
                json_extract_string(payload, '$.purpose') AS purpose,
                row_number() OVER (
                    PARTITION BY
                        json_extract_string(payload, '$.budget_kind'),
                        json_extract_string(payload, '$.pool')
                    ORDER BY source_line DESC
                ) AS rank
            FROM records
            WHERE record_type = 'budget_event'
        )
        WHERE rank = 1
        ORDER BY budget_kind, pool
        """,
    )
    configured = {
        ("search_api_credits", "discovery"): _subtract(
            config.get("max_searchapi_credits"),
            config.get("transcript_reserve_credits"),
        ),
        ("search_api_credits", "transcript"): config.get("transcript_reserve_credits"),
    }
    for row in rows:
        initial = configured.get((row.get("budget_kind"), row.get("pool")))
        row["configured"] = initial
        row["used"] = _subtract(initial, row.get("remaining"))
    return rows


def _query_summary(
    connection: duckdb.DuckDBPyConnection,
    *,
    videos: tuple[dict[str, Any], ...],
    transcripts: tuple[dict[str, Any], ...],
    api_operations: tuple[dict[str, Any], ...],
    budget_pools: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    evaluated = sum(
        video.get("final_label") in {"relevant", "irrelevant"} for video in videos
    )
    # Only a transcript-stage relevant decision is a completed positive
    # classification; provisional and operational dispositions remain separate.
    relevant = sum(
        1
        for video in videos
        if video.get("final_label") == "relevant"
        and video.get("decision_point") == "transcript"
    )
    available_transcripts = sum(
        1 for transcript in transcripts if transcript.get("is_available") is True
    )
    reported_searchapi_credits = sum(
        int(operation.get("searchapi_credits") or 0) for operation in api_operations
    )
    reported_llm_tokens = sum(
        int(operation.get("llm_input_tokens") or 0)
        + int(operation.get("llm_output_tokens") or 0)
        for operation in api_operations
    )
    committed_searchapi_credits = sum(
        int(pool.get("used") or 0)
        for pool in budget_pools
        if pool.get("budget_kind") == "search_api_credits"
    )
    searchapi_credits = max(reported_searchapi_credits, committed_searchapi_credits)
    values = (
        ("videos_discovered", "Videos discovered", len(videos), None),
        ("videos_evaluated", "Videos evaluated", evaluated, None),
        ("relevant_videos", "Relevant videos", relevant, None),
        (
            "transcripts_collected",
            "Transcripts collected",
            available_transcripts,
            None,
        ),
        ("searchapi_credits", "SearchAPI credits", searchapi_credits, "credits"),
        ("llm_tokens", "LLM tokens", reported_llm_tokens, "tokens"),
    )
    return [
        {
            "key": key,
            "label": label,
            "value": value,
            "display": f"{value:,}",
            "unit": unit,
        }
        for key, label, value, unit in values
    ]


def _is_final_relevant(video: dict[str, Any]) -> bool:
    return (
        video.get("final_label") == "relevant"
        and video.get("decision_point") == "transcript"
    )


def _presentation_video(video: dict[str, Any]) -> dict[str, Any]:
    transcript_available = video.get("transcript_available")
    if transcript_available is True:
        transcript_status = "available"
    elif transcript_available is False:
        transcript_status = "unavailable"
    else:
        transcript_status = "not_requested"
    return {
        "id": video.get("video_id"),
        "title": video.get("title") or video.get("video_id") or "Untitled video",
        "url": video.get("url"),
        "channel": video.get("channel_title") or video.get("channel_id"),
        "published_at": video.get("published_at"),
        "discovered_via": video.get("discovered_via"),
        "decision_point": video.get("decision_point"),
        "confidence": float(video.get("confidence") or 0),
        "language": video.get("transcript_language") or video.get("detected_language"),
        "reason": video.get("decision_reason"),
        "decision_label": video.get("final_label") or "pending",
        "decision_display": _presentation_decision_label(
            video.get("final_label") or "pending"
        ),
        "transcript_status": transcript_status,
    }


def _presentation_decision_label(label: str) -> str:
    """Make operational dispositions unambiguously distinct from relevance."""

    if label == "deferred_budget":
        return "Deferred \u2014 budget"
    return label.replace("_", " ")


def _presentation_transcript_item(
    video_id: str,
    title: str,
    transcript: dict[str, Any] | None,
) -> dict[str, Any]:
    if transcript is None:
        return {
            "video_id": video_id,
            "title": title,
            "status": "not_requested",
            "language": None,
            "segments": 0,
            "words": 0,
            "reason": "No transcript record was written.",
        }
    available = transcript.get("is_available") is True
    text = str(transcript.get("text") or "")
    return {
        "video_id": video_id,
        "title": title,
        "status": "available" if available else "unavailable",
        "language": transcript.get("language") or transcript.get("requested_language"),
        "segments": int(transcript.get("segment_count") or 0),
        "words": len(text.split()) if available else 0,
        "reason": None if available else transcript.get("unavailable_reason"),
    }


def _presentation_queries(
    data: DashboardData,
    relevant_ids: set[str],
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for query in data.queries:
        key = str(query.get("query") or "").strip()
        if key:
            latest[key] = query

    rows: list[dict[str, Any]] = []
    for query_text, query in latest.items():
        matching_videos = [
            video for video in data.videos if video.get("discovery_query") == query_text
        ]
        rows.append(
            {
                "query": query_text,
                "kind": query.get("query_kind") or "unknown",
                "status": query.get("status") or "unknown",
                "results": len(matching_videos),
                "relevant": sum(
                    str(video.get("video_id")) in relevant_ids
                    for video in matching_videos
                ),
            }
        )
    return rows


def _presentation_query_counts(
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any] | None,
) -> dict[str, int]:
    """Keep planned scope distinct from work executed in the current limits."""

    if checkpoint:
        return {
            "planned": int(checkpoint["planned_queries"]),
            "executed": int(checkpoint["executed_queries"]),
            "deferred": int(checkpoint["deferred_queries"]),
            "unfinished": int(checkpoint["unfinished_queries"]),
        }
    return {
        "planned": len(rows),
        "executed": sum(row.get("status") in {"partial", "executed"} for row in rows),
        "deferred": sum(row.get("status") == "deferred" for row in rows),
        "unfinished": sum(row.get("status") in {"planned", "partial"} for row in rows),
    }


def _presentation_provenance(
    discovery_sources: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    total = sum(int(source.get("videos") or 0) for source in discovery_sources)
    return [
        {
            "source": source.get("source") or "unknown",
            "count": int(source.get("videos") or 0),
            "relevant": int(source.get("relevant") or 0),
            "percent": _bounded_percent(source.get("videos"), total),
        }
        for source in discovery_sources
    ]


def _presentation_budgets(
    data: DashboardData,
    checkpoint: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    pool_rows = {
        (row.get("budget_kind"), row.get("pool")): row for row in data.budget_pools
    }
    reported_used = next(
        (
            int(metric.get("value") or 0)
            for metric in data.summary
            if metric.get("key") == "searchapi_credits"
        ),
        0,
    )
    grant = (
        int(checkpoint["max_credits"])
        if checkpoint
        else int(data.config.get("max_searchapi_credits") or 0)
    )
    total_used = int(checkpoint["credits_spent"]) if checkpoint else reported_used
    total_reserved = int(checkpoint["credits_reserved"]) if checkpoint else 0
    total_remaining = (
        int(checkpoint["credits_remaining"])
        if checkpoint
        else max(0, grant - total_used)
    )
    definitions = (
        (
            ("search_api_credits", "discovery"),
            "SearchAPI discovery",
            _subtract(
                data.config.get("max_searchapi_credits"),
                data.config.get("transcript_reserve_credits"),
            ),
            "credits",
            "blue",
        ),
        (
            ("search_api_credits", "transcript"),
            "SearchAPI transcripts",
            data.config.get("transcript_reserve_credits"),
            "credits",
            "teal",
        ),
    )
    budgets: list[dict[str, Any]] = [
        {
            "label": "SearchAPI total",
            "used": total_used,
            "limit": grant,
            "remaining": total_remaining,
            "reserved": total_reserved,
            "percent": _bounded_percent(total_used, grant),
            "unit": "credits",
            "tone": "amber" if total_remaining == 0 else "blue",
        }
    ]
    checkpoint_pools = (
        {
            ("search_api_credits", "discovery"): {
                "limit": checkpoint["discovery_capacity"],
                "used": checkpoint["discovery_spent"],
                "remaining": checkpoint["discovery_remaining"],
                "reserved": 0,
            },
            ("search_api_credits", "transcript"): {
                "limit": checkpoint["transcript_capacity"],
                "used": checkpoint["transcript_spent"],
                "remaining": checkpoint["transcript_remaining"],
                "reserved": checkpoint["credits_reserved"],
            },
        }
        if checkpoint
        else {}
    )
    for key, label, configured, unit, tone in definitions:
        if configured is None:
            continue
        exact = checkpoint_pools.get(key)
        if exact:
            limit = int(exact["limit"])
            reserved = int(exact["reserved"])
            remaining = int(exact["remaining"])
            used = int(exact["used"])
        else:
            limit = max(0, int(configured))
            row = pool_rows.get(key, {})
            reserved = (
                int(row.get("last_amount") or 0)
                if row.get("last_action") == "reserved"
                else 0
            )
            remaining = int(limit if row.get("remaining") is None else row["remaining"])
            used = max(0, limit - remaining - reserved)
        budgets.append(
            {
                "label": label,
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "reserved": reserved,
                "percent": _bounded_percent(used, limit),
                "unit": unit,
                "tone": tone,
            }
        )
    return budgets


def _checkpoint_summary(run_dir: Path) -> dict[str, Any] | None:
    """Read the optional mutable checkpoint for exact current project totals."""

    try:
        state = ProjectStateStore(run_dir).load()
    except (FileNotFoundError, ValueError):
        return None
    payload = state.model_dump(mode="json")
    try:
        budget = payload["budget"]
        discovered = payload.get("discovered_videos", {})
        terminal = set(payload.get("terminal_video_ids", []))
        deferred = set(payload.get("deferred_video_ids", []))
        max_depth = int(payload.get("max_depth", 0))
        spent = int(budget.get("discovery_spent", 0)) + int(
            budget.get("transcript_spent", 0)
        )
        reserved = sum(
            int(item.get("credits", 0)) for item in budget.get("pending", [])
        )
        pending = sum(
            video_id not in terminal and int(video.get("depth", 0)) <= max_depth
            for video_id, video in discovered.items()
        )
        all_planned = payload.get("planned_queries", [])
        max_queries = int(payload.get("max_queries", 0))
        planned = all_planned[:max_queries]
        query_progress = payload.get("query_progress", {})
        executed_queries = sum(
            int(query_progress.get(item.get("text"), {}).get("pages_completed", 0)) > 0
            for item in all_planned
        )
        unfinished_queries = sum(
            not query_progress.get(item.get("text"), {}).get("exhausted", False)
            and int(query_progress.get(item.get("text"), {}).get("pages_completed", 0))
            < int(payload.get("max_search_pages", 1))
            for item in planned
        )
        grant = int(budget["max_credits"])
        transcript_capacity = int(budget["transcript_capacity"])
        discovery_capacity = grant - transcript_capacity
        discovery_spent = int(budget.get("discovery_spent", 0))
        transcript_spent = int(budget.get("transcript_spent", 0))
        return {
            "credits_spent": spent,
            "credits_reserved": reserved,
            "credits_remaining": max(0, grant - spent - reserved),
            "discovery_capacity": discovery_capacity,
            "discovery_spent": discovery_spent,
            "discovery_remaining": max(0, discovery_capacity - discovery_spent),
            "transcript_capacity": transcript_capacity,
            "transcript_spent": transcript_spent,
            "transcript_remaining": max(
                0, transcript_capacity - transcript_spent - reserved
            ),
            "max_credits": grant,
            "pending_videos": pending,
            "deferred_videos": len(deferred),
            "planned_queries": len(all_planned),
            "executed_queries": executed_queries,
            "deferred_queries": max(0, len(all_planned) - max_queries),
            "unfinished_queries": unfinished_queries,
            "controls": {
                "max_depth": max_depth,
                "max_queries": max_queries,
                "max_search_pages": int(payload.get("max_search_pages", 1)),
                "max_channel_pages": int(payload.get("max_channel_pages", 1)),
            },
            "classifier_prompt_version": payload.get("classifier_prompt_version"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "last_status": str(payload.get("last_status", "prepared")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _latest_metric(
    metrics: tuple[dict[str, Any], ...], name: str
) -> int | float | str | None:
    for metric in metrics:
        if metric.get("name") == name:
            return metric.get("value")
    return None


def _suggested_resume_command(
    project: Path,
    *,
    status: str,
    controls: dict[str, int],
    available_credits: int,
    lifetime_grant: int,
    planned_queries: int,
    deferred_queries: int,
) -> str | None:
    """Build a complete copyable continuation command when more work is plausible."""

    pieces = [
        "uv run yt-crawl resume",
        "--project",
        shlex.quote(_display_path(project)),
    ]
    add_credits = (
        0 if available_credits > 0 else max(4, min(25, max(1, lifetime_grant // 2)))
    )
    pieces.extend(("--add-credits", str(add_credits)))

    if status == "completed":
        query_limit = min(18, planned_queries)
        candidates = []
        if deferred_queries > 0:
            candidates.append(("max_queries", query_limit))
        candidates.extend(
            (
                ("max_search_pages", 10),
                ("max_channel_pages", 10),
                ("max_depth", 5),
            )
        )
        for name, limit in candidates:
            current = controls[name]
            if current < limit:
                pieces.extend((f"--{name.replace('_', '-')}", str(current + 1)))
                break
        else:
            return None
    return " ".join(pieces)


def _continuation_reason(
    *,
    status: str,
    available_credits: int,
    has_command: bool,
) -> str:
    """Explain whether the next safe action is scope expansion or resume."""

    if status == "completed":
        if not has_command:
            return (
                "The current frontier is exhausted and every scope control is at "
                "its limit; adding credits alone cannot create more work."
            )
        if available_credits > 0:
            return (
                "The current frontier is exhausted, so this command widens scope "
                "using the remaining SearchAPI credits without adding more."
            )
        return (
            "The current frontier is exhausted and no SearchAPI credits remain, "
            "so this command widens scope and adds credits."
        )
    if available_credits > 0:
        return (
            "The saved frontier may still contain work; resume the current scope "
            "with its remaining SearchAPI credits before widening it."
        )
    return (
        "The saved frontier may still contain work but no SearchAPI credits remain, "
        "so resume the current scope with an added credit grant."
    )


def _duration_between(started_at: str | None, completed_at: str | None) -> str:
    if not started_at or not completed_at:
        return "In progress" if started_at else "—"
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return "—"
    seconds = max(0, int((end - start).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _bounded_percent(numerator: Any, denominator: Any) -> int:
    numerator_value = float(numerator or 0)
    denominator_value = float(denominator or 0)
    if denominator_value <= 0:
        return 0
    return round(max(0.0, min(100.0, numerator_value / denominator_value * 100)))


def _jsonl_file_inventory(run_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        with path.open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
        inventory.append(
            {
                "name": str(path.relative_to(run_dir)),
                "path": _display_path(path),
                "rows": rows,
                "bytes": _byte_size_display(size),
            }
        )
    return inventory


def _display_path(path: Path) -> str:
    """Prefer a portable cwd-relative path when the file is below it."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _byte_size_display(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _query_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    *,
    json_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    result = connection.execute(query)
    columns = [column[0] for column in result.description]
    rows = [dict(zip(columns, values, strict=True)) for values in result.fetchall()]
    for row in rows:
        for field in json_fields or ():
            row[field] = _decode_json(row.get(field))
    return rows


def _fetch_one_dict(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    *,
    json_fields: set[str] | None = None,
) -> dict[str, Any]:
    rows = _query_dicts(connection, query, json_fields=json_fields)
    return rows[0] if rows else {}


def _decode_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _subtract(left: Any, right: Any) -> int | None:
    if left is None or right is None:
        return None
    return int(left) - int(right)


def _percent(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value) * 100:.0f}%"


def _duration_display(value: Any) -> str | None:
    if value is None:
        return None
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


_BUILTIN_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YouTube research run {{ run_id }}</title>
  <style>
    :root { color-scheme: light dark; --bg:#f4f5f7; --panel:#fff; --ink:#17202a; --muted:#65717e; --line:#dce1e7; --accent:#9f1d2b; --good:#18794e; --bad:#b42318; --warn:#a15c00; }
    @media (prefers-color-scheme: dark) { :root { --bg:#11151a; --panel:#191f26; --ink:#edf2f7; --muted:#a8b3bf; --line:#34404c; --accent:#ff6877; --good:#61d095; --bad:#ff7b72; --warn:#f4bd61; } }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { width:min(1500px,calc(100% - 32px)); margin:28px auto 64px; }
    h1,h2,h3 { line-height:1.18; }
    h1 { margin:.15rem 0 .25rem; font-size:clamp(1.6rem,3vw,2.5rem); }
    h2 { margin:0 0 1rem; font-size:1.2rem; }
    .eyebrow { color:var(--accent); font-size:.76rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
    .muted { color:var(--muted); }
    .status { display:inline-block; margin-top:.7rem; padding:.25rem .55rem; border:1px solid var(--line); border-radius:999px; font-weight:700; }
    .status.completed { color:var(--good); } .status.failed { color:var(--bad); } .status.stopped_budget { color:var(--warn); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:22px 0; }
    .card,.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 1px 2px rgb(0 0 0/.04); }
    .card { padding:15px; } .card b { display:block; margin-top:5px; font-size:1.55rem; }
    .panel { margin:14px 0; padding:18px; overflow:hidden; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }
    dl { display:grid; grid-template-columns:minmax(115px,auto) 1fr; gap:.4rem 1rem; margin:0; }
    dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
    .table-wrap { overflow:auto; }
    table { width:100%; border-collapse:collapse; font-size:.88rem; }
    th,td { padding:.62rem .7rem; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:.73rem; letter-spacing:.04em; text-transform:uppercase; white-space:nowrap; }
    tbody tr:last-child td { border-bottom:0; }
    td.reason { min-width:260px; max-width:520px; }
    a { color:var(--accent); } code { font-size:.84em; overflow-wrap:anywhere; }
    .label { font-weight:750; } .label.relevant { color:var(--good); } .label.irrelevant { color:var(--bad); }
    details { border-top:1px solid var(--line); padding:.7rem 0; }
    details:first-of-type { border-top:0; }
    summary { cursor:pointer; font-weight:700; }
    .transcript { max-height:26rem; overflow:auto; white-space:pre-wrap; color:var(--muted); }
    .empty { color:var(--muted); font-style:italic; }
    footer { margin-top:20px; color:var(--muted); font-size:.8rem; }
    @media print { :root { color-scheme:light; } main { width:100%; } .panel,.card { box-shadow:none; break-inside:avoid; } }
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">YouTube topic research</div>
    <h1>{{ config.topic_query or "Crawler run" }}</h1>
    <div class="muted">Run <code>{{ run_id }}</code>{% if config.language %} · language {{ config.language }}{% endif %}{% if config.start_date %} · since {{ config.start_date }}{% endif %}</div>
    <div class="status {{ status }}">{{ status|replace('_',' ')|title }}</div>
    {% if status_reason %}<span class="muted"> · {{ status_reason }}</span>{% endif %}
  </header>

  <section class="cards" aria-label="Run summary">
  {% for metric in summary %}
    <div class="card"><span class="muted">{{ metric.label }}</span><b>{{ metric.display }}</b>{% if metric.unit %}<small class="muted">{{ metric.unit }}</small>{% endif %}</div>
  {% endfor %}
  </section>

  <section class="grid">
    <div class="panel"><h2>Research definition</h2><dl>
      <dt>Topic</dt><dd>{{ config.topic_query or '—' }}</dd>
      <dt>Language</dt><dd>{{ config.language or '—' }}</dd>
      <dt>Start date</dt><dd>{{ config.start_date or '—' }}</dd>
      <dt>Model</dt><dd>{{ config.model or '—' }}</dd>
      <dt>Prompt</dt><dd>{{ config.prompt_version or '—' }}</dd>
      <dt>Expanded queries</dt><dd>{{ config.expanded_queries|join(', ') if config.expanded_queries else '—' }}</dd>
    </dl></div>
    <div class="panel"><h2>Budget pools</h2>
      {% if budget_pools %}<div class="table-wrap"><table><thead><tr><th>Kind</th><th>Pool</th><th>Configured</th><th>Used</th><th>Remaining</th></tr></thead><tbody>
      {% for pool in budget_pools %}<tr><td>{{ pool.budget_kind }}</td><td>{{ pool.pool }}</td><td>{{ pool.configured if pool.configured is not none else '—' }}</td><td>{{ pool.used if pool.used is not none else '—' }}</td><td>{{ pool.remaining if pool.remaining is not none else '—' }}</td></tr>{% endfor %}
      </tbody></table></div>{% else %}<p class="empty">No budget events recorded.</p>{% endif %}
    </div>
  </section>

  <section class="panel"><h2>Videos</h2>
    {% if videos %}<div class="table-wrap"><table><thead><tr><th>Video</th><th>Channel</th><th>Discovered via</th><th>Published</th><th>Duration</th><th>Final decision</th><th>Point</th><th>Confidence</th><th>Transcript</th><th>Reason</th></tr></thead><tbody>
    {% for video in videos %}<tr>
      <td>{% if video.url %}<a href="{{ video.url }}">{{ video.title }}</a>{% else %}{{ video.title }}{% endif %}<br><code>{{ video.video_id }}</code></td>
      <td>{{ video.channel_title or video.channel_id or '—' }}</td><td>{{ video.discovered_via or '—' }}</td><td>{{ video.published_at or '—' }}</td><td>{{ video.duration_display or '—' }}</td>
      <td><span class="label {{ video.final_label or '' }}">{{ video.final_label or 'pending' }}</span></td><td>{{ video.decision_point or '—' }}</td><td>{{ video.confidence_percent or '—' }}</td><td>{% if video.transcript_available is true %}available ({{ video.transcript_segments }}){% elif video.transcript_available is false %}unavailable{% else %}—{% endif %}</td><td class="reason">{{ video.decision_reason or video.transcript_unavailable_reason or '—' }}</td>
    </tr>{% endfor %}
    </tbody></table></div>{% else %}<p class="empty">No video candidates recorded.</p>{% endif %}
  </section>

  <section class="panel"><h2>Transcripts</h2>
    {% if transcripts %}{% for transcript in transcripts %}<details><summary>{{ transcript.video_id }} · {% if transcript.is_available %}{{ transcript.segment_count }} segments, {{ transcript.transcript_characters }} characters{% else %}unavailable{% endif %}</summary>
      {% if transcript.is_available %}<p class="muted">{{ transcript.language or transcript.requested_language }} · {{ transcript.transcript_type or 'unknown type' }} · {{ transcript.transcript_duration_seconds|round(1) }} seconds</p><div class="transcript">{{ transcript.text }}</div>{% else %}<p class="empty">{{ transcript.unavailable_reason }}</p>{% endif %}
    </details>{% endfor %}{% else %}<p class="empty">No transcript attempts recorded.</p>{% endif %}
  </section>

  <section class="panel"><h2>Decision audit trail</h2>
    {% if decisions %}<div class="table-wrap"><table><thead><tr><th>Video</th><th>Point</th><th>Label</th><th>Confidence</th><th>Language</th><th>Model</th><th>Tokens</th><th>Reason</th></tr></thead><tbody>
    {% for decision in decisions %}<tr><td><code>{{ decision.video_id }}</code></td><td>{{ decision.decision_point }}</td><td><span class="label {{ decision.label }}">{{ decision.label }}</span></td><td>{{ decision.confidence_percent }}</td><td>{{ decision.detected_language or '—' }}{% if decision.language_matches is false %} (mismatch){% endif %}</td><td>{{ decision.model }}</td><td>{{ decision.llm_input_tokens + decision.llm_output_tokens }}</td><td class="reason">{{ decision.reason }}</td></tr>{% endfor %}
    </tbody></table></div>{% else %}<p class="empty">No relevance decisions recorded.</p>{% endif %}
  </section>

  <section class="grid">
    <div class="panel"><h2>Queries</h2>{% if queries %}<div class="table-wrap"><table><thead><tr><th>Query</th><th>Kind</th><th>Status</th></tr></thead><tbody>{% for query in queries %}<tr><td>{{ query.query }}</td><td>{{ query.query_kind }}</td><td>{{ query.status }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<p class="empty">No queries recorded.</p>{% endif %}</div>
    <div class="panel"><h2>API operations</h2>{% if api_operations %}<div class="table-wrap"><table><thead><tr><th>Provider</th><th>Operation</th><th>Calls</th><th>Failures</th><th>Credits</th><th>Tokens</th><th>Avg latency</th></tr></thead><tbody>{% for operation in api_operations %}<tr><td>{{ operation.provider }}</td><td>{{ operation.operation }}</td><td>{{ operation.calls }}</td><td>{{ operation.failures }}</td><td>{{ operation.searchapi_credits }}</td><td>{{ operation.llm_input_tokens + operation.llm_output_tokens }}</td><td>{{ operation.average_latency_seconds if operation.average_latency_seconds is not none else '—' }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<p class="empty">No API calls recorded.</p>{% endif %}</div>
  </section>

  {% if errors %}<section class="panel"><h2>Errors</h2><div class="table-wrap"><table><thead><tr><th>Time</th><th>Stage</th><th>Type</th><th>Target</th><th>Message</th></tr></thead><tbody>{% for error in errors %}<tr><td>{{ error.recorded_at }}</td><td>{{ error.stage }}</td><td>{{ error.exception_type or '—' }}</td><td>{{ error.video_id or error.channel_id or '—' }}</td><td>{{ error.message }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}

  <footer>Generated offline at {{ generated_at }} from the JSONL files in this run directory. No external assets or network requests are used.</footer>
</main>
</body>
</html>
"""


__all__ = [
    "DashboardData",
    "build_dashboard_context",
    "build_dashboard_data",
    "render_dashboard",
]
