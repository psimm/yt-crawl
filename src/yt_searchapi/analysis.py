"""DuckDB-only analytical views for the interactive YouTube explorer.

The crawler's JSONL files remain the source of truth. This module rebuilds a
small in-memory DuckDB database for each request, which makes the HTTP layer
live while a crawl is appending records and avoids a second database to
maintain. Topic labels are transparent keyword rules, not a hidden model.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from yt_searchapi.dashboard import (
    _JSONL_RECORD_TYPES,
    _create_normalized_views,
)

TOPIC_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "saving_budgeting",
        "label": "Saving & budgeting",
        "color": "#60a5fa",
        "keywords": (
            "sparen",
            "sparplan",
            "sparmethode",
            "budget",
            "haushalt",
            "notgroschen",
            "rücklage",
            "geld sparen",
            "einnahmen und ausgaben",
        ),
        "description": (
            "Everyday money management, saving systems, and household budgets."
        ),
    },
    {
        "id": "investing_etfs",
        "label": "Investing & ETFs",
        "color": "#34d399",
        "keywords": (
            "investieren",
            "investing",
            "etf",
            "aktien",
            "aktie",
            "geldanlage",
            "depot",
            "wertpapier",
            "fonds",
            "börse",
        ),
        "description": "Long-term investing, ETFs, securities, and portfolio basics.",
    },
    {
        "id": "retirement_pensions",
        "label": "Retirement & pensions",
        "color": "#fbbf24",
        "keywords": (
            "rente",
            "renten",
            "altersvorsorge",
            "ruhestand",
            "riester",
            "rürup",
            "rentenversicherung",
            "renteneintritt",
            "vorsorge fürs alter",
        ),
        "description": "Public pensions and private retirement planning.",
    },
    {
        "id": "debt_credit",
        "label": "Debt & credit",
        "color": "#fb7185",
        "keywords": (
            "schulden",
            "kredit",
            "kredite",
            "kreditkarte",
            "ratenkredit",
            "buy now pay later",
            "bonität",
            "schufa",
            "dispo",
            "tilgung",
        ),
        "description": "Borrowing, repayment, credit scoring, and consumer debt.",
    },
    {
        "id": "insurance_protection",
        "label": "Insurance & protection",
        "color": "#c084fc",
        "keywords": (
            "versicherung",
            "versicherungen",
            "haftpflicht",
            "berufsunfähigkeit",
            "krankenversicherung",
            "lebensversicherung",
            "risiko",
            "schutz",
        ),
        "description": "Personal risk management and insurance choices.",
    },
    {
        "id": "tax_policy",
        "label": "Taxes & policy",
        "color": "#fb923c",
        "keywords": (
            "steuer",
            "steuern",
            "steuererklärung",
            "steuerfrei",
            "freibetrag",
            "gesetz",
            "reform",
            "finanzamt",
            "staat",
        ),
        "description": "Tax rules, consumer policy, and financial regulation.",
    },
    {
        "id": "housing",
        "label": "Housing & buying",
        "color": "#2dd4bf",
        "keywords": (
            "immobilien",
            "immobilie",
            "wohnung",
            "eigenheim",
            "hauskauf",
            "baufinanzierung",
            "hypothek",
            "baufinanz",
            "miete",
        ),
        "description": "Saving for, financing, or evaluating a private home.",
    },
    {
        "id": "tools_apps",
        "label": "Tools & apps",
        "color": "#a78bfa",
        "keywords": (
            "budget app",
            "budget-app",
            "finanz app",
            "finanz-app",
            "haushaltsbuch",
            "tracking",
            "konto",
            "banking app",
            "vergleich",
        ),
        "description": "Apps, accounts, and practical tools for managing money.",
    },
)


STOPWORDS = frozenset(
    """
    aber als also am an auch auf aus bei bis das dass dem den der des die ein
    eine für im in ist mit nach nicht noch oder sein seine sie sich so über um
    von vor was wie zu zum zur ich du wir ihr man mein meine dein deine wird
    werden wurde sehr mehr nur schon denn dann diese dieser dieses einen einer
    einem eines kein keine kann können hat haben durch gegen zwischen und
    wenn sind diesem dir https http www com video videos ber
    and the of to in on with from that this your you are was is
    hier warum dich kanal welche jetzt alle uns provision wirklich instagram
    link diesen kannst deinen einfach links viele immer ohne geht weitere
    empfehlungen etwas dabei selbst abonniere entstehen youtube kommentare neue
    unser unsere unseren beim erfährst mir viel handelt fragen inhalte infos
    zeige wissen genau damit intro dar ganz mich erhalten klickst gibt kleine
    zeigen source
    """.split()
)

_ANALYSIS_RECORD_TYPES = (
    "run_config",
    "run_status",
    "query",
    "channel",
    "video_candidate",
    "relevance_decision",
    "transcript",
)


class DuckDBAnalytics:
    """Expose whitelisted analytical queries over one crawler run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"run path is not a directory: {self.run_dir}")
        self._lock = threading.RLock()
        self._fingerprint = ""
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._refresh_locked()

    @property
    def fingerprint(self) -> str:
        with self._lock:
            self._refresh_if_changed_locked()
            return self._fingerprint

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            connection = self._connection_for_request_locked()
            payload = {
                "run": self._run(connection),
                "summary": self._summary(connection),
                "funnel": self._funnel(connection),
                "timeseries": self._timeseries(connection),
                "topicTimeseries": self._topic_timeseries(connection),
                "topics": self._topics(connection),
                "channels": self._channels(connection),
                "keywords": self._keywords(connection),
                "sourceMix": self._source_mix(connection),
                "metadata": self._metadata(connection),
                "queries": self._queries(connection),
                "videos": self._videos(connection, limit=80),
                "topicDefinitions": [
                    {
                        "id": item["id"],
                        "label": item["label"],
                        "color": item["color"],
                        "description": item["description"],
                        "keywords": list(item["keywords"]),
                    }
                    for item in TOPIC_DEFINITIONS
                ],
                "server": {
                    "fingerprint": self._fingerprint,
                    "recordTypes": list(_JSONL_RECORD_TYPES),
                },
            }
            return _json_safe(payload)

    def videos(self, params: dict[str, list[str]]) -> dict[str, Any]:
        with self._lock:
            connection = self._connection_for_request_locked()
            limit = _bounded_int(params, "limit", default=50, minimum=1, maximum=250)
            offset = _bounded_int(
                params, "offset", default=0, minimum=0, maximum=100_000
            )
            rows = self._videos(connection, params=params, limit=limit, offset=offset)
            total = self._video_count(connection, params)
            return _json_safe(
                {
                    "rows": rows,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "fingerprint": self._fingerprint,
                }
            )

    def video_detail(self, video_id: str) -> dict[str, Any] | None:
        with self._lock:
            connection = self._connection_for_request_locked()
            row = _fetch_one(
                connection,
                """
                SELECT
                    v.*,
                    list(tm.label ORDER BY tm.topic_id) AS topics,
                    list(tm.topic_id ORDER BY tm.topic_id) AS topic_ids,
                    list(tm.matched_keywords ORDER BY tm.topic_id)
                        AS matched_topic_keywords
                FROM video_records v
                LEFT JOIN topic_matches tm USING (video_id)
                WHERE v.video_id = ?
                GROUP BY ALL
                """,
                [video_id],
            )
            if row is None:
                return None
            row["transcript_segments"] = _query_dicts(
                connection,
                """
                SELECT segment_index, text, start_seconds, duration_seconds
                FROM transcript_segments
                WHERE video_id = ?
                ORDER BY segment_index
                """,
                [video_id],
            )
            row["fingerprint"] = self._fingerprint
            return _json_safe(row)

    def _refresh_if_changed_locked(self) -> None:
        fingerprint = _run_fingerprint(self.run_dir)
        if fingerprint != self._fingerprint:
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        connection = duckdb.connect(database=":memory:")
        connection.execute("SET enable_progress_bar = false")
        _ingest_analysis_jsonl(connection, self.run_dir)
        _create_normalized_views(connection)
        _create_analysis_views(connection)
        old_connection = self._connection
        self._connection = connection
        self._fingerprint = _run_fingerprint(self.run_dir)
        if old_connection is not None:
            old_connection.close()

    def _connection_for_request_locked(self) -> duckdb.DuckDBPyConnection:
        self._refresh_if_changed_locked()
        assert self._connection is not None
        return self._connection

    def _run(self, connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        config = (
            _fetch_one(
                connection,
                """
            SELECT
                json_extract_string(payload, '$.topic_query') AS topic_query,
                json_extract(payload, '$.expanded_queries') AS expanded_queries,
                json_extract_string(payload, '$.language') AS language,
                json_extract_string(payload, '$.start_date') AS start_date,
                json_extract_string(payload, '$.model') AS model,
                json_extract_string(payload, '$.prompt_version') AS prompt_version
            FROM records
            WHERE record_type = 'run_config'
            ORDER BY source_file DESC, source_line DESC
            LIMIT 1
            """,
            )
            or {}
        )
        status = (
            _fetch_one(
                connection,
                """
            SELECT
                json_extract_string(payload, '$.status') AS status,
                json_extract_string(payload, '$.reason') AS reason,
                recorded_at
            FROM records
            WHERE record_type = 'run_status'
            ORDER BY source_line DESC
            LIMIT 1
            """,
            )
            or {}
        )
        started = (
            _fetch_one(
                connection,
                """
            SELECT min(recorded_at) AS started_at
            FROM records
            WHERE record_type = 'run_status'
              AND json_extract_string(payload, '$.status') = 'started'
            """,
            )
            or {}
        )
        candidates = (
            _fetch_one(
                connection,
                """
            SELECT
                min(published_at) AS first_published,
                max(published_at) AS last_published
            FROM video_records
            WHERE in_scope_date
            """,
            )
            or {}
        )
        return {
            "id": self.run_dir.name,
            "topicQuery": config.get("topic_query"),
            "expandedQueries": _decode_value(config.get("expanded_queries")) or [],
            "language": config.get("language"),
            "startDate": config.get("start_date"),
            "model": config.get("model"),
            "promptVersion": config.get("prompt_version"),
            "status": status.get("status", "unknown"),
            "statusReason": status.get("reason"),
            "startedAt": started.get("started_at"),
            "firstPublished": candidates.get("first_published"),
            "lastPublished": candidates.get("last_published"),
        }

    def _summary(self, connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        row = (
            _fetch_one(
                connection,
                """
            SELECT
                count(*) AS candidates,
                count(*) FILTER (
                    WHERE final_label = 'relevant'
                      AND decision_point = 'transcript'
                ) AS accepted,
                count(*) FILTER (WHERE final_label = 'irrelevant') AS irrelevant,
                count(*) FILTER (WHERE final_label = 'needs_transcript') AS pending,
                count(DISTINCT channel_id) AS channels,
                count(*) FILTER (WHERE views IS NOT NULL) AS videos_with_views,
                count(*) FILTER (WHERE likes IS NOT NULL) AS videos_with_likes,
                sum(coalesce(views, 0)) AS total_views,
                sum(coalesce(likes, 0)) AS total_likes,
                median(views) FILTER (WHERE views IS NOT NULL) AS median_views,
                median(likes) FILTER (WHERE likes IS NOT NULL) AS median_likes,
                avg(views) FILTER (WHERE views IS NOT NULL) AS average_views,
                avg(likes) FILTER (WHERE likes IS NOT NULL) AS average_likes,
                count(*) FILTER (WHERE transcript_available) AS transcripts_available,
                count(*) FILTER (WHERE transcript_seen) AS transcripts_seen,
                sum(transcript_words) AS transcript_words,
                sum(transcript_segments) AS transcript_segments
            FROM video_records
            """,
            )
            or {}
        )
        row["transcriptCoverage"] = _ratio(
            row.get("transcripts_available"), row.get("transcripts_seen")
        )
        row["acceptedRate"] = _ratio(row.get("accepted"), row.get("candidates"))
        row["viewsPerAccepted"] = _ratio(row.get("total_views"), row.get("accepted"))
        return row

    def _funnel(self, connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        row = (
            _fetch_one(
                connection,
                """
            SELECT
                count(*) AS discovered,
                count(*) FILTER (WHERE title IS NOT NULL) AS metadata,
                count(*) FILTER (WHERE final_label IS NOT NULL) AS decided,
                count(*) FILTER (
                    WHERE final_label = 'relevant' AND decision_point = 'transcript'
                ) AS accepted,
                count(*) FILTER (WHERE transcript_available) AS transcribed
            FROM video_records
            """,
            )
            or {}
        )
        total = int(row.get("discovered") or 0)
        return [
            {
                "label": label,
                "count": row.get(key, 0),
                "percent": _percent(row.get(key), total),
            }
            for label, key in (
                ("Discovered", "discovered"),
                ("With metadata", "metadata"),
                ("Classified", "decided"),
                ("Accepted", "accepted"),
                ("Transcribed", "transcribed"),
            )
        ]

    def _timeseries(
        self, connection: duckdb.DuckDBPyConnection
    ) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT
                published_month,
                count(*) AS videos,
                count(*) FILTER (
                    WHERE final_label = 'relevant' AND decision_point = 'transcript'
                ) AS accepted,
                count(*) FILTER (WHERE transcript_available) AS transcribed,
                sum(coalesce(views, 0)) AS views,
                median(views) FILTER (WHERE views IS NOT NULL) AS median_views,
                median(likes) FILTER (WHERE likes IS NOT NULL) AS median_likes
            FROM video_records
            WHERE published_month IS NOT NULL AND in_scope_date
            GROUP BY published_month
            ORDER BY published_month
            """,
        )

    def _topic_timeseries(
        self, connection: duckdb.DuckDBPyConnection
    ) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT
                v.published_month,
                tm.topic_id,
                tm.label,
                tm.color,
                count(DISTINCT v.video_id) AS videos,
                count(DISTINCT v.video_id) FILTER (
                    WHERE v.final_label = 'relevant'
                      AND v.decision_point = 'transcript'
                ) AS accepted
            FROM topic_matches tm
            JOIN video_records v USING (video_id)
            WHERE v.published_month IS NOT NULL AND v.in_scope_date
            GROUP BY v.published_month, tm.topic_id, tm.label, tm.color
            ORDER BY v.published_month, tm.topic_id
            """,
        )

    def _topics(self, connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT
                d.topic_id,
                d.label,
                d.color,
                d.description,
                count(DISTINCT tm.video_id) FILTER (WHERE v.in_scope_date) AS videos,
                count(DISTINCT tm.video_id) FILTER (
                    WHERE v.in_scope_date
                      AND v.final_label = 'relevant'
                      AND v.decision_point = 'transcript'
                ) AS accepted,
                sum(
                    CASE WHEN v.in_scope_date THEN coalesce(v.views, 0) ELSE 0 END
                ) AS views,
                sum(
                    CASE WHEN v.in_scope_date THEN coalesce(v.likes, 0) ELSE 0 END
                ) AS likes,
                median(v.views) FILTER (
                    WHERE v.in_scope_date AND v.views IS NOT NULL
                ) AS median_views,
                avg(v.likes) FILTER (
                    WHERE v.in_scope_date AND v.likes IS NOT NULL
                ) AS average_likes,
                count(DISTINCT tm.video_id) FILTER (
                    WHERE v.in_scope_date AND v.transcript_available
                ) AS transcribed,
                list(DISTINCT tm.matched_keywords) AS matched_keywords
            FROM topic_definitions d
            LEFT JOIN topic_matches tm ON tm.topic_id = d.topic_id
            LEFT JOIN video_records v ON v.video_id = tm.video_id
            GROUP BY d.topic_id, d.label, d.color, d.description
            ORDER BY videos DESC, d.topic_id
            """,
        )

    def _channels(self, connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            WITH topic_counts AS (
                SELECT
                    v.channel_id,
                    count(DISTINCT tm.topic_id) AS topic_count
                FROM video_records v
                LEFT JOIN topic_matches tm USING (video_id)
                GROUP BY v.channel_id
            )
            SELECT
                v.channel_id,
                coalesce(v.channel_title, c.title, v.channel_id) AS channel_title,
                c.subscribers,
                c.channel_views,
                count(*) AS videos,
                count(*) FILTER (
                    WHERE v.final_label = 'relevant'
                      AND v.decision_point = 'transcript'
                ) AS accepted,
                sum(coalesce(v.views, 0)) AS views,
                sum(coalesce(v.likes, 0)) AS likes,
                median(v.views) FILTER (WHERE v.views IS NOT NULL) AS median_views,
                avg(v.likes) FILTER (WHERE v.likes IS NOT NULL) AS average_likes,
                count(*) FILTER (WHERE v.transcript_available) AS transcribed,
                coalesce(max(topic_counts.topic_count), 0) AS topic_count
            FROM video_records v
            LEFT JOIN channel_records c USING (channel_id)
            LEFT JOIN topic_counts USING (channel_id)
            GROUP BY
                v.channel_id,
                coalesce(v.channel_title, c.title, v.channel_id),
                c.subscribers,
                c.channel_views
            ORDER BY views DESC NULLS LAST, videos DESC
            LIMIT 40
            """,
        )

    def _keywords(self, connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT word, count(*) AS mentions, count(DISTINCT video_id) AS videos
            FROM keyword_tokens
            GROUP BY word
            ORDER BY videos DESC, mentions DESC, word
            LIMIT 80
            """,
        )

    def _source_mix(
        self, connection: duckdb.DuckDBPyConnection
    ) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT
                coalesce(discovered_via, 'unknown') AS source,
                count(*) AS videos,
                count(*) FILTER (
                    WHERE final_label = 'relevant'
                      AND decision_point = 'transcript'
                ) AS accepted,
                sum(coalesce(views, 0)) AS views
            FROM video_records
            GROUP BY source
            ORDER BY videos DESC, source
            """,
        )

    def _metadata(self, connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        row = (
            _fetch_one(
                connection,
                """
            SELECT
                count(*) AS videos,
                count(*) FILTER (WHERE views IS NOT NULL) AS views,
                count(*) FILTER (WHERE likes IS NOT NULL) AS likes,
                count(*) FILTER (
                    WHERE keywords IS NOT NULL AND keywords <> 'null'
                ) AS keyword_lists,
                count(*) FILTER (WHERE category IS NOT NULL) AS categories,
                count(*) FILTER (WHERE published_at IS NOT NULL) AS exact_dates,
                count(*) FILTER (
                    WHERE description IS NOT NULL AND length(description) > 0
                ) AS descriptions,
                count(*) FILTER (WHERE transcript_seen) AS transcript_records,
                count(*) FILTER (WHERE transcript_available) AS transcripts_available
            FROM video_records
            """,
            )
            or {}
        )
        total = int(row.get("videos") or 0)
        return {
            **row,
            "coverage": {
                key: _percent(value, total)
                for key, value in (
                    ("views", row.get("views")),
                    ("likes", row.get("likes")),
                    ("keywordLists", row.get("keyword_lists")),
                    ("categories", row.get("categories")),
                    ("exactDates", row.get("exact_dates")),
                    ("descriptions", row.get("descriptions")),
                )
            },
        }

    def _queries(self, connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        return _query_dicts(
            connection,
            """
            SELECT
                json_extract_string(q.payload, '$.query') AS query,
                string_agg(
                    DISTINCT json_extract_string(q.payload, '$.query_kind'), ', '
                ) AS kind,
                string_agg(
                    DISTINCT json_extract_string(q.payload, '$.status'), ', '
                ) AS status,
                count(DISTINCT v.video_id) AS videos,
                count(DISTINCT v.video_id) FILTER (
                    WHERE v.final_label = 'relevant'
                      AND v.decision_point = 'transcript'
                ) AS accepted
            FROM records q
            LEFT JOIN video_records v
                ON v.discovery_query = json_extract_string(q.payload, '$.query')
            WHERE q.record_type = 'query'
            GROUP BY query
            ORDER BY videos DESC, query
            LIMIT 80
            """,
        )

    def _videos(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        params: dict[str, list[str]] | None = None,
        limit: int = 80,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, values = _video_filters(params or {})
        order = _video_order(params or {})
        return _query_dicts(
            connection,
            f"""
            SELECT
                v.video_id,
                v.title,
                v.description,
                v.url,
                v.channel_id,
                v.channel_title,
                v.published_at,
                v.published_month,
                v.duration_seconds,
                v.views,
                v.likes,
                v.category,
                v.keywords,
                v.thumbnail,
                v.discovered_via,
                v.discovery_query,
                v.final_label,
                v.decision_point,
                v.confidence,
                v.reason,
                v.transcript_available,
                v.transcript_words,
                v.transcript_segments,
                list(tm.label ORDER BY tm.topic_id) AS topics,
                list(tm.topic_id ORDER BY tm.topic_id) AS topic_ids
            FROM video_records v
            LEFT JOIN topic_matches tm USING (video_id)
            WHERE {where}
            GROUP BY ALL
            ORDER BY {order}
            LIMIT ? OFFSET ?
            """,
            [*values, limit, offset],
        )

    def _video_count(
        self, connection: duckdb.DuckDBPyConnection, params: dict[str, list[str]]
    ) -> int:
        where, values = _video_filters(params)
        row = (
            _fetch_one(
                connection,
                f"SELECT count(*) AS total FROM video_records v WHERE {where}",
                values,
            )
            or {}
        )
        return int(row.get("total") or 0)


def _ingest_analysis_jsonl(
    connection: duckdb.DuckDBPyConnection, run_dir: Path
) -> None:
    connection.execute(
        """
        CREATE TABLE raw_records (
            source_file VARCHAR NOT NULL,
            source_line BIGINT NOT NULL,
            payload JSON NOT NULL
        )
        """
    )
    for record_type in _ANALYSIS_RECORD_TYPES:
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


def _create_analysis_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE VIEW latest_transcripts AS
        SELECT * EXCLUDE (record_rank)
        FROM (
            SELECT
                transcript_rows.*,
                row_number() OVER (
                    PARTITION BY video_id ORDER BY source_line DESC
                ) AS record_rank
            FROM transcript_rows
        )
        WHERE record_rank = 1
        """
    )
    connection.execute(
        """
        CREATE VIEW transcript_texts AS
        SELECT
            t.video_id,
            coalesce(
                (
                    SELECT string_agg(
                        json_extract_string(segment.value, '$.text'), ' '
                        ORDER BY try_cast(segment.key AS BIGINT)
                    )
                    FROM json_each(t.segments) AS segment
                ),
                ''
            ) AS transcript_text
        FROM latest_transcripts t
        """
    )
    connection.execute(
        """
        CREATE VIEW latest_decisions AS
        SELECT * EXCLUDE (record_rank)
        FROM (
            SELECT
                decisions.*,
                row_number() OVER (
                    PARTITION BY video_id ORDER BY source_line DESC
                ) AS record_rank
            FROM decisions
        )
        WHERE record_rank = 1
        """
    )
    connection.execute(
        """
        CREATE VIEW channel_records AS
        SELECT
            channel_id,
            arg_max(title, source_line) AS title,
            arg_max(subscribers, source_line) AS subscribers,
            arg_max(channel_views, source_line) AS channel_views
        FROM (
            SELECT
                json_extract_string(payload, '$.channel_id') AS channel_id,
                json_extract_string(payload, '$.title') AS title,
                coalesce(
                    try_cast(json_extract_string(payload, '$.subscribers') AS BIGINT),
                    try_cast(
                        json_extract_string(payload, '$.raw_payload.subscribers')
                        AS BIGINT
                    )
                ) AS subscribers,
                coalesce(
                    try_cast(json_extract_string(payload, '$.views') AS BIGINT),
                    try_cast(
                        json_extract_string(payload, '$.raw_payload.views') AS BIGINT
                    )
                ) AS channel_views,
                source_line
            FROM records
            WHERE record_type = 'channel'
        )
        GROUP BY channel_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE topic_definitions (
            topic_id VARCHAR,
            label VARCHAR,
            color VARCHAR,
            description VARCHAR,
            keywords VARCHAR[]
        )
        """
    )
    for topic in TOPIC_DEFINITIONS:
        connection.execute(
            "INSERT INTO topic_definitions VALUES (?, ?, ?, ?, ?)",
            [
                topic["id"],
                topic["label"],
                topic["color"],
                topic["description"],
                list(topic["keywords"]),
            ],
        )
    connection.execute(
        """
        CREATE TEMP TABLE video_records AS
        WITH candidate_rollup AS (
            SELECT
                video_id,
                arg_max(title, source_line) AS title,
                arg_max(description, source_line) AS description,
                arg_max(url, source_line) AS url,
                arg_max(channel_id, source_line) AS channel_id,
                arg_max(channel_title, source_line) AS channel_title,
                arg_max(published_at, source_line) AS published_at,
                arg_max(duration_seconds, source_line) AS duration_seconds,
                arg_max(views, source_line) AS views,
                arg_max(likes, source_line) AS likes,
                arg_max(category, source_line) AS category,
                arg_max(NULLIF(keywords, 'null'), source_line) AS keywords,
                arg_max(thumbnail, source_line) AS thumbnail,
                arg_max(is_live_content, source_line) AS is_live_content,
                arg_max(discovered_via, source_line) AS discovered_via,
                arg_max(discovered_from_id, source_line) AS discovered_from_id,
                arg_max(discovery_query, source_line) AS discovery_query
            FROM video_candidates
            GROUP BY video_id
        )
        SELECT
            c.video_id,
            c.title,
            c.description,
            c.url,
            c.channel_id,
            c.channel_title,
            c.published_at,
            try_cast(c.published_at AS DATE) AS published_date,
            date_trunc('month', try_cast(c.published_at AS DATE)) AS published_month,
            c.duration_seconds,
            c.views,
            c.likes,
            c.category,
            c.keywords,
            c.thumbnail,
            c.is_live_content,
            c.discovered_via,
            c.discovered_from_id,
            c.discovery_query,
            try_cast(c.published_at AS DATE) >= (
                SELECT try_cast(json_extract_string(payload, '$.start_date') AS DATE)
                FROM records
                WHERE record_type = 'run_config'
                ORDER BY source_line DESC
                LIMIT 1
            ) AS in_scope_date,
            d.label AS final_label,
            d.decision_point,
            d.confidence,
            d.reason,
            t.is_available AS transcript_available,
            t.video_id IS NOT NULL AS transcript_seen,
            coalesce(tt.transcript_text, '') AS transcript_text,
            length(regexp_extract_all(
                lower(coalesce(tt.transcript_text, '')),
                '[[:alpha:]][[:alpha:]äöüß0-9-]{2,}'
            )) AS transcript_words,
            coalesce(json_array_length(t.segments), 0) AS transcript_segments,
            lower(concat_ws(
                ' ', c.title, c.description, c.category,
                cast(c.keywords AS VARCHAR), tt.transcript_text
            )) AS search_text
        FROM candidate_rollup c
        LEFT JOIN latest_decisions d USING (video_id)
        LEFT JOIN latest_transcripts t USING (video_id)
        LEFT JOIN transcript_texts tt USING (video_id)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE topic_matches AS
        SELECT
            v.video_id,
            d.topic_id,
            d.label,
            d.color,
            list(k.keyword ORDER BY k.keyword) AS matched_keywords,
            count(*) AS matched_keyword_count
        FROM video_records v
        CROSS JOIN topic_definitions d
        CROSS JOIN unnest(d.keywords) AS k(keyword)
        WHERE contains(v.search_text, lower(k.keyword))
        GROUP BY v.video_id, d.topic_id, d.label, d.color
        """
    )
    connection.execute("CREATE TEMP TABLE stopwords (word VARCHAR)")
    connection.executemany(
        "INSERT INTO stopwords VALUES (?)", [(word,) for word in STOPWORDS]
    )
    connection.execute(
        """
        CREATE TEMP TABLE keyword_tokens AS
        SELECT
            v.video_id,
            token AS word
        FROM video_records v
        CROSS JOIN unnest(regexp_extract_all(
            lower(concat_ws(' ', v.title, v.description)),
            '[[:alpha:]][[:alpha:]äöüß0-9-]{2,}'
        )) AS item(token)
        WHERE length(token) >= 3
          AND v.in_scope_date
          AND token NOT IN (SELECT word FROM stopwords)
        """
    )


def _video_filters(params: dict[str, list[str]]) -> tuple[str, list[Any]]:
    clauses = ["TRUE"]
    values: list[Any] = []
    search = _first(params, "search")
    topic = _first(params, "topic")
    channel = _first(params, "channel")
    label = _first(params, "label")
    if search:
        clauses.append("contains(v.search_text, lower(?))")
        values.append(search)
    if topic:
        clauses.append(
            "EXISTS (SELECT 1 FROM topic_matches tf "
            "WHERE tf.video_id = v.video_id AND tf.topic_id = ?) "
            "AND v.in_scope_date"
        )
        values.append(topic)
    if channel:
        clauses.append("v.channel_id = ?")
        values.append(channel)
    if label == "accepted":
        clauses.append("v.final_label = 'relevant' AND v.decision_point = 'transcript'")
    elif label:
        clauses.append("v.final_label = ?")
        values.append(label)
    return " AND ".join(clauses), values


def _video_order(params: dict[str, list[str]]) -> str:
    order = _first(params, "sort") or "views"
    direction = (
        "ASC" if (_first(params, "direction") or "desc").lower() == "asc" else "DESC"
    )
    columns = {
        "views": "views",
        "likes": "likes",
        "published": "published_at",
        "duration": "duration_seconds",
        "title": "title",
        "confidence": "confidence",
    }
    return f"{columns.get(order, columns['views'])} {direction} NULLS LAST, v.video_id"


def _run_fingerprint(run_dir: Path) -> str:
    entries = []
    for path in sorted(run_dir.glob("*.jsonl")):
        stat = path.stat()
        entries.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(entries)


def _query_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    result = connection.execute(query, params or [])
    columns = [item[0] for item in result.description]
    return [
        {
            column: _decode_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in result.fetchall()
    ]


def _fetch_one(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any] | None = None,
) -> dict[str, Any] | None:
    rows = _query_dicts(connection, query, params)
    return rows[0] if rows else None


def _decode_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, tuple):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _decode_value(item) for key, item in value.items()}
    if value == "null":
        return None
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return _decode_value(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key) or []
    return values[0].strip() if values and values[0].strip() else None


def _bounded_int(
    params: dict[str, list[str]],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(_first(params, key) or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 4)


def _percent(numerator: Any, denominator: Any) -> float:
    ratio = _ratio(numerator, denominator)
    return round((ratio or 0) * 100, 1)


__all__ = ["DuckDBAnalytics", "TOPIC_DEFINITIONS"]
