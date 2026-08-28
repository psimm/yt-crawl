"""Strict, append-only record schemas for an auditable crawler run."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Mapping

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from yt_searchapi.settings import (
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)

SCHEMA_VERSION = "1"


class RunStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    STOPPED_BUDGET = "stopped_budget"
    FAILED = "failed"


class DiscoverySource(StrEnum):
    SEARCH = "search"
    CHANNEL = "channel"
    RELATED = "related"


class QueryKind(StrEnum):
    SEED = "seed"
    INTERVIEW = "interview"
    EXPANDED = "expanded"
    CHANNEL = "channel"


class QueryStatus(StrEnum):
    PLANNED = "planned"
    PARTIAL = "partial"
    EXECUTED = "executed"
    DEFERRED = "deferred"


class RelevanceLabel(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NEEDS_TRANSCRIPT = "needs_transcript"
    DEFERRED_BUDGET = "deferred_budget"
    ERROR = "error"


class DecisionPoint(StrEnum):
    SEARCH_RESULT = "search_result"
    VIDEO_METADATA = "video_metadata"
    TRANSCRIPT = "transcript"


class InterviewExampleKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CONTEXT = "context"


class BudgetKind(StrEnum):
    SEARCH_API_CREDITS = "search_api_credits"
    # Read compatibility only. New runs never create LLM budget events.
    LEGACY_LLM_TOKENS = "llm_tokens"


class BudgetAction(StrEnum):
    SPENT = "spent"
    RESERVED = "reserved"
    RECONCILED = "reconciled"
    RELEASED = "released"
    EXHAUSTED = "exhausted"


class StrictRecord(BaseModel):
    """Shared immutable validation policy for every persisted record."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InterviewAnswerItem(StrictRecord):
    """One confirmed interview value with its user-visible provenance."""

    text: str = Field(min_length=1)
    source: Literal["direct", "suggestion", "edited_suggestion", "custom"]
    suggested_text: str | None = None
    boundary_tested: str | None = None
    suggestion_position: int | None = Field(default=None, ge=1)


class RunRecordBase(StrictRecord):
    """Lineage fields present in every JSONL row."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    recorded_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("run_id")
    @classmethod
    def _strip_run_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run_id must not be blank")
        return value


class RunConfigRecord(RunRecordBase):
    record_type: Literal["run_config"] = "run_config"
    topic_query: str = Field(min_length=1)
    expanded_queries: tuple[str, ...] = ()
    language: str = Field(min_length=1)
    start_date: date
    max_searchapi_credits: int = Field(gt=1)
    transcript_reserve_credits: int = Field(gt=0)
    session_action: Literal["start", "resume"] = "start"
    credits_added: int = Field(default=0, ge=0)
    account_remaining_credits: int | None = Field(default=None, ge=0)
    max_depth: int = Field(default=2, ge=0)
    max_queries: int = Field(default=8, ge=1)
    max_search_pages: int = Field(default=1, ge=1)
    max_channel_pages: int = Field(default=1, ge=1)
    gl: str = Field(default="us", min_length=1)
    hl: str = Field(default="en", min_length=1)
    searchapi_timeout_seconds: float = Field(default=90.0, gt=0)
    searchapi_retries: int = Field(default=DEFAULT_SEARCHAPI_RETRIES, ge=0, le=5)
    searchapi_workers: int = Field(default=DEFAULT_SEARCHAPI_WORKERS, ge=1)
    llm_workers: int = Field(default=DEFAULT_LLM_WORKERS, ge=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    topic_expansion: dict[str, JsonValue] | None = None
    classifier_system_prompt: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_pools(self) -> RunConfigRecord:
        if self.transcript_reserve_credits > self.max_searchapi_credits:
            raise ValueError(
                "transcript_reserve_credits cannot exceed max_searchapi_credits"
            )
        if (
            self.session_action == "start"
            and self.transcript_reserve_credits == self.max_searchapi_credits
        ):
            raise ValueError(
                "a new project needs at least one discovery credit; "
                "transcript_reserve_credits must be smaller than "
                "max_searchapi_credits"
            )
        return self


class RunStatusRecord(RunRecordBase):
    record_type: Literal["run_status"] = "run_status"
    status: RunStatus
    reason: str | None = None


class InterviewAnswerRecord(RunRecordBase):
    record_type: Literal["interview_answer"] = "interview_answer"
    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    answer: str
    example_kind: InterviewExampleKind
    # New runs store the confirmed values as structured items. ``answer`` and
    # ``generated_example`` remain for dashboards and older JSONL readers.
    answer_items: tuple[InterviewAnswerItem, ...] | None = None
    generated_examples: tuple[dict[str, JsonValue], ...] = ()
    generated_example: str | None = None


class QueryRecord(RunRecordBase):
    """A query plan row, including work deferred without an API request."""

    record_type: Literal["query"] = "query"
    query: str = Field(min_length=1)
    query_kind: QueryKind
    status: QueryStatus
    source_request_id: str | None = None


class DiscoveryEdgeRecord(RunRecordBase):
    """One provenance edge from a query/channel/video to a discovered target."""

    record_type: Literal["discovery_edge"] = "discovery_edge"
    video_id: str | None = None
    channel_id: str | None = None
    source_type: DiscoverySource
    source_ref: str = Field(min_length=1)
    depth: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_exactly_one_target(self) -> DiscoveryEdgeRecord:
        if (self.video_id is None) == (self.channel_id is None):
            raise ValueError("exactly one of video_id or channel_id must be set")
        return self


class ChannelRecord(RunRecordBase):
    record_type: Literal["channel"] = "channel"
    channel_id: str = Field(min_length=1)
    title: str | None = None
    url: str | None = None
    discovered_via: DiscoverySource
    discovered_from_id: str | None = None
    source_request_id: str | None = None
    subscribers: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    raw_payload: dict[str, JsonValue] | None = None


class VideoCandidateRecord(RunRecordBase):
    record_type: Literal["video_candidate"] = "video_candidate"
    video_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str | None = None
    description: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    published_at: AwareDatetime | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    discovered_via: DiscoverySource
    discovered_from_id: str | None = None
    discovery_query: str | None = None
    source_request_id: str | None = None
    views: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    category: str | None = None
    keywords: tuple[str, ...] = ()
    thumbnail: str | None = None
    is_live_content: bool | None = None
    raw_payload: dict[str, JsonValue] | None = None


class RelevanceDecisionRecord(RunRecordBase):
    record_type: Literal["relevance_decision"] = "relevance_decision"
    decision_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    label: RelevanceLabel
    decision_point: DecisionPoint
    primary_reason: str = Field(min_length=1)
    requested_language: str = Field(min_length=1)
    detected_language: str | None = None
    language_matches: bool | None
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    llm_input_tokens: int = Field(ge=0)
    llm_output_tokens: int = Field(ge=0)
    transcript_reserved: bool

    @model_validator(mode="after")
    def _require_reserve_for_relevant(self) -> RelevanceDecisionRecord:
        if (
            self.label
            in {
                RelevanceLabel.RELEVANT,
                RelevanceLabel.NEEDS_TRANSCRIPT,
            }
            and not self.transcript_reserved
        ):
            raise ValueError(
                "a relevant or needs-transcript decision must atomically reserve "
                "transcript credits"
            )
        return self


class TranscriptSegment(StrictRecord):
    text: str
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class TranscriptRecord(RunRecordBase):
    record_type: Literal["transcript"] = "transcript"
    video_id: str = Field(min_length=1)
    requested_language: str = Field(min_length=1)
    language: str | None = None
    transcript_type: str | None = None
    is_available: bool
    unavailable_reason: str | None = None
    segments: tuple[TranscriptSegment, ...] = ()
    source_request_id: str | None = None
    searchapi_credits: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_availability(self) -> TranscriptRecord:
        if self.is_available and not self.segments:
            raise ValueError("available transcripts must contain at least one segment")
        if self.is_available and self.unavailable_reason is not None:
            raise ValueError("available transcripts cannot have unavailable_reason")
        if not self.is_available and not self.unavailable_reason:
            raise ValueError("unavailable transcripts require unavailable_reason")
        return self

    @property
    def text(self) -> str:
        """Return all segments joined without duplicating text in JSONL."""

        return " ".join(segment.text.strip() for segment in self.segments).strip()


class ApiCallRecord(RunRecordBase):
    record_type: Literal["api_call"] = "api_call"
    provider: Literal["searchapi", "openai"]
    operation: str = Field(min_length=1)
    request_id: str | None = None
    status: Literal["success", "error", "cache_hit", "not_sent"]
    searchapi_credits: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_cached_input_tokens: int = Field(default=0, ge=0)
    llm_cache_write_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    llm_model: str | None = None
    llm_estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0)
    error: str | None = None


class BudgetEventRecord(RunRecordBase):
    record_type: Literal["budget_event"] = "budget_event"
    budget_kind: BudgetKind
    pool: str = Field(min_length=1)
    action: BudgetAction
    amount: int = Field(ge=0)
    remaining: int = Field(ge=0)
    reservation_id: str | None = None
    purpose: str = Field(min_length=1)


class RunMetricRecord(RunRecordBase):
    record_type: Literal["run_metric"] = "run_metric"
    name: str = Field(min_length=1)
    value: int | float | str | bool
    unit: str | None = None


class RunErrorRecord(RunRecordBase):
    record_type: Literal["run_error"] = "run_error"
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    exception_type: str | None = None
    video_id: str | None = None
    channel_id: str | None = None
    retryable: bool = False


RunRecord = Annotated[
    RunConfigRecord
    | RunStatusRecord
    | InterviewAnswerRecord
    | QueryRecord
    | DiscoveryEdgeRecord
    | ChannelRecord
    | VideoCandidateRecord
    | RelevanceDecisionRecord
    | TranscriptRecord
    | ApiCallRecord
    | BudgetEventRecord
    | RunMetricRecord
    | RunErrorRecord,
    Field(discriminator="record_type"),
]

RUN_RECORD_ADAPTER: TypeAdapter[RunRecord] = TypeAdapter(RunRecord)


def validate_run_record(value: object) -> RunRecord:
    """Validate decoded JSON data against the discriminated record union."""

    if isinstance(value, Mapping) and value.get("record_type") == "run_config":
        value = dict(value)
        value.pop("llm_discovery_tokens", None)
        value.pop("llm_transcript_tokens", None)
    elif (
        isinstance(value, Mapping)
        and value.get("record_type") == "relevance_decision"
    ):
        value = dict(value)
        value.setdefault("primary_reason", value.get("reason"))
        for retired in (
            "reason",
            "confidence",
            "published_after_start_date",
            "criteria",
            "positive_example_ids",
            "negative_example_ids",
        ):
            value.pop(retired, None)
    return RUN_RECORD_ADAPTER.validate_python(value)
