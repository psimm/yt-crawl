"""Budget-first SearchAPI crawler for auditable YouTube topic research."""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel

from yt_crawl.budget import (
    BudgetExceededError,
    SearchApiCreditBudget,
    TranscriptCreditReservation,
)
from yt_crawl.classifier import (
    ClassificationJob,
    RelevanceClassifier,
    RelevanceDecision,
    VideoCandidate,
    compact_relevance_decision,
)
from yt_crawl.client import (
    SearchApiClient,
    SearchApiError,
    is_retryable_searchapi_error,
)
from yt_crawl.dates import (
    is_on_or_after_start_date,
    parse_publication_date,
    select_transcript_name,
)
from yt_crawl.llm_runtime import LoggedLlmClient
from yt_crawl.models import youtube_video
from yt_crawl.prompts import (
    CLASSIFIER_PROMPT_VERSION,
    CompiledClassifierPrompt,
    LlmUsage,
    TopicExpansion,
)
from yt_crawl.records import (
    ApiCallRecord,
    BudgetAction,
    BudgetEventRecord,
    BudgetKind,
    ChannelRecord,
    DecisionPoint,
    DiscoveryEdgeRecord,
    DiscoverySource,
    QueryKind,
    QueryRecord,
    QueryStatus,
    RelevanceDecisionRecord,
    RelevanceLabel,
    RunErrorRecord,
    RunMetricRecord,
    RunStatus,
    RunStatusRecord,
    TranscriptRecord,
    TranscriptSegment,
    VideoCandidateRecord,
)
from yt_crawl.runtime_events import (
    CrawlProgressCallback,
    CrawlProgressSnapshot,
    RuntimeEvent,
    RuntimeEventCallback,
)
from yt_crawl.settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)
from yt_crawl.state import (
    BudgetState,
    CrawlProjectState,
    PageProgress,
    ProjectStateStore,
    validate_pending_transcript_decisions,
    validate_resumable_classifier_prompt,
)
from yt_crawl.storage import JsonlRunWriter

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    topic_query: str
    language: str
    start_date: date
    max_depth: int = 2
    max_queries: int = 8
    max_search_pages: int = 1
    max_channel_pages: int = 1
    transcript_excerpt_chars: int = 12_000
    gl: str = "us"
    hl: str | None = None
    searchapi_timeout_seconds: float = 90.0
    searchapi_retries: int = DEFAULT_SEARCHAPI_RETRIES
    searchapi_workers: int = DEFAULT_SEARCHAPI_WORKERS
    llm_workers: int = DEFAULT_LLM_WORKERS
    model: str = DEFAULT_LLM_MODEL
    llm_api_base: str | None = None

    def __post_init__(self) -> None:
        if not self.topic_query.strip():
            raise ValueError("topic_query must not be blank")
        if not self.language.strip():
            raise ValueError("language must not be blank")
        for name in ("max_queries", "max_search_pages", "max_channel_pages"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_depth < 0:
            raise ValueError("max_depth must not be negative")
        if self.transcript_excerpt_chars < 1_000:
            raise ValueError("transcript_excerpt_chars must be at least 1000")
        if not self.gl.strip():
            raise ValueError("gl must not be blank")
        if self.hl is not None and not self.hl.strip():
            raise ValueError("hl must not be blank when supplied")
        if self.searchapi_timeout_seconds <= 0:
            raise ValueError("searchapi_timeout_seconds must be greater than zero")
        if not 0 <= self.searchapi_retries <= 5:
            raise ValueError("searchapi_retries must be between 0 and 5")
        if self.searchapi_workers < 1:
            raise ValueError("searchapi_workers must be at least 1")
        if self.llm_workers < 1:
            raise ValueError("llm_workers must be at least 1")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.llm_api_base is not None and not self.llm_api_base.strip():
            raise ValueError("llm_api_base must not be blank when supplied")

    @property
    def interface_language(self) -> str:
        """SearchAPI interface language, derived from the selected content language."""

        if self.hl is not None:
            return self.hl.strip()
        return self.language.strip().replace("_", "-").split("-", 1)[0].lower()


@dataclass(frozen=True, slots=True)
class CrawlSummary:
    status: RunStatus
    stop_reason: str
    videos_discovered: int
    videos_evaluated: int
    relevant_videos: int
    transcripts_collected: int
    channels_expanded: int
    queries_executed: int
    pending_videos: int
    transcripts_unavailable: int = 0


@dataclass(frozen=True, slots=True)
class _DiscoveredVideo:
    video_id: str
    title: str
    description: str | None
    channel_id: str | None
    channel_title: str | None
    published_time: str | None
    duration_text: str | None
    source: DiscoverySource
    source_ref: str
    depth: int


@dataclass(frozen=True, slots=True)
class _DiscoveredChannel:
    channel_id: str
    title: str | None
    source: DiscoverySource
    source_ref: str
    depth: int


@dataclass(frozen=True, slots=True)
class _PlannedQuery:
    text: str
    kind: QueryKind


@dataclass(frozen=True, slots=True)
class _TranscriptDispatch:
    discovered: _DiscoveredVideo
    detail: youtube_video.SearchResponse
    candidate: VideoCandidate
    transcript_name: str
    reservation: TranscriptCreditReservation
    cache_hit: bool
    request: dict[str, object]


@dataclass(frozen=True, slots=True)
class _MetadataClassification:
    discovered: _DiscoveredVideo
    detail: youtube_video.SearchResponse
    candidate: VideoCandidate
    transcript_name: str


@dataclass(frozen=True, slots=True)
class _TranscriptClassification:
    discovered: _DiscoveredVideo
    detail: youtube_video.SearchResponse
    candidate: VideoCandidate
    transcript: TranscriptRecord


class ResearchCrawler:
    """Crawl one interviewed research definition within a SearchAPI grant."""

    def __init__(
        self,
        *,
        config: CrawlConfig,
        expansion: TopicExpansion,
        classifier_prompt: CompiledClassifierPrompt,
        searchapi: SearchApiClient,
        classifier: RelevanceClassifier,
        llm_client: LoggedLlmClient,
        search_budget: SearchApiCreditBudget,
        writer: JsonlRunWriter,
        on_progress: ProgressCallback | None = None,
        on_api_event: RuntimeEventCallback | None = None,
        on_crawl_progress: CrawlProgressCallback | None = None,
        state_store: ProjectStateStore | None = None,
        resume_state: CrawlProjectState | None = None,
    ) -> None:
        self.config = config
        self.expansion = expansion
        self.prompt = classifier_prompt
        self.searchapi = searchapi
        self.classifier = classifier
        self.llm_client = llm_client
        self.search_budget = search_budget
        self.writer = writer
        self.on_progress = on_progress or (lambda _stage, _message: None)
        self.on_api_event = on_api_event or (lambda _event: None)
        self.on_crawl_progress = on_crawl_progress or (lambda _snapshot: None)
        self.state_store = state_store

        self._video_queue: deque[_DiscoveredVideo] = deque()
        self._channel_queue: deque[_DiscoveredChannel] = deque()
        self._queued_video_ids: set[str] = set()
        self._queued_channel_ids: set[str] = set()
        self._evaluated_video_ids: set[str] = set()
        self._finalized_video_ids: set[str] = set()
        self._dispositioned_video_ids: set[str] = set()
        self._terminal_video_ids: set[str] = set()
        self._dashboard_pending_video_ids: set[str] = set()
        self._deferred_video_ids: set[str] = set()
        self._expanded_channel_ids: set[str] = set()
        self._planned_queries = self._build_query_plan()
        self._discovered_videos: dict[str, _DiscoveredVideo] = {}
        self._discovered_channels: dict[str, _DiscoveredChannel] = {}
        self._query_progress: dict[str, PageProgress] = {}
        self._channel_progress: dict[str, PageProgress] = {}
        self._started_queries: set[str] = set()
        self._executed_queries: set[str] = set()
        self._relevant_ids: set[str] = set()
        self._transcript_ids: set[str] = set()
        self._unavailable_transcript_ids = self._load_unavailable_transcript_ids()
        self._pending_transcript_contexts: dict[str, dict[str, Any]] = {}
        self._stop_reason = "frontier_exhausted"
        self._planned_records_written = False
        self._checkpoint_dirty = True
        provider_workers = int(getattr(searchapi, "max_workers", 1))
        self._searchapi_workers = max(
            1, min(self.config.searchapi_workers, provider_workers)
        )
        supports_llm_batches = callable(
            getattr(self.classifier, "classify_many_with_usage", None)
        )
        self._pipeline_batch_size = max(
            self._searchapi_workers,
            self.config.llm_workers if supports_llm_batches else 1,
        )
        if resume_state is not None:
            self._restore_state(resume_state)
            self._refresh_frontier()
        self._commit_state("prepared", force=True)

    def run(self) -> CrawlSummary:
        """Continue seed search, classification, and graph expansion."""

        logger.info("Crawler session started run_id={}", self.writer.run_id)
        # A resumed run gets its own current outcome; do not leak the reason a
        # previous attempt failed or exhausted its then-current grant.
        self._stop_reason = "frontier_exhausted"
        self.writer.append(
            RunStatusRecord(run_id=self.writer.run_id, status=RunStatus.STARTED)
        )
        if not self._planned_records_written:
            for query in self._planned_queries:
                self.writer.append(
                    QueryRecord(
                        run_id=self.writer.run_id,
                        query=query.text,
                        query_kind=query.kind,
                        status=QueryStatus.PLANNED,
                    )
                )
            self._planned_records_written = True
        self._deferred_video_ids.clear()
        self._mark_state_dirty()
        self._commit_state("running")

        try:
            while True:
                self._refresh_frontier()
                if self._pipeline_batch_size > 1 and self._pending_transcript_contexts:
                    self._process_pending_transcript_batch()
                    continue
                if self._video_queue:
                    if self._pipeline_batch_size > 1:
                        candidates = self._pop_video_batch()
                        self._mark_state_dirty()
                        try:
                            self._process_video_batch(candidates)
                        except Exception:
                            for candidate in reversed(candidates):
                                if (
                                    candidate.video_id not in self._terminal_video_ids
                                    and candidate.video_id not in self._queued_video_ids
                                ):
                                    self._video_queue.appendleft(candidate)
                                    self._queued_video_ids.add(candidate.video_id)
                            self._mark_state_dirty()
                            self._commit_state("running")
                            raise
                        self._commit_state("running")
                        continue
                    candidate = self._video_queue.popleft()
                    self._queued_video_ids.discard(candidate.video_id)
                    self._mark_state_dirty()
                    try:
                        self._process_video(candidate)
                    except Exception:
                        if candidate.video_id not in self._terminal_video_ids:
                            self._video_queue.appendleft(candidate)
                            self._queued_video_ids.add(candidate.video_id)
                            self._mark_state_dirty()
                        self._commit_state("running")
                        raise
                    self._commit_state("running")
                    continue
                if self._channel_queue:
                    if self._searchapi_workers > 1:
                        channels = self._pop_channel_batch()
                        self._mark_state_dirty()
                        try:
                            self._expand_channel_batch(channels)
                        except Exception:
                            for channel in reversed(channels):
                                if (
                                    self._channel_needs_work(channel.channel_id)
                                    and channel.channel_id
                                    not in self._queued_channel_ids
                                ):
                                    self._channel_queue.appendleft(channel)
                                    self._queued_channel_ids.add(channel.channel_id)
                            self._mark_state_dirty()
                            self._commit_state("running")
                            raise
                        self._commit_state("running")
                        continue
                    channel = self._channel_queue.popleft()
                    self._queued_channel_ids.discard(channel.channel_id)
                    self._mark_state_dirty()
                    try:
                        self._expand_channel(channel)
                    except Exception:
                        self._channel_queue.appendleft(channel)
                        self._queued_channel_ids.add(channel.channel_id)
                        self._mark_state_dirty()
                        self._commit_state("running")
                        raise
                    self._commit_state("running")
                    continue
                query = self._next_query_needing_work()
                if query is not None:
                    if self._searchapi_workers > 1:
                        self._execute_query_batch(self._queries_needing_work())
                    else:
                        self._execute_query(query)
                    continue
                break
        except BudgetExceededError as exc:
            self._stop_reason = str(exc)
            status = RunStatus.STOPPED_BUDGET
            self._record_budget_deferred_videos()
        except Exception as exc:
            self._stop_reason = str(exc)
            status = RunStatus.FAILED
            self._record_error("crawler", exc)
        else:
            status = RunStatus.COMPLETED

        self._record_deferred_queries()
        summary = self._summary(status)
        self._write_summary(summary)
        self.writer.append(
            RunStatusRecord(
                run_id=self.writer.run_id,
                status=status,
                reason=summary.stop_reason,
            )
        )
        self._mark_state_dirty()
        self._commit_state(status.value, force=True)
        logger.info(
            "Crawler session finished run_id={} status={} reason={}",
            self.writer.run_id,
            status.value,
            "crawler_failed" if status is RunStatus.FAILED else summary.stop_reason,
        )
        return summary

    def _emit_api_event(self, event: RuntimeEvent) -> None:
        try:
            self.on_api_event(event)
        except Exception as exc:
            logger.error(
                "Runtime display callback failed provider={} operation={} "
                "phase={} error_type={}",
                event.provider,
                event.operation,
                event.phase,
                type(exc).__name__,
            )

    def _execute_query(self, query: _PlannedQuery) -> None:
        progress = self._query_progress.setdefault(query.text, PageProgress())
        if (
            progress.exhausted
            or progress.pages_completed >= self.config.max_search_pages
        ):
            return
        self._progress("search", f"Searching: {query.text}")
        while progress.pages_completed < self.config.max_search_pages:
            request = {
                "q": query.text,
                "gl": self.config.gl,
                "hl": self.config.interface_language,
            }
            if progress.next_page_token:
                request["sp"] = progress.next_page_token
            response = self._discovery_call(
                "youtube_search",
                "youtube",
                request,
                lambda request=request: self.searchapi.search([request])[0],
            )
            request_id = _request_id(response)
            if progress.first_request_id is None:
                progress.first_request_id = request_id
            self._started_queries.add(query.text)
            self.writer.append(
                QueryRecord(
                    run_id=self.writer.run_id,
                    query=query.text,
                    query_kind=query.kind,
                    status=QueryStatus.PARTIAL,
                    source_request_id=progress.first_request_id,
                )
            )
            self._extract_search_results(response, query.text, request_id)
            next_token = _next_page_token(response)
            previous_token = progress.next_page_token
            progress.pages_completed += 1
            progress.next_page_token = next_token
            progress.exhausted = not next_token or next_token == previous_token
            self._mark_state_dirty()
            if progress.exhausted:
                break
        self._executed_queries.add(query.text)
        self.writer.append(
            QueryRecord(
                run_id=self.writer.run_id,
                query=query.text,
                query_kind=query.kind,
                status=QueryStatus.EXECUTED,
                source_request_id=progress.first_request_id,
            )
        )
        self._mark_state_dirty()
        self._commit_state("running")

    def _execute_query_batch(self, queries: list[_PlannedQuery]) -> None:
        """Fetch one page per independent query concurrently, then commit in order."""

        selected = queries[: self._searchapi_workers]
        requests: list[dict[str, object]] = []
        for query in selected:
            progress = self._query_progress.setdefault(query.text, PageProgress())
            request: dict[str, object] = {
                "q": query.text,
                "gl": self.config.gl,
                "hl": self.config.interface_language,
            }
            if progress.next_page_token:
                request["sp"] = progress.next_page_token
            requests.append(request)
        self._progress("search", f"Searching {len(requests)} query variants")
        completed, blocked = self._discovery_batch(
            "youtube_search",
            "youtube",
            requests,
            lambda prepared: self.searchapi.search(prepared, return_exceptions=True),
        )
        first_error: Exception | None = None
        for query, result in zip(selected, completed, strict=False):
            if isinstance(result, Exception):
                if first_error is None:
                    first_error = result
                continue
            progress = self._query_progress[query.text]
            request_id = _request_id(result)
            if progress.first_request_id is None:
                progress.first_request_id = request_id
            self._started_queries.add(query.text)
            self.writer.append(
                QueryRecord(
                    run_id=self.writer.run_id,
                    query=query.text,
                    query_kind=query.kind,
                    status=QueryStatus.PARTIAL,
                    source_request_id=progress.first_request_id,
                )
            )
            self._extract_search_results(result, query.text, request_id)
            next_token = _next_page_token(result)
            previous_token = progress.next_page_token
            progress.pages_completed += 1
            progress.next_page_token = next_token
            progress.exhausted = not next_token or next_token == previous_token
            if (
                progress.exhausted
                or progress.pages_completed >= self.config.max_search_pages
            ):
                self._executed_queries.add(query.text)
                self.writer.append(
                    QueryRecord(
                        run_id=self.writer.run_id,
                        query=query.text,
                        query_kind=query.kind,
                        status=QueryStatus.EXECUTED,
                        source_request_id=progress.first_request_id,
                    )
                )
            self._mark_state_dirty()
        self._commit_state("running")
        if blocked is not None:
            raise blocked
        if first_error is not None:
            raise first_error

    def _extract_search_results(
        self, response: Any, query: str, request_id: str | None
    ) -> None:
        items: list[Any] = list(response.videos or [])
        for section in response.sections or []:
            items.extend(section.items or [])
        for shorts in response.shorts or []:
            items.extend(shorts.items or [])
        for playlist in response.playlists or []:
            items.extend(playlist.videos or [])
        items.extend(response.movies or [])
        featured = response.featured_channel
        if featured is not None:
            if featured.highlighted_video is not None:
                items.append(featured.highlighted_video)
            items.extend(featured.videos or [])
            self._discover_channel(
                _DiscoveredChannel(
                    channel_id=featured.id or "",
                    title=featured.title,
                    source=DiscoverySource.SEARCH,
                    source_ref=query,
                    depth=1,
                ),
                request_id=request_id,
                raw=featured,
            )

        for item in items:
            self._discover_video_from_model(
                item,
                source=DiscoverySource.SEARCH,
                source_ref=query,
                depth=0,
                request_id=request_id,
            )

        for channel in response.channels or []:
            self._discover_channel(
                _DiscoveredChannel(
                    channel_id=channel.id or "",
                    title=channel.title,
                    source=DiscoverySource.SEARCH,
                    source_ref=query,
                    depth=1,
                ),
                request_id=request_id,
                raw=channel,
            )

    def _pop_video_batch(self) -> list[_DiscoveredVideo]:
        candidates: list[_DiscoveredVideo] = []
        while self._video_queue and len(candidates) < self._pipeline_batch_size:
            candidate = self._video_queue.popleft()
            self._queued_video_ids.discard(candidate.video_id)
            candidates.append(candidate)
        return candidates

    def _process_video_batch(self, candidates: list[_DiscoveredVideo]) -> None:
        """Fetch details within provider capacity, then classify concurrently."""

        fresh = [
            candidate
            for candidate in candidates
            if candidate.video_id not in self._terminal_video_ids
            and candidate.video_id not in self._pending_transcript_contexts
        ]
        requests = [
            {
                "video_id": candidate.video_id,
                "gl": self.config.gl,
                "hl": self.config.interface_language,
            }
            for candidate in fresh
        ]
        completed, blocked = self._discovery_batch(
            "youtube_video",
            "youtube_video",
            requests,
            lambda prepared: self.searchapi.video(prepared, return_exceptions=True),
        )
        first_error: Exception | None = None
        classification_work: list[_MetadataClassification] = []
        for candidate, result in zip(fresh, completed, strict=False):
            if isinstance(result, Exception):
                self._record_error("video_detail", result, video_id=candidate.video_id)
                self._record_error_disposition(
                    candidate.video_id,
                    "video_detail_error",
                    point=DecisionPoint.VIDEO_METADATA,
                    terminal=False,
                )
                if first_error is None:
                    first_error = result
                continue
            work = self._prepare_metadata_classification(candidate, result)
            if work is not None:
                classification_work.append(work)
        classification_error = self._classify_metadata_batch(
            classification_work, defer_transcript=True
        )
        if first_error is None:
            first_error = classification_error
        self._mark_state_dirty()
        self._commit_state("running")
        if self._pending_transcript_contexts:
            self._process_pending_transcript_batch()
        if blocked is not None:
            raise blocked
        if first_error is not None:
            raise first_error

    def _prepare_metadata_classification(
        self,
        discovered: _DiscoveredVideo,
        detail: youtube_video.SearchResponse,
    ) -> _MetadataClassification | None:
        """Apply deterministic gates and build one metadata classification job."""

        self._evaluated_video_ids.add(discovered.video_id)
        self._mark_state_dirty()
        video = detail.video
        if video is None:
            self._record_deterministic_rejection(
                discovered.video_id,
                "video_detail_missing",
                language_matches=None,
            )
            return None
        channel_id = detail.channel.id if detail.channel else discovered.channel_id
        channel_title = (
            detail.channel.name if detail.channel else discovered.channel_title
        )
        title = video.title or discovered.title
        date_evidence = parse_publication_date(video.published_time)
        published_ok = is_on_or_after_start_date(date_evidence, self.config.start_date)
        published_at = _date_to_datetime(date_evidence.parsed)
        self.writer.append(
            VideoCandidateRecord(
                run_id=self.writer.run_id,
                video_id=discovered.video_id,
                title=title,
                description=video.description,
                channel_id=channel_id,
                channel_title=channel_title,
                published_at=(
                    published_at if date_evidence.certainty == "exact" else None
                ),
                duration_seconds=video.length_seconds,
                discovered_via=discovered.source,
                discovered_from_id=discovered.source_ref,
                discovery_query=(
                    discovered.source_ref
                    if discovered.source is DiscoverySource.SEARCH
                    else None
                ),
                source_request_id=_request_id(detail),
                views=video.views,
                likes=video.likes,
                category=video.category,
                keywords=tuple(video.keywords or ()),
                thumbnail=video.thumbnail,
                is_live_content=video.is_live_content,
                raw_payload=video.model_dump(mode="json"),
            )
        )
        if not published_ok:
            reason = (
                "published_before_start_date"
                if date_evidence.certainty == "exact"
                else "publication_date_not_proven"
            )
            self._record_deterministic_rejection(
                discovered.video_id,
                f"{reason}: {date_evidence.raw!r}",
                language_matches=None,
            )
            return None
        languages = [
            (language.name, language.lang)
            for language in detail.available_transcripts_languages or []
        ]
        transcript_name = select_transcript_name(self.config.language, languages)
        if transcript_name is None:
            self._record_deterministic_rejection(
                discovered.video_id,
                "requested_language_transcript_not_available",
                language_matches=False,
            )
            return None
        candidate = VideoCandidate(
            video_id=discovered.video_id,
            title=title,
            description=video.description,
            channel_id=channel_id,
            channel_title=channel_title,
            published_at=published_at,
            duration_text=(
                str(video.length_seconds) if video.length_seconds is not None else None
            ),
            discovery_source=discovered.source.value,
            discovery_reference=discovered.source_ref,
        )
        return _MetadataClassification(
            discovered=discovered,
            detail=detail,
            candidate=candidate,
            transcript_name=transcript_name,
        )

    def _classify_metadata_batch(
        self,
        work: list[_MetadataClassification],
        *,
        defer_transcript: bool,
    ) -> Exception | None:
        if not work:
            return None
        batch_method = getattr(self.classifier, "classify_many_with_usage", None)
        if callable(batch_method):
            results = batch_method(
                [
                    ClassificationJob(candidate=item.candidate, stage="metadata")
                    for item in work
                ],
                self.prompt,
                max_workers=self.config.llm_workers,
                return_exceptions=True,
            )
        else:
            results = []
            for item in work:
                try:
                    with self.llm_client.call_context(
                        "discovery", "classify_video_metadata"
                    ):
                        result = self.classifier.classify_with_usage(
                            item.candidate, self.prompt, stage="metadata"
                        )
                except Exception as exc:
                    result = exc
                results.append(result)
        first_error: Exception | None = None
        for item, result in zip(work, results, strict=True):
            video_id = item.discovered.video_id
            if isinstance(result, Exception):
                self._record_error("metadata_classification", result, video_id=video_id)
                self._record_error_disposition(
                    video_id,
                    "metadata_classification_error",
                    point=DecisionPoint.VIDEO_METADATA,
                    terminal=False,
                )
                if first_error is None:
                    first_error = result
                continue
            if result.output.decision == "irrelevant":
                self._record_llm_decision(
                    video_id,
                    result.output,
                    result.usage,
                    response_id=result.response_id,
                    point=DecisionPoint.VIDEO_METADATA,
                    transcript_reserved=False,
                )
                continue
            self._pending_transcript_contexts[video_id] = {
                "detail": item.detail.model_dump(mode="json"),
                "candidate": item.candidate.model_dump(mode="json"),
                "transcript_name": item.transcript_name,
                "transcript_dispatch_pending": False,
                "metadata_decision_recorded": False,
                "metadata_decision": result.output.model_dump(mode="json"),
                "metadata_usage": result.usage.model_dump(mode="json"),
                "metadata_response_id": result.response_id,
            }
            self._mark_state_dirty()
            if not defer_transcript:
                self._process_pending_transcript_batch()
        return first_error

    def _process_video(
        self,
        discovered: _DiscoveredVideo,
        *,
        prefetched_detail: youtube_video.SearchResponse | None = None,
        defer_transcript: bool = False,
    ) -> None:
        if discovered.video_id in self._terminal_video_ids:
            return
        pending_context = self._pending_transcript_contexts.get(discovered.video_id)
        if pending_context is not None:
            if defer_transcript:
                return
            self._process_pending_transcript_batch()
            return
        self._progress("video", f"Inspecting {discovered.title}")

        logger.info(
            "Inspecting candidate run_id={} video_id={}",
            self.writer.run_id,
            discovered.video_id,
        )
        with logger.contextualize(
            video_id=discovered.video_id, run_id=self.writer.run_id
        ):
            try:
                detail = prefetched_detail or self._discovery_call(
                    "youtube_video",
                    "youtube_video",
                    {
                        "video_id": discovered.video_id,
                        "gl": self.config.gl,
                        "hl": self.config.interface_language,
                    },
                    lambda: self.searchapi.video(
                        [
                            {
                                "video_id": discovered.video_id,
                                "gl": self.config.gl,
                                "hl": self.config.interface_language,
                            }
                        ]
                    )[0],
                )
            except BudgetExceededError:
                raise
            except Exception as exc:
                self._record_error("video_detail", exc, video_id=discovered.video_id)
                self._record_error_disposition(
                    discovered.video_id,
                    "video_detail_error",
                    point=DecisionPoint.VIDEO_METADATA,
                    terminal=False,
                )
                raise

            work = self._prepare_metadata_classification(discovered, detail)
            if work is None:
                return
            first_error = self._classify_metadata_batch(
                [work], defer_transcript=defer_transcript
            )
            if first_error is not None:
                raise first_error

    def _classify_transcript_batch(
        self, work: list[_TranscriptClassification]
    ) -> Exception | None:
        if not work:
            return None
        candidates = [
            item.candidate.model_copy(
                update={
                    "transcript_excerpt": _sample_transcript(
                        item.transcript.text, self.config.transcript_excerpt_chars
                    )
                }
            )
            for item in work
        ]
        batch_method = getattr(self.classifier, "classify_many_with_usage", None)
        if callable(batch_method):
            results = batch_method(
                [
                    ClassificationJob(candidate=candidate, stage="transcript")
                    for candidate in candidates
                ],
                self.prompt,
                max_workers=self.config.llm_workers,
                return_exceptions=True,
            )
        else:
            results = []
            for candidate in candidates:
                try:
                    with self.llm_client.call_context(
                        "transcript", "classify_video_transcript"
                    ):
                        result = self.classifier.classify_with_usage(
                            candidate, self.prompt, stage="transcript"
                        )
                except Exception as exc:
                    result = exc
                results.append(result)
        first_error: Exception | None = None
        for item, result in zip(work, results, strict=True):
            video_id = item.discovered.video_id
            if isinstance(result, Exception):
                self._record_error(
                    "transcript_classification", result, video_id=video_id
                )
                self._record_error_disposition(
                    video_id,
                    "transcript_classification_error",
                    point=DecisionPoint.TRANSCRIPT,
                    transcript_reserved=True,
                    terminal=False,
                )
                if first_error is None:
                    first_error = result
                continue
            self._record_llm_decision(
                video_id,
                result.output,
                result.usage,
                response_id=result.response_id,
                point=DecisionPoint.TRANSCRIPT,
                transcript_reserved=True,
            )
            self._pending_transcript_contexts.pop(video_id, None)
            self._mark_state_dirty()
            if result.output.decision != "relevant":
                continue
            self._relevant_ids.add(video_id)
            self._progress("relevant", f"Relevant: {item.candidate.title}")
            self._enqueue_related(item.detail, item.discovered.depth + 1)
            channel_id = item.candidate.channel_id
            if channel_id:
                self._discover_channel(
                    _DiscoveredChannel(
                        channel_id=channel_id,
                        title=item.candidate.channel_title,
                        source=DiscoverySource.RELATED,
                        source_ref=video_id,
                        depth=item.discovered.depth + 1,
                    ),
                    request_id=_request_id(item.detail),
                    raw=item.detail.channel,
                )
        return first_error

    def _record_pending_metadata_decision(
        self,
        video_id: str,
        *,
        metadata_result: Any | None = None,
    ) -> None:
        context = self._pending_transcript_contexts[video_id]
        if context.get("metadata_decision_recorded"):
            return
        if metadata_result is not None:
            decision = metadata_result.output
            usage = metadata_result.usage
            response_id = metadata_result.response_id
        elif context.get("metadata_decision") is not None:
            decision = compact_relevance_decision(
                context["metadata_decision"],
                requested_language=self.config.language,
            )
            usage = LlmUsage.model_validate(context["metadata_usage"])
            response_id = context.get("metadata_response_id")
        else:
            # Legacy checkpoints recorded this decision before they could save
            # the richer continuation fields.
            context["metadata_decision_recorded"] = True
            self._mark_state_dirty()
            return
        self._record_llm_decision(
            video_id,
            decision,
            usage,
            response_id=response_id,
            point=DecisionPoint.VIDEO_METADATA,
            transcript_reserved=True,
        )
        context["metadata_decision_recorded"] = True
        self._mark_state_dirty()

    def _process_pending_transcript_batch(self) -> None:
        """Dispatch checkpointed independent transcripts as one bounded batch."""

        work: list[_TranscriptDispatch] = []
        classifications: list[_TranscriptClassification] = []
        blocked: BudgetExceededError | None = None
        for video_id in list(self._pending_transcript_contexts):
            if len(work) + len(classifications) >= self._pipeline_batch_size:
                break
            discovered = self._discovered_videos.get(video_id)
            if discovered is None or video_id in self._terminal_video_ids:
                continue
            context = self._pending_transcript_contexts[video_id]
            detail = youtube_video.SearchResponse.model_validate(context["detail"])
            candidate = VideoCandidate.model_validate(context["candidate"])
            transcript_name = str(context["transcript_name"])
            saved_transcript = self._saved_transcript(video_id)
            if saved_transcript is not None:
                classifications.append(
                    _TranscriptClassification(
                        discovered=discovered,
                        detail=detail,
                        candidate=candidate,
                        transcript=saved_transcript,
                    )
                )
                continue
            request: dict[str, object] = {
                "video_id": video_id,
                "lang": self.config.language,
                "transcript_name": transcript_name,
                "only_available": False,
            }
            cache_hit = self._is_searchapi_cached("youtube_transcripts", request)
            reservation = self.search_budget.pending_transcript(video_id)
            if reservation is not None and context.get(
                "transcript_dispatch_pending", True
            ):
                snapshot = self.search_budget.charge_failed_transcript(reservation)
                context["transcript_dispatch_pending"] = False
                self.writer.append(
                    BudgetEventRecord(
                        run_id=self.writer.run_id,
                        budget_kind=BudgetKind.SEARCH_API_CREDITS,
                        pool="transcript",
                        action=BudgetAction.RECONCILED,
                        amount=reservation.credits,
                        remaining=snapshot.transcript_remaining,
                        reservation_id=reservation.reservation_id,
                        purpose=f"transcript:{video_id}:interrupted",
                    )
                )
                reservation = None
                self._mark_state_dirty()
            if reservation is None:
                try:
                    reservation = self.search_budget.mark_relevant(
                        video_id, transcript_credits=0 if cache_hit else 1
                    )
                except BudgetExceededError as exc:
                    blocked = exc
                    break
                self._record_transcript_reservation(reservation)
            context["transcript_dispatch_pending"] = False
            self._mark_state_dirty()
            self._record_pending_metadata_decision(video_id)
            work.append(
                _TranscriptDispatch(
                    discovered=discovered,
                    detail=detail,
                    candidate=candidate,
                    transcript_name=transcript_name,
                    reservation=reservation,
                    cache_hit=cache_hit,
                    request=request,
                )
            )
        if not work:
            classification_error = self._classify_transcript_batch(classifications)
            self._commit_state("running")
            if classification_error is not None:
                raise classification_error
            if blocked is not None:
                raise blocked
            return
        final_errors: dict[str, Exception] = {}
        pending = work
        for attempt in range(self.config.searchapi_retries + 1):
            if attempt:
                self._sleep_before_searchapi_retry("youtube_transcripts", attempt)
            responses, starts = self._dispatch_transcript_batch(pending)
            retry_work: list[_TranscriptDispatch] = []
            for item, response, started in zip(pending, responses, starts, strict=True):
                video_id = item.discovered.video_id
                try:
                    transcript = self._apply_transcript_result(
                        item,
                        response,
                        started=started,
                    )
                except Exception as exc:
                    final_errors[video_id] = exc
                    if (
                        attempt < self.config.searchapi_retries
                        and blocked is None
                        and is_retryable_searchapi_error(exc)
                    ):
                        try:
                            retry_work.append(self._prepare_transcript_retry(item))
                        except BudgetExceededError as budget_exc:
                            blocked = budget_exc
                    continue
                final_errors.pop(video_id, None)
                if transcript is None:
                    self._pending_transcript_contexts.pop(video_id, None)
                    self._mark_state_dirty()
                    continue
                classifications.append(
                    _TranscriptClassification(
                        discovered=item.discovered,
                        detail=item.detail,
                        candidate=item.candidate,
                        transcript=transcript,
                    )
                )
            if not retry_work:
                break
            logger.warning(
                "Retrying {} failed transcript request(s) retry={}/{}",
                len(retry_work),
                attempt + 1,
                self.config.searchapi_retries,
            )
            pending = retry_work
        classification_error = self._classify_transcript_batch(classifications)
        self._commit_state("running")
        if blocked is not None:
            raise blocked
        if final_errors:
            raise next(iter(final_errors.values()))
        if classification_error is not None:
            raise classification_error

    def _dispatch_transcript_batch(
        self, work: list[_TranscriptDispatch]
    ) -> tuple[list[Any | Exception], list[float]]:
        self._progress("transcript", f"Collecting {len(work)} transcripts")
        for item in work:
            self._pending_transcript_contexts[item.discovered.video_id][
                "transcript_dispatch_pending"
            ] = True
        self._mark_state_dirty()
        # This is the checkpoint-before-provider boundary. An interruption after
        # it is conservatively treated as a potentially billed dispatch.
        self._commit_state("running", force=True)
        starts = [time.perf_counter() for _item in work]
        for item in work:
            self._emit_api_event(
                RuntimeEvent(
                    provider="searchapi",
                    operation="youtube_transcripts",
                    phase="started",
                    cache_hit=item.cache_hit,
                )
            )
        try:
            requests = [item.request for item in work]
            if len(requests) == 1:
                responses = list(self.searchapi.transcripts(requests))
            else:
                responses = list(
                    self.searchapi.transcripts(requests, return_exceptions=True)
                )
        except Exception as exc:
            responses = [exc for _item in work]
        if len(responses) != len(work):
            mismatch = RuntimeError(
                "youtube_transcripts returned an unexpected batch size"
            )
            responses = [mismatch for _item in work]
        return responses, starts

    def _prepare_transcript_retry(
        self, item: _TranscriptDispatch
    ) -> _TranscriptDispatch:
        video_id = item.discovered.video_id
        cache_hit = self._is_searchapi_cached("youtube_transcripts", item.request)
        reservation = self.search_budget.mark_relevant(
            video_id, transcript_credits=0 if cache_hit else 1
        )
        self._record_transcript_reservation(reservation)
        self._pending_transcript_contexts[video_id]["transcript_dispatch_pending"] = (
            False
        )
        self._mark_state_dirty()
        return replace(item, reservation=reservation, cache_hit=cache_hit)

    def _apply_transcript_result(
        self,
        item: _TranscriptDispatch,
        response: Any | Exception,
        *,
        started: float,
    ) -> TranscriptRecord | None:
        """Reconcile and persist one already-dispatched transcript result."""

        video_id = item.discovered.video_id
        if isinstance(response, Exception):
            snapshot = self.search_budget.charge_failed_transcript(item.reservation)
            self._pending_transcript_contexts[video_id][
                "transcript_dispatch_pending"
            ] = False
            self._mark_state_dirty()
            self.writer.append(
                ApiCallRecord(
                    run_id=self.writer.run_id,
                    provider="searchapi",
                    operation="youtube_transcripts",
                    status="error",
                    searchapi_credits=item.reservation.credits,
                    latency_seconds=time.perf_counter() - started,
                    error=str(response),
                )
            )
            self._emit_api_event(
                RuntimeEvent(
                    provider="searchapi",
                    operation="youtube_transcripts",
                    phase="finished",
                    status="error",
                    cache_hit=item.cache_hit,
                    error=str(response),
                )
            )
            self.writer.append(
                BudgetEventRecord(
                    run_id=self.writer.run_id,
                    budget_kind=BudgetKind.SEARCH_API_CREDITS,
                    pool="transcript",
                    action=BudgetAction.RECONCILED,
                    amount=item.reservation.credits,
                    remaining=snapshot.transcript_remaining,
                    reservation_id=item.reservation.reservation_id,
                    purpose=f"transcript:{video_id}:error",
                )
            )
            self._record_error("transcript", response, video_id=video_id)
            self._record_error_disposition(
                video_id,
                "transcript_request_error",
                point=DecisionPoint.TRANSCRIPT,
                transcript_reserved=True,
                terminal=False,
            )
            raise response

        snapshot = self.search_budget.reconcile_transcript(
            item.reservation, actual_credits=item.reservation.credits
        )
        self._pending_transcript_contexts[video_id]["transcript_dispatch_pending"] = (
            False
        )
        self._mark_state_dirty()
        request_id = _request_id(response)
        segments = tuple(
            TranscriptSegment(
                text=segment.text or "",
                start_seconds=max(0.0, float(segment.start or 0.0)),
                duration_seconds=max(0.0, float(segment.duration or 0.0)),
            )
            for segment in response.transcripts or []
            if segment.text
        )
        available = bool(segments) and not response.error
        unavailable_reason = (
            None if available else (response.error or "empty_transcript")
        )
        api_status = (
            "cache_hit" if item.cache_hit else ("success" if available else "error")
        )
        self.writer.append(
            ApiCallRecord(
                run_id=self.writer.run_id,
                provider="searchapi",
                operation="youtube_transcripts",
                request_id=request_id,
                status=api_status,
                searchapi_credits=item.reservation.credits,
                latency_seconds=time.perf_counter() - started,
                error=unavailable_reason,
            )
        )
        self._emit_api_event(
            RuntimeEvent(
                provider="searchapi",
                operation="youtube_transcripts",
                phase="finished",
                status=api_status,
                cache_hit=item.cache_hit,
                error=unavailable_reason,
            )
        )
        self.writer.append(
            BudgetEventRecord(
                run_id=self.writer.run_id,
                budget_kind=BudgetKind.SEARCH_API_CREDITS,
                pool="transcript",
                action=BudgetAction.RECONCILED,
                amount=item.reservation.credits,
                remaining=snapshot.transcript_remaining,
                reservation_id=item.reservation.reservation_id,
                purpose=f"transcript:{video_id}",
            )
        )
        record = TranscriptRecord(
            run_id=self.writer.run_id,
            video_id=video_id,
            requested_language=self.config.language,
            language=self.config.language if available else None,
            transcript_type=None,
            is_available=available,
            unavailable_reason=unavailable_reason,
            segments=segments,
            source_request_id=request_id,
            searchapi_credits=item.reservation.credits,
        )
        self.writer.append(record)
        if not available:
            self._unavailable_transcript_ids.add(video_id)
            self._record_error(
                "transcript",
                RuntimeError(unavailable_reason or "transcript unavailable"),
                video_id=video_id,
            )
            self._record_error_disposition(
                video_id,
                "transcript_response_error" if response.error else "empty_transcript",
                point=DecisionPoint.TRANSCRIPT,
                transcript_reserved=True,
            )
            return None
        self._transcript_ids.add(video_id)
        self._unavailable_transcript_ids.discard(video_id)
        self._mark_state_dirty()
        return record

    def _load_unavailable_transcript_ids(self) -> set[str]:
        """Restore latest confirmed transcript availability from the audit stream."""

        path = self.writer.path_for("transcript")
        if not path.is_file():
            return set()
        latest: dict[str, bool] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = TranscriptRecord.model_validate_json(line)
            latest[record.video_id] = record.is_available
        return {video_id for video_id, available in latest.items() if not available}

    def _saved_transcript(self, video_id: str) -> TranscriptRecord | None:
        if video_id not in self._transcript_ids:
            return None
        path = self.writer.path_for("transcript")
        if not path.is_file():
            return None
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            record = TranscriptRecord.model_validate_json(line)
            if record.video_id == video_id and record.is_available:
                return record
        return None

    def _pop_channel_batch(self) -> list[_DiscoveredChannel]:
        channels: list[_DiscoveredChannel] = []
        while self._channel_queue and len(channels) < self._searchapi_workers:
            channel = self._channel_queue.popleft()
            self._queued_channel_ids.discard(channel.channel_id)
            channels.append(channel)
        return channels

    def _expand_channel_batch(self, channels: list[_DiscoveredChannel]) -> None:
        """Fetch one page per independent channel, preserving page-chain order."""

        selected = [
            channel
            for channel in channels
            if channel.channel_id
            and channel.depth <= self.config.max_depth
            and self._channel_needs_work(channel.channel_id)
        ]
        requests: list[dict[str, object]] = []
        for channel in selected:
            progress = self._channel_progress.setdefault(
                channel.channel_id, PageProgress()
            )
            request: dict[str, object] = {
                "channel_id": channel.channel_id,
                "gl": self.config.gl,
                "hl": self.config.interface_language,
            }
            if progress.next_page_token:
                request["next_page_token"] = progress.next_page_token
            requests.append(request)
        self._progress("channel", f"Scanning {len(requests)} channels")
        completed, blocked = self._discovery_batch(
            "youtube_channel_videos",
            "youtube_channel_videos",
            requests,
            lambda prepared: self.searchapi.channel_videos(
                prepared, return_exceptions=True
            ),
        )
        first_error: Exception | None = None
        for channel, result in zip(selected, completed, strict=False):
            if isinstance(result, Exception):
                self._record_error("channel", result, channel_id=channel.channel_id)
                if self._skip_empty_channel_result(channel, result):
                    continue
                if first_error is None:
                    first_error = result
                continue
            self._apply_channel_response(channel, result)
        if blocked is not None:
            raise blocked
        if first_error is not None:
            raise first_error

    def _expand_channel(self, channel: _DiscoveredChannel) -> None:
        if not channel.channel_id or channel.depth > self.config.max_depth:
            return
        progress = self._channel_progress.setdefault(channel.channel_id, PageProgress())
        if (
            progress.exhausted
            or progress.pages_completed >= self.config.max_channel_pages
        ):
            return
        self._progress(
            "channel", f"Scanning channel {channel.title or channel.channel_id}"
        )
        while progress.pages_completed < self.config.max_channel_pages:
            request = {
                "channel_id": channel.channel_id,
                "gl": self.config.gl,
                "hl": self.config.interface_language,
            }
            if progress.next_page_token:
                request["next_page_token"] = progress.next_page_token
            try:
                response = self._discovery_call(
                    "youtube_channel_videos",
                    "youtube_channel_videos",
                    request,
                    lambda request=request: self.searchapi.channel_videos([request])[0],
                )
            except BudgetExceededError:
                raise
            except Exception as exc:
                self._record_error("channel", exc, channel_id=channel.channel_id)
                if self._skip_empty_channel_result(channel, exc):
                    return
                raise
            self._apply_channel_response(channel, response)
            if progress.exhausted:
                break

    def _skip_empty_channel_result(
        self, channel: _DiscoveredChannel, exc: Exception
    ) -> bool:
        if not _is_empty_channel_result_error(exc):
            return False
        progress = self._channel_progress.setdefault(channel.channel_id, PageProgress())
        progress.pages_completed += 1
        progress.next_page_token = None
        progress.exhausted = True
        self._progress(
            "channel",
            f"No videos returned for {channel.title or channel.channel_id}; skipping",
        )
        self._mark_state_dirty()
        return True

    def _apply_channel_response(
        self, channel: _DiscoveredChannel, response: Any
    ) -> None:
        progress = self._channel_progress[channel.channel_id]
        self._expanded_channel_ids.add(channel.channel_id)
        request_id = _request_id(response)
        if response.channel is not None:
            self.writer.append(
                ChannelRecord(
                    run_id=self.writer.run_id,
                    channel_id=channel.channel_id,
                    title=response.channel.title or channel.title,
                    discovered_via=channel.source,
                    discovered_from_id=channel.source_ref,
                    source_request_id=request_id,
                    subscribers=response.channel.subscribers,
                    views=None,
                    raw_payload=response.channel.model_dump(mode="json"),
                )
            )
        for video in response.videos or []:
            self._discover_video_from_model(
                video,
                source=DiscoverySource.CHANNEL,
                source_ref=channel.channel_id,
                depth=channel.depth,
                request_id=request_id,
            )
        next_token = _next_page_token(response)
        previous_token = progress.next_page_token
        progress.pages_completed += 1
        progress.next_page_token = next_token
        progress.exhausted = not next_token or next_token == previous_token
        self._mark_state_dirty()

    def _enqueue_related(self, detail: Any, depth: int) -> None:
        recommended = detail.recommended_videos
        if recommended is None:
            return
        source_id = detail.video.id if detail.video else "unknown"
        request_id = _request_id(detail)
        for video in recommended.videos or []:
            self._discover_video_from_model(
                video,
                source=DiscoverySource.RELATED,
                source_ref=source_id,
                depth=depth,
                request_id=request_id,
            )

    def _discover_video_from_model(
        self,
        item: Any,
        *,
        source: DiscoverySource,
        source_ref: str,
        depth: int,
        request_id: str | None,
    ) -> None:
        video_id = getattr(item, "id", None)
        title = getattr(item, "title", None)
        if not video_id or not title:
            return
        channel = getattr(item, "channel", None)
        discovered = _DiscoveredVideo(
            video_id=video_id,
            title=title,
            description=getattr(item, "description", None),
            channel_id=getattr(channel, "id", None) if channel else None,
            channel_title=getattr(channel, "title", None) if channel else None,
            published_time=getattr(item, "published_time", None),
            duration_text=(
                getattr(item, "length", None) or getattr(item, "duration", None)
            ),
            source=source,
            source_ref=source_ref,
            depth=depth,
        )
        previous = self._discovered_videos.get(video_id)
        if previous is None or discovered.depth < previous.depth:
            self._discovered_videos[video_id] = discovered
        current = self._discovered_videos[video_id]
        if (
            current.depth <= self.config.max_depth
            and video_id not in self._terminal_video_ids
        ):
            self._dashboard_pending_video_ids.add(video_id)
        else:
            self._dashboard_pending_video_ids.discard(video_id)
        self.writer.append(
            DiscoveryEdgeRecord(
                run_id=self.writer.run_id,
                video_id=video_id,
                source_type=source,
                source_ref=source_ref,
                depth=depth,
            )
        )
        self.writer.append(
            VideoCandidateRecord(
                run_id=self.writer.run_id,
                video_id=video_id,
                title=title,
                url=getattr(item, "link", None),
                description=discovered.description,
                channel_id=discovered.channel_id,
                channel_title=discovered.channel_title,
                duration_seconds=_duration_seconds(discovered.duration_text),
                discovered_via=source,
                discovered_from_id=source_ref,
                discovery_query=source_ref
                if source is DiscoverySource.SEARCH
                else None,
                source_request_id=request_id,
                raw_payload=_model_payload(item),
            )
        )
        if (
            depth <= self.config.max_depth
            and video_id not in self._queued_video_ids
            and video_id not in self._terminal_video_ids
        ):
            self._queued_video_ids.add(video_id)
            self._video_queue.append(discovered)
        if (
            channel is not None
            and discovered.channel_id
            and source is not DiscoverySource.CHANNEL
        ):
            self._discover_channel(
                _DiscoveredChannel(
                    channel_id=discovered.channel_id,
                    title=discovered.channel_title,
                    source=source,
                    source_ref=source_ref,
                    depth=depth + 1,
                ),
                request_id=request_id,
                raw=channel,
            )

    def _discover_channel(
        self,
        channel: _DiscoveredChannel,
        *,
        request_id: str | None,
        raw: Any | None = None,
    ) -> None:
        if not channel.channel_id:
            return
        previous = self._discovered_channels.get(channel.channel_id)
        if previous is None or channel.depth < previous.depth:
            self._discovered_channels[channel.channel_id] = channel
        self.writer.append(
            DiscoveryEdgeRecord(
                run_id=self.writer.run_id,
                channel_id=channel.channel_id,
                source_type=channel.source,
                source_ref=channel.source_ref,
                depth=channel.depth,
            )
        )
        self.writer.append(
            ChannelRecord(
                run_id=self.writer.run_id,
                channel_id=channel.channel_id,
                title=channel.title,
                url=getattr(raw, "link", None),
                discovered_via=channel.source,
                discovered_from_id=channel.source_ref,
                source_request_id=request_id,
                raw_payload=_model_payload(raw),
            )
        )
        if (
            channel.depth <= self.config.max_depth
            and channel.channel_id not in self._queued_channel_ids
            and self._channel_needs_work(channel.channel_id)
        ):
            self._queued_channel_ids.add(channel.channel_id)
            self._channel_queue.append(channel)

    def _discovery_call(
        self,
        operation: str,
        engine: str,
        request: dict[str, object],
        call: Callable[[], Any],
    ) -> Any:
        """Run one discovery request with separately budgeted retry attempts."""

        for attempt in range(self.config.searchapi_retries + 1):
            if attempt:
                self._sleep_before_searchapi_retry(operation, attempt)
            try:
                return self._discovery_attempt(operation, engine, request, call)
            except BudgetExceededError:
                raise
            except Exception as exc:
                if (
                    attempt >= self.config.searchapi_retries
                    or not is_retryable_searchapi_error(exc)
                ):
                    raise
                logger.warning(
                    "Retrying SearchAPI operation={} after error_type={} retry={}/{}",
                    operation,
                    type(exc).__name__,
                    attempt + 1,
                    self.config.searchapi_retries,
                )
        raise AssertionError("unreachable SearchAPI retry loop")

    def _discovery_attempt(
        self,
        operation: str,
        engine: str,
        request: dict[str, object],
        call: Callable[[], Any],
    ) -> Any:
        cache_hit = self._is_searchapi_cached(engine, request)
        if cache_hit:
            snapshot = self.search_budget.snapshot()
        else:
            snapshot = self.search_budget.spend_discovery()
            self._mark_state_dirty()
        self.writer.append(
            BudgetEventRecord(
                run_id=self.writer.run_id,
                budget_kind=BudgetKind.SEARCH_API_CREDITS,
                pool="discovery",
                action=BudgetAction.SPENT,
                amount=0 if cache_hit else 1,
                remaining=snapshot.discovery_remaining,
                purpose=f"{operation}:cache_hit" if cache_hit else operation,
            )
        )
        # A dispatched timeout is still charged locally. Commit the complete
        # current frontier and charge before entering the provider.
        self._commit_state("running", force=True)
        started = time.perf_counter()
        self._emit_api_event(
            RuntimeEvent(
                provider="searchapi",
                operation=operation,
                phase="started",
                cache_hit=cache_hit,
            )
        )
        try:
            response = call()
        except Exception as exc:
            self.writer.append(
                ApiCallRecord(
                    run_id=self.writer.run_id,
                    provider="searchapi",
                    operation=operation,
                    status="error",
                    searchapi_credits=0 if cache_hit else 1,
                    latency_seconds=time.perf_counter() - started,
                    error=str(exc),
                )
            )
            self._emit_api_event(
                RuntimeEvent(
                    provider="searchapi",
                    operation=operation,
                    phase="finished",
                    status="error",
                    cache_hit=cache_hit,
                    error=str(exc),
                )
            )
            raise
        self.writer.append(
            ApiCallRecord(
                run_id=self.writer.run_id,
                provider="searchapi",
                operation=operation,
                request_id=_request_id(response),
                status="cache_hit" if cache_hit else "success",
                searchapi_credits=0 if cache_hit else 1,
                latency_seconds=time.perf_counter() - started,
                error=getattr(response, "error", None),
            )
        )
        self._emit_api_event(
            RuntimeEvent(
                provider="searchapi",
                operation=operation,
                phase="finished",
                status="cache_hit" if cache_hit else "success",
                cache_hit=cache_hit,
                error=getattr(response, "error", None),
            )
        )
        return response

    def _discovery_batch(
        self,
        operation: str,
        engine: str,
        requests: list[dict[str, object]],
        call: Callable[[list[dict[str, object]]], list[Any]],
    ) -> tuple[list[Any | Exception], BudgetExceededError | None]:
        """Retry only failed batch members, charging every provider attempt."""

        outcomes: list[Any | Exception | None] = [None] * len(requests)
        pending_indices = list(range(len(requests)))
        blocked: BudgetExceededError | None = None
        for attempt in range(self.config.searchapi_retries + 1):
            if attempt:
                self._sleep_before_searchapi_retry(operation, attempt)
            attempted, attempt_blocked = self._discovery_batch_attempt(
                operation,
                engine,
                [requests[index] for index in pending_indices],
                call,
            )
            attempted_indices = pending_indices[: len(attempted)]
            for index, result in zip(attempted_indices, attempted, strict=True):
                outcomes[index] = result
            if attempt_blocked is not None:
                blocked = attempt_blocked
                break
            pending_indices = [
                index
                for index in attempted_indices
                if isinstance(outcomes[index], Exception)
                and is_retryable_searchapi_error(outcomes[index])
            ]
            if not pending_indices:
                break
            if attempt < self.config.searchapi_retries:
                logger.warning(
                    "Retrying {} failed SearchAPI batch item(s) operation={} "
                    "retry={}/{}",
                    len(pending_indices),
                    operation,
                    attempt + 1,
                    self.config.searchapi_retries,
                )

        completed = [result for result in outcomes if result is not None]
        return completed, blocked

    def _discovery_batch_attempt(
        self,
        operation: str,
        engine: str,
        requests: list[dict[str, object]],
        call: Callable[[list[dict[str, object]]], list[Any]],
    ) -> tuple[list[Any | Exception], BudgetExceededError | None]:
        """Pre-charge, dispatch, and audit one bounded batch attempt."""

        prepared: list[dict[str, object]] = []
        cache_hits: list[bool] = []
        started: list[float] = []
        blocked: BudgetExceededError | None = None
        for request in requests:
            cache_hit = self._is_searchapi_cached(engine, request)
            try:
                if cache_hit:
                    snapshot = self.search_budget.snapshot()
                else:
                    snapshot = self.search_budget.spend_discovery()
                    self._mark_state_dirty()
            except BudgetExceededError as exc:
                blocked = exc
                break
            self.writer.append(
                BudgetEventRecord(
                    run_id=self.writer.run_id,
                    budget_kind=BudgetKind.SEARCH_API_CREDITS,
                    pool="discovery",
                    action=BudgetAction.SPENT,
                    amount=0 if cache_hit else 1,
                    remaining=snapshot.discovery_remaining,
                    purpose=f"{operation}:cache_hit" if cache_hit else operation,
                )
            )
            prepared.append(request)
            cache_hits.append(cache_hit)
        if not prepared:
            return [], blocked
        # Charge and checkpoint the whole attempt once. No provider member is
        # dispatched until every prepared request is durably represented.
        self._commit_state("running", force=True)
        for cache_hit in cache_hits:
            started.append(time.perf_counter())
            self._emit_api_event(
                RuntimeEvent(
                    provider="searchapi",
                    operation=operation,
                    phase="started",
                    cache_hit=cache_hit,
                )
            )
        try:
            results = list(call(prepared))
        except Exception as exc:
            results = [exc for _request in prepared]
        if len(results) != len(prepared):
            mismatch = RuntimeError(
                f"{operation} returned {len(results)} results for "
                f"{len(prepared)} requests"
            )
            results = [mismatch for _request in prepared]
        for result, cache_hit, item_started in zip(
            results, cache_hits, started, strict=True
        ):
            if isinstance(result, Exception):
                status = "error"
                request_id = None
                error = str(result)
            else:
                status = "cache_hit" if cache_hit else "success"
                request_id = _request_id(result)
                error = getattr(result, "error", None)
            self.writer.append(
                ApiCallRecord(
                    run_id=self.writer.run_id,
                    provider="searchapi",
                    operation=operation,
                    request_id=request_id,
                    status=status,
                    searchapi_credits=0 if cache_hit else 1,
                    latency_seconds=time.perf_counter() - item_started,
                    error=error,
                )
            )
            self._emit_api_event(
                RuntimeEvent(
                    provider="searchapi",
                    operation=operation,
                    phase="finished",
                    status=status,
                    cache_hit=cache_hit,
                    error=error,
                )
            )
        return results, blocked

    def _sleep_before_searchapi_retry(self, operation: str, retry: int) -> None:
        maximum = min(30.0, float(2 ** (retry - 1)))
        delay = random.uniform(maximum / 2, maximum)
        self._progress(
            "retry",
            f"Retrying {operation} in {delay:.1f}s "
            f"({retry}/{self.config.searchapi_retries})",
        )
        time.sleep(delay)

    def _is_searchapi_cached(self, engine: str, request: dict[str, object]) -> bool:
        checker = getattr(self.searchapi, "is_cached", None)
        return bool(checker(engine, request)) if callable(checker) else False

    def _record_transcript_reservation(
        self, reservation: TranscriptCreditReservation
    ) -> None:
        self.writer.append(
            BudgetEventRecord(
                run_id=self.writer.run_id,
                budget_kind=BudgetKind.SEARCH_API_CREDITS,
                pool="transcript",
                action=BudgetAction.RESERVED,
                amount=reservation.credits,
                remaining=self.search_budget.snapshot().transcript_remaining,
                reservation_id=reservation.reservation_id,
                purpose=f"transcript:{reservation.video_id}",
            )
        )

    def _record_llm_decision(
        self,
        video_id: str,
        decision: RelevanceDecision,
        usage: LlmUsage,
        *,
        response_id: str | None,
        point: DecisionPoint,
        transcript_reserved: bool,
    ) -> None:
        if decision.decision == "relevant" and point is DecisionPoint.VIDEO_METADATA:
            # This is a lifecycle state, not a third model decision. The
            # metadata result has earned a transcript reservation but cannot
            # become a final inclusion until transcript classification.
            label = RelevanceLabel.NEEDS_TRANSCRIPT
        elif decision.decision == "relevant":
            label = RelevanceLabel.RELEVANT
        else:
            label = RelevanceLabel.IRRELEVANT
        self.writer.append(
            RelevanceDecisionRecord(
                run_id=self.writer.run_id,
                decision_id=uuid4().hex,
                video_id=video_id,
                label=label,
                decision_point=point,
                primary_reason=decision.primary_reason,
                requested_language=self.config.language,
                detected_language=decision.detected_language,
                language_matches=(
                    None
                    if decision.language_match == "unknown"
                    else decision.language_match == "match"
                ),
                model=self.classifier.model,
                prompt_version=CLASSIFIER_PROMPT_VERSION,
                prompt_sha256=self.prompt.prompt_sha256,
                response_id=response_id,
                llm_input_tokens=usage.input_tokens,
                llm_output_tokens=usage.output_tokens,
                transcript_reserved=transcript_reserved,
            )
        )
        self._dispositioned_video_ids.add(video_id)
        if label in {RelevanceLabel.RELEVANT, RelevanceLabel.IRRELEVANT}:
            self._finalized_video_ids.add(video_id)
            self._mark_video_terminal(video_id)

    def _mark_video_terminal(self, video_id: str) -> None:
        """Keep terminal scheduling state and dashboard pending state aligned."""

        self._terminal_video_ids.add(video_id)
        self._dashboard_pending_video_ids.discard(video_id)

    def _record_deterministic_rejection(
        self,
        video_id: str,
        reason: str,
        *,
        language_matches: bool | None,
        point: DecisionPoint = DecisionPoint.VIDEO_METADATA,
        transcript_reserved: bool = False,
    ) -> None:
        self.writer.append(
            RelevanceDecisionRecord(
                run_id=self.writer.run_id,
                decision_id=uuid4().hex,
                video_id=video_id,
                label=RelevanceLabel.IRRELEVANT,
                decision_point=point,
                primary_reason=reason,
                requested_language=self.config.language,
                language_matches=language_matches,
                model="deterministic-gates",
                prompt_version="gates-v1",
                prompt_sha256=self.prompt.prompt_sha256,
                llm_input_tokens=0,
                llm_output_tokens=0,
                transcript_reserved=transcript_reserved,
            )
        )
        self._dispositioned_video_ids.add(video_id)
        self._finalized_video_ids.add(video_id)
        self._mark_video_terminal(video_id)

    def _record_error_disposition(
        self,
        video_id: str,
        reason: str,
        *,
        point: DecisionPoint,
        transcript_reserved: bool = False,
        terminal: bool = True,
    ) -> None:
        self.writer.append(
            RelevanceDecisionRecord(
                run_id=self.writer.run_id,
                decision_id=uuid4().hex,
                video_id=video_id,
                label=RelevanceLabel.ERROR,
                decision_point=point,
                primary_reason=reason,
                requested_language=self.config.language,
                language_matches=None,
                model="not-classified-error",
                prompt_version="error-v1",
                prompt_sha256=self.prompt.prompt_sha256,
                llm_input_tokens=0,
                llm_output_tokens=0,
                transcript_reserved=transcript_reserved,
            )
        )
        self._dispositioned_video_ids.add(video_id)
        if terminal:
            self._mark_video_terminal(video_id)

    def _record_budget_deferred_videos(self) -> None:
        self._record_unprocessed_videos(
            RelevanceLabel.DEFERRED_BUDGET, self._stop_reason
        )

    def _record_unprocessed_videos(self, label: RelevanceLabel, reason: str) -> None:
        pending_ids = {
            video_id
            for video_id, video in self._discovered_videos.items()
            if video.depth <= self.config.max_depth
            and video_id not in self._terminal_video_ids
        }
        if label is RelevanceLabel.DEFERRED_BUDGET:
            pending_ids -= self._deferred_video_ids
        for video_id in sorted(pending_ids):
            self.writer.append(
                RelevanceDecisionRecord(
                    run_id=self.writer.run_id,
                    decision_id=uuid4().hex,
                    video_id=video_id,
                    label=label,
                    decision_point=DecisionPoint.SEARCH_RESULT,
                    primary_reason=reason,
                    requested_language=self.config.language,
                    language_matches=None,
                    model="not-classified",
                    prompt_version="disposition-v1",
                    prompt_sha256=self.prompt.prompt_sha256,
                    llm_input_tokens=0,
                    llm_output_tokens=0,
                    transcript_reserved=False,
                )
            )
            self._dispositioned_video_ids.add(video_id)
            if label is RelevanceLabel.DEFERRED_BUDGET:
                self._deferred_video_ids.add(video_id)
            else:
                self._mark_video_terminal(video_id)

    def _record_error(
        self,
        stage: str,
        exc: Exception,
        *,
        video_id: str | None = None,
        channel_id: str | None = None,
    ) -> None:
        self.writer.append(
            RunErrorRecord(
                run_id=self.writer.run_id,
                stage=stage,
                message=str(exc),
                exception_type=type(exc).__name__,
                video_id=video_id,
                channel_id=channel_id,
                retryable=is_retryable_searchapi_error(exc),
            )
        )

    def _build_query_plan(self) -> tuple[_PlannedQuery, ...]:
        search_queries = self.expansion.search_queries
        values: list[_PlannedQuery] = [_PlannedQuery(search_queries[0], QueryKind.SEED)]
        values.extend(
            _PlannedQuery(query, QueryKind.EXPANDED) for query in search_queries[1:]
        )
        values.extend(
            _PlannedQuery(query, QueryKind.CHANNEL)
            for query in self.expansion.channel_discovery_queries
        )
        deduplicated: list[_PlannedQuery] = []
        seen: set[str] = set()
        for item in values:
            key = item.text.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(_PlannedQuery(item.text.strip(), item.kind))
        # Persist the complete interviewed plan. ``max_queries`` is a view over
        # this list so a later resume can expose more queries without another
        # interview or expansion call.
        return tuple(deduplicated)

    def _next_query_needing_work(self) -> _PlannedQuery | None:
        queries = self._queries_needing_work()
        return queries[0] if queries else None

    def _queries_needing_work(self) -> list[_PlannedQuery]:
        queries: list[_PlannedQuery] = []
        for query in self._planned_queries[: self.config.max_queries]:
            progress = self._query_progress.get(query.text)
            if progress is None:
                queries.append(query)
            elif (
                not progress.exhausted
                and progress.pages_completed < self.config.max_search_pages
            ):
                queries.append(query)
        return queries

    def _channel_needs_work(self, channel_id: str) -> bool:
        progress = self._channel_progress.get(channel_id)
        return progress is None or (
            not progress.exhausted
            and progress.pages_completed < self.config.max_channel_pages
        )

    def _refresh_frontier(self) -> None:
        """Expose persisted work made eligible by monotonic control increases."""

        for video in self._discovered_videos.values():
            if (
                video.depth <= self.config.max_depth
                and video.video_id not in self._terminal_video_ids
                and video.video_id not in self._queued_video_ids
            ):
                self._video_queue.append(video)
                self._queued_video_ids.add(video.video_id)
        for channel in self._discovered_channels.values():
            if (
                channel.depth <= self.config.max_depth
                and channel.channel_id not in self._queued_channel_ids
                and self._channel_needs_work(channel.channel_id)
            ):
                self._channel_queue.append(channel)
                self._queued_channel_ids.add(channel.channel_id)

    def _restore_state(self, state: CrawlProjectState) -> None:
        if state.run_id != self.writer.run_id:
            raise ValueError(
                f"checkpoint run ID {state.run_id!r} does not match "
                f"project {self.writer.run_id!r}"
            )
        validate_resumable_classifier_prompt(state)
        validate_pending_transcript_decisions(state)
        self._planned_queries = tuple(
            _PlannedQuery(item["text"], QueryKind(item["kind"]))
            for item in state.planned_queries
        )
        self._video_queue = deque(_video_from_state(item) for item in state.video_queue)
        self._channel_queue = deque(
            _channel_from_state(item) for item in state.channel_queue
        )
        self._discovered_videos = {
            key: _video_from_state(item)
            for key, item in state.discovered_videos.items()
        }
        self._discovered_channels = {
            key: _channel_from_state(item)
            for key, item in state.discovered_channels.items()
        }
        self._query_progress = {
            key: value.model_copy() for key, value in state.query_progress.items()
        }
        self._channel_progress = {
            key: value.model_copy() for key, value in state.channel_progress.items()
        }
        self._queued_video_ids = set(state.queued_video_ids)
        self._queued_channel_ids = set(state.queued_channel_ids)
        self._evaluated_video_ids = set(state.evaluated_video_ids)
        self._finalized_video_ids = set(state.finalized_video_ids)
        self._dispositioned_video_ids = set(state.dispositioned_video_ids)
        self._terminal_video_ids = set(state.terminal_video_ids)
        self._dashboard_pending_video_ids = {
            video_id
            for video_id, video in self._discovered_videos.items()
            if video.depth <= self.config.max_depth
            and video_id not in self._terminal_video_ids
        }
        self._deferred_video_ids = set(state.deferred_video_ids)
        self._relevant_ids = set(state.relevant_ids)
        self._transcript_ids = set(state.transcript_ids)
        self._pending_transcript_contexts = {
            key: dict(value) for key, value in state.pending_transcript_contexts.items()
        }
        self._expanded_channel_ids = {
            key
            for key, progress in self._channel_progress.items()
            if progress.pages_completed > 0
        }
        self._started_queries = {
            key
            for key, progress in self._query_progress.items()
            if progress.pages_completed > 0
        }
        self._executed_queries = set(self._started_queries)
        self._stop_reason = state.stop_reason
        self._planned_records_written = state.planned_records_written

    def _mark_state_dirty(self) -> None:
        """Record that the next checkpoint must include in-memory mutations."""

        self._checkpoint_dirty = True

    def _commit_state(self, last_status: str, *, force: bool = False) -> None:
        """Atomically persist all accumulated mutations at a safe boundary."""

        if self.state_store is None or (not force and not self._checkpoint_dirty):
            return
        budget_state = BudgetState.model_validate(self.search_budget.export_state())
        state = CrawlProjectState(
            run_id=self.writer.run_id,
            topic_query=self.config.topic_query,
            language=self.config.language,
            start_date=self.config.start_date,
            gl=self.config.gl,
            hl=self.config.interface_language,
            searchapi_timeout_seconds=self.config.searchapi_timeout_seconds,
            searchapi_retries=self.config.searchapi_retries,
            searchapi_workers=self.config.searchapi_workers,
            llm_workers=self.config.llm_workers,
            model=self.config.model,
            llm_api_base=self.config.llm_api_base,
            transcript_excerpt_chars=self.config.transcript_excerpt_chars,
            max_depth=self.config.max_depth,
            max_queries=self.config.max_queries,
            max_search_pages=self.config.max_search_pages,
            max_channel_pages=self.config.max_channel_pages,
            expansion=self.expansion.model_dump(mode="json"),
            classifier_system_prompt=self.prompt.system_prompt,
            prompt_sha256=self.prompt.prompt_sha256,
            classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
            planned_queries=[
                {"text": item.text, "kind": item.kind.value}
                for item in self._planned_queries
            ],
            budget=budget_state,
            video_queue=[_video_to_state(item) for item in self._video_queue],
            channel_queue=[_channel_to_state(item) for item in self._channel_queue],
            discovered_videos={
                key: _video_to_state(item)
                for key, item in self._discovered_videos.items()
            },
            discovered_channels={
                key: _channel_to_state(item)
                for key, item in self._discovered_channels.items()
            },
            query_progress=self._query_progress,
            channel_progress=self._channel_progress,
            queued_video_ids=sorted(self._queued_video_ids),
            queued_channel_ids=sorted(self._queued_channel_ids),
            evaluated_video_ids=sorted(self._evaluated_video_ids),
            finalized_video_ids=sorted(self._finalized_video_ids),
            dispositioned_video_ids=sorted(self._dispositioned_video_ids),
            terminal_video_ids=sorted(self._terminal_video_ids),
            deferred_video_ids=sorted(self._deferred_video_ids),
            relevant_ids=sorted(self._relevant_ids),
            transcript_ids=sorted(self._transcript_ids),
            pending_transcript_contexts=self._pending_transcript_contexts,
            stop_reason=self._stop_reason,
            last_status=last_status,
            planned_records_written=self._planned_records_written,
        )
        self.state_store.save(state)
        self._checkpoint_dirty = False
        self._publish_crawl_progress()

    def _publish_crawl_progress(self) -> None:
        queries_planned = min(self.config.max_queries, len(self._planned_queries))
        queries_started = sum(
            progress.pages_completed > 0 for progress in self._query_progress.values()
        )
        queries_done = sum(
            progress.exhausted
            or progress.pages_completed >= self.config.max_search_pages
            for progress in self._query_progress.values()
        )
        channels_exhausted = sum(
            progress.exhausted for progress in self._channel_progress.values()
        )
        channels_page_capped = sum(
            not progress.exhausted
            and progress.pages_completed >= self.config.max_channel_pages
            for progress in self._channel_progress.values()
        )
        snapshot = CrawlProgressSnapshot(
            discovered=len(self._discovered_videos),
            evaluated=len(self._evaluated_video_ids),
            relevant=len(self._relevant_ids),
            transcripts=len(self._transcript_ids),
            pending=len(self._dashboard_pending_video_ids),
            queries_done=min(queries_done, queries_planned),
            queries_started=min(queries_started, queries_planned),
            queries_planned=queries_planned,
            channels_done=channels_exhausted + channels_page_capped,
            channels_discovered=len(self._discovered_channels),
            channels_exhausted=channels_exhausted,
            channels_page_capped=channels_page_capped,
        )
        try:
            self.on_crawl_progress(snapshot)
        except Exception as exc:
            logger.error(
                "Crawl progress callback failed error_type={}",
                type(exc).__name__,
            )

    def _record_deferred_queries(self) -> None:
        for query in self._planned_queries:
            if (
                query.text not in self._executed_queries
                and query.text not in self._started_queries
            ):
                self.writer.append(
                    QueryRecord(
                        run_id=self.writer.run_id,
                        query=query.text,
                        query_kind=query.kind,
                        status=QueryStatus.DEFERRED,
                    )
                )

    def _summary(self, status: RunStatus) -> CrawlSummary:
        pending = {
            video_id
            for video_id, video in self._discovered_videos.items()
            if video.depth <= self.config.max_depth
            and video_id not in self._terminal_video_ids
        }
        return CrawlSummary(
            status=status,
            stop_reason=self._stop_reason,
            videos_discovered=len(self._discovered_videos),
            videos_evaluated=len(self._evaluated_video_ids),
            relevant_videos=len(self._relevant_ids),
            transcripts_collected=len(self._transcript_ids),
            transcripts_unavailable=len(self._unavailable_transcript_ids),
            channels_expanded=len(self._expanded_channel_ids),
            queries_executed=len(self._executed_queries),
            pending_videos=len(pending),
        )

    def _write_summary(self, summary: CrawlSummary) -> None:
        for name, value in (
            ("videos_discovered", summary.videos_discovered),
            ("videos_evaluated", summary.videos_evaluated),
            ("relevant_videos", summary.relevant_videos),
            ("transcripts_collected", summary.transcripts_collected),
            ("transcripts_unavailable", summary.transcripts_unavailable),
            ("channels_expanded", summary.channels_expanded),
            ("queries_executed", summary.queries_executed),
            ("pending_videos", summary.pending_videos),
            ("stop_reason", summary.stop_reason),
        ):
            self.writer.append(
                RunMetricRecord(run_id=self.writer.run_id, name=name, value=value)
            )

    def _progress(self, stage: str, message: str) -> None:
        self.on_progress(stage, message)
        logger.debug(
            "Crawl progress run_id={} stage={} message={}",
            self.writer.run_id,
            stage,
            message,
        )


def _request_id(response: Any) -> str | None:
    metadata = getattr(response, "search_metadata", None)
    return getattr(metadata, "id", None) if metadata is not None else None


def _is_empty_channel_result_error(exc: Exception) -> bool:
    return (
        isinstance(exc, SearchApiError)
        and exc.status_code == 200
        and "youtube channel videos didn't return any results" in exc.message.casefold()
    )


def _next_page_token(response: Any) -> str | None:
    pagination = getattr(response, "pagination", None)
    return getattr(pagination, "next_page_token", None) if pagination else None


def _model_payload(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return None


def _duration_seconds(value: str | None) -> int | None:
    if not value:
        return None
    parts = value.strip().split(":")
    if not all(part.isdigit() for part in parts) or len(parts) not in {2, 3}:
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _date_to_datetime(value: date | None) -> datetime | None:
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC) if value else None


def _sample_transcript(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    middle = len(text) // 2
    return "\n\n[...middle sample...]\n\n".join(
        (
            text[:third],
            text[middle - third // 2 : middle + third // 2],
            text[-third:],
        )
    )


def _video_to_state(video: _DiscoveredVideo) -> dict[str, Any]:
    return {
        "video_id": video.video_id,
        "title": video.title,
        "description": video.description,
        "channel_id": video.channel_id,
        "channel_title": video.channel_title,
        "published_time": video.published_time,
        "duration_text": video.duration_text,
        "source": video.source.value,
        "source_ref": video.source_ref,
        "depth": video.depth,
    }


def _video_from_state(value: dict[str, Any]) -> _DiscoveredVideo:
    return _DiscoveredVideo(
        video_id=str(value["video_id"]),
        title=str(value["title"]),
        description=value.get("description"),
        channel_id=value.get("channel_id"),
        channel_title=value.get("channel_title"),
        published_time=value.get("published_time"),
        duration_text=value.get("duration_text"),
        source=DiscoverySource(str(value["source"])),
        source_ref=str(value["source_ref"]),
        depth=int(value["depth"]),
    )


def _channel_to_state(channel: _DiscoveredChannel) -> dict[str, Any]:
    return {
        "channel_id": channel.channel_id,
        "title": channel.title,
        "source": channel.source.value,
        "source_ref": channel.source_ref,
        "depth": channel.depth,
    }


def _channel_from_state(value: dict[str, Any]) -> _DiscoveredChannel:
    return _DiscoveredChannel(
        channel_id=str(value["channel_id"]),
        title=value.get("title"),
        source=DiscoverySource(str(value["source"])),
        source_ref=str(value["source_ref"]),
        depth=int(value["depth"]),
    )


__all__ = ["CrawlConfig", "CrawlSummary", "ResearchCrawler"]
