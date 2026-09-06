import json
import threading
import time
from contextlib import contextmanager
from datetime import date
from io import StringIO
from types import SimpleNamespace

import httpx
from rich.console import Console

from yt_crawl.budget import SearchApiCreditBudget
from yt_crawl.classifier import RelevanceClassifier, RelevanceDecision
from yt_crawl.client import SearchApiClient, SearchApiError
from yt_crawl.crawler import CrawlConfig, ResearchCrawler, _DiscoveredChannel
from yt_crawl.models import (
    youtube,
    youtube_channel_videos,
    youtube_transcripts,
    youtube_video,
)
from yt_crawl.prompts import (
    CompiledClassifierPrompt,
    LlmCallResult,
    LlmUsage,
    TopicExpansion,
)
from yt_crawl.records import DiscoverySource, RunStatus
from yt_crawl.run_tui import RunDashboard
from yt_crawl.settings import DEFAULT_LLM_WORKERS, DEFAULT_SEARCHAPI_WORKERS
from yt_crawl.state import ProjectStateStore
from yt_crawl.storage import JsonlRunWriter

CLASSIFY_PROMPT_SHA256 = (
    "4d553b0b48b4bb151321bd8aa5d9904477ecfdbec8dea4f2554cef83409c8ae6"
)


class FakeSearchApi:
    def __init__(self, *, transcript_language: str = "en") -> None:
        self.search_calls = 0
        self.video_calls = 0
        self.transcript_calls = 0
        self.transcript_language = transcript_language

    def search(self, requests):
        self.search_calls += 1
        return [
            youtube.SearchResponse(
                videos=[
                    youtube.Video(
                        id="video-1",
                        title="A focused topic interview",
                        link="https://youtube.test/watch?v=video-1",
                        published_time="1 month ago",
                    )
                ]
            )
        ]

    def video(self, requests):
        self.video_calls += 1
        return [
            youtube_video.SearchResponse(
                video=youtube_video.Video(
                    id="video-1",
                    title="A focused topic interview",
                    length_seconds=600,
                    published_time="July 1, 2026",
                    description="A substantive discussion of the research topic.",
                ),
                available_transcripts_languages=[
                    youtube_video.TranscriptLanguage(
                        name=(
                            "English (auto-generated)"
                            if self.transcript_language == "en"
                            else "Deutsch (automatisch erzeugt)"
                        ),
                        lang=self.transcript_language,
                    )
                ],
            )
        ]

    def transcripts(self, requests):
        self.transcript_calls += 1
        return [
            youtube_transcripts.SearchResponse(
                transcripts=[
                    youtube_transcripts.Transcript(
                        text="This interview discusses the topic in depth.",
                        start=0,
                        duration=4,
                    )
                ]
            )
        ]


class MultiCandidateSearchApi(FakeSearchApi):
    def search(self, requests):
        self.search_calls += 1
        video_id = f"video-{self.search_calls}"
        return [
            youtube.SearchResponse(
                videos=[
                    youtube.Video(
                        id=video_id,
                        title=f"Candidate {self.search_calls}",
                        link=f"https://youtube.test/watch?v={video_id}",
                    )
                ]
            )
        ]

    def video(self, requests):
        self.video_calls += 1
        video_id = requests[0]["video_id"]
        return [
            youtube_video.SearchResponse(
                video=youtube_video.Video(
                    id=video_id,
                    title=f"Detail {video_id}",
                    published_time="July 1, 2026",
                ),
                available_transcripts_languages=[
                    youtube_video.TranscriptLanguage(
                        name="Deutsch (automatisch erzeugt)", lang="de"
                    )
                ],
            )
        ]


class ChannelSearchApi(FakeSearchApi):
    def __init__(self) -> None:
        super().__init__()
        self.channel_calls = 0

    def search(self, requests):
        self.search_calls += 1
        return [
            youtube.SearchResponse(
                videos=[
                    youtube.Video(
                        id="video-1",
                        title="A focused topic interview",
                        link="https://youtube.test/watch?v=video-1",
                        published_time="1 month ago",
                        channel=youtube.VideoChannel(
                            id="channel-1", title="Topic channel"
                        ),
                    )
                ]
            )
        ]

    def channel_videos(self, requests):
        self.channel_calls += 1
        return [
            youtube_channel_videos.SearchResponse(
                channel=youtube_channel_videos.Channel(
                    id="channel-1",
                    title="Topic channel",
                    subscribers=123,
                ),
                videos=[],
            )
        ]


class EmptyChannelSearchApi(FakeSearchApi):
    def __init__(self) -> None:
        super().__init__()
        self.channel_calls = 0

    def channel_videos(self, requests, *, return_exceptions=False):
        self.channel_calls += len(requests)
        errors = [
            SearchApiError(200, "YouTube Channel Videos didn't return any results.")
            for _request in requests
        ]
        if return_exceptions:
            return errors
        raise errors[0]


class FailingTranscriptSearchApi(FakeSearchApi):
    def transcripts(self, requests):
        self.transcript_calls += 1
        raise TimeoutError("transcript outcome unknown")


class FailOnceTranscriptSearchApi(FakeSearchApi):
    def transcripts(self, requests):
        self.transcript_calls += 1
        if self.transcript_calls == 1:
            raise httpx.ReadError("[Errno 54] Connection reset by peer")
        return [
            youtube_transcripts.SearchResponse(
                transcripts=[
                    youtube_transcripts.Transcript(
                        text="A substantive discussion of the research topic.",
                        start=0,
                        duration=4,
                    )
                ]
            )
        ]


class PermanentTranscriptFailureSearchApi(FakeSearchApi):
    def transcripts(self, requests):
        self.transcript_calls += 1
        raise SearchApiError(400, "invalid transcript request")


class UnavailableTranscriptSearchApi(FakeSearchApi):
    def __init__(self, *, error: str | None) -> None:
        super().__init__()
        self.error = error

    def transcripts(self, requests):
        self.transcript_calls += 1
        return [
            youtube_transcripts.SearchResponse(
                transcripts=[],
                error=self.error,
            )
        ]


class FakeClassifier:
    model = "gpt-5.6-luna"

    def classify_with_usage(self, candidate, prompt, *, stage):
        if stage == "metadata":
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="en",
                primary_reason="topic_match",
            )
        else:
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="en",
                primary_reason="topic_match",
            )
        return LlmCallResult(
            output=decision,
            usage=LlmUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        )


class FakeLlm:
    @contextmanager
    def call_context(self, *args, **kwargs):
        yield


class ConcurrentResponses:
    def __init__(
        self,
        *,
        fail_once: tuple[str, str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._active = {"metadata": 0, "transcript": 0}
        self.peak = {"metadata": 0, "transcript": 0}
        self.fail_once = fail_once

    def parse(self, **kwargs):
        payload = json.loads(kwargs["input"][1]["content"][0]["text"])
        stage = payload["classification_stage"]
        video_id = payload["candidate"]["video_id"]
        with self._lock:
            self._active[stage] += 1
            self.peak[stage] = max(self.peak[stage], self._active[stage])
        time.sleep(0.04)
        with self._lock:
            self._active[stage] -= 1
            should_fail = self.fail_once == (stage, video_id)
            if should_fail:
                self.fail_once = None
        if should_fail:
            raise RuntimeError(f"temporary {stage} failure for {video_id}")
        if stage == "metadata":
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="de",
                primary_reason="topic_match",
            )
        else:
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="de",
                primary_reason="topic_match",
            )
        return SimpleNamespace(
            output_parsed=decision,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=None,
            ),
            id=f"resp-{stage}-{video_id}",
        )


class CachedBatchSearchApi:
    max_workers = 2

    def __init__(self) -> None:
        self.cached: set[tuple[str, str]] = set()

    def is_cached(self, engine, request):
        identifier = str(
            request.get("q") or request.get("video_id") or request.get("channel_id")
        )
        return (engine, identifier) in self.cached

    def search(self, requests, *, return_exceptions=False):
        for request in requests:
            self.cached.add(("youtube", request["q"]))
        return [
            youtube.SearchResponse(
                videos=[
                    youtube.Video(id="video-1", title="First"),
                    youtube.Video(id="video-2", title="Second"),
                ]
            )
            for _request in requests
        ]

    def video(self, requests, *, return_exceptions=False):
        responses = []
        for request in requests:
            video_id = request["video_id"]
            self.cached.add(("youtube_video", video_id))
            responses.append(
                youtube_video.SearchResponse(
                    video=youtube_video.Video(
                        id=video_id,
                        title=f"Detail {video_id}",
                        published_time="July 1, 2026",
                    ),
                    available_transcripts_languages=[
                        youtube_video.TranscriptLanguage(name="Deutsch", lang="de")
                    ],
                )
            )
        return responses

    def transcripts(self, requests, *, return_exceptions=False):
        responses = []
        for request in requests:
            self.cached.add(("youtube_transcripts", request["video_id"]))
            responses.append(
                youtube_transcripts.SearchResponse(
                    transcripts=[
                        youtube_transcripts.Transcript(
                            text="Ausführliche Diskussion.", start=0, duration=4
                        )
                    ]
                )
            )
        return responses


class CountingProjectStateStore(ProjectStateStore):
    def __init__(self, project_dir) -> None:
        super().__init__(project_dir)
        self.save_count = 0

    def save(self, state) -> None:
        self.save_count += 1
        super().save(state)


def make_crawler(
    tmp_path,
    api: FakeSearchApi,
    *,
    config: CrawlConfig | None = None,
    on_api_event=None,
    on_crawl_progress=None,
    search_budget: SearchApiCreditBudget | None = None,
    state_store: ProjectStateStore | None = None,
    resume_state=None,
) -> ResearchCrawler:
    return ResearchCrawler(
        config=config
        or CrawlConfig(
            topic_query="research topic",
            language="en",
            start_date=date(2026, 1, 1),
            max_queries=2,
            max_depth=0,
            searchapi_retries=0,
        ),
        expansion=TopicExpansion(
            topic_interpretation="Research topic",
            inclusion_criteria=("substantive topic",),
            exclusion_criteria=("passing mention",),
            search_queries=("research topic interview",),
            channel_discovery_queries=("research topic channel",),
            ambiguity_rules=(),
        ),
        classifier_prompt=CompiledClassifierPrompt(
            system_prompt="Classify.", prompt_sha256=CLASSIFY_PROMPT_SHA256
        ),
        searchapi=api,
        classifier=FakeClassifier(),
        llm_client=FakeLlm(),
        search_budget=search_budget or SearchApiCreditBudget(5, 2),
        writer=JsonlRunWriter(tmp_path, "run-1"),
        on_api_event=on_api_event,
        on_crawl_progress=on_crawl_progress,
        state_store=state_store,
        resume_state=resume_state,
    )


def test_crawler_transcribes_before_finishing_candidate(tmp_path) -> None:
    api = FakeSearchApi()

    summary = make_crawler(tmp_path, api).run()

    assert summary.status is RunStatus.COMPLETED
    assert summary.relevant_videos == 1
    assert summary.transcripts_collected == 1
    assert summary.transcripts_unavailable == 0
    assert api.search_calls == 2
    assert api.video_calls == 1
    assert api.transcript_calls == 1
    assert (tmp_path / "run-1" / "transcript.jsonl").exists()


def test_searchapi_events_balance_across_crawler_calls(tmp_path) -> None:
    events = []

    summary = make_crawler(tmp_path, FakeSearchApi(), on_api_event=events.append).run()

    assert summary.status is RunStatus.COMPLETED
    starts = [event for event in events if event.phase == "started"]
    finishes = [event for event in events if event.phase == "finished"]
    assert len(starts) == len(finishes) == 4
    assert [event.operation for event in starts] == [
        "youtube_search",
        "youtube_video",
        "youtube_transcripts",
        "youtube_search",
    ]
    assert all(event.status in {"success", "cache_hit"} for event in finishes)


def test_crawler_publishes_preaggregated_progress_snapshots(tmp_path) -> None:
    snapshots = []
    state_store = ProjectStateStore(tmp_path / "run-1")

    summary = make_crawler(
        tmp_path,
        FakeSearchApi(),
        on_crawl_progress=snapshots.append,
        state_store=state_store,
    ).run()

    assert snapshots
    latest = snapshots[-1]
    assert latest.discovered == summary.videos_discovered
    assert latest.evaluated == summary.videos_evaluated
    assert latest.relevant == summary.relevant_videos
    assert latest.transcripts == summary.transcripts_collected
    assert latest.pending == summary.pending_videos
    assert latest.queries_done == summary.queries_executed


def test_crawl_progress_callback_cannot_fail_the_crawl(tmp_path) -> None:
    def fail_callback(_snapshot) -> None:
        raise RuntimeError("display failed")

    summary = make_crawler(
        tmp_path,
        FakeSearchApi(),
        on_crawl_progress=fail_callback,
        state_store=ProjectStateStore(tmp_path / "run-1"),
    ).run()

    assert summary.status is RunStatus.COMPLETED


def test_metadata_relevant_is_provisional_until_transcript_classification(
    tmp_path,
) -> None:
    api = FakeSearchApi()

    summary = make_crawler(tmp_path, api).run()

    decisions = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "relevance_decision.jsonl")
        .read_text()
        .splitlines()
    ]
    assert summary.relevant_videos == 1
    assert api.transcript_calls == 1
    assert [(row["decision_point"], row["label"]) for row in decisions] == [
        ("video_metadata", "needs_transcript"),
        ("transcript", "relevant"),
    ]
    assert decisions[0]["primary_reason"] == "topic_match"
    assert "confidence" not in decisions[0]
    assert "criteria" not in decisions[0]
    assert "published_after_start_date" not in decisions[0]


def test_max_depth_zero_does_not_expand_search_result_channels(tmp_path) -> None:
    api = ChannelSearchApi()

    summary = make_crawler(tmp_path, api).run()

    assert summary.status is RunStatus.COMPLETED
    assert summary.channels_expanded == 0
    assert api.channel_calls == 0


def test_max_depth_one_expands_search_result_channels(tmp_path) -> None:
    api = ChannelSearchApi()
    crawler = make_crawler(tmp_path, api)
    crawler.config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=1,
    )
    crawler.search_budget = SearchApiCreditBudget(6, 2)

    summary = crawler.run()

    assert summary.status is RunStatus.COMPLETED
    assert summary.channels_expanded == 1
    assert api.channel_calls == 1
    channels = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "channel.jsonl").read_text().splitlines()
    ]
    expanded_channel = next(
        channel for channel in channels if channel["subscribers"] is not None
    )
    assert expanded_channel["subscribers"] == 123
    assert expanded_channel["views"] is None


def test_empty_channel_result_is_skipped_in_sequential_expansion(tmp_path) -> None:
    api = EmptyChannelSearchApi()
    crawler = make_crawler(tmp_path, api)
    channel = _DiscoveredChannel(
        channel_id="channel-empty",
        title="Empty channel",
        source=DiscoverySource.SEARCH,
        source_ref="research topic",
        depth=0,
    )

    crawler._expand_channel(channel)

    progress = crawler._channel_progress[channel.channel_id]
    assert progress.pages_completed == 1
    assert progress.exhausted is True
    assert api.channel_calls == 1


def test_empty_channel_result_does_not_abort_channel_batch(tmp_path) -> None:
    api = EmptyChannelSearchApi()
    crawler = make_crawler(tmp_path, api)
    channel = _DiscoveredChannel(
        channel_id="channel-empty",
        title="Empty channel",
        source=DiscoverySource.SEARCH,
        source_ref="research topic",
        depth=0,
    )

    crawler._expand_channel_batch([channel])

    progress = crawler._channel_progress[channel.channel_id]
    assert progress.pages_completed == 1
    assert progress.exhausted is True
    assert api.channel_calls == 1
    error = json.loads(
        (tmp_path / "run-1" / "run_error.jsonl").read_text().splitlines()[-1]
    )
    assert error["channel_id"] == channel.channel_id
    assert error["exception_type"] == "SearchApiError"


def test_wrong_language_is_rejected_without_transcript_spend(tmp_path) -> None:
    api = FakeSearchApi(transcript_language="de")

    summary = make_crawler(tmp_path, api).run()

    assert summary.status is RunStatus.COMPLETED
    assert summary.relevant_videos == 0
    assert summary.transcripts_collected == 0
    assert api.transcript_calls == 0


def test_tight_budget_marks_uninspected_candidate_deferred(tmp_path) -> None:
    api = MultiCandidateSearchApi()
    crawler = make_crawler(tmp_path, api)
    crawler.search_budget = SearchApiCreditBudget(4, 1)

    summary = crawler.run()

    assert summary.status is RunStatus.STOPPED_BUDGET
    assert summary.videos_discovered == 2
    assert summary.videos_evaluated == 1
    assert summary.pending_videos == 1
    assert api.video_calls == 1
    decisions = (tmp_path / "run-1" / "relevance_decision.jsonl").read_text()
    assert '"label":"irrelevant"' in decisions
    assert '"label":"deferred_budget"' in decisions


def test_deferred_budget_candidate_is_requeued_on_resume(tmp_path) -> None:
    project = tmp_path / "run-1"
    state_store = ProjectStateStore(project)
    api = MultiCandidateSearchApi()
    first_budget = SearchApiCreditBudget(4, 1)

    first = make_crawler(
        tmp_path,
        api,
        search_budget=first_budget,
        state_store=state_store,
    ).run()

    assert first.status is RunStatus.STOPPED_BUDGET
    first_state = state_store.load()
    assert "video-2" not in first_state.terminal_video_ids
    assert first_state.deferred_video_ids == ["video-2"]

    resumed_budget = SearchApiCreditBudget.restore(first_state.budget.model_dump())
    resumed_budget.expand(5, 1)
    second = make_crawler(
        tmp_path,
        api,
        search_budget=resumed_budget,
        state_store=state_store,
        resume_state=first_state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert second.pending_videos == 0
    assert api.video_calls == 2
    decisions = [
        json.loads(line)
        for line in (project / "relevance_decision.jsonl").read_text().splitlines()
    ]
    assert [(row["video_id"], row["label"]) for row in decisions] == [
        ("video-1", "irrelevant"),
        ("video-2", "deferred_budget"),
        ("video-2", "irrelevant"),
    ]
    final_state = state_store.load()
    assert "video-2" in final_state.terminal_video_ids
    assert final_state.deferred_video_ids == []


def test_transcript_dispatch_error_is_charged_but_remains_retryable(
    tmp_path,
) -> None:
    api = FailingTranscriptSearchApi()
    runtime_events = []
    crawler = make_crawler(tmp_path, api, on_api_event=runtime_events.append)

    summary = crawler.run()

    assert summary.status is RunStatus.FAILED
    assert summary.pending_videos == 1
    snapshot = crawler.search_budget.snapshot()
    assert snapshot.transcript_reserved == 0
    assert snapshot.transcript_spent == 1
    budget_events = (tmp_path / "run-1" / "budget_event.jsonl").read_text()
    decisions = (tmp_path / "run-1" / "relevance_decision.jsonl").read_text()
    assert "transcript:video-1:error" in budget_events
    assert '"label":"error"' in decisions
    assert not (tmp_path / "run-1" / "transcript.jsonl").exists()
    errors = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "run_error.jsonl").read_text().splitlines()
    ]
    assert {error["stage"] for error in errors} == {"transcript", "crawler"}
    assert all(error["retryable"] is True for error in errors)
    assert sum(event.phase == "started" for event in runtime_events) == sum(
        event.phase == "finished" for event in runtime_events
    )
    assert runtime_events[-1].status == "error"


def test_transient_transcript_failure_retries_and_charges_each_attempt(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("yt_crawl.crawler.time.sleep", lambda _delay: None)
    api = FailOnceTranscriptSearchApi()
    budget = SearchApiCreditBudget(5, 2)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=1,
    )

    summary = make_crawler(tmp_path, api, config=config, search_budget=budget).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.transcript_calls == 2
    assert budget.snapshot().transcript_spent == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "api_call.jsonl").read_text().splitlines()
        if '"operation":"youtube_transcripts"' in line
    ]
    assert [row["status"] for row in rows] == ["error", "success"]
    assert sum(row["searchapi_credits"] for row in rows) == 2


def test_transcript_retry_exhaustion_fails_after_budgeted_attempts(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("yt_crawl.crawler.time.sleep", lambda _delay: None)
    api = FailingTranscriptSearchApi()
    budget = SearchApiCreditBudget(5, 2)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=1,
    )

    summary = make_crawler(tmp_path, api, config=config, search_budget=budget).run()

    assert summary.status is RunStatus.FAILED
    assert api.transcript_calls == 2
    assert budget.snapshot().transcript_spent == 2


def test_nonretryable_transcript_failure_is_not_repeated(tmp_path) -> None:
    api = PermanentTranscriptFailureSearchApi()
    budget = SearchApiCreditBudget(5, 2)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=2,
    )

    summary = make_crawler(tmp_path, api, config=config, search_budget=budget).run()

    assert summary.status is RunStatus.FAILED
    assert api.transcript_calls == 1
    assert budget.snapshot().transcript_spent == 1


def test_retry_stops_before_unfunded_transcript_attempt(tmp_path) -> None:
    api = FailingTranscriptSearchApi()
    budget = SearchApiCreditBudget(3, 1)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=1,
    )

    summary = make_crawler(tmp_path, api, config=config, search_budget=budget).run()

    assert summary.status is RunStatus.STOPPED_BUDGET
    assert api.transcript_calls == 1
    assert budget.snapshot().transcript_spent == 1


def test_discovery_batch_retries_only_failed_members(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("yt_crawl.crawler.time.sleep", lambda _delay: None)
    budget = SearchApiCreditBudget(5, 2)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=1,
    )
    crawler = make_crawler(
        tmp_path,
        FakeSearchApi(),
        config=config,
        search_budget=budget,
    )
    calls: list[list[str]] = []

    def dispatch(requests):
        calls.append([str(request["video_id"]) for request in requests])
        return [
            (
                httpx.ConnectError("temporary DNS failure")
                if request["video_id"] == "video-1" and len(calls) == 1
                else youtube_video.SearchResponse()
            )
            for request in requests
        ]

    results, blocked = crawler._discovery_batch(
        "youtube_video",
        "youtube_video",
        [{"video_id": "video-1"}, {"video_id": "video-2"}],
        dispatch,
    )

    assert blocked is None
    assert all(not isinstance(result, Exception) for result in results)
    assert calls == [["video-1", "video-2"], ["video-1"]]
    assert budget.snapshot().discovery_spent == 3


def test_discovery_batch_checkpoints_all_charges_once_before_dispatch(tmp_path) -> None:
    budget = SearchApiCreditBudget(8, 2)
    state_store = CountingProjectStateStore(tmp_path / "coalesced-checkpoint")
    crawler = make_crawler(
        tmp_path,
        FakeSearchApi(),
        search_budget=budget,
        state_store=state_store,
    )
    state_store.save_count = 0
    observed_spend: list[int] = []
    observed_writes: list[int] = []

    def dispatch(requests):
        observed_spend.append(state_store.load().budget.discovery_spent)
        observed_writes.append(state_store.save_count)
        return [youtube_video.SearchResponse() for _request in requests]

    results, blocked = crawler._discovery_batch_attempt(
        "youtube_video",
        "youtube_video",
        [
            {"video_id": "video-1"},
            {"video_id": "video-2"},
            {"video_id": "video-3"},
        ],
        dispatch,
    )

    assert blocked is None
    assert len(results) == 3
    assert observed_spend == [3]
    assert observed_writes == [1]
    assert state_store.save_count == 1


def test_single_discovery_retry_is_separately_charged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("yt_crawl.crawler.time.sleep", lambda _delay: None)
    budget = SearchApiCreditBudget(4, 1)
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=1,
    )
    crawler = make_crawler(
        tmp_path,
        FakeSearchApi(),
        config=config,
        search_budget=budget,
    )
    attempts = 0

    def dispatch():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary DNS failure")
        return youtube_video.SearchResponse()

    response = crawler._discovery_call(
        "youtube_video",
        "youtube_video",
        {"video_id": "video-1"},
        dispatch,
    )

    assert isinstance(response, youtube_video.SearchResponse)
    assert attempts == 2
    assert budget.snapshot().discovery_spent == 2


def test_unavailable_transcript_response_is_an_error_not_irrelevant(tmp_path) -> None:
    api = UnavailableTranscriptSearchApi(error="Transcript disabled by uploader")
    crawler = make_crawler(tmp_path, api)

    summary = crawler.run()

    transcripts = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "transcript.jsonl").read_text().splitlines()
    ]
    decisions = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "relevance_decision.jsonl")
        .read_text()
        .splitlines()
    ]
    errors = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "run_error.jsonl").read_text().splitlines()
    ]
    assert summary.status is RunStatus.COMPLETED
    assert summary.transcripts_collected == 0
    assert summary.transcripts_unavailable == 1
    assert transcripts[0]["is_available"] is False
    assert transcripts[0]["unavailable_reason"] == "Transcript disabled by uploader"
    assert [decision["label"] for decision in decisions] == [
        "needs_transcript",
        "error",
    ]
    assert decisions[-1]["primary_reason"] == "transcript_response_error"
    assert errors[-1]["message"] == "Transcript disabled by uploader"
    restored = make_crawler(tmp_path, api)
    assert restored._unavailable_transcript_ids == {"video-1"}


def test_empty_transcript_response_is_an_error_not_irrelevant(tmp_path) -> None:
    crawler = make_crawler(tmp_path, UnavailableTranscriptSearchApi(error=None))

    summary = crawler.run()

    transcript = json.loads(
        (tmp_path / "run-1" / "transcript.jsonl").read_text().splitlines()[0]
    )
    decisions = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "relevance_decision.jsonl")
        .read_text()
        .splitlines()
    ]
    assert summary.status is RunStatus.COMPLETED
    assert transcript["is_available"] is False
    assert transcript["unavailable_reason"] == "empty_transcript"
    assert summary.transcripts_unavailable == 1
    assert [decision["label"] for decision in decisions] == [
        "needs_transcript",
        "error",
    ]
    assert decisions[-1]["primary_reason"] == "empty_transcript"


def test_llm_batching_is_independent_of_searchapi_worker_limit(
    tmp_path,
) -> None:
    lock = threading.Lock()
    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        engine = request.url.params["engine"]
        with lock:
            captured.append(request)
            active[engine] = active.get(engine, 0) + 1
            peak[engine] = max(peak.get(engine, 0), active[engine])
        if engine in {"youtube_video", "youtube_transcripts"}:
            time.sleep(0.04)
        with lock:
            active[engine] -= 1
        if engine == "youtube":
            payload = youtube.SearchResponse(
                videos=[
                    youtube.Video(id="video-1", title="Erstes Video"),
                    youtube.Video(id="video-2", title="Zweites Video"),
                ]
            )
        elif engine == "youtube_video":
            video_id = request.url.params["video_id"]
            payload = youtube_video.SearchResponse(
                video=youtube_video.Video(
                    id=video_id,
                    title=f"Details {video_id}",
                    published_time="July 1, 2026",
                ),
                available_transcripts_languages=[
                    youtube_video.TranscriptLanguage(name="Deutsch", lang="de")
                ],
            )
        else:
            payload = youtube_transcripts.SearchResponse(
                transcripts=[
                    youtube_transcripts.Transcript(
                        text="Eine ausführliche Diskussion.", start=0, duration=4
                    )
                ]
            )
        return httpx.Response(
            200, json=payload.model_dump(mode="json"), request=request
        )

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    budget = SearchApiCreditBudget(7, 2)
    llm_responses = ConcurrentResponses()
    try:
        with SearchApiClient(
            "test-key",
            client=http,
            max_retries=0,
            max_workers=1,
            cache_dir=tmp_path / "project" / ".cache" / "searchapi",
            cache_ttl=None,
        ) as api:
            crawler = ResearchCrawler(
                config=CrawlConfig(
                    topic_query="Finanzplanung",
                    language="de",
                    start_date=date(2026, 1, 1),
                    max_queries=1,
                    max_depth=0,
                    gl="de",
                    hl=None,
                    searchapi_workers=1,
                    searchapi_timeout_seconds=0.5,
                    llm_workers=2,
                ),
                expansion=TopicExpansion(
                    topic_interpretation="Finanzplanung",
                    inclusion_criteria=("substantive Behandlung",),
                    exclusion_criteria=("beiläufige Erwähnung",),
                    search_queries=("persönliche Finanzplanung",),
                    channel_discovery_queries=("Finanzplanung Kanal",),
                    ambiguity_rules=(),
                ),
                classifier_prompt=CompiledClassifierPrompt(
                    system_prompt="Classify.", prompt_sha256=CLASSIFY_PROMPT_SHA256
                ),
                searchapi=api,
                classifier=RelevanceClassifier(
                    SimpleNamespace(responses=llm_responses)
                ),
                llm_client=FakeLlm(),
                search_budget=budget,
                writer=JsonlRunWriter(tmp_path, "parallel-run"),
            )
            summary = crawler.run()
    finally:
        http.close()

    assert summary.status is RunStatus.COMPLETED
    assert summary.videos_evaluated == 2
    assert summary.transcripts_collected == 2
    assert peak["youtube_video"] == 1
    assert peak["youtube_transcripts"] == 1
    assert llm_responses.peak == {"metadata": 2, "transcript": 2}
    snapshot = budget.snapshot()
    assert snapshot.discovery_spent == 3
    assert snapshot.transcript_spent == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "parallel-run" / "api_call.jsonl")
        .read_text()
        .splitlines()
        if '"provider":"searchapi"' in line
    ]
    assert len(rows) == 5
    assert sum(row["searchapi_credits"] for row in rows) == 5
    decisions = [
        json.loads(line)
        for line in (tmp_path / "parallel-run" / "relevance_decision.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["video_id"] for row in decisions] == [
        "video-1",
        "video-2",
        "video-1",
        "video-2",
    ]
    localized = [
        request
        for request in captured
        if request.url.params["engine"] != "youtube_transcripts"
    ]
    assert all(request.url.params["gl"] == "de" for request in localized)
    assert all(request.url.params["hl"] == "de" for request in localized)
    transcript_requests = [
        request
        for request in captured
        if request.url.params["engine"] == "youtube_transcripts"
    ]
    assert all(request.url.params["lang"] == "de" for request in transcript_requests)


def test_dashboard_caps_queued_searchapi_batch_at_provider_limit(tmp_path) -> None:
    lock = threading.Lock()
    active = 0
    transport_peak = 0
    details_started = threading.Event()
    release_details = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, transport_peak
        engine = request.url.params["engine"]
        with lock:
            active += 1
            transport_peak = max(transport_peak, active)
        try:
            if engine == "youtube_video":
                details_started.set()
                assert release_details.wait(timeout=5)
                video_id = request.url.params["video_id"]
                payload = youtube_video.SearchResponse(
                    video=youtube_video.Video(
                        id=video_id,
                        title=f"Details {video_id}",
                        published_time="July 1, 2026",
                    ),
                    available_transcripts_languages=[
                        youtube_video.TranscriptLanguage(name="Deutsch", lang="de")
                    ],
                )
            elif engine == "youtube":
                payload = youtube.SearchResponse(
                    videos=[
                        youtube.Video(id=f"video-{number}", title=f"Video {number}")
                        for number in range(1, 5)
                    ]
                )
            else:
                payload = youtube_transcripts.SearchResponse(
                    transcripts=[
                        youtube_transcripts.Transcript(
                            text="Eine ausführliche Diskussion.", start=0, duration=4
                        )
                    ]
                )
            return httpx.Response(
                200, json=payload.model_dump(mode="json"), request=request
            )
        finally:
            with lock:
                active -= 1

    project = tmp_path / "dashboard-batch"
    console_output = StringIO()
    console = Console(file=console_output, force_terminal=False, width=120)
    dashboard = RunDashboard(
        mode="START NEW PROJECT",
        project_dir=project,
        budget=SearchApiCreditBudget(9, 4),
        state_store=ProjectStateStore(project),
        searchapi_concurrency=2,
        openai_concurrency=4,
        console=console,
    )
    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    summaries = []
    failures = []
    try:
        with SearchApiClient(
            "test-key",
            client=http,
            max_retries=0,
            max_workers=2,
            cache_dir=project / ".cache" / "searchapi",
            cache_ttl=None,
        ) as api:
            crawler = ResearchCrawler(
                config=CrawlConfig(
                    topic_query="Finanzplanung",
                    language="de",
                    start_date=date(2026, 1, 1),
                    max_queries=1,
                    max_depth=0,
                    searchapi_workers=2,
                    llm_workers=4,
                ),
                expansion=TopicExpansion(
                    topic_interpretation="Finanzplanung",
                    inclusion_criteria=("substantive Behandlung",),
                    exclusion_criteria=("beiläufige Erwähnung",),
                    search_queries=("persönliche Finanzplanung",),
                    channel_discovery_queries=("Finanzplanung Kanal",),
                    ambiguity_rules=(),
                ),
                classifier=RelevanceClassifier(
                    SimpleNamespace(responses=ConcurrentResponses())
                ),
                classifier_prompt=CompiledClassifierPrompt(
                    system_prompt="Classify.", prompt_sha256=CLASSIFY_PROMPT_SHA256
                ),
                llm_client=FakeLlm(),
                searchapi=api,
                search_budget=dashboard.budget,
                writer=JsonlRunWriter(project.parent, project.name),
                on_api_event=dashboard.on_event,
                state_store=ProjectStateStore(project),
            )

            def run_crawler() -> None:
                try:
                    summaries.append(crawler.run())
                except BaseException as exc:  # pragma: no cover - assertion detail
                    failures.append(exc)

            thread = threading.Thread(target=run_crawler, daemon=True)
            thread.start()
            assert details_started.wait(timeout=2)
            console.print(dashboard.render())
            rendered = console_output.getvalue()
            assert "SearchAPI 2/2 · 2 queued" in rendered
            assert "SearchAPI 4/2" not in rendered
            assert transport_peak == 2
            release_details.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
    finally:
        release_details.set()
        http.close()

    assert not failures
    assert summaries[0].status is RunStatus.COMPLETED
    assert summaries[0].videos_evaluated == 4
    assert transport_peak == 2


def test_config_derives_interface_language_unless_explicitly_overridden() -> None:
    derived = CrawlConfig(
        topic_query="topic",
        language="pt-BR",
        start_date=date(2026, 1, 1),
        hl=None,
    )
    explicit = CrawlConfig(
        topic_query="topic",
        language="pt-BR",
        start_date=date(2026, 1, 1),
        hl="en",
    )

    assert derived.interface_language == "pt"
    assert explicit.interface_language == "en"
    assert derived.searchapi_workers == DEFAULT_SEARCHAPI_WORKERS
    assert derived.llm_workers == DEFAULT_LLM_WORKERS


def test_batched_classification_failure_keeps_successes_and_resumes_failed_item(
    tmp_path,
) -> None:
    project = tmp_path / "classification-resume"
    api = CachedBatchSearchApi()
    responses = ConcurrentResponses(fail_once=("metadata", "video-1"))
    classifier = RelevanceClassifier(SimpleNamespace(responses=responses))
    config = CrawlConfig(
        topic_query="Finanzplanung",
        language="de",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_workers=2,
        llm_workers=2,
    )
    expansion = TopicExpansion(
        topic_interpretation="Finanzplanung",
        inclusion_criteria=("substantive Behandlung",),
        exclusion_criteria=("beiläufige Erwähnung",),
        search_queries=("persönliche Finanzplanung",),
        channel_discovery_queries=("Finanzplanung Kanal",),
        ambiguity_rules=(),
    )
    prompt = CompiledClassifierPrompt(
        system_prompt="Classify.", prompt_sha256=CLASSIFY_PROMPT_SHA256
    )
    first_budget = SearchApiCreditBudget(7, 2)
    first = ResearchCrawler(
        config=config,
        expansion=expansion,
        classifier_prompt=prompt,
        searchapi=api,
        classifier=classifier,
        llm_client=FakeLlm(),
        search_budget=first_budget,
        writer=JsonlRunWriter(project.parent, project.name),
        state_store=ProjectStateStore(project),
    ).run()

    assert first.status is RunStatus.FAILED
    assert first.relevant_videos == 1
    state = ProjectStateStore(project).load()
    assert "video-1" not in state.terminal_video_ids
    assert "video-2" in state.terminal_video_ids

    resumed_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    second = ResearchCrawler(
        config=config,
        expansion=expansion,
        classifier_prompt=prompt,
        searchapi=api,
        classifier=classifier,
        llm_client=FakeLlm(),
        search_budget=resumed_budget,
        writer=JsonlRunWriter(project.parent, project.name),
        state_store=ProjectStateStore(project),
        resume_state=state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert second.relevant_videos == 2
    assert resumed_budget.snapshot().discovery_spent == 3
    assert resumed_budget.snapshot().transcript_spent == 2
    assert responses.peak["metadata"] == 2
    decisions = [
        json.loads(line)
        for line in (project / "relevance_decision.jsonl").read_text().splitlines()
    ]
    relevant = [row["video_id"] for row in decisions if row["label"] == "relevant"]
    assert relevant == ["video-2", "video-1"]
