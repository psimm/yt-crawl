import json
from contextlib import contextmanager
from datetime import date

import pytest

from yt_searchapi.budget import SearchApiCreditBudget
from yt_searchapi.classifier import RelevanceDecision
from yt_searchapi.crawler import CrawlConfig, ResearchCrawler
from yt_searchapi.models import youtube, youtube_transcripts, youtube_video
from yt_searchapi.prompts import (
    CLASSIFIER_PROMPT_VERSION,
    CompiledClassifierPrompt,
    LlmCallResult,
    LlmUsage,
    TopicExpansion,
)
from yt_searchapi.records import RunStatus
from yt_searchapi.state import ProjectStateStore, validate_pending_transcript_decisions
from yt_searchapi.storage import JsonlRunWriter

CLASSIFY_PROMPT_SHA256 = (
    "4d553b0b48b4bb151321bd8aa5d9904477ecfdbec8dea4f2554cef83409c8ae6"
)


class NoopLlm:
    @contextmanager
    def call_context(self, *_args, **_kwargs):
        yield


class UnusedClassifier:
    model = "unused"

    def classify_with_usage(self, *_args, **_kwargs):
        raise AssertionError("language gate should reject before classification")


class ResumeApi:
    def __init__(self, *, paginated: bool = False) -> None:
        self.search_calls = 0
        self.video_calls = 0
        self.paginated = paginated

    def search(self, requests):
        self.search_calls += 1
        request = requests[0]
        if self.paginated:
            token = None if request.get("sp") else f"next:{request['q']}"
            return [
                youtube.SearchResponse(
                    videos=[],
                    pagination=youtube.Pagination(next_page_token=token),
                )
            ]
        video_id = f"video-{self.search_calls}"
        return [
            youtube.SearchResponse(
                videos=[youtube.Video(id=video_id, title=f"Candidate {video_id}")]
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
                    youtube_video.TranscriptLanguage(name="Deutsch", lang="de")
                ],
            )
        ]


class CachedEmptyApi(ResumeApi):
    def is_cached(self, engine, request):
        return engine == "youtube"

    def search(self, requests):
        self.search_calls += 1
        return [youtube.SearchResponse(videos=[])]


class FailSecondPageOnceApi(ResumeApi):
    def __init__(self) -> None:
        super().__init__(paginated=True)
        self.requests = []
        self.failed = False

    def search(self, requests):
        self.search_calls += 1
        request = dict(requests[0])
        self.requests.append(request)
        if request.get("sp") and not self.failed:
            self.failed = True
            raise TimeoutError("network cut after first page")
        token = None if request.get("sp") else f"next:{request['q']}"
        return [
            youtube.SearchResponse(
                videos=[], pagination=youtube.Pagination(next_page_token=token)
            )
        ]


class FailVideoDetailOnceApi(ResumeApi):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def video(self, requests):
        self.video_calls += 1
        if not self.failed:
            self.failed = True
            raise TimeoutError("video detail outcome unknown")
        video_id = requests[0]["video_id"]
        return [
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
        ]


class FailTranscriptOnceApi(ResumeApi):
    def __init__(self) -> None:
        super().__init__()
        self.transcript_calls = 0
        self.failed = False

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
                        name="English (auto-generated)", lang="en"
                    )
                ],
            )
        ]

    def transcripts(self, requests):
        self.transcript_calls += 1
        if not self.failed:
            self.failed = True
            raise TimeoutError("transcript outcome unknown")
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


class InterruptTranscriptOnceApi(FailTranscriptOnceApi):
    def is_cached(self, engine, request):
        return engine == "youtube_transcripts" and self.transcript_calls > 0

    def transcripts(self, requests):
        self.transcript_calls += 1
        if self.transcript_calls == 1:
            raise KeyboardInterrupt("interrupted after transcript dispatch")
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


class InterruptUncachedTranscriptOnceApi(FailTranscriptOnceApi):
    def transcripts(self, requests):
        self.transcript_calls += 1
        if self.transcript_calls == 1:
            raise KeyboardInterrupt("interrupted after transcript dispatch")
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


class CountingClassifier:
    model = "test-classifier"

    def __init__(self) -> None:
        self.metadata_calls = 0
        self.transcript_calls = 0

    def classify_with_usage(self, candidate, prompt, *, stage):
        if stage == "metadata":
            self.metadata_calls += 1
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="en",
                primary_reason="topic_match",
            )
        else:
            self.transcript_calls += 1
            decision = RelevanceDecision(
                decision="relevant",
                language_match="match",
                detected_language="en",
                primary_reason="topic_match",
            )
        return LlmCallResult(
            output=decision,
            usage=LlmUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


EXPANSION = TopicExpansion(
    topic_interpretation="Research topic",
    inclusion_criteria=("substantive",),
    exclusion_criteria=("passing mention",),
    search_queries=("third query", "fourth query"),
    channel_discovery_queries=("channel query",),
    ambiguity_rules=(),
)
PROMPT = CompiledClassifierPrompt(
    system_prompt="Classify.", prompt_sha256=CLASSIFY_PROMPT_SHA256
)


def test_pending_legacy_classifier_output_requires_explicit_migration(tmp_path) -> None:
    crawler = make_project_crawler(
        tmp_path / "legacy-pending",
        FailTranscriptOnceApi(),
        SearchApiCreditBudget(5, 1),
        classifier=CountingClassifier(),
    )
    crawler.run()
    state = ProjectStateStore(tmp_path / "legacy-pending").load()
    assert state.classifier_prompt_version == CLASSIFIER_PROMPT_VERSION
    (context,) = state.pending_transcript_contexts.values()
    context["metadata_decision"]["decision"] = "needs_transcript"

    with pytest.raises(ValueError, match="Migrate crawl_state.json explicitly"):
        validate_pending_transcript_decisions(state)


def make_project_crawler(
    project,
    api,
    budget,
    *,
    config=None,
    resume_state=None,
    classifier=None,
):
    config = config or CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_depth=0,
        searchapi_retries=0,
    )
    return ResearchCrawler(
        config=config,
        expansion=EXPANSION,
        classifier_prompt=PROMPT,
        searchapi=api,
        classifier=classifier or UnusedClassifier(),
        llm_client=NoopLlm(),
        search_budget=budget,
        writer=JsonlRunWriter(project.parent, project.name),
        state_store=ProjectStateStore(project),
        resume_state=resume_state,
    )


def test_budget_stopped_project_resumes_without_repeating_searches(tmp_path) -> None:
    project = tmp_path / "project"
    api = ResumeApi()
    first = make_project_crawler(project, api, SearchApiCreditBudget(4, 1)).run()

    assert first.status is RunStatus.STOPPED_BUDGET
    assert api.search_calls == 2
    assert api.video_calls == 1

    state = ProjectStateStore(project).load()
    restored_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    restored_budget.expand(8, 3)
    second = make_project_crawler(
        project,
        api,
        restored_budget,
        resume_state=state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert second.pending_videos == 0
    assert api.search_calls == 2
    assert api.video_calls == 2
    assert restored_budget.snapshot().discovery_spent == 4


def test_increasing_query_limit_exposes_next_query_only(tmp_path) -> None:
    project = tmp_path / "queries"
    api = ResumeApi(paginated=True)
    budget = SearchApiCreditBudget(10, 3)
    first = make_project_crawler(project, api, budget).run()
    assert first.status is RunStatus.COMPLETED
    assert api.search_calls == 2

    state = ProjectStateStore(project).load()
    expanded_config = CrawlConfig(
        topic_query=state.topic_query,
        language=state.language,
        start_date=state.start_date,
        max_queries=3,
        max_depth=0,
    )
    second = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget.restore(state.budget.model_dump()),
        config=expanded_config,
        resume_state=state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert api.search_calls == 3


def test_increasing_page_limit_continues_from_saved_tokens(tmp_path) -> None:
    project = tmp_path / "pages"
    api = ResumeApi(paginated=True)
    budget = SearchApiCreditBudget(10, 3)
    make_project_crawler(project, api, budget).run()
    assert api.search_calls == 2

    state = ProjectStateStore(project).load()
    expanded_config = CrawlConfig(
        topic_query=state.topic_query,
        language=state.language,
        start_date=state.start_date,
        max_queries=2,
        max_search_pages=2,
        max_depth=0,
    )
    summary = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget.restore(state.budget.model_dump()),
        config=expanded_config,
        resume_state=state,
    ).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.search_calls == 4


def test_cache_hits_are_audited_and_do_not_spend_local_credits(tmp_path) -> None:
    project = tmp_path / "cached"
    api = CachedEmptyApi()
    budget = SearchApiCreditBudget(4, 1)

    summary = make_project_crawler(project, api, budget).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.search_calls == 2
    assert budget.snapshot().discovery_spent == 0
    audit = (project / "api_call.jsonl").read_text(encoding="utf-8")
    assert audit.count('"status":"cache_hit"') == 2
    assert '"searchapi_credits":0' in audit


def test_failed_session_resumes_at_saved_page_token(tmp_path) -> None:
    project = tmp_path / "network-error"
    api = FailSecondPageOnceApi()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=2,
        max_search_pages=2,
        max_depth=0,
        searchapi_retries=0,
    )
    budget = SearchApiCreditBudget(10, 3)
    first = make_project_crawler(project, api, budget, config=config).run()
    assert first.status is RunStatus.FAILED

    state = ProjectStateStore(project).load()
    second = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget.restore(state.budget.model_dump()),
        config=config,
        resume_state=state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    seed_first_pages = [
        request
        for request in api.requests
        if request["q"] == "third query" and "sp" not in request
    ]
    assert len(seed_first_pages) == 1
    assert second.stop_reason == "frontier_exhausted"
    assert ProjectStateStore(project).load().stop_reason == "frontier_exhausted"


def test_failed_video_detail_is_retried_without_repeating_search(tmp_path) -> None:
    project = tmp_path / "video-detail-retry"
    api = FailVideoDetailOnceApi()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_retries=0,
    )
    first = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget(5, 1),
        config=config,
    ).run()

    assert first.status is RunStatus.FAILED
    state = ProjectStateStore(project).load()
    second = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget.restore(state.budget.model_dump()),
        config=config,
        resume_state=state,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert second.stop_reason == "frontier_exhausted"
    assert api.search_calls == 1
    assert api.video_calls == 2
    assert second.videos_evaluated == 1


def test_failed_transcript_dispatch_resumes_at_transcript_boundary(tmp_path) -> None:
    project = tmp_path / "transcript-retry"
    api = FailTranscriptOnceApi()
    classifier = CountingClassifier()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_retries=0,
    )
    first_budget = SearchApiCreditBudget(5, 2)
    first = make_project_crawler(
        project,
        api,
        first_budget,
        config=config,
        classifier=classifier,
    ).run()

    assert first.status is RunStatus.FAILED
    assert first_budget.snapshot().transcript_spent == 1
    state = ProjectStateStore(project).load()
    assert state.budget.completed_video_ids == []
    second_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    second = make_project_crawler(
        project,
        api,
        second_budget,
        config=config,
        resume_state=state,
        classifier=classifier,
    ).run()

    assert second.status is RunStatus.COMPLETED
    assert second.stop_reason == "frontier_exhausted"
    assert api.search_calls == 1
    assert api.video_calls == 1
    assert api.transcript_calls == 2
    assert classifier.metadata_calls == 1
    assert classifier.transcript_calls == 1
    assert second_budget.snapshot().transcript_spent == 2
    transcript_lines = (project / "transcript.jsonl").read_text().splitlines()
    assert len(transcript_lines) == 1


def test_pending_transcript_reservation_survives_interruption_and_cache_change(
    tmp_path,
) -> None:
    project = tmp_path / "pending-transcript"
    api = InterruptTranscriptOnceApi()
    classifier = CountingClassifier()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_retries=0,
    )
    crawler = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget(5, 2),
        config=config,
        classifier=classifier,
    )

    with pytest.raises(KeyboardInterrupt, match="after transcript dispatch"):
        crawler.run()

    state = ProjectStateStore(project).load()
    assert state.budget.pending == [
        {
            "reservation_id": state.budget.pending[0]["reservation_id"],
            "video_id": "video-1",
            "credits": 1,
        }
    ]
    assert "video-1" in state.pending_transcript_contexts
    assert state.pending_transcript_contexts["video-1"]["transcript_dispatch_pending"]
    restored_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    summary = make_project_crawler(
        project,
        api,
        restored_budget,
        config=config,
        resume_state=state,
        classifier=classifier,
    ).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.search_calls == 1
    assert api.video_calls == 1
    assert api.transcript_calls == 2
    assert classifier.metadata_calls == 1
    assert classifier.transcript_calls == 1
    assert restored_budget.snapshot().transcript_spent == 1
    transcript = json.loads((project / "transcript.jsonl").read_text().splitlines()[0])
    assert transcript["searchapi_credits"] == 0
    events = (project / "budget_event.jsonl").read_text()
    assert "transcript:video-1:interrupted" in events


def test_interrupted_uncached_transcript_retry_charges_both_dispatches(
    tmp_path,
) -> None:
    project = tmp_path / "uncached-transcript-retry"
    api = InterruptUncachedTranscriptOnceApi()
    classifier = CountingClassifier()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_retries=0,
    )
    crawler = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget(5, 2),
        config=config,
        classifier=classifier,
    )

    with pytest.raises(KeyboardInterrupt, match="after transcript dispatch"):
        crawler.run()

    state = ProjectStateStore(project).load()
    restored_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    summary = make_project_crawler(
        project,
        api,
        restored_budget,
        config=config,
        resume_state=state,
        classifier=classifier,
    ).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.transcript_calls == 2
    assert restored_budget.snapshot().transcript_spent == 2
    events = (project / "budget_event.jsonl").read_text()
    assert "transcript:video-1:interrupted" in events


def test_interrupted_transcript_retry_borrows_one_unspent_discovery_credit(
    tmp_path,
) -> None:
    project = tmp_path / "blocked-transcript-retry"
    api = InterruptUncachedTranscriptOnceApi()
    classifier = CountingClassifier()
    config = CrawlConfig(
        topic_query="research topic",
        language="en",
        start_date=date(2026, 1, 1),
        max_queries=1,
        max_depth=0,
        searchapi_retries=0,
    )
    crawler = make_project_crawler(
        project,
        api,
        SearchApiCreditBudget(4, 1),
        config=config,
        classifier=classifier,
    )

    with pytest.raises(KeyboardInterrupt, match="after transcript dispatch"):
        crawler.run()

    state = ProjectStateStore(project).load()
    restored_budget = SearchApiCreditBudget.restore(state.budget.model_dump())
    summary = make_project_crawler(
        project,
        api,
        restored_budget,
        config=config,
        resume_state=state,
        classifier=classifier,
    ).run()

    assert summary.status is RunStatus.COMPLETED
    assert api.transcript_calls == 2
    snapshot = restored_budget.snapshot()
    assert snapshot.transcript_spent == 2
    assert snapshot.transcript_capacity == 2
    assert snapshot.discovery_capacity == 2
    assert snapshot.total_committed == 4
    assert snapshot.total_remaining == 0
    resumed_state = ProjectStateStore(project).load()
    assert resumed_state.budget.pending == []
    assert resumed_state.pending_transcript_contexts == {}
