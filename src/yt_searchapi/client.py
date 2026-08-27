"""Shared SearchAPI client for YouTube engines."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Required, TypedDict, TypeVar, overload

import httpx
from diskcache import Cache
from loguru import logger
from pydantic import BaseModel

from yt_searchapi.models import (
    youtube,
    youtube_channel,
    youtube_channel_videos,
    youtube_comments,
    youtube_transcripts,
    youtube_video,
)
from yt_searchapi.models.account import AccountResponse
from yt_searchapi.settings import DEFAULT_SEARCHAPI_WORKERS

T = TypeVar("T", bound=BaseModel)
SearchParams = (
    youtube.SearchParameters
    | youtube_video.SearchParameters
    | youtube_transcripts.SearchParameters
    | youtube_comments.SearchParameters
    | youtube_channel.SearchParameters
    | youtube_channel_videos.SearchParameters
)


class SearchRequest(TypedDict, total=False):
    q: Required[str]
    sp: str
    gl: str
    hl: str


class VideoRequest(TypedDict, total=False):
    video_id: Required[str]
    gl: str
    hl: str


class TranscriptsRequest(TypedDict, total=False):
    video_id: Required[str]
    lang: str
    transcript_type: str
    transcript_name: str
    only_available: bool


class CommentsRequest(TypedDict, total=False):
    video_id: Required[str]
    gl: str
    hl: str
    next_page_token: str


class ChannelRequest(TypedDict, total=False):
    channel_id: Required[str]
    gl: str
    hl: str


class ChannelVideosRequest(TypedDict, total=False):
    channel_id: Required[str]
    gl: str
    hl: str
    next_page_token: str


BASE_URL = "https://www.searchapi.io/api/v1"


def _normalize_provider_counts(value: object) -> object:
    """Repair near-integer subscriber counts caused by provider float math."""
    if isinstance(value, list):
        return [_normalize_provider_counts(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _normalize_provider_counts(item) for key, item in value.items()}
    subscribers = normalized.get("subscribers")
    if isinstance(subscribers, float) and math.isfinite(subscribers):
        nearest = round(subscribers)
        tolerance = max(1e-9, 4 * math.ulp(subscribers))
        if abs(subscribers - nearest) <= tolerance:
            normalized["subscribers"] = nearest
    return normalized


class SearchApiError(Exception):
    """Raised when SearchAPI returns a non-success HTTP response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"{status_code}: {message}")


def is_retryable_searchapi_error(exc: BaseException) -> bool:
    """Return whether a failed SearchAPI operation is safe to try again."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.TransportError | TimeoutError | ConnectionError):
            return True
        if isinstance(current, SearchApiError):
            return _is_retryable_status(current.status_code)
        current = current.__cause__ or current.__context__
    return False


class SearchApiClient:
    """Typed client over SearchAPI YouTube engines on GET /search."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl: int | None = None,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
        max_workers: int = DEFAULT_SEARCHAPI_WORKERS,
        jsonl_prefix: str | Path | None = None,
        return_exceptions: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._cache = Cache(str(cache_dir)) if cache_dir is not None else None
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_workers = max_workers
        self._jsonl_prefix = Path(jsonl_prefix) if jsonl_prefix is not None else None
        self._jsonl_lock = threading.Lock()
        self._return_exceptions = return_exceptions
        logger.debug(
            "SearchApiClient ready base_url={} cache={} jsonl_prefix={} "
            "max_workers={} max_retries={} return_exceptions={}",
            base_url.rstrip("/"),
            cache_dir,
            self._jsonl_prefix,
            max_workers,
            max_retries,
            return_exceptions,
        )

    @property
    def max_workers(self) -> int:
        """Maximum number of independent provider requests in one batch."""

        return self._max_workers

    @property
    def timeout(self) -> float:
        """Per-request SearchAPI timeout in seconds."""

        return self._timeout

    def close(self) -> None:
        """Close the underlying HTTP client and cache if this instance created them."""
        if self._owns_client:
            self._client.close()
        if self._cache is not None:
            self._cache.close()
        logger.debug("SearchApiClient closed")

    def __enter__(self) -> SearchApiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @overload
    def search(
        self,
        requests: Sequence[SearchRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube.SearchResponse]: ...

    @overload
    def search(
        self,
        requests: Sequence[SearchRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[True],
    ) -> list[youtube.SearchResponse | BaseException]: ...

    def search(
        self,
        requests: Sequence[SearchRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: bool | None = None,
    ) -> list[youtube.SearchResponse] | list[youtube.SearchResponse | BaseException]:
        """Search YouTube; set max_pages to follow pagination via sp."""
        params_list = [
            youtube.SearchParameters(
                engine="youtube",
                q=r["q"],
                sp=r.get("sp"),
                gl=r.get("gl"),
                hl=r.get("hl"),
            )
            for r in requests
        ]
        if max_pages is None:
            return self._get_many(
                youtube.SearchResponse, params_list, return_exceptions=return_exceptions
            )
        return self._paginate_many(
            youtube.SearchResponse,
            params_list,
            max_pages,
            token_field="sp",
            return_exceptions=return_exceptions,
        )

    @overload
    def video(
        self,
        requests: Sequence[VideoRequest],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube_video.SearchResponse]: ...

    @overload
    def video(
        self,
        requests: Sequence[VideoRequest],
        *,
        return_exceptions: Literal[True],
    ) -> list[youtube_video.SearchResponse | BaseException]: ...

    def video(
        self,
        requests: Sequence[VideoRequest],
        *,
        return_exceptions: bool | None = None,
    ) -> (
        list[youtube_video.SearchResponse]
        | list[youtube_video.SearchResponse | BaseException]
    ):
        """Fetch YouTube video details."""
        return self._get_many(
            youtube_video.SearchResponse,
            [
                youtube_video.SearchParameters(
                    engine="youtube_video",
                    video_id=r["video_id"],
                    gl=r.get("gl"),
                    hl=r.get("hl"),
                )
                for r in requests
            ],
            return_exceptions=return_exceptions,
        )

    @overload
    def transcripts(
        self,
        requests: Sequence[TranscriptsRequest],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube_transcripts.SearchResponse]: ...

    @overload
    def transcripts(
        self,
        requests: Sequence[TranscriptsRequest],
        *,
        return_exceptions: Literal[True],
    ) -> list[youtube_transcripts.SearchResponse | BaseException]: ...

    def transcripts(
        self,
        requests: Sequence[TranscriptsRequest],
        *,
        return_exceptions: bool | None = None,
    ) -> (
        list[youtube_transcripts.SearchResponse]
        | list[youtube_transcripts.SearchResponse | BaseException]
    ):
        """Fetch YouTube video transcripts."""
        return self._get_many(
            youtube_transcripts.SearchResponse,
            [
                youtube_transcripts.SearchParameters(
                    engine="youtube_transcripts",
                    video_id=r["video_id"],
                    lang=r.get("lang"),
                    transcript_type=r.get("transcript_type"),
                    transcript_name=r.get("transcript_name"),
                    only_available=r.get("only_available"),
                )
                for r in requests
            ],
            return_exceptions=return_exceptions,
        )

    @overload
    def comments(
        self,
        requests: Sequence[CommentsRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube_comments.SearchResponse]: ...

    @overload
    def comments(
        self,
        requests: Sequence[CommentsRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[True],
    ) -> list[youtube_comments.SearchResponse | BaseException]: ...

    def comments(
        self,
        requests: Sequence[CommentsRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: bool | None = None,
    ) -> (
        list[youtube_comments.SearchResponse]
        | list[youtube_comments.SearchResponse | BaseException]
    ):
        """Fetch YouTube video comments; set max_pages to follow pagination."""
        params_list = [
            youtube_comments.SearchParameters(
                engine="youtube_comments",
                video_id=r["video_id"],
                gl=r.get("gl"),
                hl=r.get("hl"),
                next_page_token=r.get("next_page_token"),
            )
            for r in requests
        ]
        if max_pages is None:
            return self._get_many(
                youtube_comments.SearchResponse,
                params_list,
                return_exceptions=return_exceptions,
            )
        return self._paginate_many(
            youtube_comments.SearchResponse,
            params_list,
            max_pages,
            token_field="next_page_token",
            return_exceptions=return_exceptions,
        )

    @overload
    def channel(
        self,
        requests: Sequence[ChannelRequest],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube_channel.SearchResponse]: ...

    @overload
    def channel(
        self,
        requests: Sequence[ChannelRequest],
        *,
        return_exceptions: Literal[True],
    ) -> list[youtube_channel.SearchResponse | BaseException]: ...

    def channel(
        self,
        requests: Sequence[ChannelRequest],
        *,
        return_exceptions: bool | None = None,
    ) -> (
        list[youtube_channel.SearchResponse]
        | list[youtube_channel.SearchResponse | BaseException]
    ):
        """Fetch YouTube channel details."""
        return self._get_many(
            youtube_channel.SearchResponse,
            [
                youtube_channel.SearchParameters(
                    engine="youtube_channel",
                    channel_id=r["channel_id"],
                    gl=r.get("gl"),
                    hl=r.get("hl"),
                )
                for r in requests
            ],
            return_exceptions=return_exceptions,
        )

    @overload
    def channel_videos(
        self,
        requests: Sequence[ChannelVideosRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[False] = False,
    ) -> list[youtube_channel_videos.SearchResponse]: ...

    @overload
    def channel_videos(
        self,
        requests: Sequence[ChannelVideosRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: Literal[True],
    ) -> list[youtube_channel_videos.SearchResponse | BaseException]: ...

    def channel_videos(
        self,
        requests: Sequence[ChannelVideosRequest],
        *,
        max_pages: int | None = None,
        return_exceptions: bool | None = None,
    ) -> (
        list[youtube_channel_videos.SearchResponse]
        | list[youtube_channel_videos.SearchResponse | BaseException]
    ):
        """List channel videos; set max_pages to follow pagination."""
        params_list = [
            youtube_channel_videos.SearchParameters(
                engine="youtube_channel_videos",
                channel_id=r["channel_id"],
                gl=r.get("gl"),
                hl=r.get("hl"),
                next_page_token=r.get("next_page_token"),
            )
            for r in requests
        ]
        if max_pages is None:
            return self._get_many(
                youtube_channel_videos.SearchResponse,
                params_list,
                return_exceptions=return_exceptions,
            )
        return self._paginate_many(
            youtube_channel_videos.SearchResponse,
            params_list,
            max_pages,
            token_field="next_page_token",
            return_exceptions=return_exceptions,
        )

    def me(self) -> AccountResponse:
        """Fetch account credits and hourly API usage from GET /me."""
        response, elapsed = self._request_with_retries("/me", label="endpoint=/me")
        result = AccountResponse.model_validate(response.json())
        logger.info("fetched endpoint=/me elapsed={:.2f}s", elapsed)
        return result

    def is_cached(self, engine: str, params: dict[str, object]) -> bool:
        """Return whether one engine request can be served without a credit.

        Callers must perform this check before reserving or spending the credit
        for that request. Each project uses one client and no parallel runs.
        """

        if self._cache is None:
            return False
        cleaned = {"engine": engine, **params}
        cleaned = {key: value for key, value in cleaned.items() if value is not None}
        return self._cache.get(_cache_key(cleaned)) is not None

    def _request_with_retries(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
        label: str,
    ) -> tuple[httpx.Response, float]:
        """GET path with retries on 429, 5xx, and transport errors."""
        started = time.perf_counter()
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            logger.debug(
                "GET {} {} attempt={}/{} params={}",
                path,
                label,
                attempt + 1,
                self._max_retries + 1,
                params,
            )
            try:
                response = self._client.get(path, params=params, timeout=self._timeout)
            except httpx.RequestError as exc:
                if attempt >= self._max_retries:
                    elapsed = time.perf_counter() - started
                    logger.error(
                        "request failed {} elapsed={:.2f}s error_type={}",
                        label,
                        elapsed,
                        type(exc).__name__,
                    )
                    raise
                delay = self._retry_backoff * (2**attempt)
                logger.warning(
                    "transport error {} attempt={}/{}, sleeping {}s error_type={}",
                    label,
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    type(exc).__name__,
                )
                time.sleep(delay)
                continue
            if (
                not _is_retryable_status(response.status_code)
                or attempt >= self._max_retries
            ):
                break
            delay = self._retry_backoff * (2**attempt)
            logger.warning(
                "retryable status {} {} attempt={}/{}, sleeping {}s",
                response.status_code,
                label,
                attempt + 1,
                self._max_retries + 1,
                delay,
            )
            time.sleep(delay)
        assert response is not None
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            message = _error_message(response)
            logger.error(
                "request failed {} status={} elapsed={:.2f}s",
                label,
                response.status_code,
                elapsed,
            )
            raise SearchApiError(response.status_code, message)
        return response, elapsed

    def _resolve_return_exceptions(self, return_exceptions: bool | None) -> bool:
        """Resolve per-call flag, falling back to the client constructor setting."""
        return (
            self._return_exceptions if return_exceptions is None else return_exceptions
        )

    def _paginate_one(
        self,
        model: type[T],
        params: SearchParams,
        max_pages: int,
        *,
        token_field: str,
        return_exceptions: bool = False,
    ) -> list[T] | list[T | BaseException]:
        """Fetch up to max_pages for one request, following next_page_token.

        When ``return_exceptions`` is true and a page fails mid-chain,
        returns pages fetched so far plus the exception as the last element.
        """
        pages: list[T | BaseException] = []
        current = params
        for page_num in range(1, max_pages + 1):
            has_token = bool(getattr(current, token_field, None))
            logger.info(
                "fetching page engine={} page={}/{} has_token={}",
                getattr(current, "engine", None),
                page_num,
                max_pages,
                has_token,
            )
            try:
                resp = self._get(model, current)
            except Exception as exc:
                if return_exceptions:
                    logger.error(
                        "pagination page failed engine={} page={} error_type={}",
                        getattr(current, "engine", None),
                        page_num,
                        type(exc).__name__,
                    )
                    pages.append(exc)
                    return pages
                raise
            pages.append(resp)
            next_token = _next_page_token(resp)
            if not next_token:
                logger.debug(
                    "pagination exhausted engine={} page={}",
                    getattr(current, "engine", None),
                    page_num,
                )
                break
            if page_num >= max_pages:
                break
            current = current.model_copy(update={token_field: next_token})
        return pages

    @overload
    def _paginate_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        max_pages: int,
        *,
        token_field: str,
        return_exceptions: Literal[False] = False,
    ) -> list[T]: ...

    @overload
    def _paginate_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        max_pages: int,
        *,
        token_field: str,
        return_exceptions: Literal[True],
    ) -> list[T | BaseException]: ...

    def _paginate_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        max_pages: int,
        *,
        token_field: str,
        return_exceptions: bool | None = None,
    ) -> list[T] | list[T | BaseException]:
        """Paginate each request (pages sequential; requests via max_workers pool).

        When ``return_exceptions`` is true, a failed request chain contributes
        one exception in the flat result (or prior pages plus that exception if
        failure was mid-chain) instead of aborting the batch.
        """
        if max_pages < 1:
            raise ValueError(f"max_pages must be >= 1 when set, got {max_pages!r}")
        if not params_list:
            return []
        catch = self._resolve_return_exceptions(return_exceptions)
        logger.info(
            "paginating {} request(s) max_pages={} max_workers={}",
            len(params_list),
            max_pages,
            self._max_workers,
        )
        started = time.perf_counter()

        if catch:

            def worker(p: SearchParams) -> list[T | BaseException]:
                try:
                    return self._paginate_one(
                        model,
                        p,
                        max_pages,
                        token_field=token_field,
                        return_exceptions=True,
                    )
                except Exception as exc:
                    logger.error(
                        "pagination chain failed engine={} error_type={}",
                        getattr(p, "engine", None),
                        type(exc).__name__,
                    )
                    return [exc]
        else:

            def worker(p: SearchParams) -> list[T | BaseException]:
                return self._paginate_one(
                    model,
                    p,
                    max_pages,
                    token_field=token_field,
                    return_exceptions=False,
                )

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            page_lists = list(pool.map(worker, params_list))
        results = [page for pages in page_lists for page in pages]
        elapsed = time.perf_counter() - started
        logger.info(
            "paginated {} request(s) -> {} page(s) elapsed={:.2f}s",
            len(params_list),
            len(results),
            elapsed,
        )
        return results

    def _get(self, model: type[T], params: SearchParams) -> T:
        cleaned = params.model_dump(exclude_none=True)
        engine = cleaned.get("engine")
        key = _cache_key(cleaned)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("cache hit engine={} params={}", engine, cleaned)
                return model.model_validate(_normalize_provider_counts(cached))
        response, elapsed = self._request_with_retries(
            "/search",
            params=cleaned,
            label=f"engine={engine}",
        )
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            message = str(data["error"])
            logger.error(
                "request failed engine={} status={} elapsed={:.2f}s",
                engine,
                response.status_code,
                elapsed,
            )
            raise SearchApiError(response.status_code, message)
        result = model.model_validate(_normalize_provider_counts(data))
        if self._cache is not None:
            self._cache.set(
                key,
                result.model_dump(mode="json"),
                expire=self._cache_ttl,
            )
            logger.debug("cache set engine={}", engine)
        logger.info(
            "fetched engine={} elapsed={:.2f}s params={}",
            engine,
            elapsed,
            cleaned,
        )
        self._append_jsonl(engine, result)
        return result

    @overload
    def _get_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[T]: ...

    @overload
    def _get_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        *,
        return_exceptions: Literal[True],
    ) -> list[T | BaseException]: ...

    def _get_many(
        self,
        model: type[T],
        params_list: Sequence[SearchParams],
        *,
        return_exceptions: bool | None = None,
    ) -> list[T] | list[T | BaseException]:
        """Fetch many search responses in parallel.

        When ``return_exceptions`` is true (constructor flag or keyword), failed
        items are returned as exceptions in order instead of raising — like
        ``asyncio.gather(..., return_exceptions=True)``.
        """
        if not params_list:
            return []
        catch = self._resolve_return_exceptions(return_exceptions)
        logger.info(
            "fetching {} request(s) with max_workers={}",
            len(params_list),
            self._max_workers,
        )
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            if catch:

                def _one(params: SearchParams) -> T | BaseException:
                    try:
                        return self._get(model, params)
                    except Exception as exc:
                        logger.error(
                            "batch item failed engine={} error_type={}",
                            getattr(params, "engine", None),
                            type(exc).__name__,
                        )
                        return exc

                results = list(pool.map(_one, params_list))
            else:
                results = list(pool.map(lambda p: self._get(model, p), params_list))
        elapsed = time.perf_counter() - started
        logger.info(
            "fetched {} request(s) elapsed={:.2f}s",
            len(results),
            elapsed,
        )
        return results

    def _append_jsonl(self, engine: object, result: BaseModel) -> None:
        if self._jsonl_prefix is None or not isinstance(engine, str):
            return
        stem = (
            engine.removeprefix("youtube_") if engine.startswith("youtube_") else engine
        )
        path = self._jsonl_prefix / f"{stem}.jsonl"
        line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        with self._jsonl_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        logger.debug("appended jsonl path={}", path)


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _cache_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _next_page_token(response: BaseModel) -> str | None:
    pagination = getattr(response, "pagination", None)
    if pagination is None:
        return None
    token = getattr(pagination, "next_page_token", None)
    return token if token else None


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(data, dict) and data.get("error") is not None:
        return str(data["error"])
    return response.text or response.reason_phrase
