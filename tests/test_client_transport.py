import threading
import time
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError

from yt_searchapi.client import SearchApiClient
from yt_searchapi.settings import DEFAULT_SEARCHAPI_WORKERS


def test_transcript_request_uses_exact_language_without_fallback() -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        client = SearchApiClient("test-key", client=http, max_retries=0)
        assert client.max_workers == DEFAULT_SEARCHAPI_WORKERS
        client.transcripts(
            [
                {
                    "video_id": "video-1",
                    "lang": "de",
                    "transcript_name": "Deutsch (automatisch erzeugt)",
                    "only_available": False,
                }
            ]
        )
    finally:
        http.close()

    assert len(captured) == 1
    query = parse_qs(captured[0].url.query.decode(), keep_blank_values=True)
    assert query == {
        "engine": ["youtube_transcripts"],
        "video_id": ["video-1"],
        "lang": ["de"],
        "transcript_name": ["Deutsch (automatisch erzeugt)"],
        "only_available": ["false"],
    }


def test_project_cache_has_unlimited_ttl_and_avoids_second_request(tmp_path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"videos": []})

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    request = {"q": "heat pumps", "gl": "us", "hl": "en"}
    try:
        with SearchApiClient(
            "test-key",
            client=http,
            cache_dir=tmp_path / ".cache" / "searchapi",
            cache_ttl=None,
            max_retries=0,
        ) as client:
            assert client.is_cached("youtube", request) is False
            client.search([request])
            assert client.is_cached("youtube", request) is True
            client.search([request])
    finally:
        http.close()

    assert len(requests) == 1


def test_malformed_response_is_not_cached(tmp_path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"videos": "not-a-list"})
        return httpx.Response(200, json={"videos": []})

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    request = {"q": "heat pumps", "gl": "us", "hl": "en"}
    try:
        with SearchApiClient(
            "test-key",
            client=http,
            cache_dir=tmp_path / ".cache" / "searchapi",
            cache_ttl=None,
            max_retries=0,
        ) as client:
            with pytest.raises(ValidationError):
                client.search([request])
            assert client.is_cached("youtube", request) is False

            result = client.search([request])
            assert result[0].videos == []
            assert client.is_cached("youtube", request) is True
    finally:
        http.close()

    assert len(requests) == 2


def test_independent_video_and_transcript_requests_use_bounded_parallelism() -> None:
    lock = threading.Lock()
    active = 0
    peaks: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active
        engine = request.url.params["engine"]
        with lock:
            active += 1
            peaks[engine] = max(peaks.get(engine, 0), active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return httpx.Response(200, json={}, request=request)

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        client = SearchApiClient("test-key", client=http, max_retries=0, max_workers=2)
        client.video([{"video_id": "v1"}, {"video_id": "v2"}])
        client.transcripts(
            [
                {"video_id": "v1", "lang": "de"},
                {"video_id": "v2", "lang": "de"},
            ]
        )
        client.channel_videos([{"channel_id": "c1"}, {"channel_id": "c2"}])
    finally:
        http.close()

    assert peaks == {
        "youtube_video": 2,
        "youtube_transcripts": 2,
        "youtube_channel_videos": 2,
    }


def test_timeout_applies_to_injected_transport_and_is_not_retried() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        timeout = request.extensions["timeout"]
        assert timeout == {
            "connect": 0.25,
            "read": 0.25,
            "write": 0.25,
            "pool": 0.25,
        }
        raise httpx.ReadTimeout("transcript timed out", request=request)

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        client = SearchApiClient("test-key", client=http, timeout=0.25, max_retries=0)
        with pytest.raises(httpx.ReadTimeout, match="timed out"):
            client.transcripts([{"video_id": "v1", "lang": "de"}])
    finally:
        http.close()

    assert len(requests) == 1


def test_every_localized_endpoint_preserves_gl_and_hl() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={}, request=request)

    http = httpx.Client(
        base_url="https://searchapi.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        client = SearchApiClient("test-key", client=http, max_retries=0)
        client.search([{"q": "Finanzplanung", "gl": "de", "hl": "de"}])
        client.video([{"video_id": "v1", "gl": "de", "hl": "de"}])
        client.comments([{"video_id": "v1", "gl": "de", "hl": "de"}])
        client.channel([{"channel_id": "c1", "gl": "de", "hl": "de"}])
        client.channel_videos([{"channel_id": "c1", "gl": "de", "hl": "de"}])
    finally:
        http.close()

    assert [request.url.params["engine"] for request in requests] == [
        "youtube",
        "youtube_video",
        "youtube_comments",
        "youtube_channel",
        "youtube_channel_videos",
    ]
    for request in requests:
        assert request.url.params["gl"] == "de"
        assert request.url.params["hl"] == "de"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": 0}, "timeout"),
        ({"max_workers": 0}, "max_workers"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_invalid_transport_controls_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SearchApiClient("test-key", **kwargs)
