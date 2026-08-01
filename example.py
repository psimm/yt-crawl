"""Example usage of SearchApiClient for each YouTube engine.

Warning: running this file performs real, billable SearchAPI requests.
"""

from pathlib import Path

from loguru import logger

from yt_searchapi import SearchApiClient
from yt_searchapi.settings import Settings

SEARCH_QUERIES = [
    "lofi hip hop",
    "python tutorial",
    "space documentary",
]
VIDEO_IDS = [
    "dQw4w9WgXcQ",  # Rick Astley
    "jNQXAC9IVRw",  # Me at the zoo
    "9bZkp7q19f0",  # Gangnam Style
]
CHANNEL_IDS = [
    "UCuAXFkgsw1L7xaCfnd5JJOw",  # Rick Astley
    "UCX6OQ3DkcsbYNE6H8uQQuVA",  # MrBeast
    "UCBR8-60-B28hp2BmDPdntcQ",  # YouTube
]
OUT_DIR = Path("output")


def main() -> None:
    settings = Settings()
    if not settings.searchapi_api_key:
        raise RuntimeError("SEARCHAPI_API_KEY is required to run this billable example")
    OUT_DIR.mkdir(exist_ok=True)
    logger.info("starting example run jsonl_prefix={}", OUT_DIR)

    with SearchApiClient(
        settings.searchapi_api_key,
        cache_dir=".cache/searchapi",
        jsonl_prefix=OUT_DIR,
    ) as client:
        client.search([{"q": q, "gl": "us", "hl": "en"} for q in SEARCH_QUERIES])
        client.video([{"video_id": v, "gl": "us", "hl": "en"} for v in VIDEO_IDS])
        client.transcripts([{"video_id": v, "lang": "en"} for v in VIDEO_IDS])
        client.comments([{"video_id": v, "gl": "us", "hl": "en"} for v in VIDEO_IDS])
        client.channel([{"channel_id": c, "gl": "us", "hl": "en"} for c in CHANNEL_IDS])
        client.channel_videos(
            [{"channel_id": c, "gl": "us", "hl": "en"} for c in CHANNEL_IDS]
        )

    logger.info("finished example run")


if __name__ == "__main__":
    main()
