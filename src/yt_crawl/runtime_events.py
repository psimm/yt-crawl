"""Typed, presentation-only events for provider request activity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Provider = Literal["searchapi", "openai"]
RuntimeEventPhase = Literal["started", "finished"]
RuntimeEventStatus = Literal["success", "error", "cache_hit"]


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One provider operation transition emitted around an audited call."""

    provider: Provider
    operation: str
    phase: RuntimeEventPhase
    status: RuntimeEventStatus | None = None
    cache_hit: bool = False
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CrawlProgressSnapshot:
    """Small immutable crawler state intended only for live presentation."""

    discovered: int = 0
    evaluated: int = 0
    relevant: int = 0
    transcripts: int = 0
    pending: int = 0
    queries_done: int = 0
    queries_started: int = 0
    queries_planned: int = 0
    channels_done: int = 0
    channels_discovered: int = 0
    channels_exhausted: int = 0
    channels_page_capped: int = 0


RuntimeEventCallback = Callable[[RuntimeEvent], None]
CrawlProgressCallback = Callable[[CrawlProgressSnapshot], None]


__all__ = [
    "CrawlProgressCallback",
    "CrawlProgressSnapshot",
    "RuntimeEvent",
    "RuntimeEventCallback",
]
