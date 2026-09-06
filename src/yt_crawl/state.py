"""Validated project checkpoint used by ``yt-crawl resume``."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yt_crawl.classifier import compact_relevance_decision
from yt_crawl.prompts import CLASSIFIER_PROMPT_VERSION
from yt_crawl.settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)

LEGACY_CLASSIFIER_PROMPT_VERSION = "relevance-v1"


class CheckpointPromptIntegrityError(ValueError):
    """A checkpoint's stored classifier prompt cannot be trusted."""


def checkpoint_recovery_message(project_dir: str | Path) -> str:
    """Explain how to recover when preparation never produced a checkpoint."""

    project = Path(project_dir).expanduser().resolve()
    return (
        f"No valid resumable checkpoint is available in {project}. "
        "Research preparation may not have finished. Start again with a new "
        "project path: `uv run yt-crawl start --project <new-path> ...`."
    )


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetState(StateModel):
    max_credits: int = Field(ge=0)
    transcript_capacity: int = Field(ge=0)
    discovery_spent: int = Field(ge=0)
    transcript_spent: int = Field(ge=0)
    pending: list[dict[str, Any]] = []
    completed_video_ids: list[str] = []


class PageProgress(StateModel):
    pages_completed: int = Field(default=0, ge=0)
    next_page_token: str | None = None
    exhausted: bool = False
    first_request_id: str | None = None


class CrawlProjectState(StateModel):
    """Complete operational state for one project.

    JSONL remains the immutable audit trail. This file is deliberately mutable:
    it is the scheduler checkpoint needed to continue without replaying work.
    """

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1)
    topic_query: str = Field(min_length=1)
    language: str = Field(min_length=1)
    start_date: date
    gl: str
    hl: str
    searchapi_timeout_seconds: float = Field(default=90.0, gt=0)
    searchapi_retries: int = Field(default=DEFAULT_SEARCHAPI_RETRIES, ge=0, le=5)
    searchapi_workers: int = Field(default=DEFAULT_SEARCHAPI_WORKERS, ge=1)
    llm_workers: int = Field(default=DEFAULT_LLM_WORKERS, ge=1)
    model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1)
    llm_api_base: str | None = None
    transcript_excerpt_chars: int
    max_depth: int = Field(ge=0)
    max_queries: int = Field(ge=1)
    max_search_pages: int = Field(ge=1)
    max_channel_pages: int = Field(ge=1)
    expansion: dict[str, Any]
    classifier_system_prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classifier_prompt_version: str = Field(
        default=LEGACY_CLASSIFIER_PROMPT_VERSION,
        min_length=1,
    )
    planned_queries: list[dict[str, str]]
    budget: BudgetState

    video_queue: list[dict[str, Any]] = []
    channel_queue: list[dict[str, Any]] = []
    discovered_videos: dict[str, dict[str, Any]] = {}
    discovered_channels: dict[str, dict[str, Any]] = {}
    query_progress: dict[str, PageProgress] = {}
    channel_progress: dict[str, PageProgress] = {}
    queued_video_ids: list[str] = []
    queued_channel_ids: list[str] = []
    evaluated_video_ids: list[str] = []
    finalized_video_ids: list[str] = []
    dispositioned_video_ids: list[str] = []
    terminal_video_ids: list[str] = []
    deferred_video_ids: list[str] = []
    relevant_ids: list[str] = []
    transcript_ids: list[str] = []
    pending_transcript_contexts: dict[str, dict[str, Any]] = {}
    stop_reason: str = "not_started"
    last_status: str = "prepared"
    planned_records_written: bool = False


def validate_pending_transcript_decisions(state: CrawlProjectState) -> None:
    """Require stored pending model outputs to match the current binary schema.

    Checkpoint mutation is intentionally left to an explicit migration: a
    legacy deferred model response cannot safely be reinterpreted as a current
    binary judgment. Contexts that predate stored model output remain supported
    by the crawler's existing continuation path.
    """

    for video_id, context in state.pending_transcript_contexts.items():
        payload = context.get("metadata_decision")
        if payload is None:
            continue
        try:
            decision = compact_relevance_decision(
                payload,
                requested_language=state.language,
            )
        except Exception as exc:
            raise ValueError(
                f"Pending metadata decision for {video_id!r} uses an obsolete "
                "classifier schema. Migrate crawl_state.json explicitly before "
                "resuming this project."
            ) from exc
        if decision.decision != "relevant":
            raise ValueError(
                f"Pending metadata decision for {video_id!r} must be a binary "
                "relevant result before transcript continuation."
            )


def validate_classifier_prompt_integrity(state: CrawlProjectState) -> None:
    """Verify that a checkpoint's prompt text and hash describe one artifact."""

    actual = hashlib.sha256(state.classifier_system_prompt.encode("utf-8")).hexdigest()
    if actual != state.prompt_sha256:
        raise CheckpointPromptIntegrityError(
            "crawl_state.json classifier prompt hash does not match its stored "
            "classifier_system_prompt. Repair or migrate the checkpoint before "
            "resuming."
        )


def validate_resumable_classifier_prompt(state: CrawlProjectState) -> None:
    """Reject legacy or tampered classifier prompts before resume funding."""

    validate_classifier_prompt_integrity(state)
    if state.classifier_prompt_version != CLASSIFIER_PROMPT_VERSION:
        raise CheckpointPromptIntegrityError(
            "crawl_state.json uses classifier prompt version "
            f"{state.classifier_prompt_version!r}; {CLASSIFIER_PROMPT_VERSION!r} "
            "is required for binary relevance decisions. Migrate the checkpoint "
            "before resuming."
        )


class ProjectStateStore:
    """Load and atomically replace a project's operational checkpoint."""

    filename = "crawl_state.json"

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.path = self.project_dir / self.filename

    def load(self) -> CrawlProjectState:
        if not self.path.is_file():
            raise FileNotFoundError(checkpoint_recovery_message(self.project_dir))
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read crawler checkpoint {self.path}: {exc}"
            ) from exc
        state = CrawlProjectState.model_validate(payload)
        validate_classifier_prompt_integrity(state)
        return state

    def save(self, state: CrawlProjectState) -> None:
        validate_classifier_prompt_integrity(state)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(state.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(self.project_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


__all__ = [
    "BudgetState",
    "CheckpointPromptIntegrityError",
    "CrawlProjectState",
    "LEGACY_CLASSIFIER_PROMPT_VERSION",
    "PageProgress",
    "ProjectStateStore",
    "checkpoint_recovery_message",
    "validate_classifier_prompt_integrity",
    "validate_pending_transcript_decisions",
    "validate_resumable_classifier_prompt",
]
