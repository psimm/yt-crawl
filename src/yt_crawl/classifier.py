"""Strict Responses-API video relevance classifier."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, Sequence

from pydantic import Field

from yt_crawl.prompts import (
    DEFAULT_LLM_MODEL,
    CompiledClassifierPrompt,
    LlmCallResult,
    StrictModel,
    extract_usage,
)


class VideoCandidate(StrictModel):
    """Evidence available at the point a relevance decision is made."""

    video_id: str
    title: str
    description: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    published_at: datetime | None = None
    duration_text: str | None = None
    transcript_excerpt: str | None = None
    discovery_source: Literal["search", "channel", "related"]
    discovery_reference: str


class RelevanceDecision(StrictModel):
    """Auditable staged decision returned by the model."""

    decision: Literal["relevant", "irrelevant"]
    language_match: Literal["match", "mismatch", "unknown"]
    detected_language: str | None = Field(
        pattern=r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$",
        description="BCP-47-like language code such as de, en, or pt-BR",
    )
    primary_reason: Literal[
        "topic_match",
        "off_topic",
        "wrong_language",
        "insufficient_evidence",
        "excluded_scope",
    ]


@dataclass(frozen=True, slots=True)
class ClassificationJob:
    """One independently auditable item for bounded concurrent classification."""

    candidate: VideoCandidate
    stage: Literal["metadata", "transcript"]


class _Responses(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _Responses


class RelevanceClassifier:
    """Classify candidates using an injected OpenAI-compatible client."""

    def __init__(
        self,
        client: _OpenAIClient,
        *,
        model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self._client = client
        self.model = model

    def classify(
        self,
        candidate: VideoCandidate,
        prompt: CompiledClassifierPrompt,
        *,
        stage: Literal["metadata", "transcript"],
    ) -> RelevanceDecision:
        return self.classify_with_usage(candidate, prompt, stage=stage).output

    def classify_with_usage(
        self,
        candidate: VideoCandidate,
        prompt: CompiledClassifierPrompt,
        *,
        stage: Literal["metadata", "transcript"],
    ) -> LlmCallResult[RelevanceDecision]:
        """Classify once and expose token usage for the crawl budget ledger."""

        candidate_json = json.dumps(
            {
                "classification_stage": stage,
                # Publication time is an operational gate evaluated by the
                # crawler before classification. Keeping it out of the model
                # payload makes a second, qualitative date cutoff impossible.
                "candidate": candidate.model_dump(
                    mode="json", exclude={"published_at"}
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        response = self._client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt.system_prompt,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": candidate_json}],
                },
            ],
            # Disable GPT-5.6's automatic suffix checkpoint. Only the fixed
            # developer instructions and labeled examples above are cacheable;
            # the video-specific user content is never part of a cache write.
            prompt_cache_options={"mode": "explicit"},
            prompt_cache_key=prompt.prompt_sha256,
            text_format=RelevanceDecision,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("Responses API returned no parsed RelevanceDecision")
        decision = (
            parsed
            if isinstance(parsed, RelevanceDecision)
            else RelevanceDecision.model_validate(parsed)
        )
        decision = _normalize_decision_consistency(decision)
        _validate_decision_consistency(decision)
        return LlmCallResult(
            output=decision,
            usage=extract_usage(response),
            response_id=getattr(response, "id", None),
        )

    def classify_many_with_usage(
        self,
        jobs: Sequence[ClassificationJob],
        prompt: CompiledClassifierPrompt,
        *,
        max_workers: int,
        return_exceptions: bool = False,
    ) -> tuple[LlmCallResult[RelevanceDecision] | Exception, ...]:
        """Classify independent jobs in order with a bounded worker pool.

        Audited clients receive one context per worker so concurrent calls keep
        their operation and local audit rows isolated. Plain injected test clients
        remain supported without requiring the auditing facade.
        """

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if not jobs:
            return ()

        call_context = getattr(self._client, "call_context", None)

        def classify_one(
            job: ClassificationJob,
        ) -> LlmCallResult[RelevanceDecision] | Exception:
            context = (
                call_context(
                    "transcript" if job.stage == "transcript" else "discovery",
                    (
                        "classify_video_transcript"
                        if job.stage == "transcript"
                        else "classify_video_metadata"
                    ),
                )
                if callable(call_context)
                else nullcontext()
            )
            try:
                with context:
                    return self.classify_with_usage(
                        job.candidate,
                        prompt,
                        stage=job.stage,
                    )
            except Exception as exc:
                if return_exceptions:
                    return exc
                raise

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(classify_one, jobs))


def _validate_decision_consistency(
    decision: RelevanceDecision,
) -> None:
    """Reject structurally valid but logically inconsistent model output."""

    if decision.language_match == "unknown":
        if decision.detected_language is not None:
            raise ValueError("Unknown language matches require detected_language=None")
    elif decision.detected_language is None:
        raise ValueError("Known language matches require detected_language")
    if decision.decision == "relevant":
        if decision.language_match != "match":
            raise ValueError("Relevant decisions require language_match='match'")
        if decision.primary_reason != "topic_match":
            raise ValueError("Relevant decisions require primary_reason='topic_match'")
    elif decision.primary_reason == "topic_match":
        raise ValueError("Irrelevant decisions cannot use primary_reason='topic_match'")


def _normalize_decision_consistency(
    decision: RelevanceDecision,
) -> RelevanceDecision:
    """Repair the one harmless language-field contradiction models emit.

    ``language_match='unknown'`` means the model did not have enough reliable
    language evidence. A simultaneously supplied code is therefore not
    auditable evidence and must not turn an otherwise usable decision into a
    failed crawl item.
    """

    if decision.language_match == "unknown" and decision.detected_language is not None:
        return decision.model_copy(update={"detected_language": None})
    return decision


def compact_relevance_decision(
    value: RelevanceDecision | dict[str, Any],
    *,
    requested_language: str,
) -> RelevanceDecision:
    """Load current or legacy pending output into the compact decision schema."""

    if isinstance(value, RelevanceDecision):
        return value
    compact = {
        key: value.get(key)
        for key in (
            "decision",
            "language_match",
            "detected_language",
            "primary_reason",
        )
    }
    try:
        decision = RelevanceDecision.model_validate(compact)
    except ValueError:
        if compact["language_match"] != "match" or not isinstance(
            compact["detected_language"], str
        ):
            raise
        # Legacy pending responses sometimes used names such as "German".
        # A positive match lets us replace that spelling with the already
        # validated requested code without reclassifying the video.
        requested_parts = requested_language.strip().replace("_", "-").split("-", 1)
        compact["detected_language"] = "-".join(
            [requested_parts[0].lower(), *requested_parts[1:]]
        )
        decision = RelevanceDecision.model_validate(compact)
    decision = _normalize_decision_consistency(decision)
    _validate_decision_consistency(decision)
    return decision


__all__ = [
    "ClassificationJob",
    "RelevanceClassifier",
    "RelevanceDecision",
    "VideoCandidate",
    "compact_relevance_decision",
]
