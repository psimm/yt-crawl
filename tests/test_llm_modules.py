"""Offline tests for the injected Responses-API LLM layer."""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

from yt_crawl.classifier import (
    ClassificationJob,
    RelevanceClassifier,
    RelevanceDecision,
    VideoCandidate,
    compact_relevance_decision,
)
from yt_crawl.interview import (
    INTERVIEW_QUESTIONS,
    InterviewPlanner,
    InterviewSuggestions,
    SuggestedExample,
    build_topic_brief,
    conduct_interview,
)
from yt_crawl.prompts import (
    DEFAULT_LLM_MODEL,
    TopicBrief,
    TopicExpander,
    TopicExpansion,
    compile_classifier_prompt,
)


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed: object) -> None:
        self.responses = FakeResponses(parsed)


def sample_brief() -> TopicBrief:
    return TopicBrief(
        topic_query="heat pumps in apartment buildings",
        language="German",
        research_goal="Understand practical retrofit experience.",
        positive_examples=("Interview mit einer WEG nach Einbau einer Wärmepumpe",),
        negative_examples=("Werbung, die Wärmepumpen nur im Vorbeigehen erwähnt",),
        must_include=("Mehrfamilienhaus",),
        edge_case_guidance=("Gemischte Systeme nur bei praktischer Erfahrung",),
        verbatim_answers=(),
    )


def sample_expansion() -> TopicExpansion:
    return TopicExpansion(
        topic_interpretation="Practical heat-pump retrofits in shared housing",
        inclusion_criteria=("Substantive retrofit experience",),
        exclusion_criteria=("Pure product advertising",),
        search_queries=("Wärmepumpe Mehrfamilienhaus Interview",),
        channel_discovery_queries=("Kanal Wärmepumpe Mehrfamilienhaus",),
        ambiguity_rules=("Require concrete apartment-building evidence",),
    )


def sample_suggestions() -> InterviewSuggestions:
    return InterviewSuggestions(
        examples=tuple(
            SuggestedExample(
                label="positive",
                example=f"P{index}",
                boundary_tested=f"positive-{index}",
            )
            for index in range(1, 11)
        )
        + tuple(
            SuggestedExample(
                label="negative",
                example=f"N{index}",
                boundary_tested=f"negative-{index}",
            )
            for index in range(1, 11)
        )
    )


class PromptTests(unittest.TestCase):
    def test_compilation_is_deterministic_and_preserves_examples(self) -> None:
        first = compile_classifier_prompt(sample_brief(), sample_expansion())
        second = compile_classifier_prompt(sample_brief(), sample_expansion())

        self.assertEqual(first, second)
        self.assertEqual(len(first.prompt_sha256), 64)
        self.assertIn(sample_brief().positive_examples[0], first.system_prompt)
        self.assertIn(sample_brief().negative_examples[0], first.system_prompt)

    def test_expander_uses_default_model_and_strict_schema(self) -> None:
        expansion = sample_expansion()
        client = FakeClient(expansion.model_dump(mode="json"))

        result = TopicExpander(client).expand(sample_brief())

        self.assertEqual(result, expansion)
        (call,) = client.responses.calls
        self.assertEqual(call["model"], DEFAULT_LLM_MODEL)
        self.assertIs(call["text_format"], TopicExpansion)
        request = json.loads(call["input"])
        self.assertEqual(request["topic_query"], sample_brief().topic_query)
        self.assertEqual(call["prompt_cache_options"], {"mode": "explicit"})

    def test_expansion_forbids_unknown_fields(self) -> None:
        payload = sample_expansion().model_dump()
        payload["surprise"] = "not allowed"
        with self.assertRaises(ValidationError):
            TopicExpansion.model_validate(payload)

    def test_responses_schema_forbids_extra_and_requires_every_field(self) -> None:
        schema = TopicExpansion.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))


class InterviewTests(unittest.TestCase):
    def test_question_order_is_fixed(self) -> None:
        self.assertEqual(
            [question.question_id for question in INTERVIEW_QUESTIONS],
            [
                "research_goal",
                "language",
                "start_date",
                "positive_examples",
                "negative_examples",
            ],
        )

    def test_suggestions_use_responses_parse(self) -> None:
        suggestions = sample_suggestions()
        client = FakeClient(suggestions)

        result = InterviewPlanner(client).suggest_examples("topic", "English")

        self.assertEqual(result, suggestions)
        (call,) = client.responses.calls
        self.assertEqual(call["model"], "gpt-5.6-luna")
        self.assertIs(call["text_format"], InterviewSuggestions)
        self.assertEqual(call["prompt_cache_options"], {"mode": "explicit"})
        self.assertNotIn("prompt_cache_key", call)
        self.assertNotIn("prompt_cache_breakpoint", call["instructions"])
        self.assertNotIn("prompt_cache_breakpoint", call["input"])

    def test_interview_preserves_verbatim_user_examples(self) -> None:
        suggestions = sample_suggestions()
        answers: dict[str, str | tuple[str, ...]] = {
            "research_goal": "  Keep my spacing exactly.  ",
            "language": "de",
            "start_date": "2024-01-01",
            "positive_examples": ("yes: café", "EXACT second"),
            "negative_examples": ("no: adjacent",),
        }
        seen: list[str] = []

        def ask(question: Any, _visible: Any) -> str | tuple[str, ...]:
            seen.append(question.question_id)
            return answers[question.question_id]

        brief = conduct_interview(
            ask,
            topic_query="topic",
            language="English",
            suggestions=suggestions,
        )

        self.assertEqual(seen, [q.question_id for q in INTERVIEW_QUESTIONS])
        self.assertEqual(brief.research_goal, "  Keep my spacing exactly.  ")
        self.assertEqual(brief.language, "de")
        self.assertEqual(brief.positive_examples, ("yes: café", "EXACT second"))
        self.assertEqual(brief.negative_examples, ("no: adjacent",))
        self.assertEqual(brief.must_include, ())
        self.assertEqual(brief.edge_case_guidance, ())
        self.assertNotIn(
            "start_date", {answer.question_id for answer in brief.verbatim_answers}
        )

    def test_build_topic_brief_requires_every_fixed_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing interview answer"):
            build_topic_brief(
                topic_query="topic",
                language="English",
                answers={"research_goal": "goal"},
            )

    def test_query_fields_explicitly_require_selected_language(self) -> None:
        schema = TopicExpansion.model_json_schema()["properties"]
        self.assertIn("requested language", schema["search_queries"]["description"])
        self.assertIn(
            "requested language",
            schema["channel_discovery_queries"]["description"],
        )


class ClassifierTests(unittest.TestCase):
    def test_classifier_passes_candidate_and_returns_decision(self) -> None:
        decision = RelevanceDecision(
            decision="relevant",
            language_match="match",
            detected_language="de",
            primary_reason="topic_match",
        )
        client = FakeClient(decision)
        candidate = VideoCandidate(
            video_id="abc123",
            title="Wärmepumpe im Mehrfamilienhaus",
            description="Interview nach dem Einbau",
            channel_id="channel-1",
            channel_title="Energie Praxis",
            published_at=datetime(2026, 1, 2, tzinfo=UTC),
            duration_text="12:34",
            transcript_excerpt="Wir berichten heute über den Einbau ...",
            discovery_source="search",
            discovery_reference="Wärmepumpe Mehrfamilienhaus Interview",
        )

        prompt = compile_classifier_prompt(sample_brief(), sample_expansion())
        result = RelevanceClassifier(client).classify(
            candidate,
            prompt,
            stage="transcript",
        )

        self.assertEqual(result, decision)
        (call,) = client.responses.calls
        self.assertEqual(call["model"], DEFAULT_LLM_MODEL)
        self.assertIs(call["text_format"], RelevanceDecision)
        self.assertNotIn("instructions", call)
        self.assertEqual(call["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual(call["prompt_cache_key"], prompt.prompt_sha256)
        self.assertEqual(call["input"][0]["role"], "developer")
        stable = call["input"][0]["content"][0]
        self.assertEqual(stable["text"], prompt.system_prompt)
        self.assertEqual(
            stable["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )
        self.assertEqual(call["input"][1]["role"], "user")
        request = json.loads(call["input"][1]["content"][0]["text"])
        self.assertEqual(request["classification_stage"], "transcript")
        self.assertEqual(request["candidate"]["video_id"], "abc123")
        self.assertNotIn("published_at", request["candidate"])
        markers = sum(
            "prompt_cache_breakpoint" in block
            for message in call["input"]
            for block in message["content"]
        )
        self.assertEqual(markers, 1)

    def test_classifier_exposes_bounded_ordered_batch_hook(self) -> None:
        decision = RelevanceDecision(
            decision="irrelevant",
            language_match="mismatch",
            detected_language="en",
            primary_reason="wrong_language",
        )
        client = FakeClient(decision)
        jobs = tuple(
            ClassificationJob(
                candidate=VideoCandidate(
                    video_id=f"video-{index}",
                    title=f"Video {index}",
                    discovery_source="search",
                    discovery_reference="query",
                ),
                stage="metadata",
            )
            for index in range(3)
        )

        results = RelevanceClassifier(client).classify_many_with_usage(
            jobs,
            compile_classifier_prompt(sample_brief(), sample_expansion()),
            max_workers=2,
        )

        self.assertEqual(tuple(result.output for result in results), (decision,) * 3)
        video_ids = tuple(
            json.loads(call["input"][1]["content"][0]["text"])["candidate"]["video_id"]
            for call in client.responses.calls
        )
        self.assertEqual(set(video_ids), {"video-0", "video-1", "video-2"})

    def test_classifier_rejects_inconsistent_relevant_decision(self) -> None:
        inconsistent = {
            "decision": "relevant",
            "language_match": "unknown",
            "detected_language": None,
            "primary_reason": "topic_match",
        }
        client = FakeClient(inconsistent)
        candidate = VideoCandidate(
            video_id="abc123",
            title="Unknown language",
            discovery_source="related",
            discovery_reference="seed-video",
        )

        with self.assertRaisesRegex(ValueError, "language_match"):
            RelevanceClassifier(client).classify(
                candidate,
                compile_classifier_prompt(sample_brief(), sample_expansion()),
                stage="transcript",
            )

    def test_metadata_stage_returns_binary_provisional_relevant_decision(self) -> None:
        provisional = RelevanceDecision(
            decision="relevant",
            language_match="match",
            detected_language="de",
            primary_reason="topic_match",
        )
        client = FakeClient(provisional)
        candidate = VideoCandidate(
            video_id="maybe-1",
            title="Potentially relevant",
            discovery_source="channel",
            discovery_reference="channel-1",
        )

        result = RelevanceClassifier(client).classify_with_usage(
            candidate,
            compile_classifier_prompt(sample_brief(), sample_expansion()),
            stage="metadata",
        )

        self.assertEqual(result.output, provisional)
        self.assertEqual(result.usage.total_tokens, 0)

    def test_schema_rejects_non_binary_decision_and_omits_retired_score(self) -> None:
        non_binary = {
            "decision": "needs_transcript",
            "language_match": "unknown",
            "detected_language": None,
            "primary_reason": "insufficient_evidence",
        }
        schema = RelevanceDecision.model_json_schema()
        retired_score_field = "topicality" + "_score"

        with self.assertRaises(ValidationError):
            RelevanceDecision.model_validate(non_binary)
        self.assertNotIn(retired_score_field, schema["properties"])
        self.assertNotIn("confidence", schema["properties"])
        self.assertNotIn("matched_criteria", schema["properties"])
        self.assertNotIn("evidence", schema["properties"])
        self.assertNotIn("needs_transcript", str(schema))
        self.assertEqual(
            set(schema["properties"]),
            {"decision", "language_match", "detected_language", "primary_reason"},
        )

    def test_detected_language_requires_a_language_code(self) -> None:
        with self.assertRaises(ValidationError):
            RelevanceDecision(
                decision="irrelevant",
                language_match="mismatch",
                detected_language="English",
                primary_reason="wrong_language",
            )

    def test_legacy_pending_decision_is_projected_to_compact_schema(self) -> None:
        decision = compact_relevance_decision(
            {
                "decision": "relevant",
                "language_match": "match",
                "detected_language": "English",
                "primary_reason": "topic_match",
                "confidence": 0.98,
                "matched_criteria": ["criterion-1"],
                "evidence": ["title"],
            },
            requested_language="en",
        )

        self.assertEqual(
            decision.model_dump(),
            {
                "decision": "relevant",
                "language_match": "match",
                "detected_language": "en",
                "primary_reason": "topic_match",
            },
        )

    def test_unknown_language_discards_contradictory_detected_language(self) -> None:
        client = FakeClient(
            {
                "decision": "irrelevant",
                "language_match": "unknown",
                "detected_language": "de",
                "primary_reason": "insufficient_evidence",
            }
        )
        candidate = VideoCandidate(
            video_id="abc123",
            title="Unknown language",
            discovery_source="related",
            discovery_reference="seed-video",
        )

        decision = RelevanceClassifier(client).classify(
            candidate,
            compile_classifier_prompt(sample_brief(), sample_expansion()),
            stage="metadata",
        )

        self.assertEqual(decision.language_match, "unknown")
        self.assertIsNone(decision.detected_language)

    def test_legacy_unknown_language_discards_contradictory_detected_language(
        self,
    ) -> None:
        decision = compact_relevance_decision(
            {
                "decision": "irrelevant",
                "language_match": "unknown",
                "detected_language": "de",
                "primary_reason": "insufficient_evidence",
            },
            requested_language="de",
        )

        self.assertIsNone(decision.detected_language)

    def test_prompt_contains_no_non_binary_decision_or_retired_score(self) -> None:
        prompt = compile_classifier_prompt(sample_brief(), sample_expansion())
        retired_score_field = "topicality" + "_score"

        self.assertNotIn("needs_transcript", prompt.system_prompt)
        self.assertNotIn(retired_score_field, prompt.system_prompt)
        self.assertIn(
            "only decisions are relevant and irrelevant", prompt.system_prompt
        )


if __name__ == "__main__":
    unittest.main()
