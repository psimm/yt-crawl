from __future__ import annotations

from collections import deque
from pathlib import Path

from yt_searchapi.interview_ui import PromptChoice
from yt_searchapi.settings import (
    DEFAULT_LLM_WORKERS,
    DEFAULT_SEARCHAPI_RETRIES,
    DEFAULT_SEARCHAPI_WORKERS,
)
from yt_searchapi.start_settings_ui import (
    BASE_START_SETTING_QUESTIONS,
    collect_missing_start_settings,
    float_setting,
    integer_setting,
)


class TextOnlyPrompts:
    def __init__(self, answers: list[str]) -> None:
        self.answers = deque(answers)
        self.calls: list[dict[str, object]] = []

    def text(
        self,
        message: str,
        *,
        default: str = "",
        required: bool = True,
        instruction: str,
        validate=None,
        invalid_message: str | None = None,
    ) -> str:
        answer = self.answers.popleft()
        assert validate is None or validate(answer)
        self.calls.append(
            {
                "message": message,
                "default": default,
                "required": required,
                "instruction": instruction,
                "invalid_message": invalid_message,
            }
        )
        return answer

    def checkbox(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> tuple[str, ...]:
        raise AssertionError("not used")

    def select(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> str:
        raise AssertionError("not used")


def test_only_omitted_start_settings_are_asked_with_dynamic_count() -> None:
    prompts = TextOnlyPrompts(["de", "2025-02-03", "de"])
    provided = {
        "topic": "personal finance",
        "project": Path("project"),
        "max_credits": 20,
        "language": None,
        "start_date": None,
        "max_depth": 0,
        "max_queries": 2,
        "max_search_pages": 1,
        "max_channel_pages": 1,
        "gl": "de",
        "hl": None,
    }

    result = collect_missing_start_settings(
        provided, BASE_START_SETTING_QUESTIONS, prompts=prompts
    )

    assert result == {
        **provided,
        "language": "de",
        "start_date": "2025-02-03",
        "hl": "de",
    }
    assert [call["message"] for call in prompts.calls] == [
        "Question 1/3 — Which language should the videos be in?",
        "Question 2/3 — Include videos published on or after which date?",
        "Question 3/3 — Which interface language should SearchAPI request?",
    ]
    assert prompts.calls[-1]["default"] == "de"


def test_explicit_falsy_values_are_not_treated_as_missing() -> None:
    prompts = TextOnlyPrompts([])
    questions = (
        integer_setting("max_depth", "Depth?", "Enter depth.", minimum=0, maximum=5),
    )

    result = collect_missing_start_settings(
        {"max_depth": 0}, questions, prompts=prompts
    )

    assert result == {"max_depth": 0}
    assert prompts.calls == []


def test_extra_timeout_retry_and_parallelism_questions_are_schema_driven() -> None:
    prompts = TextOnlyPrompts(
        ["30.5", str(DEFAULT_SEARCHAPI_RETRIES), "6", str(DEFAULT_LLM_WORKERS)]
    )
    questions = (
        float_setting(
            "searchapi_timeout",
            "SearchAPI timeout?",
            "Seconds.",
            default=90,
        ),
        integer_setting(
            "searchapi_retries",
            "SearchAPI retries?",
            "Retries.",
            minimum=0,
            maximum=5,
            default=DEFAULT_SEARCHAPI_RETRIES,
        ),
        integer_setting(
            "searchapi_workers",
            "SearchAPI concurrency?",
            "Workers.",
            minimum=1,
            maximum=32,
            default=DEFAULT_SEARCHAPI_WORKERS,
        ),
        integer_setting(
            "llm_workers",
            "LLM concurrency?",
            "Workers.",
            minimum=1,
            maximum=32,
            default=DEFAULT_LLM_WORKERS,
        ),
    )

    result = collect_missing_start_settings(
        {question.key: None for question in questions}, questions, prompts=prompts
    )

    assert result == {
        "searchapi_timeout": 30.5,
        "searchapi_retries": DEFAULT_SEARCHAPI_RETRIES,
        "searchapi_workers": 6,
        "llm_workers": DEFAULT_LLM_WORKERS,
    }
