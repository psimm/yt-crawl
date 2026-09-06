"""Interactive collection of start settings omitted from the command line."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias

from yt_crawl.interview_ui import InquirerPrompts, InterviewPrompts

SettingValue = str | int | float | Path
SettingParser: TypeAlias = Callable[[str], SettingValue]
DefaultFactory: TypeAlias = Callable[[Mapping[str, SettingValue]], str]


@dataclass(frozen=True, slots=True)
class StartSettingQuestion:
    """One independently testable start setting prompt."""

    key: str
    prompt: str
    instruction: str
    parser: SettingParser = str
    default: str | DefaultFactory = ""
    validate: Callable[[str], bool] | None = None
    invalid_message: str | None = None

    def default_for(self, values: Mapping[str, SettingValue]) -> str:
        return self.default(values) if callable(self.default) else self.default


def collect_missing_start_settings(
    provided: Mapping[str, SettingValue | None],
    questions: Sequence[StartSettingQuestion],
    *,
    prompts: InterviewPrompts | None = None,
) -> dict[str, SettingValue]:
    """Ask only for values represented by ``None`` and return the resolved mapping.

    CLI adapters must use ``None`` for omitted options. Falsy values such as zero are
    considered explicitly supplied and are preserved.
    """

    prompt_adapter = prompts or InquirerPrompts()
    resolved: dict[str, SettingValue] = {
        key: value for key, value in provided.items() if value is not None
    }
    missing = [question for question in questions if provided.get(question.key) is None]
    total = len(missing)
    for position, question in enumerate(missing, start=1):
        raw = prompt_adapter.text(
            f"Question {position}/{total} — {question.prompt}",
            default=question.default_for(resolved),
            required=True,
            instruction=question.instruction,
            validate=question.validate,
            invalid_message=question.invalid_message,
        )
        resolved[question.key] = question.parser(raw.strip())
    return resolved


def text_setting(
    key: str,
    prompt: str,
    instruction: str,
    *,
    default: str | DefaultFactory = "",
    parser: SettingParser = str,
    validate: Callable[[str], bool] | None = None,
    invalid_message: str | None = None,
) -> StartSettingQuestion:
    """Build a text-backed setting for a caller-specific schema."""

    return StartSettingQuestion(
        key=key,
        prompt=prompt,
        instruction=instruction,
        parser=parser,
        default=default,
        validate=validate,
        invalid_message=invalid_message,
    )


def integer_setting(
    key: str,
    prompt: str,
    instruction: str,
    *,
    minimum: int,
    maximum: int | None = None,
    default: int | None = None,
) -> StartSettingQuestion:
    """Build an integer setting with one inclusive range validator."""

    def valid(value: str) -> bool:
        try:
            parsed = int(value.strip())
        except ValueError:
            return False
        return parsed >= minimum and (maximum is None or parsed <= maximum)

    range_copy = f"at least {minimum}"
    if maximum is not None:
        range_copy = f"between {minimum} and {maximum}"
    return StartSettingQuestion(
        key=key,
        prompt=prompt,
        instruction=instruction,
        parser=int,
        default="" if default is None else str(default),
        validate=valid,
        invalid_message=f"Enter a whole number {range_copy}.",
    )


def float_setting(
    key: str,
    prompt: str,
    instruction: str,
    *,
    minimum_exclusive: float = 0,
    default: float | None = None,
) -> StartSettingQuestion:
    """Build a positive floating-point setting, suitable for request timeouts."""

    def valid(value: str) -> bool:
        try:
            return float(value.strip()) > minimum_exclusive
        except ValueError:
            return False

    return StartSettingQuestion(
        key=key,
        prompt=prompt,
        instruction=instruction,
        parser=float,
        default="" if default is None else str(default),
        validate=valid,
        invalid_message=f"Enter a number greater than {minimum_exclusive:g}.",
    )


BASE_START_SETTING_QUESTIONS: tuple[StartSettingQuestion, ...] = (
    text_setting(
        "topic",
        "What topic or research question should be crawled?",
        "Enter a concise topic or question.",
    ),
    text_setting(
        "project",
        "Where should the new project be stored?",
        "Enter a new project directory.",
        parser=lambda value: Path(value).expanduser(),
    ),
    integer_setting(
        "max_credits",
        "How many SearchAPI credits may this first run use?",
        "You can add another credit grant when resuming later.",
        minimum=4,
    ),
    text_setting(
        "language",
        "Which language should the videos be in?",
        "Use a language code such as en or de.",
        validate=lambda value: bool(
            re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", value.strip())
        ),
        invalid_message="Enter a language code such as en or de.",
    ),
    text_setting(
        "start_date",
        "Include videos published on or after which date?",
        "Use YYYY-MM-DD.",
        validate=lambda value: _is_iso_date(value),
        invalid_message="Enter a valid date in YYYY-MM-DD format.",
    ),
    integer_setting(
        "max_depth",
        "How many related-channel graph levels should be followed?",
        "0 searches only initial results; higher values follow related channels.",
        minimum=0,
        maximum=5,
        default=2,
    ),
    integer_setting(
        "max_queries",
        "How many query variants may be used?",
        "Choose between 2 and 18.",
        minimum=2,
        maximum=18,
        default=8,
    ),
    integer_setting(
        "max_search_pages",
        "How many pages may be fetched per search query?",
        "Choose between 1 and 10.",
        minimum=1,
        maximum=10,
        default=1,
    ),
    integer_setting(
        "max_channel_pages",
        "How many pages may be fetched per discovered channel?",
        "Choose at least 1.",
        minimum=1,
        default=1,
    ),
    text_setting(
        "gl",
        "Which country should SearchAPI use?",
        "Use a two-letter country code such as us or de.",
        default="us",
        validate=lambda value: bool(re.fullmatch(r"[A-Za-z]{2}", value.strip())),
        invalid_message="Enter a two-letter country code such as us or de.",
    ),
    text_setting(
        "hl",
        "Which interface language should SearchAPI request?",
        "Use a language code such as en or de.",
        default=lambda values: str(values.get("language", "")),
        validate=lambda value: bool(
            re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", value.strip())
        ),
        invalid_message="Enter a language code such as en or de.",
    ),
)


def _is_iso_date(value: str) -> bool:
    stripped = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped) is None:
        return False
    try:
        date.fromisoformat(stripped)
    except ValueError:
        return False
    return True


__all__ = [
    "BASE_START_SETTING_QUESTIONS",
    "StartSettingQuestion",
    "collect_missing_start_settings",
    "float_setting",
    "integer_setting",
    "text_setting",
]
