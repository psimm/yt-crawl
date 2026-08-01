"""Small, testable terminal UI for confirming a research setup."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from InquirerPy import inquirer
from InquirerPy.base import Choice
from InquirerPy.prompts.checkbox import CheckboxPrompt
from pydantic import Field, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yt_searchapi.interview import InterviewSuggestions, SuggestedExample
from yt_searchapi.prompts import StrictModel, TopicBrief

InterviewSection = Literal[
    "research_goal",
    "language",
    "start_date",
    "positive_examples",
    "negative_examples",
]
ItemSource = Literal["direct", "suggestion", "edited_suggestion", "custom"]
SuggestionLoader = Callable[[str], InterviewSuggestions]
INTERVIEW_CANCELLED_MESSAGE = (
    "interview cancelled; no partial interview answers were saved and no crawl "
    "checkpoint was created. Start again with a different new --project path "
    "because the attempted path now contains audit files."
)
_SECTION_LABELS: dict[InterviewSection, str] = {
    "research_goal": "Research goal",
    "language": "Video language",
    "start_date": "Earliest publication date",
    "positive_examples": "Definitely in scope",
    "negative_examples": "Definitely out of scope",
}


class InterviewCancelled(Exception):
    """Raised when the user explicitly cancels the whole interview."""


class InterviewItem(StrictModel):
    """One confirmed value plus provenance for the saved audit record."""

    text: str = Field(min_length=1)
    source: ItemSource
    suggested_text: str | None = None
    boundary_tested: str | None = None
    suggestion_position: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_suggestion_provenance(self) -> InterviewItem:
        suggestion_fields = (
            self.suggested_text,
            self.boundary_tested,
            self.suggestion_position,
        )
        if self.source in {"suggestion", "edited_suggestion"}:
            if any(value is None for value in suggestion_fields):
                raise ValueError("suggestion items require complete provenance")
        elif any(value is not None for value in suggestion_fields):
            raise ValueError("direct/custom items cannot claim suggestion provenance")
        return self


class ConfirmedInterview(StrictModel):
    """The five confirmed setup answers used by the crawl."""

    research_goal: InterviewItem
    language: InterviewItem
    start_date: InterviewItem
    positive_examples: tuple[InterviewItem, ...] = ()
    negative_examples: tuple[InterviewItem, ...] = ()

    @model_validator(mode="after")
    def _validate_language_and_date(self) -> ConfirmedInterview:
        if not _is_language_code(self.language.text):
            raise ValueError("language must be a code such as en or de")
        if not _is_iso_date(self.start_date.text):
            raise ValueError("start_date must use ISO format YYYY-MM-DD")
        return self

    @property
    def language_code(self) -> str:
        return self.language.text.strip()

    @property
    def publication_start_date(self) -> date:
        return date.fromisoformat(self.start_date.text.strip())

    def values(self) -> dict[InterviewSection, str | tuple[str, ...]]:
        return {
            "research_goal": self.research_goal.text,
            "language": self.language_code,
            "start_date": self.publication_start_date.isoformat(),
            "positive_examples": tuple(item.text for item in self.positive_examples),
            "negative_examples": tuple(item.text for item in self.negative_examples),
        }

    def to_topic_brief(self, *, topic_query: str) -> TopicBrief:
        values = self.values()
        return TopicBrief.model_validate(
            {
                "topic_query": topic_query,
                "language": self.language_code,
                "research_goal": values["research_goal"],
                "positive_examples": values["positive_examples"],
                "negative_examples": values["negative_examples"],
                # Kept empty for compatibility with existing prompt/checkpoint schemas.
                "must_include": (),
                "edge_case_guidance": (),
                "verbatim_answers": tuple(
                    {
                        "question_id": section,
                        "answer": _json_answer(value),
                    }
                    for section, value in values.items()
                ),
            }
        )

    def items_for(self, section: InterviewSection) -> tuple[InterviewItem, ...]:
        value = getattr(self, section)
        return (value,) if isinstance(value, InterviewItem) else value


@dataclass(frozen=True, slots=True)
class PromptChoice:
    value: str
    label: str
    checked: bool = False


@dataclass(frozen=True, slots=True)
class CheckboxAnswer:
    """Selected values plus text edited directly on the checkbox screen."""

    selected: tuple[str, ...]
    edits: tuple[tuple[str, str], ...] = ()

    def edited_text(self, value: str, fallback: str) -> str:
        return dict(self.edits).get(value, fallback)


@dataclass(frozen=True, slots=True)
class _EditRequest:
    value: str
    selected: tuple[str, ...]


class _EditableCheckboxPrompt(CheckboxPrompt):
    """Checkbox prompt that reports an edit request for its highlighted row."""

    def __init__(self, **kwargs) -> None:
        keybindings = dict(kwargs.pop("keybindings", {}) or {})
        keybindings["edit-highlighted"] = [{"key": "e"}]
        super().__init__(keybindings=keybindings, **kwargs)
        self.kb_func_lookup["edit-highlighted"] = [
            {"func": self._handle_edit_highlighted}
        ]

    def _handle_edit_highlighted(self, event) -> None:
        selection = self.content_control.selection
        selected = tuple(
            str(choice["value"])
            for choice in self.content_control.choices
            if choice["enabled"]
        )
        event.app.exit(
            result=_EditRequest(value=str(selection["value"]), selected=selected)
        )


class InterviewPrompts(Protocol):
    """Minimal prompt surface that unit tests can replace without a terminal."""

    def checkbox(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> CheckboxAnswer | tuple[str, ...]: ...

    def select(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> str: ...

    def text(
        self,
        message: str,
        *,
        default: str = "",
        required: bool = True,
        instruction: str,
        validate: Callable[[str], bool] | None = None,
        invalid_message: str | None = None,
    ) -> str: ...


class InquirerPrompts:
    """InquirerPy adapter with explicit keys and Ctrl-C cancellation."""

    def checkbox(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> CheckboxAnswer:
        labels = {choice.value: choice.label for choice in choices}
        selected = {choice.value for choice in choices if choice.checked}
        edits: dict[str, str] = {}
        cursor_value: str | None = None
        while True:
            prompt = _EditableCheckboxPrompt(
                message=message,
                choices=[
                    Choice(
                        choice.value,
                        name=edits.get(choice.value, choice.label),
                        enabled=choice.value in selected,
                    )
                    for choice in choices
                ],
                default=cursor_value,
                instruction="(Space: toggle, E: edit, Enter: continue)",
                long_instruction=f"{instruction}  Ctrl-C cancels the interview.",
                cycle=False,
                raise_keyboard_interrupt=True,
                transformer=lambda values: f"{len(values)} selected",
            )
            result = prompt.execute()
            if not isinstance(result, _EditRequest):
                return CheckboxAnswer(
                    selected=tuple(str(value) for value in result),
                    edits=tuple(edits.items()),
                )
            selected = set(result.selected)
            cursor_value = result.value
            edited = self.text(
                f"Edit highlighted example — {labels[result.value]}",
                default=edits.get(result.value, labels[result.value]),
                required=True,
                instruction="Enter replacement text.",
            )
            edits[result.value] = edited

    def select(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> str:
        return str(
            inquirer.select(
                message=message,
                choices=[Choice(choice.value, name=choice.label) for choice in choices],
                instruction="(Up/Down: move, Enter: choose)",
                long_instruction=f"{instruction}  Ctrl-C cancels the interview.",
                cycle=False,
                raise_keyboard_interrupt=True,
            ).execute()
        )

    def text(
        self,
        message: str,
        *,
        default: str = "",
        required: bool = True,
        instruction: str,
        validate: Callable[[str], bool] | None = None,
        invalid_message: str | None = None,
    ) -> str:
        validator = validate
        if validator is None and required:

            def required_text(value: str) -> bool:
                return bool(value.strip())

            validator = required_text
        return str(
            inquirer.text(
                message=message,
                default=default,
                instruction="(Enter: save)",
                long_instruction=f"{instruction}  Ctrl-C cancels the interview.",
                validate=validator,
                invalid_message=invalid_message
                or "Enter some text, or press Ctrl-C to cancel.",
                raise_keyboard_interrupt=True,
            ).execute()
        )


@dataclass(frozen=True, slots=True)
class _IndexedSuggestion:
    position: int
    value: SuggestedExample


class TerminalInterview:
    """Collect, review, and confirm the research-boundary answers."""

    def __init__(
        self,
        suggestions: InterviewSuggestions | None = None,
        *,
        suggestion_loader: SuggestionLoader | None = None,
        language_prefill: str = "",
        start_date_prefill: str = "",
        ask_language: bool = True,
        ask_start_date: bool = True,
        prompts: InterviewPrompts | None = None,
        console: Console | None = None,
    ) -> None:
        if suggestions is None and suggestion_loader is None:
            raise ValueError("suggestions or suggestion_loader is required")
        self.suggestions = suggestions
        self.suggestion_loader = suggestion_loader
        self.language_prefill = language_prefill
        self.start_date_prefill = start_date_prefill
        self.ask_language = ask_language
        self.ask_start_date = ask_start_date
        if not ask_language and not _is_language_code(language_prefill):
            raise ValueError("a valid language_prefill is required when not asking")
        if not ask_start_date and not _is_iso_date(start_date_prefill):
            raise ValueError("a valid start_date_prefill is required when not asking")
        sections: list[InterviewSection] = ["research_goal"]
        if ask_language:
            sections.append("language")
        if ask_start_date:
            sections.append("start_date")
        sections.extend(("positive_examples", "negative_examples"))
        self._question_positions = {
            section: position for position, section in enumerate(sections, start=1)
        }
        self._question_total = len(sections)
        self.prompts = prompts or InquirerPrompts()
        self.console = console or Console()

    def run(self) -> ConfirmedInterview:
        self.console.print(
            Panel.fit(
                f"{self._question_total} questions, then a review. Ctrl-C cancels.",
                title="Set up research",
            )
        )
        goal = self._collect_goal()
        language = (
            self._collect_language(self.language_prefill)
            if self.ask_language
            else InterviewItem(text=self.language_prefill.strip(), source="direct")
        )
        start_date = (
            self._collect_start_date(self.start_date_prefill)
            if self.ask_start_date
            else InterviewItem(text=self.start_date_prefill.strip(), source="direct")
        )
        self._load_suggestions(language.text)
        draft: dict[InterviewSection, InterviewItem | tuple[InterviewItem, ...]] = {
            "research_goal": goal,
            "language": language,
            "start_date": start_date,
            "positive_examples": self._collect_examples("positive", ()),
            "negative_examples": self._collect_examples("negative", ()),
        }

        while True:
            confirmed = ConfirmedInterview.model_validate(draft)
            self._render_review(confirmed)
            action = self.prompts.select(
                "Review · Ready to start?",
                (
                    PromptChoice("confirm", "Confirm and start crawl"),
                    PromptChoice("research_goal", "Edit research goal"),
                    PromptChoice("language", "Edit video language"),
                    PromptChoice("start_date", "Edit earliest publication date"),
                    PromptChoice("positive_examples", "Edit in-scope examples"),
                    PromptChoice("negative_examples", "Edit out-of-scope examples"),
                    PromptChoice("cancel", "Cancel"),
                ),
                instruction="Review or change an answer.",
            )
            if action == "confirm":
                self.console.print(
                    "[bold green]Interview complete.[/bold green] "
                    "Preparing search queries…"
                )
                return confirmed
            if action == "cancel":
                raise InterviewCancelled(INTERVIEW_CANCELLED_MESSAGE)
            section = action
            if section == "research_goal":
                current = _one_item(draft[section])
                draft[section] = self._collect_goal(current.text, revision=True)
            elif section == "language":
                current = _one_item(draft[section])
                revised = self._collect_language(current.text, revision=True)
                draft[section] = revised
                if revised.text.strip() != current.text.strip():
                    self._load_suggestions(revised.text)
                    for example_section in (
                        "positive_examples",
                        "negative_examples",
                    ):
                        current_examples = draft[example_section]
                        assert isinstance(current_examples, tuple)
                        draft[example_section] = tuple(
                            InterviewItem(text=item.text, source="custom")
                            for item in current_examples
                        )
                    self.console.print(
                        "[dim]Suggestions refreshed for the new language.[/dim]"
                    )
            elif section == "start_date":
                current = _one_item(draft[section])
                draft[section] = self._collect_start_date(current.text, revision=True)
            elif section in {"positive_examples", "negative_examples"}:
                items = draft[section]
                assert isinstance(items, tuple)
                draft[section] = self._collect_examples(
                    "positive" if section == "positive_examples" else "negative",
                    items,
                    revision=True,
                )
            else:  # pragma: no cover - adapters return one of the supplied values
                raise ValueError(f"unknown interview action {action!r}")

    def _load_suggestions(self, language: str) -> None:
        if self.suggestion_loader is not None:
            self.suggestions = self.suggestion_loader(language.strip())
        assert self.suggestions is not None

    def _collect_goal(
        self, current: str = "", *, revision: bool = False
    ) -> InterviewItem:
        value = self.prompts.text(
            self._message(
                "research_goal",
                "What should this research help you understand or decide?",
                revision=revision,
            ),
            default=current,
            required=True,
            instruction="One or two sentences.",
        )
        return InterviewItem(text=value, source="direct")

    def _collect_language(
        self, current: str = "", *, revision: bool = False
    ) -> InterviewItem:
        value = self.prompts.text(
            self._message(
                "language",
                "Which language should the videos be in?",
                revision=revision,
            ),
            default=current,
            required=True,
            instruction="Use a language code such as en or de.",
            validate=_is_language_code,
            invalid_message="Enter a language code such as en or de.",
        )
        return InterviewItem(text=value.strip(), source="direct")

    def _collect_start_date(
        self, current: str = "", *, revision: bool = False
    ) -> InterviewItem:
        value = self.prompts.text(
            self._message(
                "start_date",
                "Include videos published on or after which date?",
                revision=revision,
            ),
            default=current,
            required=True,
            instruction="Use YYYY-MM-DD.",
            validate=_is_iso_date,
            invalid_message="Enter a valid date in YYYY-MM-DD format.",
        )
        return InterviewItem(text=value.strip(), source="direct")

    def _collect_examples(
        self,
        label: Literal["positive", "negative"],
        current: tuple[InterviewItem, ...],
        *,
        revision: bool = False,
    ) -> tuple[InterviewItem, ...]:
        assert self.suggestions is not None
        section: InterviewSection = (
            "positive_examples" if label == "positive" else "negative_examples"
        )
        indexed = tuple(
            _IndexedSuggestion(position, example)
            for position, example in enumerate(
                (
                    example
                    for example in self.suggestions.examples
                    if example.label == label
                ),
                start=1,
            )
        )
        current_suggestions = {
            item.suggestion_position: item
            for item in current
            if item.suggestion_position is not None
        }
        answer = _checkbox_answer(
            self.prompts.checkbox(
                self._message(
                    section,
                    (
                        "Which examples are definitely in scope?"
                        if label == "positive"
                        else "Which near-miss examples are definitely out of scope?"
                    ),
                    revision=revision,
                ),
                tuple(
                    PromptChoice(
                        str(item.position),
                        (
                            current_suggestions[item.position].text
                            if item.position in current_suggestions
                            else item.value.example
                        ),
                        checked=item.position in current_suggestions,
                    )
                    for item in indexed
                ),
                instruction="Select any that fit. Press E to edit the highlighted row.",
            )
        )
        selected = set(answer.selected)
        items: list[InterviewItem] = []
        for item in indexed:
            if str(item.position) not in selected:
                continue
            previous = current_suggestions.get(item.position)
            existing_text = previous.text if previous else item.value.example
            text = answer.edited_text(
                str(item.position),
                existing_text,
            )
            items.append(
                InterviewItem(
                    text=text,
                    source=(
                        "suggestion"
                        if text == item.value.example
                        else "edited_suggestion"
                    ),
                    suggested_text=item.value.example,
                    boundary_tested=item.value.boundary_tested,
                    suggestion_position=item.position,
                )
            )

        custom = tuple(item for item in current if item.source == "custom")
        if custom:
            custom_answer = _checkbox_answer(
                self.prompts.checkbox(
                    self._message(
                        section, "Keep or remove custom examples", revision=revision
                    ),
                    tuple(
                        PromptChoice(str(index), item.text, checked=True)
                        for index, item in enumerate(custom, start=1)
                    ),
                    instruction=(
                        "Uncheck to remove. Press E to edit the highlighted row."
                    ),
                )
            )
            retained = set(custom_answer.selected)
            for index, item in enumerate(custom, start=1):
                if str(index) not in retained:
                    continue
                text = custom_answer.edited_text(str(index), item.text)
                items.append(InterviewItem(text=text, source="custom"))

        singular = "in-scope example" if label == "positive" else "out-of-scope example"
        while self._add_or_finish(section, singular, revision=revision):
            text = self.prompts.text(
                self._message(section, f"Custom {singular}", revision=revision),
                required=True,
                instruction="Enter one example.",
            )
            items.append(InterviewItem(text=text, source="custom"))
        return tuple(items)

    def _add_or_finish(
        self,
        section: InterviewSection,
        singular: str,
        *,
        revision: bool,
    ) -> bool:
        return (
            self.prompts.select(
                self._message(
                    section, f"Add another custom {singular}?", revision=revision
                ),
                (
                    PromptChoice("done", "Done with this section"),
                    PromptChoice("add", "Add one example"),
                ),
                instruction="Choose Add or Done.",
            )
            == "add"
        )

    def _message(self, section: InterviewSection, text: str, *, revision: bool) -> str:
        if revision:
            return f"Review · Edit {_SECTION_LABELS[section]} — {text}"
        position = self._question_positions[section]
        return f"Question {position}/{self._question_total} — {text}"

    def _render_review(self, confirmed: ConfirmedInterview) -> None:
        self.console.print()
        self.console.print(
            Panel.fit("Check these answers before the crawl starts.", title="Review")
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("Question", no_wrap=True)
        table.add_column("Answer")
        table.add_column("Source", no_wrap=True)
        for section in _SECTION_LABELS:
            items = confirmed.items_for(section)
            if not items:
                table.add_row(_SECTION_LABELS[section], Text("None", style="dim"), "—")
                continue
            for index, item in enumerate(items):
                source = {
                    "direct": "Your answer",
                    "custom": "Your example",
                    "suggestion": "Suggested",
                    "edited_suggestion": "Suggested, edited",
                }[item.source]
                table.add_row(
                    _SECTION_LABELS[section] if index == 0 else "",
                    Text(item.text),
                    source,
                )
        self.console.print(table)


def _one_item(value: InterviewItem | tuple[InterviewItem, ...]) -> InterviewItem:
    assert isinstance(value, InterviewItem)
    return value


def _checkbox_answer(value: CheckboxAnswer | tuple[str, ...]) -> CheckboxAnswer:
    if isinstance(value, CheckboxAnswer):
        return value
    return CheckboxAnswer(selected=tuple(value))


def _is_language_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", value.strip()))


def _is_iso_date(value: str) -> bool:
    stripped = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped) is None:
        return False
    try:
        date.fromisoformat(stripped)
    except ValueError:
        return False
    return True


def _json_answer(value: str | tuple[str, ...]) -> str:
    items = (value,) if isinstance(value, str) else value
    return json.dumps(items, ensure_ascii=False)


__all__ = [
    "CheckboxAnswer",
    "ConfirmedInterview",
    "INTERVIEW_CANCELLED_MESSAGE",
    "InquirerPrompts",
    "InterviewCancelled",
    "InterviewItem",
    "InterviewPrompts",
    "PromptChoice",
    "TerminalInterview",
]
