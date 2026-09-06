from __future__ import annotations

import json
from collections import deque
from contextlib import nullcontext
from io import StringIO

import pytest
import typer
from rich.console import Console

import yt_crawl.cli as cli
import yt_crawl.interview_ui as interview_ui
from yt_crawl.cli import _record_confirmed_interview
from yt_crawl.interview import InterviewSuggestions, SuggestedExample
from yt_crawl.interview_ui import (
    INTERVIEW_CANCELLED_MESSAGE,
    CheckboxAnswer,
    ConfirmedInterview,
    InquirerPrompts,
    InterviewCancelled,
    InterviewItem,
    PromptChoice,
    TerminalInterview,
)
from yt_crawl.storage import JsonlRunWriter


class FakePrompts:
    def __init__(
        self,
        *,
        checkboxes: list[CheckboxAnswer | tuple[str, ...]],
        selects: list[str],
        texts: list[str],
    ) -> None:
        self.checkboxes = deque(checkboxes)
        self.selects = deque(selects)
        self.texts = deque(texts)
        self.calls: list[dict[str, object]] = []

    def checkbox(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> CheckboxAnswer | tuple[str, ...]:
        self.calls.append(
            {
                "kind": "checkbox",
                "message": message,
                "instruction": instruction,
                "choices": choices,
            }
        )
        return self.checkboxes.popleft()

    def select(
        self,
        message: str,
        choices: tuple[PromptChoice, ...],
        *,
        instruction: str,
    ) -> str:
        self.calls.append(
            {
                "kind": "select",
                "message": message,
                "instruction": instruction,
                "choices": choices,
            }
        )
        return self.selects.popleft()

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
        self.calls.append(
            {
                "kind": "text",
                "message": message,
                "instruction": instruction,
                "default": default,
                "required": required,
                "validate": validate,
                "invalid_message": invalid_message,
            }
        )
        return self.texts.popleft()

    def assert_consumed(self) -> None:
        assert not self.checkboxes
        assert not self.selects
        assert not self.texts


@pytest.fixture
def suggestions() -> InterviewSuggestions:
    return InterviewSuggestions(
        examples=tuple(
            SuggestedExample(
                label="positive",
                example=f"Positive {index}",
                boundary_tested=f"positive-boundary-{index}",
            )
            for index in range(1, 11)
        )
        + tuple(
            SuggestedExample(
                label="negative",
                example=f"Negative {index}",
                boundary_tested=f"negative-boundary-{index}",
            )
            for index in range(1, 11)
        )
    )


def test_five_question_flow_selection_edit_custom_and_confirmation(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[
            CheckboxAnswer(selected=("1", "2"), edits=(("2", "Edited positive 2"),)),
            (),
        ],
        selects=["add", "done", "add", "done", "confirm"],
        texts=[
            "  Goal exactly as entered.  ",
            "de",
            "2025-01-02",
            " custom positive ",
            "custom negative",
        ],
    )
    console = Console(record=True, width=140)

    result = TerminalInterview(suggestions, prompts=prompts, console=console).run()

    prompts.assert_consumed()
    assert result.research_goal.text == "  Goal exactly as entered.  "
    assert result.language_code == "de"
    assert result.publication_start_date.isoformat() == "2025-01-02"
    assert [item.text for item in result.positive_examples] == [
        "Positive 1",
        "Edited positive 2",
        " custom positive ",
    ]
    assert [item.source for item in result.positive_examples] == [
        "suggestion",
        "edited_suggestion",
        "custom",
    ]
    assert result.negative_examples[0].source == "custom"
    output = console.export_text()
    assert "5 questions, then a review" in output
    assert "Review" in output
    assert "Interview complete" in output

    brief = result.to_topic_brief(topic_query="topic")
    assert brief.language == "de"
    assert brief.must_include == ()
    assert brief.edge_case_guidance == ()
    decoded = {
        answer.question_id: json.loads(answer.answer)
        for answer in brief.verbatim_answers
    }
    assert "start_date" not in decoded


def test_inquirer_checkbox_e_edit_reopens_same_choice_list(monkeypatch) -> None:
    created: list[dict[str, object]] = []
    results = deque(
        [
            interview_ui._EditRequest(value="2", selected=("1",)),
            ["1", "2"],
        ]
    )

    class StubEditableCheckbox:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        def execute(self):
            return results.popleft()

    monkeypatch.setattr(interview_ui, "_EditableCheckboxPrompt", StubEditableCheckbox)
    prompts = InquirerPrompts()
    edit_defaults = []

    def edit_text(*_args, **kwargs):
        edit_defaults.append(kwargs["default"])
        return "Edited two"

    monkeypatch.setattr(prompts, "text", edit_text)

    answer = prompts.checkbox(
        "Examples",
        (
            PromptChoice("1", "One", checked=True),
            PromptChoice("2", "Previously edited"),
        ),
        instruction="Choose.",
    )

    assert answer == CheckboxAnswer(selected=("1", "2"), edits=(("2", "Edited two"),))
    assert len(created) == 2
    second_choices = created[1]["choices"]
    assert second_choices[0].enabled is True
    assert second_choices[1].name == "Edited two"
    assert created[1]["default"] == "2"
    assert edit_defaults == ["Previously edited"]


def test_every_initial_and_nested_prompt_shows_parent_progress_without_boundary_copy(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[("1",), ()],
        selects=["add", "done", "done", "confirm"],
        texts=["Goal", "en", "2026-01-01", "Custom"],
    )

    TerminalInterview(suggestions, prompts=prompts, console=Console()).run()

    setup_calls = [
        call
        for call in prompts.calls
        if not str(call["message"]).startswith("Review ·")
    ]
    assert all("Question " in str(call["message"]) for call in setup_calls)
    assert any("Question 1/5 —" in str(c["message"]) for c in setup_calls)
    assert any("Question 5/5 —" in str(c["message"]) for c in setup_calls)
    assert all(
        "remaining" not in str(call["message"]).casefold() for call in setup_calls
    )
    example_calls = [c for c in prompts.calls if c["kind"] == "checkbox"]
    assert len(example_calls[0]["choices"]) == 10
    assert len(example_calls[1]["choices"]) == 10
    visible_copy = " ".join(
        choice.label for call in example_calls for choice in call["choices"]
    )
    assert "boundary" not in visible_copy.casefold()


def test_initial_goal_has_no_default_claim_and_prefills_are_editable(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[(), ()],
        selects=["done", "done", "confirm"],
        texts=["Goal", "fr", "2024-03-04"],
    )

    TerminalInterview(
        suggestions,
        language_prefill="de",
        start_date_prefill="2024-01-01",
        prompts=prompts,
        console=Console(),
    ).run()

    text_calls = [call for call in prompts.calls if call["kind"] == "text"]
    assert text_calls[0]["default"] == ""
    assert text_calls[1]["default"] == "de"
    assert text_calls[2]["default"] == "2024-01-01"
    all_copy = " ".join(
        f"{call['message']} {call['instruction']}" for call in prompts.calls
    )
    assert "displayed default" not in all_copy.casefold()


def test_review_revision_uses_review_label_not_numbered_question(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[
            (),
            (),
            CheckboxAnswer(selected=("1",), edits=(("1", "Rewritten positive"),)),
        ],
        selects=["done", "done", "positive_examples", "done", "confirm"],
        texts=["Goal", "en", "2026-01-01"],
    )

    result = TerminalInterview(suggestions, prompts=prompts, console=Console()).run()

    assert result.positive_examples[0].source == "edited_suggestion"
    revision_calls = [
        call
        for call in prompts.calls
        if "Edit Definitely in scope" in str(call["message"])
    ]
    assert revision_calls
    assert all(str(call["message"]).startswith("Review ·") for call in revision_calls)


def test_reopened_suggestion_uses_current_edit_and_preserves_provenance(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[
            CheckboxAnswer(selected=("1",), edits=(("1", "First edit"),)),
            (),
            CheckboxAnswer(selected=("1",)),
            CheckboxAnswer(selected=("1",), edits=(("1", "Second edit"),)),
        ],
        selects=[
            "done",
            "done",
            "positive_examples",
            "done",
            "positive_examples",
            "done",
            "confirm",
        ],
        texts=["Goal", "en", "2026-01-01"],
    )

    result = TerminalInterview(suggestions, prompts=prompts, console=Console()).run()

    revision_checkboxes = [
        call
        for call in prompts.calls
        if call["kind"] == "checkbox" and str(call["message"]).startswith("Review ·")
    ]
    assert revision_checkboxes[0]["choices"][0] == PromptChoice(
        "1", "First edit", checked=True
    )
    assert revision_checkboxes[1]["choices"][0] == PromptChoice(
        "1", "First edit", checked=True
    )
    assert result.positive_examples == (
        InterviewItem(
            text="Second edit",
            source="edited_suggestion",
            suggested_text="Positive 1",
            boundary_tested="positive-boundary-1",
            suggestion_position=1,
        ),
    )


def test_prefilled_language_and_start_date_can_be_skipped(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[(), ()],
        selects=["done", "done", "confirm"],
        texts=["Goal"],
    )
    console = Console(record=True)

    result = TerminalInterview(
        suggestions,
        language_prefill="de",
        start_date_prefill="2025-01-02",
        ask_language=False,
        ask_start_date=False,
        prompts=prompts,
        console=console,
    ).run()

    prompts.assert_consumed()
    assert result.language_code == "de"
    assert result.publication_start_date.isoformat() == "2025-01-02"
    setup_messages = [
        str(call["message"])
        for call in prompts.calls
        if not str(call["message"]).startswith("Review ·")
    ]
    assert any("Question 1/3" in message for message in setup_messages)
    assert any("Question 3/3" in message for message in setup_messages)
    assert all("language" not in message.casefold() for message in setup_messages)
    assert "3 questions, then a review" in console.export_text()


def test_skipped_prefills_must_be_valid(suggestions: InterviewSuggestions) -> None:
    with pytest.raises(ValueError, match="language_prefill"):
        TerminalInterview(
            suggestions,
            language_prefill="",
            start_date_prefill="2025-01-02",
            ask_language=False,
        )

    with pytest.raises(ValueError, match="start_date_prefill"):
        TerminalInterview(
            suggestions,
            language_prefill="de",
            start_date_prefill="not-a-date",
            ask_start_date=False,
        )


def test_language_revision_refreshes_suggestions_and_keeps_examples_as_custom(
    suggestions: InterviewSuggestions,
) -> None:
    refreshed = InterviewSuggestions(
        examples=tuple(
            SuggestedExample(
                label="positive",
                example=f"Positif {index}",
                boundary_tested=f"p-{index}",
            )
            for index in range(1, 11)
        )
        + tuple(
            SuggestedExample(
                label="negative",
                example=f"Negatif {index}",
                boundary_tested=f"n-{index}",
            )
            for index in range(1, 11)
        )
    )
    loaded_languages: list[str] = []

    def load(language: str) -> InterviewSuggestions:
        loaded_languages.append(language)
        return suggestions if language == "en" else refreshed

    prompts = FakePrompts(
        checkboxes=[("1",), ()],
        selects=["done", "done", "language", "confirm"],
        texts=["Goal", "en", "2026-01-01", "fr"],
    )

    result = TerminalInterview(
        suggestion_loader=load, prompts=prompts, console=Console()
    ).run()

    assert loaded_languages == ["en", "fr"]
    assert result.language_code == "fr"
    assert result.positive_examples == (
        InterviewItem(text="Positive 1", source="custom"),
    )


def test_explicit_cancel_never_returns_partial_answers(
    suggestions: InterviewSuggestions,
) -> None:
    prompts = FakePrompts(
        checkboxes=[(), ()],
        selects=["done", "done", "cancel"],
        texts=["Partial goal", "en", "2026-01-01"],
    )

    with pytest.raises(InterviewCancelled, match="no partial interview answers"):
        TerminalInterview(suggestions, prompts=prompts, console=Console()).run()

    prompts.assert_consumed()


@pytest.mark.parametrize(
    "interruption",
    [InterviewCancelled(INTERVIEW_CANCELLED_MESSAGE), KeyboardInterrupt()],
    ids=["cancel", "ctrl-c"],
)
def test_interrupted_preparation_writes_no_interview_answer_records(
    monkeypatch, tmp_path, interruption
) -> None:
    class CancelledInterview:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self):
            raise interruption

    class FakeClient:
        def call_context(self, *_args, **_kwargs):
            return nullcontext()

    monkeypatch.setattr(cli, "TerminalInterview", CancelledInterview)
    terminal_output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=terminal_output, color_system=None, force_terminal=False),
    )
    writer = JsonlRunWriter(tmp_path, "project")

    with pytest.raises(typer.Abort):
        cli._prepare_research(
            writer=writer,
            client=FakeClient(),
            topic="topic",
            language_hint="en",
            start_date_hint="2026-01-01",
            max_credits=4,
            transcript_reserve=1,
            account_remaining_credits=10,
            max_depth=1,
            max_queries=2,
            max_search_pages=1,
            max_channel_pages=1,
        )

    assert not (writer.run_dir / "interview_answer.jsonl").exists()
    assert not (writer.run_dir / "crawl_state.json").exists()
    errors = [
        json.loads(line)
        for line in (writer.run_dir / "run_error.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    statuses = [
        json.loads(line)
        for line in (writer.run_dir / "run_status.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert errors[-1]["message"] == INTERVIEW_CANCELLED_MESSAGE
    assert statuses[-1]["reason"] == INTERVIEW_CANCELLED_MESSAGE
    terminal_text = " ".join(terminal_output.getvalue().split())
    assert INTERVIEW_CANCELLED_MESSAGE in terminal_text
    assert "no crawl checkpoint was created" in terminal_text
    assert "different new --project path" in terminal_text


def test_confirmed_answers_are_persisted_as_five_structured_rows(
    tmp_path, suggestions: InterviewSuggestions
) -> None:
    confirmed = ConfirmedInterview(
        research_goal=InterviewItem(text=" exact goal ", source="direct"),
        language=InterviewItem(text="de", source="direct"),
        start_date=InterviewItem(text="2025-02-03", source="direct"),
        positive_examples=(
            InterviewItem(
                text="Edited text",
                source="edited_suggestion",
                suggested_text="Positive 1",
                boundary_tested="positive-boundary-1",
                suggestion_position=1,
            ),
            InterviewItem(text="custom || remains literal", source="custom"),
        ),
        negative_examples=(),
    )
    writer = JsonlRunWriter(tmp_path, "project")

    _record_confirmed_interview(writer, confirmed, suggestions)

    rows = [
        json.loads(line)
        for line in (writer.run_dir / "interview_answer.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["question_id"] for row in rows] == [
        "research_goal",
        "language",
        "start_date",
        "positive_examples",
        "negative_examples",
    ]
    by_question = {row["question_id"]: row for row in rows}
    assert json.loads(by_question["language"]["answer"]) == ["de"]
    assert json.loads(by_question["start_date"]["answer"]) == ["2025-02-03"]
    positive = by_question["positive_examples"]
    assert json.loads(positive["answer"]) == [
        "Edited text",
        "custom || remains literal",
    ]
    assert positive["answer_items"][0]["source"] == "edited_suggestion"
    assert positive["generated_examples"][0]["example"] == "Positive 1"
