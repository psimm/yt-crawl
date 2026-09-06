"""Strict publication-date and language gates for crawler candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

import dateparser

DateCertainty = Literal["exact", "relative", "unknown"]

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DAY = re.compile(r"\b(?:[0-2]?\d|3[01])(?:st|nd|rd|th)?\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_AMBIGUOUS_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}\b")
_RELATIVE = re.compile(
    r"\b(?:ago|yesterday|today|hour|day|week|month|year)s?\b", re.IGNORECASE
)
_DATE_PREFIX = re.compile(
    r"^(?:premiered|published(?:\s+on)?|streamed\s+live\s+on)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PublicationDateEvidence:
    """Parsed date plus whether it is safe to use for a strict date boundary."""

    raw: str | None
    parsed: date | None
    certainty: DateCertainty


def parse_publication_date(
    value: str | None, *, relative_base: datetime | None = None
) -> PublicationDateEvidence:
    """Parse SearchAPI's date string without treating relative dates as exact.

    Search and channel results commonly contain values such as ``"2 years ago"``.
    They are useful for prioritisation but unsafe at an exact ``start_date`` boundary.
    Video-detail values containing a day, month and four-digit year are accepted as
    exact. Unknown or coarse values are retained as evidence and rejected by the
    strict gate.
    """

    if value is None or not value.strip():
        return PublicationDateEvidence(value, None, "unknown")

    raw = value.strip()
    if _AMBIGUOUS_NUMERIC_DATE.search(raw):
        return PublicationDateEvidence(raw, None, "unknown")
    parseable = _DATE_PREFIX.sub("", raw)
    base = relative_base or datetime.now(timezone.utc)
    parsed = dateparser.parse(
        parseable,
        settings={
            "RELATIVE_BASE": base,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
            "PREFER_DATES_FROM": "past",
        },
    )
    parsed_date = parsed.date() if parsed is not None else None
    if parsed_date is None:
        return PublicationDateEvidence(raw, None, "unknown")

    if _RELATIVE.search(raw):
        return PublicationDateEvidence(raw, parsed_date, "relative")

    has_exact_components = bool(_ISO_DATE.search(raw)) or (
        bool(_YEAR.search(raw)) and bool(_DAY.search(raw))
    )
    certainty: DateCertainty = "exact" if has_exact_components else "unknown"
    return PublicationDateEvidence(raw, parsed_date, certainty)


def is_on_or_after_start_date(
    evidence: PublicationDateEvidence, start_date: date
) -> bool:
    """Return true only for an exact, in-range publication date."""

    return (
        evidence.certainty == "exact"
        and evidence.parsed is not None
        and evidence.parsed >= start_date
    )


def language_code_matches(requested: str, available: str | None) -> bool:
    """Match BCP-47-like codes while accepting regional variants of a language."""

    if not available:
        return False
    requested_parts = requested.casefold().replace("_", "-").split("-")
    available_parts = available.casefold().replace("_", "-").split("-")
    return requested_parts[0] == available_parts[0]


def select_transcript_name(
    requested_language: str, available: list[tuple[str | None, str | None]]
) -> str | None:
    """Choose a named source transcript and never fall back to another language."""

    matches = [
        (name, code)
        for name, code in available
        if name and language_code_matches(requested_language, code)
    ]
    if not matches:
        return None

    requested = requested_language.casefold().replace("_", "-")
    matches.sort(
        key=lambda item: (
            item[1].casefold().replace("_", "-") != requested if item[1] else True,
            "auto-generated" in item[0].casefold(),
            item[0].casefold(),
        )
    )
    return matches[0][0]
