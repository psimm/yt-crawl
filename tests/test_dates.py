from datetime import date, datetime, timezone

from yt_crawl.dates import (
    is_on_or_after_start_date,
    language_code_matches,
    parse_publication_date,
    select_transcript_name,
)


def test_exact_date_passes_boundary() -> None:
    evidence = parse_publication_date("Premiered July 31, 2026")

    assert evidence.parsed == date(2026, 7, 31)
    assert evidence.certainty == "exact"
    assert is_on_or_after_start_date(evidence, date(2026, 7, 1))


def test_relative_date_is_not_accepted_as_exact() -> None:
    evidence = parse_publication_date(
        "2 months ago", relative_base=datetime(2026, 7, 31, tzinfo=timezone.utc)
    )

    assert evidence.certainty == "relative"
    assert not is_on_or_after_start_date(evidence, date(2026, 1, 1))


def test_ambiguous_numeric_date_is_not_treated_as_exact() -> None:
    evidence = parse_publication_date("07/01/2026")

    assert evidence.certainty == "unknown"
    assert evidence.parsed is None


def test_language_matching_and_named_selection() -> None:
    assert language_code_matches("en", "en-US")
    assert not language_code_matches("de", "en")
    available = [
        ("English (auto-generated)", "en"),
        ("English (United States)", "en-US"),
        ("Deutsch", "de"),
    ]

    assert select_transcript_name("en-US", available) == "English (United States)"
    assert select_transcript_name("fr", available) is None
