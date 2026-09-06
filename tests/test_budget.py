from concurrent.futures import ThreadPoolExecutor

import pytest

from yt_crawl.budget import (
    BudgetExceededError,
    ImmediateTranscriptRequiredError,
    SearchApiCreditBudget,
    UnknownReservationError,
)


def test_searchapi_reserve_is_inaccessible_to_discovery() -> None:
    budget = SearchApiCreditBudget(max_credits=10, transcript_reserve_credits=4)

    budget.spend_discovery(6)

    with pytest.raises(BudgetExceededError):
        budget.spend_discovery()
    snapshot = budget.snapshot()
    assert snapshot.discovery_remaining == 0
    assert snapshot.transcript_remaining == 4
    assert snapshot.total_remaining == 4


def test_transcript_reservation_borrows_only_unspent_discovery_capacity() -> None:
    budget = SearchApiCreditBudget(max_credits=50, transcript_reserve_credits=17)
    budget.spend_discovery(27)
    first = budget.mark_relevant("video-1", transcript_credits=17)
    budget.reconcile_transcript(first)

    second = budget.mark_relevant("video-2")
    snapshot = budget.snapshot()

    assert second.credits == 1
    assert snapshot.transcript_capacity == 18
    assert snapshot.transcript_spent == 17
    assert snapshot.transcript_reserved == 1
    assert snapshot.discovery_capacity == 32
    assert snapshot.discovery_remaining == 5
    assert snapshot.total_committed == 45
    assert snapshot.total_remaining == 5


def test_repeated_transcript_borrowing_stops_when_total_grant_is_committed() -> None:
    budget = SearchApiCreditBudget(max_credits=5, transcript_reserve_credits=1)
    budget.spend_discovery(2)
    first = budget.mark_relevant("video-1")
    budget.reconcile_transcript(first)

    second = budget.mark_relevant("video-2")
    budget.reconcile_transcript(second)
    third = budget.mark_relevant("video-3")
    budget.reconcile_transcript(third)

    snapshot = budget.snapshot()
    assert snapshot.transcript_capacity == 3
    assert snapshot.discovery_capacity == 2
    assert snapshot.total_committed == 5
    assert snapshot.total_remaining == 0
    with pytest.raises(
        BudgetExceededError,
        match=r"transcript pool.*0 unspent discovery.*0 total credits unspent",
    ):
        budget.mark_relevant("video-4")


def test_cache_hit_reconciliation_keeps_borrowed_capacity_without_spending() -> None:
    budget = SearchApiCreditBudget(max_credits=4, transcript_reserve_credits=1)
    budget.spend_discovery(2)
    first = budget.mark_relevant("video-1")
    budget.reconcile_transcript(first)

    borrowed = budget.mark_relevant("video-2")
    snapshot = budget.reconcile_transcript(borrowed, actual_credits=0)

    assert snapshot.transcript_capacity == 2
    assert snapshot.discovery_capacity == 2
    assert snapshot.transcript_spent == 1
    assert snapshot.transcript_reserved == 0
    assert snapshot.total_committed == 3
    assert snapshot.total_remaining == 1


def test_borrowed_transcript_capacity_survives_export_and_restore() -> None:
    budget = SearchApiCreditBudget(max_credits=4, transcript_reserve_credits=1)
    budget.spend_discovery(2)
    first = budget.mark_relevant("video-1")
    budget.reconcile_transcript(first)
    budget.mark_relevant("video-2")

    restored = SearchApiCreditBudget.restore(budget.export_state())
    snapshot = restored.snapshot()

    assert snapshot.transcript_capacity == 2
    assert snapshot.discovery_capacity == 2
    assert snapshot.transcript_spent == 1
    assert snapshot.transcript_reserved == 1
    assert snapshot.discovery_spent == 2
    assert snapshot.total_committed == 4
    assert snapshot.total_remaining == 0


def test_restore_supports_a_grant_fully_transferred_to_transcripts() -> None:
    budget = SearchApiCreditBudget(max_credits=4, transcript_reserve_credits=1)
    budget.mark_relevant("video-1", transcript_credits=4)

    restored = SearchApiCreditBudget.restore(budget.export_state())
    snapshot = restored.snapshot()

    assert snapshot.transcript_capacity == 4
    assert snapshot.discovery_capacity == 0
    assert snapshot.transcript_reserved == 4
    assert snapshot.total_committed == 4
    assert snapshot.total_remaining == 0

    restored.expand(max_credits=8, transcript_reserve_credits=4)
    expanded = restored.snapshot()
    assert expanded.discovery_capacity == 4
    assert expanded.transcript_capacity == 4
    assert expanded.transcript_reserved == 4
    assert expanded.total_remaining == 4


def test_expand_cannot_return_transferred_capacity_to_discovery() -> None:
    budget = SearchApiCreditBudget(max_credits=10, transcript_reserve_credits=3)
    reservation = budget.mark_relevant("video-1", transcript_credits=7)
    budget.reconcile_transcript(reservation, actual_credits=0)

    before = budget.snapshot()
    assert before.transcript_capacity == 7
    assert before.discovery_capacity == 3

    with pytest.raises(ValueError, match="one-way transferred capacity"):
        budget.expand(max_credits=11, transcript_reserve_credits=4)

    rejected = budget.snapshot()
    assert rejected.max_credits == 10
    assert rejected.transcript_capacity == 7
    assert rejected.discovery_capacity == 3

    budget.expand(max_credits=11, transcript_reserve_credits=7)
    expanded = budget.snapshot()
    assert expanded.transcript_capacity == 7
    assert expanded.discovery_capacity == 4
    assert expanded.total_remaining == 11


def test_concurrent_transcript_reservations_borrow_without_exceeding_grant() -> None:
    budget = SearchApiCreditBudget(max_credits=12, transcript_reserve_credits=2)
    budget.spend_discovery(4)

    def try_reserve(index: int) -> bool:
        try:
            budget.mark_relevant(f"video-{index}")
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(try_reserve, range(20)))

    snapshot = budget.snapshot()
    assert sum(outcomes) == 8
    assert snapshot.transcript_capacity == 8
    assert snapshot.discovery_capacity == 4
    assert snapshot.transcript_reserved == 8
    assert snapshot.total_committed == 12
    assert snapshot.total_remaining == 0


def test_relevant_video_must_be_transcribed_before_more_discovery() -> None:
    budget = SearchApiCreditBudget(max_credits=10, transcript_reserve_credits=4)
    reservation = budget.mark_relevant("video-1")

    with pytest.raises(ImmediateTranscriptRequiredError, match="video-1"):
        budget.spend_discovery()

    snapshot = budget.reconcile_transcript(reservation, actual_credits=1)
    assert snapshot.transcript_spent == 1
    assert snapshot.transcript_reserved == 0
    assert snapshot.pending_transcript_video_ids == ()
    assert budget.spend_discovery().discovery_spent == 1


def test_transcript_hold_is_idempotent_and_can_be_released() -> None:
    budget = SearchApiCreditBudget(max_credits=8, transcript_reserve_credits=3)
    first = budget.mark_relevant("video-1", transcript_credits=2)
    second = budget.mark_relevant("video-1", transcript_credits=2)

    assert first == second
    assert budget.snapshot().transcript_reserved == 2
    snapshot = budget.release_transcript(first.reservation_id)
    assert snapshot.transcript_reserved == 0
    assert snapshot.transcript_remaining == 3

    with pytest.raises(UnknownReservationError):
        budget.release_transcript(first)


def test_bounded_transcript_batch_borrows_after_exhausting_its_reserve() -> None:
    budget = SearchApiCreditBudget(max_credits=8, transcript_reserve_credits=3)
    budget.mark_relevant("video-1")
    budget.mark_relevant("video-2")
    budget.mark_relevant("video-3")
    budget.mark_relevant("video-4")

    with pytest.raises(ImmediateTranscriptRequiredError, match="video-1"):
        budget.spend_discovery()

    snapshot = budget.snapshot()
    assert snapshot.pending_transcript_video_ids == (
        "video-1",
        "video-2",
        "video-3",
        "video-4",
    )
    assert snapshot.transcript_capacity == 4
    assert snapshot.discovery_capacity == 4


def test_transcript_reconciliation_is_atomic_when_actual_cost_is_too_high() -> None:
    budget = SearchApiCreditBudget(max_credits=8, transcript_reserve_credits=3)
    first = budget.mark_relevant("video-1", transcript_credits=2)

    with pytest.raises(BudgetExceededError):
        budget.reconcile_transcript(first, actual_credits=9)

    assert budget.snapshot().pending_transcript_video_ids == ("video-1",)
    budget.reconcile_transcript(first)


def test_transcript_reconciliation_can_borrow_for_an_underestimated_cost() -> None:
    budget = SearchApiCreditBudget(max_credits=8, transcript_reserve_credits=3)
    reservation = budget.mark_relevant("video-1", transcript_credits=2)

    snapshot = budget.reconcile_transcript(reservation, actual_credits=4)

    assert snapshot.transcript_capacity == 4
    assert snapshot.discovery_capacity == 4
    assert snapshot.transcript_spent == 4
    assert snapshot.total_committed == 4


def test_searchapi_discovery_spending_is_thread_safe() -> None:
    budget = SearchApiCreditBudget(max_credits=60, transcript_reserve_credits=10)

    def try_spend(_: int) -> bool:
        try:
            budget.spend_discovery()
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(try_spend, range(100)))

    assert sum(outcomes) == 50
    assert budget.snapshot().discovery_spent == 50


def test_searchapi_budget_restore_and_expand_preserves_spending() -> None:
    budget = SearchApiCreditBudget(max_credits=6, transcript_reserve_credits=2)
    budget.spend_discovery(3)
    reservation = budget.mark_relevant("video-1")
    budget.reconcile_transcript(reservation, actual_credits=1)

    restored = SearchApiCreditBudget.restore(budget.export_state())
    restored.expand(max_credits=10, transcript_reserve_credits=3)

    snapshot = restored.snapshot()
    assert snapshot.max_credits == 10
    assert snapshot.discovery_spent == 3
    assert snapshot.transcript_spent == 1
    assert snapshot.discovery_remaining == 4
    assert snapshot.transcript_remaining == 2


def test_resize_overwrites_remaining_without_reopening_spent_credits() -> None:
    budget = SearchApiCreditBudget(max_credits=70, transcript_reserve_credits=20)
    budget.spend_discovery(10)
    reservation = budget.mark_relevant("video-1", transcript_credits=10)
    budget.reconcile_transcript(reservation)

    budget.resize(max_credits=20, transcript_reserve_credits=10)
    zeroed = budget.snapshot()
    assert zeroed.total_remaining == 0
    assert zeroed.discovery_spent == 10
    assert zeroed.transcript_spent == 10
    assert zeroed.discovery_capacity == 10
    assert zeroed.transcript_capacity == 10

    budget.resize(max_credits=120, transcript_reserve_credits=40)
    expanded = budget.snapshot()
    assert expanded.total_remaining == 100
    assert expanded.discovery_spent == 10
    assert expanded.transcript_spent == 10
    assert expanded.max_credits == 120


def test_resize_cannot_go_below_spent_or_reserved_credits() -> None:
    budget = SearchApiCreditBudget(max_credits=12, transcript_reserve_credits=4)
    budget.spend_discovery(3)
    budget.mark_relevant("video-1", transcript_credits=2)

    with pytest.raises(ValueError, match="already spent or reserved"):
        budget.resize(max_credits=5, transcript_reserve_credits=1)
    with pytest.raises(ValueError, match="already spent"):
        budget.resize(max_credits=6, transcript_reserve_credits=4)
    rejected = budget.snapshot()
    assert rejected.max_credits == 12
    assert rejected.transcript_reserved == 2


def test_restore_accepts_a_zero_remaining_overwrite() -> None:
    restored = SearchApiCreditBudget.restore(
        {
            "max_credits": 0,
            "transcript_capacity": 0,
            "discovery_spent": 0,
            "transcript_spent": 0,
            "pending": [],
            "completed_video_ids": [],
        }
    )
    snapshot = restored.snapshot()
    assert snapshot.max_credits == 0
    assert snapshot.total_remaining == 0
    assert snapshot.discovery_capacity == 0
    assert snapshot.transcript_capacity == 0


@pytest.mark.parametrize(
    ("max_credits", "reserve"),
    [(0, 0), (10, 0), (10, 10), (10, 11), (True, 1)],
)
def test_invalid_searchapi_budget_configuration(max_credits: int, reserve: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        SearchApiCreditBudget(max_credits, reserve)
