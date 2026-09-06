"""Conservative, resumable SearchAPI credit accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from uuid import uuid4


class BudgetError(RuntimeError):
    """Base class for budget accounting failures."""


class BudgetExceededError(BudgetError):
    """Raised when an operation would exceed available credit capacity."""


class ImmediateTranscriptRequiredError(BudgetError):
    """Raised when discovery is attempted before pending transcripts finish."""


class UnknownReservationError(BudgetError):
    """Raised when a reservation is missing or has already been reconciled."""


@dataclass(frozen=True, slots=True)
class TranscriptCreditReservation:
    """Credits held for the immediate transcript request for one video."""

    reservation_id: str
    video_id: str
    credits: int


@dataclass(frozen=True, slots=True)
class SearchApiBudgetSnapshot:
    """Immutable point-in-time view of SearchAPI credit accounting."""

    max_credits: int
    discovery_capacity: int
    discovery_spent: int
    discovery_remaining: int
    transcript_capacity: int
    transcript_spent: int
    transcript_reserved: int
    transcript_remaining: int
    total_committed: int
    total_remaining: int
    pending_transcript_video_ids: tuple[str, ...]

    @property
    def can_discover(self) -> bool:
        """Whether at least one discovery credit can be spent right now."""

        return not self.pending_transcript_video_ids and self.discovery_remaining > 0


class SearchApiCreditBudget:
    """Protect a fixed SearchAPI transcript reserve from discovery fan-out.

    ``mark_relevant`` atomically holds transcript credits and activates the
    immediate-transcription guard. Multiple independent transcript requests may
    be reserved as one bounded batch. While any such reservation is pending,
    ``spend_discovery`` fails.  The caller must then fetch the transcript and
    call ``reconcile_transcript`` (or explicitly abandon it with
    ``release_transcript``) before exploring another search result, channel,
    or related-video page.

    Discovery credits can never consume the transcript reserve.  Once that
    reserve is full, however, a transcript reservation may transfer *unspent*
    discovery capacity into the transcript pool.  This
    avoids stranding an otherwise usable grant while preserving the protected
    transcript capacity against discovery fan-out.
    """

    def __init__(self, max_credits: int, transcript_reserve_credits: int) -> None:
        _require_positive_int("max_credits", max_credits)
        _require_positive_int("transcript_reserve_credits", transcript_reserve_credits)
        if transcript_reserve_credits >= max_credits:
            raise ValueError(
                "transcript_reserve_credits must be smaller than max_credits "
                "so at least one discovery request is possible"
            )

        self._max_credits = max_credits
        self._transcript_capacity = transcript_reserve_credits
        self._discovery_capacity = max_credits - transcript_reserve_credits
        self._discovery_spent = 0
        self._transcript_spent = 0
        self._pending: dict[str, TranscriptCreditReservation] = {}
        self._completed_video_ids: set[str] = set()
        self._lock = threading.RLock()

    @classmethod
    def restore(cls, state: dict[str, object]) -> SearchApiCreditBudget:
        """Restore persisted accounting without reopening spent credits."""

        max_credits = int(state["max_credits"])
        transcript_capacity = int(state["transcript_capacity"])
        _require_nonnegative_int("max_credits", max_credits)
        _require_nonnegative_int("transcript_capacity", transcript_capacity)
        if transcript_capacity > max_credits:
            raise ValueError(
                "persisted transcript capacity cannot exceed the SearchAPI grant"
            )

        # Persisted grants may exhaust either pool, including a zero remaining
        # overwrite. Fresh configuration still requires a discovery credit.
        budget = cls(2, 1)
        budget._max_credits = max_credits
        budget._transcript_capacity = transcript_capacity
        budget._discovery_capacity = max_credits - transcript_capacity
        budget._discovery_spent = int(state.get("discovery_spent", 0))
        budget._transcript_spent = int(state.get("transcript_spent", 0))
        pending = state.get("pending", [])
        if not isinstance(pending, list):
            raise ValueError("persisted SearchAPI pending reservations must be a list")
        for item in pending:
            if not isinstance(item, dict):
                raise ValueError("invalid persisted SearchAPI reservation")
            reservation = TranscriptCreditReservation(
                reservation_id=str(item["reservation_id"]),
                video_id=str(item["video_id"]),
                credits=int(item["credits"]),
            )
            budget._pending[reservation.video_id] = reservation
        completed = state.get("completed_video_ids", [])
        if not isinstance(completed, list):
            raise ValueError("persisted completed transcript IDs must be a list")
        budget._completed_video_ids = {str(item) for item in completed}
        snapshot = budget.snapshot()
        if snapshot.discovery_remaining < 0 or snapshot.transcript_remaining < 0:
            raise ValueError("persisted SearchAPI budget spends exceed its grant")
        return budget

    def export_state(self) -> dict[str, object]:
        """Return the JSON-compatible accounting needed for resume."""

        with self._lock:
            return {
                "max_credits": self._max_credits,
                "transcript_capacity": self._transcript_capacity,
                "discovery_spent": self._discovery_spent,
                "transcript_spent": self._transcript_spent,
                "pending": [
                    {
                        "reservation_id": item.reservation_id,
                        "video_id": item.video_id,
                        "credits": item.credits,
                    }
                    for item in self._pending.values()
                ],
                "completed_video_ids": sorted(self._completed_video_ids),
            }

    def expand(self, max_credits: int, transcript_reserve_credits: int) -> None:
        """Increase a grant without moving transcript capacity back to discovery.

        The requested transcript capacity may allocate newly granted credits,
        but it can never be lower than the current capacity.  This preserves
        every prior discovery-to-transcript transfer, as well as spent and
        currently reserved credits.
        """

        _require_positive_int("max_credits", max_credits)
        _require_positive_int("transcript_reserve_credits", transcript_reserve_credits)
        with self._lock:
            if max_credits < self._max_credits:
                raise ValueError("SearchAPI max_credits cannot be decreased")
            if transcript_reserve_credits >= max_credits:
                raise ValueError(
                    "transcript_reserve_credits must be smaller than max_credits"
                )
            if transcript_reserve_credits < self._transcript_capacity:
                raise ValueError(
                    "transcript capacity cannot be lower than its current "
                    "one-way transferred capacity"
                )
            minimum_transcript = (
                self._transcript_spent + self._pending_credits_unlocked()
            )
            minimum_discovery = self._discovery_spent
            if transcript_reserve_credits < minimum_transcript:
                raise ValueError(
                    "transcript reserve cannot be lower than credits already spent "
                    "or reserved"
                )
            if max_credits - transcript_reserve_credits < minimum_discovery:
                raise ValueError(
                    "discovery capacity cannot be lower than credits already spent"
                )
            self._max_credits = max_credits
            self._transcript_capacity = transcript_reserve_credits
            self._discovery_capacity = max_credits - transcript_reserve_credits

    def resize(self, max_credits: int, transcript_reserve_credits: int) -> None:
        """Set grant and pool capacities without reopening spent or reserved credits."""

        _require_nonnegative_int("max_credits", max_credits)
        _require_nonnegative_int(
            "transcript_reserve_credits", transcript_reserve_credits
        )
        with self._lock:
            if transcript_reserve_credits > max_credits:
                raise ValueError(
                    "transcript_reserve_credits cannot exceed max_credits"
                )
            minimum_transcript = (
                self._transcript_spent + self._pending_credits_unlocked()
            )
            minimum_discovery = self._discovery_spent
            if transcript_reserve_credits < minimum_transcript:
                raise ValueError(
                    "transcript reserve cannot be lower than credits already spent "
                    "or reserved"
                )
            if max_credits - transcript_reserve_credits < minimum_discovery:
                raise ValueError(
                    "discovery capacity cannot be lower than credits already spent"
                )
            self._max_credits = max_credits
            self._transcript_capacity = transcript_reserve_credits
            self._discovery_capacity = max_credits - transcript_reserve_credits

    def spend_discovery(self, credits: int = 1) -> SearchApiBudgetSnapshot:
        """Atomically charge a search, video, channel, or related-video request."""

        _require_positive_int("credits", credits)
        with self._lock:
            self._raise_if_transcript_pending()
            remaining = self._discovery_capacity - self._discovery_spent
            if credits > remaining:
                raise BudgetExceededError(
                    f"discovery requires {credits} credits but only {remaining} remain"
                )
            self._discovery_spent += credits
            return self._snapshot_unlocked()

    def mark_relevant(
        self, video_id: str, transcript_credits: int = 1
    ) -> TranscriptCreditReservation:
        """Reserve credits and require this relevant video's transcript next.

        Repeating the call for an already-pending video is idempotent when the
        requested credit amount is unchanged.  This makes retrying persistence
        around the decision point safe without double-reserving credits.
        """

        video_id = _require_nonempty("video_id", video_id)
        _require_nonnegative_int("transcript_credits", transcript_credits)
        with self._lock:
            existing = self._pending.get(video_id)
            if existing is not None:
                if existing.credits != transcript_credits:
                    raise BudgetError(
                        f"video {video_id!r} already reserves {existing.credits} "
                        f"transcript credits, not {transcript_credits}"
                    )
                return existing
            if video_id in self._completed_video_ids:
                raise BudgetError(
                    f"video {video_id!r} already has a reconciled transcript"
                )
            self._transfer_discovery_capacity_to_transcripts_unlocked(
                transcript_credits
            )
            reservation = TranscriptCreditReservation(
                reservation_id=uuid4().hex,
                video_id=video_id,
                credits=transcript_credits,
            )
            self._pending[video_id] = reservation
            return reservation

    reserve_transcript = mark_relevant

    def pending_transcript(self, video_id: str) -> TranscriptCreditReservation | None:
        """Return the existing hold for ``video_id``, if one is pending."""

        video_id = _require_nonempty("video_id", video_id)
        with self._lock:
            return self._pending.get(video_id)

    def reconcile_transcript(
        self,
        reservation: TranscriptCreditReservation | str,
        *,
        actual_credits: int | None = None,
    ) -> SearchApiBudgetSnapshot:
        """Finalize a transcript reservation after its request completes.

        ``actual_credits=0`` is useful for a cache hit or a request that was not
        billed.  If the real cost exceeds the reservation, the extra charge
        may transfer only currently unspent discovery capacity to transcripts.
        """

        reservation_id = _reservation_id(reservation)
        if actual_credits is not None:
            _require_nonnegative_int("actual_credits", actual_credits)
        with self._lock:
            video_id, active = self._find_pending_unlocked(reservation_id)
            actual = active.credits if actual_credits is None else actual_credits
            extra = max(0, actual - active.credits)
            self._transfer_discovery_capacity_to_transcripts_unlocked(extra)

            del self._pending[video_id]
            self._transcript_spent += actual
            self._completed_video_ids.add(video_id)
            return self._snapshot_unlocked()

    complete_transcript = reconcile_transcript

    def charge_failed_transcript(
        self,
        reservation: TranscriptCreditReservation | str,
    ) -> SearchApiBudgetSnapshot:
        """Charge an uncertain dispatch without marking transcript work complete."""

        reservation_id = _reservation_id(reservation)
        with self._lock:
            video_id, active = self._find_pending_unlocked(reservation_id)
            del self._pending[video_id]
            self._transcript_spent += active.credits
            return self._snapshot_unlocked()

    def release_transcript(
        self, reservation: TranscriptCreditReservation | str
    ) -> SearchApiBudgetSnapshot:
        """Release an unspent transcript hold after explicitly abandoning it."""

        reservation_id = _reservation_id(reservation)
        with self._lock:
            video_id, _ = self._find_pending_unlocked(reservation_id)
            del self._pending[video_id]
            return self._snapshot_unlocked()

    cancel_transcript = release_transcript

    def assert_can_discover(self) -> None:
        """Raise if immediate transcription or exhausted discovery blocks work."""

        with self._lock:
            self._raise_if_transcript_pending()
            if self._discovery_spent >= self._discovery_capacity:
                raise BudgetExceededError("no discovery credits remain")

    def snapshot(self) -> SearchApiBudgetSnapshot:
        """Return a consistent, immutable accounting snapshot."""

        with self._lock:
            return self._snapshot_unlocked()

    def _raise_if_transcript_pending(self) -> None:
        if self._pending:
            pending = ", ".join(sorted(self._pending))
            raise ImmediateTranscriptRequiredError(
                "transcript collection must be reconciled before discovery "
                f"continues; pending video IDs: {pending}"
            )

    def _find_pending_unlocked(
        self, reservation_id: str
    ) -> tuple[str, TranscriptCreditReservation]:
        for video_id, reservation in self._pending.items():
            if reservation.reservation_id == reservation_id:
                return video_id, reservation
        raise UnknownReservationError(
            f"unknown or already reconciled transcript reservation {reservation_id!r}"
        )

    def _pending_credits_unlocked(self) -> int:
        return sum(item.credits for item in self._pending.values())

    def _transcript_available_unlocked(self) -> int:
        return (
            self._transcript_capacity
            - self._transcript_spent
            - self._pending_credits_unlocked()
        )

    def _transfer_discovery_capacity_to_transcripts_unlocked(
        self, required_credits: int
    ) -> None:
        """Meet a transcript need by moving only unspent discovery capacity."""

        transcript_available = self._transcript_available_unlocked()
        shortfall = required_credits - transcript_available
        if shortfall <= 0:
            return

        discovery_unspent = self._discovery_capacity - self._discovery_spent
        total_unspent = transcript_available + discovery_unspent
        if shortfall > discovery_unspent:
            raise BudgetExceededError(
                "transcript pool requires "
                f"{required_credits} credits but has {transcript_available} unspent; "
                f"only {discovery_unspent} unspent discovery credits can transfer "
                f"({total_unspent} total credits unspent)"
            )

        self._transcript_capacity += shortfall
        self._discovery_capacity -= shortfall

    def _snapshot_unlocked(self) -> SearchApiBudgetSnapshot:
        transcript_reserved = self._pending_credits_unlocked()
        transcript_remaining = (
            self._transcript_capacity - self._transcript_spent - transcript_reserved
        )
        discovery_remaining = self._discovery_capacity - self._discovery_spent
        total_committed = (
            self._discovery_spent + self._transcript_spent + transcript_reserved
        )
        return SearchApiBudgetSnapshot(
            max_credits=self._max_credits,
            discovery_capacity=self._discovery_capacity,
            discovery_spent=self._discovery_spent,
            discovery_remaining=discovery_remaining,
            transcript_capacity=self._transcript_capacity,
            transcript_spent=self._transcript_spent,
            transcript_reserved=transcript_reserved,
            transcript_remaining=transcript_remaining,
            total_committed=total_committed,
            total_remaining=self._max_credits - total_committed,
            pending_transcript_video_ids=tuple(sorted(self._pending)),
        )


def _reservation_id(reservation: object) -> str:
    if isinstance(reservation, str):
        return _require_nonempty("reservation_id", reservation)
    reservation_id = getattr(reservation, "reservation_id", None)
    if not isinstance(reservation_id, str):
        raise TypeError("reservation must be a reservation object or reservation ID")
    return _require_nonempty("reservation_id", reservation_id)


def _require_positive_int(name: str, value: int) -> None:
    _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")


def _require_nonempty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized
