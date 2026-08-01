"""Append-only JSONL persistence for crawler run records."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path

from yt_searchapi.records import RunRecordBase


class RunIdMismatchError(ValueError):
    """Raised when a record is sent to another run's writer."""


class JsonlRunWriter:
    """Write each strict record type to its own append-only JSONL file.

    The writer opens files in append mode for every call, so a restarted run
    never truncates earlier evidence.  A lock prevents lines from interleaving
    between threads that share this writer.  Records are flushed before the
    call returns; pass ``fsync=True`` when durability across a process or host
    crash is more important than throughput.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        fsync: bool = False,
    ) -> None:
        self._run_id = _safe_run_id(run_id)
        self._run_dir = Path(output_dir).expanduser().resolve() / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._lock = threading.RLock()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def path_for(self, record_type: str) -> Path:
        """Return the stable output path for a validated record type."""

        if not record_type or not record_type.replace("_", "").isalnum():
            raise ValueError(f"unsafe record type {record_type!r}")
        return self._run_dir / f"{record_type}.jsonl"

    def append(self, record: RunRecordBase) -> Path:
        """Validate lineage and append exactly one JSON object plus newline."""

        if not isinstance(record, RunRecordBase):
            raise TypeError("record must be a validated RunRecordBase instance")
        if record.run_id != self._run_id:
            raise RunIdMismatchError(
                f"record run_id {record.run_id!r} does not match writer run_id "
                f"{self._run_id!r}"
            )
        record_type = getattr(record, "record_type", None)
        if not isinstance(record_type, str):
            raise TypeError("record must declare a string record_type")
        path = self.path_for(record_type)
        line = record.model_dump_json() + "\n"

        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
        return path

    def append_many(self, records: Iterable[RunRecordBase]) -> tuple[Path, ...]:
        """Append multiple records while holding one thread lock."""

        paths: list[Path] = []
        with self._lock:
            for record in records:
                paths.append(self.append(record))
        return tuple(paths)

    def jsonl_paths(self) -> tuple[Path, ...]:
        """List JSONL files currently present for this run."""

        with self._lock:
            return tuple(sorted(self._run_dir.glob("*.jsonl")))


def _safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty")
    if normalized in {".", ".."} or Path(normalized).name != normalized:
        raise ValueError("run_id must be a single safe path component")
    return normalized
