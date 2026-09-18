"""Read and write demo traces as JSON Lines.

A trace is the deterministic recording of one demo run — one
:class:`~cipoc.demo.events.DemoEvent` per line. Replaying a trace drives the
exact same UI pipeline as a live run, which is what makes the record-then-replay
presentation path safe and scrubbable.

The format is plain JSONL (not a single JSON array) so a live recording can be
appended one line at a time via :class:`TraceWriter` without holding the whole
run in memory or rewriting the file, and so a partial trace from an interrupted
run is still readable up to the last complete line.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from types import TracebackType
from typing import Iterable, Iterator

from .events import DemoEvent


def write_trace(path: str | Path, events: Iterable[DemoEvent]) -> int:
    """Write ``events`` to ``path`` as JSONL. Returns the number written."""
    with TraceWriter(path) as writer:
        for event in events:
            writer.write(event)
        return writer.count


def read_trace(path: str | Path) -> list[DemoEvent]:
    """Load a whole trace into memory (replay mode)."""
    return list(iter_trace(path))


def iter_trace(path: str | Path) -> Iterator[DemoEvent]:
    """Stream a trace line by line, skipping blank lines."""
    previous = -1
    previous_t = 0.0
    with Path(path).open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b"\n"):
                    warnings.warn("Ignoring incomplete final demo trace line.", stacklevel=2)
                    break
                raise ValueError(f"Malformed demo trace line {number}.") from None
            event = DemoEvent.from_dict(data)
            if event.seq != previous + 1 or not math.isfinite(event.t) or event.t < previous_t:
                raise ValueError(f"Invalid demo event sequence/timing at line {number}.")
            previous, previous_t = event.seq, event.t
            yield event


class TraceWriter:
    """Append :class:`DemoEvent`s to a JSONL trace as they are produced.

    Used by the live/record path: each event is flushed immediately so an
    interrupted run leaves a usable partial trace. Usable as a context manager.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self.count = 0

    def write(self, event: DemoEvent) -> None:
        self._handle.write(json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["write_trace", "read_trace", "iter_trace", "TraceWriter"]
