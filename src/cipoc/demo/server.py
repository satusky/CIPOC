"""Phase 2 — FastAPI app serving the demo: REST state + SSE controls.

The server owns a :class:`DemoSession` (replay of a recorded trace, or a live
run) and exposes it to the browser frontend (Phase 3) two ways:

* **REST** for pulling static-per-run data (the whole event list, the presenter
  step list) and any point-in-time :class:`~cipoc.demo.state.DemoSnapshot`
  (replay a prefix of the trace up to a cursor).
* **SSE** (``/api/stream``) for pushing cursor changes so every connected viewer
  follows the presenter, and — in live mode — for pushing events as the graph
  produces them. Control endpoints (``next`` / ``prev`` / ``goto`` / ``play`` /
  ``pause``) move the presenter cursor and broadcast the move.

The presenter cursor is a *step* index (see :mod:`cipoc.demo.steps`): advancing a
step replays the trace up to that step's ``end_seq``, which is a natural pause
boundary. Because a recorded trace is a fully-known ordered list, replay
navigation is pure and scrubbable; live mode only lets the cursor advance as far
as the graph has produced.

``fastapi`` / ``uvicorn`` are demo-only dependencies (the ``demo`` extra), so this
module is imported only when actually serving — never by ``cipoc.demo`` itself.
"""

from __future__ import annotations

import asyncio
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cipoc.demo.events import DemoEvent
from cipoc.demo.mapping import overview_block_map
from cipoc.demo.serialize import to_jsonable
from cipoc.demo.state import DemoState, replay
from cipoc.demo.steps import Step, build_steps
from cipoc.demo.trace import read_trace


WEB_DIR = Path(__file__).resolve().parent / "web"

# The simplified Panel-1 topology: the "overview" chart authored in the repo's
# machine-readable flowcharts. Served to the frontend so the map is always drawn
# from the same source of truth the rest of the project renders from.
FLOWCHARTS_JSON = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "visualization"
    / "agent_flowcharts.json"
)


@lru_cache(maxsize=1)
def overview_chart() -> dict[str, Any]:
    """Return the ``overview`` flowchart (elements + style + layout + colors)."""
    data = json.loads(FLOWCHARTS_JSON.read_text())
    chart = next(c for c in data["charts"] if c.get("id") == "overview")
    return {
        "elements": chart["elements"],
        "style": data.get("style", []),
        "layout": chart.get("layout", {"name": "breadthfirst"}),
        "agent_colors": data.get("metadata", {}).get("agent_colors", {}),
        "kind_colors": data.get("metadata", {}).get("kind_colors", {}),
        "title": chart.get("title", "Workflow"),
        # fine agent_system node ID -> coarse overview block the map highlights.
        "coarse_map": overview_block_map(),
    }


class DemoSession:
    """A replayable demo: a fixed event list, its steps, and a presenter cursor.

    Snapshots are produced by replaying a prefix of the trace, so any cursor
    position is reachable directly. Per-step snapshots are memoized because they
    are the positions the UI navigates; arbitrary-``seq`` scrubbing recomputes on
    demand. Cursor moves are thread-safe so control endpoints and an SSE push can
    interleave.
    """

    mode = "replay"

    def __init__(
        self,
        events: Iterable[DemoEvent],
        *,
        description: str = "CIPOC extraction",
        target_groups: Any = None,
        group_hierarchy: Any = None,
    ) -> None:
        self.description = description
        self._state_kwargs: dict[str, Any] = {"description": description}
        if target_groups is not None:
            self._state_kwargs["target_groups"] = target_groups
        if group_hierarchy is not None:
            self._state_kwargs["group_hierarchy"] = group_hierarchy

        self._lock = threading.RLock()
        self._events: list[DemoEvent] = list(events)
        self._steps: list[Step] = build_steps(self._events)
        self._step_snapshots: dict[int, dict[str, Any]] = {}
        self._cursor = 0
        self.playing = False

    # --- Static run data --------------------------------------------------

    @property
    def events(self) -> list[DemoEvent]:
        return self._events

    @property
    def steps(self) -> list[Step]:
        return self._steps

    def meta(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "description": self.description,
                "num_events": len(self._events),
                "num_steps": len(self._steps),
                "cursor": self._cursor,
                "playing": self.playing,
                "finished": self._is_last_cursor(),
            }

    def events_payload(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def steps_payload(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self._steps]

    # --- Snapshots --------------------------------------------------------

    def _replay_to_seq(self, seq: int) -> DemoState:
        with self._lock:
            prefix = [event for event in self._events if event.seq <= seq]
        return replay(prefix, **self._state_kwargs)

    def snapshot_at_seq(self, seq: int) -> dict[str, Any]:
        return self._replay_to_seq(seq).snapshot().to_dict()

    def case_at_seq(self, seq: int) -> Any:
        case = self._replay_to_seq(seq).latest_case
        return to_jsonable(case) if case is not None else None

    def notes(self) -> dict[str, Any]:
        """Compact ``note_id -> {note_type, date, content}`` from the latest case.

        The frontend needs raw note text to highlight extractor evidence spans
        inline (Phase 4). Note ``content`` is immutable once a note is scanned, so
        the newest corpus serves every cursor position; keys are stringified to
        survive JSON round-tripping. Empty until the first note has been scanned.
        """
        seq = self._events[-1].seq if self._events else 0
        case = self._replay_to_seq(seq).latest_case
        corpus = getattr(case, "note_corpus", None) or {}
        return {
            str(note_id): {
                "note_id": getattr(note, "note_id", note_id),
                "note_type": getattr(note, "note_type", None),
                "date": getattr(note, "date", None),
                "content": getattr(note, "content", None),
            }
            for note_id, note in corpus.items()
        }

    def step_snapshot(self, index: int) -> dict[str, Any]:
        with self._lock:
            if not self._steps:
                return replay((), **self._state_kwargs).snapshot().to_dict()
            index = _clamp(index, 0, len(self._steps) - 1)
            if index not in self._step_snapshots:
                self._step_snapshots[index] = self.snapshot_at_seq(self._steps[index].end_seq)
            return self._step_snapshots[index]

    # --- Cursor -----------------------------------------------------------

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def _max_cursor(self) -> int:
        return max(0, len(self._steps) - 1)

    def _is_last_cursor(self) -> bool:
        return self._cursor >= self._max_cursor()

    def goto(self, index: int) -> dict[str, Any]:
        with self._lock:
            self._cursor = _clamp(index, 0, self._max_cursor())
            return self._view()

    def next(self) -> dict[str, Any]:
        with self._lock:
            if self._is_last_cursor():
                self.playing = False
            return self.goto(self._cursor + 1)

    def prev(self) -> dict[str, Any]:
        with self._lock:
            return self.goto(self._cursor - 1)

    def set_playing(self, playing: bool) -> dict[str, Any]:
        with self._lock:
            self.playing = playing and not self._is_last_cursor()
            return self._view()

    def view(self) -> dict[str, Any]:
        with self._lock:
            return self._view()

    def _view(self) -> dict[str, Any]:
        step = self._steps[self._cursor] if self._steps else None
        return {
            "cursor": self._cursor,
            "playing": self.playing,
            "at_end": self._is_last_cursor(),
            "step": step.to_dict() if step is not None else None,
            "snapshot": self.step_snapshot(self._cursor),
        }


class LiveDemoSession(DemoSession):
    """A demo driven by a running graph rather than a finished trace.

    Starts a background thread that consumes an event iterator (typically
    :func:`cipoc.demo.stream.run_demo_stream`, which also records a trace),
    appending each :class:`DemoEvent` and re-deriving the step list as the run
    progresses. The presenter cursor can only advance as far as the graph has
    produced, so ``next`` naturally gates the reveal of live output.

    Push-on-event: when a produced event opens a new presenter step (or the run
    finishes), the session notifies a listener (wired by :func:`build_app` to the
    :class:`Broadcaster`) so every open browser learns more steps are available
    without polling. The reveal itself stays presenter-gated — the notification
    just refreshes the step list and re-enables ``Next`` / keeps auto-play going.
    """

    mode = "live"

    def __init__(
        self,
        events: Iterable[DemoEvent],
        *,
        description: str = "CIPOC extraction (live)",
        target_groups: Any = None,
        group_hierarchy: Any = None,
    ) -> None:
        super().__init__(
            (),
            description=description,
            target_groups=target_groups,
            group_hierarchy=group_hierarchy,
        )
        self._source = iter(events)
        self._done = False
        self._thread: threading.Thread | None = None
        self._listener: Any = None

    def set_listener(self, listener: Any) -> None:
        """Register a ``callable(message: dict)`` invoked as the run progresses."""
        self._listener = listener

    def start(self) -> None:
        """Begin consuming the event source on a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="cipoc-demo-live", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            for event in self._source:
                self.append(event)
        finally:
            with self._lock:
                self._done = True
            self._emit_live()

    def append(self, event: DemoEvent) -> None:
        """Add one produced event and refresh the derived step list."""
        with self._lock:
            prev_steps = len(self._steps)
            self._events.append(event)
            last = self._max_cursor()
            self._steps = build_steps(self._events)
            # The final (in-progress) step's end grows as events arrive, so its
            # memoized snapshot is stale; drop it. Earlier steps are frozen.
            self._step_snapshots.pop(last, None)
            self._step_snapshots.pop(self._max_cursor(), None)
            grew = len(self._steps) != prev_steps
        # Notify outside the lock. A new step becomes available to reveal (grew),
        # but also any content-bearing event advances the in-progress frontier
        # step's snapshot — so a presenter watching that frontier (e.g. the
        # collapsed "Characterize notes" step) sees per-note results fill in as
        # they arrive. The frontend only re-renders panels when the cursor is
        # actually on the frontier, so frozen earlier steps stay undisturbed.
        if grew or event.type != "run_start":
            self._emit_live()

    def _emit_live(self) -> None:
        listener = self._listener
        if listener is None:
            return
        try:
            listener({"type": "live", **self.meta()})
        except Exception:
            # A dead/failing listener must never break the graph-driving thread.
            pass

    def meta(self) -> dict[str, Any]:
        data = super().meta()
        with self._lock:
            data["done"] = self._done
        return data


class Broadcaster:
    """Fan-out of JSON messages to all connected SSE subscribers.

    Each subscriber gets its own :class:`asyncio.Queue`; :meth:`publish` puts the
    message on every queue. Used to push presenter cursor moves (and, in live
    mode, produced events) to every open browser so viewers stay in lockstep.

    ``publish`` is called from two kinds of thread: FastAPI's sync control
    endpoints (a threadpool) and — in live mode — the graph-driving daemon thread.
    Neither is the event-loop thread that owns the subscriber queues, so once the
    loop is known (bound the first time an SSE client connects) deliveries are
    marshalled onto it with :meth:`~asyncio.loop.call_soon_threadsafe`. Before any
    client has connected there are no subscribers, so a direct put is harmless.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that owns the subscriber queues (idempotent)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _deliver(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(message)

    def publish(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._deliver, message)
        else:
            self._deliver(message)


def build_app(session: DemoSession) -> FastAPI:
    """Build the FastAPI app serving ``session`` (replay or live)."""
    app = FastAPI(title="CIPOC Demo", docs_url=None, redoc_url=None)
    broadcaster = Broadcaster()
    app.state.session = session
    app.state.broadcaster = broadcaster

    # Live mode pushes step-availability notifications to every viewer; wire the
    # session's listener to the broadcaster (which marshals onto the event loop).
    if isinstance(session, LiveDemoSession):
        session.set_listener(broadcaster.publish)

    def _broadcast_view() -> dict[str, Any]:
        view = session.view()
        broadcaster.publish({"type": "cursor", **view})
        return view

    # --- Static run data ---
    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(session.meta())

    @app.get("/api/events")
    def events() -> JSONResponse:
        return JSONResponse(session.events_payload())

    @app.get("/api/steps")
    def steps() -> JSONResponse:
        return JSONResponse(session.steps_payload())

    @app.get("/api/graph")
    def graph() -> JSONResponse:
        return JSONResponse(overview_chart())

    # --- Point-in-time state ---
    @app.get("/api/snapshot")
    def snapshot(seq: int | None = None) -> JSONResponse:
        if seq is None:
            seq = session.events[-1].seq if session.events else 0
        return JSONResponse(session.snapshot_at_seq(seq))

    @app.get("/api/step/{index}")
    def step(index: int) -> JSONResponse:
        if not session.steps:
            raise HTTPException(status_code=404, detail="No steps in this run.")
        if index < 0 or index >= len(session.steps):
            raise HTTPException(status_code=404, detail="Step index out of range.")
        return JSONResponse(session.step_snapshot(index))

    @app.get("/api/case")
    def case(seq: int | None = None) -> JSONResponse:
        if seq is None:
            seq = session.events[-1].seq if session.events else 0
        return JSONResponse(session.case_at_seq(seq))

    @app.get("/api/notes")
    def notes() -> JSONResponse:
        return JSONResponse(session.notes())

    # --- Cursor + controls ---
    @app.get("/api/cursor")
    def cursor() -> JSONResponse:
        return JSONResponse(session.view())

    @app.post("/api/next")
    def next_step() -> JSONResponse:
        session.next()
        return JSONResponse(_broadcast_view())

    @app.post("/api/prev")
    def prev_step() -> JSONResponse:
        session.prev()
        return JSONResponse(_broadcast_view())

    @app.post("/api/goto/{index}")
    def goto(index: int) -> JSONResponse:
        session.goto(index)
        return JSONResponse(_broadcast_view())

    @app.post("/api/play")
    def play() -> JSONResponse:
        session.set_playing(True)
        return JSONResponse(_broadcast_view())

    @app.post("/api/pause")
    def pause() -> JSONResponse:
        session.set_playing(False)
        return JSONResponse(_broadcast_view())

    # --- SSE ---
    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        # First SSE connection binds the running loop so cross-thread publishes
        # (control endpoints, the live daemon) marshal onto it safely.
        broadcaster.bind_loop(asyncio.get_running_loop())
        queue = broadcaster.subscribe()

        async def event_source():
            # Seed the new subscriber with the current view so it renders at once.
            yield _sse({"type": "cursor", **session.view()})
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(message)
            finally:
                broadcaster.unsubscribe(queue)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    # --- Frontend (Phase 3 fills web/; serve a placeholder until then) ---
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:
        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse(_PLACEHOLDER_HTML)

    return app


def _sse(message: dict[str, Any]) -> str:
    return f"data: {json.dumps(message)}\n\n"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


_PLACEHOLDER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>CIPOC Demo</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>CIPOC demo server</h1>
<p>The backend is running. The web frontend lands in Phase 3.</p>
<p>Meanwhile the API is live:</p>
<ul>
  <li><code>GET /api/meta</code></li>
  <li><code>GET /api/events</code>, <code>GET /api/steps</code></li>
  <li><code>GET /api/step/{index}</code>, <code>GET /api/snapshot?seq=N</code></li>
  <li><code>POST /api/next</code>, <code>/api/prev</code>, <code>/api/goto/{index}</code></li>
  <li><code>GET /api/stream</code> (SSE)</li>
</ul>
</body></html>
"""


def load_replay_session(trace_path: str | Path, *, description: str | None = None) -> DemoSession:
    """Build a replay :class:`DemoSession` from a recorded trace file."""
    events = read_trace(trace_path)
    label = description or f"Replay of {Path(trace_path).name}"
    return DemoSession(events, description=label)


__all__ = [
    "DemoSession",
    "LiveDemoSession",
    "Broadcaster",
    "build_app",
    "load_replay_session",
    "overview_chart",
    "WEB_DIR",
]
