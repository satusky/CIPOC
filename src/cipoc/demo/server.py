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
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from cipoc.demo.events import DemoEvent
from cipoc.demo.mapping import overview_block_map
from cipoc.demo.serialize import to_jsonable
from cipoc.demo.state import DemoState, replay
from cipoc.demo.steps import Step, StepBuilder
from cipoc.demo.stream import DemoRun
from cipoc.demo.trace import read_trace
from cipoc.export import OmopExporter
from cipoc.export.models import NOTE_FIELDS, NOTE_NLP_FIELDS


WEB_DIR = Path(__file__).resolve().parent / "web"

# OMOP NOTE.person_id is required and ``Case`` carries no patient identity — it is
# a per-case extraction result, not a record about a person. The demo therefore
# supplies one. It has to be a real value rather than ``None``: a blank person_id
# fails ``OmopNoteRow`` validation, which would send every NOTE row to the error
# list and make a working export look broken on stage.
DEMO_PERSON_ID = 1

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
        artifact: dict[str, Any] | None = None,
    ) -> None:
        self.description = description
        self._state_kwargs: dict[str, Any] = {"description": description}
        if target_groups is not None:
            self._state_kwargs["target_groups"] = target_groups
        if group_hierarchy is not None:
            self._state_kwargs["group_hierarchy"] = group_hierarchy

        self._lock = threading.RLock()
        self._events: list[DemoEvent] = []
        self._builder = StepBuilder()
        self._steps = self._builder.steps
        self._latest = DemoState(**self._state_kwargs)
        self._checkpoints: OrderedDict[int, DemoState] = OrderedDict()
        self._step_snapshots: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._cursor = 0
        self.playing = False
        self.follow_live = False
        self.artifact = artifact
        for event in events:
            self._append(event)

    def _append(self, event: DemoEvent) -> None:
        with self._lock:
            previous_frontier = self._max_cursor()
            self._latest.ingest(event)
            self._builder.ingest(event)
            self._events.append(event)
            self._step_snapshots.pop(previous_frontier, None)
            self._step_snapshots.pop(self._max_cursor(), None)
            if event.seq % 128 == 0:
                self._checkpoints[event.seq] = deepcopy(self._latest)
                if len(self._checkpoints) > 4:
                    self._checkpoints.popitem(last=False)
            if self.follow_live:
                self._cursor = self._max_cursor()

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
                "follow_live": self.follow_live,
                "status": self._replay_status(),
                "artifact_available": self.artifact is not None,
                "issues": (self._events[-1].payload or {}).get("issues", []) if self._events and self._events[-1].type in ("run_end", "run_error") else [],
            }

    def _replay_status(self) -> str:
        if not self._events:
            return "empty"
        return {"run_end": "completed", "run_error": "failed"}.get(self._events[-1].type, "incomplete")

    def events_payload(self, after: int = -1) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events[max(0, after + 1):]
        return [event.to_dict() for event in events]

    def steps_payload(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self._steps]

    # --- Snapshots --------------------------------------------------------

    def _replay_to_seq(self, seq: int) -> DemoState:
        with self._lock:
            if self._events and seq >= self._events[-1].seq:
                return deepcopy(self._latest)
            eligible = [key for key in self._checkpoints if key <= seq]
            start = max(eligible) if eligible else -1
            state = deepcopy(self._checkpoints[start]) if eligible else DemoState(**self._state_kwargs)
            prefix = self._events[start + 1:seq + 1]
        for event in prefix:
            state.ingest(event)
        return state

    def snapshot_at_seq(self, seq: int) -> dict[str, Any]:
        return self._replay_to_seq(seq).snapshot().to_dict()

    def case_at_seq(self, seq: int) -> Any:
        case = self._replay_to_seq(seq).latest_case
        return to_jsonable(case) if case is not None else None

    def has_item(self, item_id: int) -> bool:
        """Whether this run ever requested ``item_id``.

        Read from the newest state, not the cursor's: the variable table is
        planned once and does not shrink, so an item that is real later is real
        now, and a 404 that depended on where the presenter is standing would be
        a worse answer than an empty table.
        """
        seq = self._events[-1].seq if self._events else 0
        case_state = self._replay_to_seq(seq).latest_case
        if case_state is None:
            return False
        return item_id in case_state.variable_results

    def omop_at_seq(self, item_id: int, seq: int) -> dict[str, Any]:
        """The OMOP rows one variable would export, as of ``seq``.

        The demo's closing point is that a coded value is not the deliverable —
        the NOTE_NLP row is. This runs the real :class:`OmopExporter` rather than
        describing it, so what the browser shows and what ``scripts/export_omop.py``
        writes cannot drift apart.

        NOTE rows are narrowed to the notes this variable's spans actually cite,
        which is a *presentation* choice and so belongs here rather than in the
        exporter: a real export writes the whole corpus, and ``build`` is handed
        the whole corpus too so its dangling-reference check stays honest. Both
        counts are reported so the modal can say which it is showing.
        """
        case_state = self._replay_to_seq(seq).latest_case
        # Null until the first note has been scanned. The variables pane — and so
        # the button — exists from the planning step onward, which is earlier.
        if case_state is None:
            return _empty_omop(item_id, seq)

        notes = list((getattr(case_state, "note_corpus", None) or {}).values())
        case = case_state.to_case()
        tables = OmopExporter(person_id=DEMO_PERSON_ID, nlp_date=self.nlp_date()).build(
            notes=notes,
            case=case,
            item_ids=[item_id],
        )

        cited = {str(row.note_id) for row in tables.note_nlp_rows}
        shown = [row for row in tables.note_rows if str(row.note_id) in cited]
        return {
            "item_id": item_id,
            "seq": seq,
            "person_id": DEMO_PERSON_ID,
            "preview": True,
            "status": _variable_status(case, item_id),
            "note_nlp": _table(NOTE_NLP_FIELDS, tables.note_nlp_rows),
            "note": _table(NOTE_FIELDS, shown)
            | {"shown": len(shown), "total": len(tables.note_rows)},
            "errors": [error.model_dump() for error in tables.errors],
        }

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
                if len(self._step_snapshots) > 12:
                    self._step_snapshots.popitem(last=False)
            return self._step_snapshots[index]

    def nlp_date(self) -> str:
        if self.artifact:
            return self.artifact["run"]["started_at"][:10]
        start = (self._events[0].payload or {}) if self._events else {}
        return start.get("started_at", "1970-01-01")[:10]

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
            self.follow_live = False
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
            self.follow_live = False
            self.playing = playing and not self._is_last_cursor()
            return self._view()

    def follow(self) -> dict[str, Any]:
        with self._lock:
            self.follow_live = self.mode == "live"
            self.playing = False
            self._cursor = self._max_cursor()
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
            "follow_live": self.follow_live,
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
        events: Iterable[DemoEvent] | DemoRun,
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
        self.job = events if isinstance(events, DemoRun) else None
        self._source = None if self.job else iter(events)
        self._done = False
        self._thread: threading.Thread | None = None
        self._listener: Any = None

    def set_listener(self, listener: Any) -> None:
        """Register a ``callable(message: dict)`` invoked as the run progresses."""
        self._listener = listener

    def start(self) -> None:
        """Start once; browser reconnects never restart execution."""
        with self._lock:
            if self._thread is not None:
                return
            self.follow_live = True
            self._thread = threading.Thread(target=self._run, name="cipoc-demo-live")
            self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        try:
            if self.job:
                self.job.execute(self.append)
                self.artifact = self.job.artifact
            else:
                for event in self._source:
                    self.append(event)
        except Exception as error:
            if self.job:
                self.job.status = "failed"
                self.job.error = f"Live worker failed ({type(error).__name__})."
        finally:
            with self._lock:
                self._done = True
            self._emit_live()

    def append(self, event: DemoEvent) -> None:
        """Add one produced event and refresh the derived step list."""
        self._append(event)
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
            if self.job:
                data.update(self.job.meta())
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
        self._publish_lock = threading.Lock()
        self._pending: dict[str, Any] | None = None
        self._scheduled = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that owns the subscriber queues (idempotent)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _deliver(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)

    def publish(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is not None:
            # Bound the event loop's callback backlog as well as each queue.
            with self._publish_lock:
                self._pending = message
                if self._scheduled:
                    return
                self._scheduled = True
            try:
                loop.call_soon_threadsafe(self._flush)
            except RuntimeError:
                with self._publish_lock:
                    self._scheduled = False
        else:
            self._deliver(message)

    def _flush(self) -> None:
        with self._publish_lock:
            message, self._pending = self._pending, None
            self._scheduled = False
        if message is not None:
            self._deliver(message)


def build_app(session: DemoSession) -> FastAPI:
    """Build the FastAPI app serving ``session`` (replay or live)."""
    broadcaster = Broadcaster()

    @asynccontextmanager
    async def lifespan(app):
        broadcaster.bind_loop(asyncio.get_running_loop())
        async def playback():
            while True:
                await asyncio.sleep(3)
                if session.playing:
                    await asyncio.to_thread(session.next)
                    broadcaster.publish({"type": "refresh"})
        clock = asyncio.create_task(playback())
        try:
            yield
        finally:
            clock.cancel()
            try:
                await clock
            except asyncio.CancelledError:
                pass
            if isinstance(session, LiveDemoSession):
                await asyncio.to_thread(session.join)

    app = FastAPI(title="CIPOC Demo", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.session = session
    app.state.broadcaster = broadcaster

    # Live mode pushes step-availability notifications to every viewer; wire the
    # session's listener to the broadcaster (which marshals onto the event loop).
    if isinstance(session, LiveDemoSession):
        session.set_listener(lambda message: broadcaster.publish({"type": "refresh"}))

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def _broadcast_view() -> dict[str, Any]:
        view = session.view()
        broadcaster.publish({"type": "refresh"})
        return view

    # --- Static run data ---
    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(session.meta())

    @app.get("/api/events")
    def events(after: int = -1) -> JSONResponse:
        return JSONResponse(session.events_payload(after))

    @app.post("/api/start")
    def start() -> JSONResponse:
        if not isinstance(session, LiveDemoSession):
            raise HTTPException(409, "This is a replay.")
        session.start()
        return JSONResponse(session.meta())

    @app.post("/api/follow")
    def follow() -> JSONResponse:
        session.follow()
        return JSONResponse(_broadcast_view())

    @app.get("/api/artifact")
    def artifact() -> JSONResponse:
        value = session.job.artifact if isinstance(session, LiveDemoSession) and session.job else session.artifact
        if value is None:
            raise HTTPException(409, "No canonical artifact is available yet.")
        filename = "run_result.json" if value["run"]["status"] == "completed" else "run_failure.json"
        return JSONResponse(value, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/api/recording")
    def recording() -> Response:
        content = "".join(json.dumps(event.to_dict(), allow_nan=False) + "\n" for event in session.events)
        return Response(content, media_type="application/x-ndjson", headers={
            "Content-Disposition": 'attachment; filename="demo_trace.jsonl"',
        })

    @app.get("/api/export")
    def export(person_id: int) -> Response:
        import csv
        import io
        import zipfile
        from cipoc.models import OrchestratorRunResult
        value = session.job.artifact if isinstance(session, LiveDemoSession) and session.job else session.artifact
        if not value or value["run"]["status"] != "completed":
            raise HTTPException(409, "A completed canonical run is required.")
        result = OrchestratorRunResult.model_validate(value)
        tables = OmopExporter(person_id=person_id, nlp_date=result.run.started_at.date()).build(
            notes=result.corpus.note_corpus.values(), case=result.case,
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, fields, rows in (("note.csv", NOTE_FIELDS, tables.note_rows), ("note_nlp.csv", NOTE_NLP_FIELDS, tables.note_nlp_rows)):
                text = io.StringIO(newline="")
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row.model_dump() for row in rows)
                bundle.writestr(name, text.getvalue())
            bundle.writestr("omop_errors.json", json.dumps({"errors": [e.model_dump() for e in tables.errors]}))
        return Response(buffer.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": 'attachment; filename="omop_staging.zip"',
        })

    @app.get("/api/steps")
    def steps() -> JSONResponse:
        return JSONResponse(session.steps_payload())

    @app.get("/api/update")
    def update(after: int = -1) -> JSONResponse:
        # One revision for cursor, steps, and incremental events prevents the
        # map from displaying a snapshot newer than its local event history.
        with session._lock:
            value = {"meta": session.meta(), "view": session.view(),
                     "steps": session.steps_payload(), "events": session.events_payload(after)}
        return JSONResponse(value)

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

    @app.get("/api/omop/{item_id}")
    def omop(item_id: int, seq: int | None = None) -> JSONResponse:
        """OMOP NOTE / NOTE_NLP rows for one variable at one cursor position.

        A variable with no valid extraction — structured-data, not-found,
        blocked, or simply not reached yet — is a 200 with empty tables, not a
        404. "This coding produces no NOTE_NLP row, and here is why" is a thing
        the demo wants to be able to say. Only an item the case never requested
        is a 404.
        """
        if seq is None:
            seq = session.events[-1].seq if session.events else 0
        if not session.has_item(item_id):
            raise HTTPException(status_code=404, detail=f"No variable {item_id} in this run.")
        return JSONResponse(session.omop_at_seq(item_id, seq))

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
            yield _sse({"type": "refresh"})
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


def _table(fields: tuple[str, ...], rows: Iterable[Any]) -> dict[str, Any]:
    """Column-oriented shape: field order comes from the row model, once, here.

    The browser renders a table straight from ``columns`` + positional ``rows``
    and never has to know an OMOP column name, so the CSV header and the on-screen
    header cannot disagree.
    """
    return {
        "columns": list(fields),
        "rows": [[_cell(row_data.get(field)) for field in fields] for row_data in (row.model_dump() for row in rows)],
    }


def _cell(value: Any) -> str:
    """Every cell reaches the browser as a string; ``None`` renders as blank."""
    return "" if value is None else str(value)


def _empty_omop(item_id: int, seq: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "seq": seq,
        "person_id": DEMO_PERSON_ID,
        "status": None,
        "note_nlp": {"columns": list(NOTE_NLP_FIELDS), "rows": []},
        "note": {"columns": list(NOTE_FIELDS), "rows": [], "shown": 0, "total": 0},
        "errors": [],
    }


def _variable_status(case: Any, item_id: int) -> str | None:
    """The variable's own status, so an empty table can explain itself."""
    result = case.variable_results.get(item_id)
    if result is None:
        return None
    status = result.status
    return getattr(status, "value", status)


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


def load_replay_session(trace_path: str | Path, *, description: str | None = None, artifact_path: Path | None = None) -> DemoSession:
    """Build a replay :class:`DemoSession` from a recorded trace file."""
    events = read_trace(trace_path)
    label = description or f"Replay of {Path(trace_path).name}"
    artifact = None
    if artifact_path is not None:
        from cipoc.models import OrchestratorRunResult, OrchestratorRunFailure
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
        model = OrchestratorRunResult if "case" in raw else OrchestratorRunFailure
        artifact = model.model_validate(raw).model_dump(mode="json")
        recorded_id = (events[-1].payload or {}).get("run_id") if events else None
        if recorded_id != artifact["run"]["run_id"]:
            raise ValueError("The recording and canonical artifact must have the same run UUID.")
    return DemoSession(events, description=label, artifact=artifact)


__all__ = [
    "DemoSession",
    "LiveDemoSession",
    "Broadcaster",
    "build_app",
    "load_replay_session",
    "overview_chart",
    "WEB_DIR",
]
