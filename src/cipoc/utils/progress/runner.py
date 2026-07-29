"""LangGraph stream runner and progress-renderer selection."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time
from typing import Any, AsyncIterator, Callable, Iterable, Mapping, Sequence, TextIO

from langgraph.graph.state import CompiledStateGraph

from cipoc.tools import GroupNode

from .events import ProgressEvent, normalize
from .layout import build_rows, render_lines
from .model import ProgressModel, Snapshot, TaskKind
from .renderers import AnsiAltScreen, NotebookDisplay, PlainLog, Renderer


_MIN_TTY_HEIGHT = 12
_PLAIN_POLL_INTERVAL = 0.05
_REPAINT_JOIN_TIMEOUT = 1.0

#: What the dashboard itself needs: durable state plus task lifecycle. Callers
#: wanting per-node writes (``astream_results``) add "updates" on top; the model
#: ignores that kind, so widening the request never changes what is painted.
DEFAULT_STREAM_MODE = ("values", "tasks")


def _notebook_kernel_present() -> bool:
    """Return whether execution is inside a Jupyter-compatible IPython kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        return False
    if shell is None:
        return False
    return (
        getattr(shell, "kernel", None) is not None
        or shell.__class__.__name__ == "ZMQInteractiveShell"
        or bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    )


def _select_renderer(
    stream: TextIO,
    *,
    notebook: bool | None = None,
    size_provider: Callable[[], os.terminal_size] | None = None,
) -> Renderer:
    """Choose notebook, full-screen TTY, or append-only output."""
    if notebook is None:
        notebook = _notebook_kernel_present()
    if notebook:
        return NotebookDisplay()

    try:
        interactive = bool(stream.isatty())
    except (AttributeError, OSError):
        interactive = False
    if not interactive:
        return PlainLog(stream)

    size_provider = size_provider or (
        lambda: shutil.get_terminal_size((100, 24))
    )
    try:
        height = int(size_provider().lines)
    except (AttributeError, OSError, ValueError):
        height = 24
    if height < _MIN_TTY_HEIGHT:
        return PlainLog(stream)
    return AnsiAltScreen(stream, size_provider=size_provider)


class _RepaintLoop:
    """Paint the latest immutable snapshot without blocking graph ingestion."""

    def __init__(self, renderer: Renderer, snapshot: Snapshot):
        self.renderer = renderer
        self.snapshot = snapshot
        self.tick = 0
        self.error: Exception | None = None
        self.cleanup_interrupt: BaseException | None = None
        self._stop = threading.Event()
        self._pending: queue.Queue[Snapshot] | None = (
            queue.Queue() if isinstance(renderer, PlainLog) else None
        )
        if self._pending is not None:
            self._pending.put(snapshot)
        self._thread = threading.Thread(
            target=self._run,
            name="cipoc-progress-renderer",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def publish(self, snapshot: Snapshot) -> None:
        # A reference assignment is atomic in CPython. The model remains owned by
        # the stream thread; the painter only ever sees frozen snapshots.
        self.snapshot = snapshot
        if self._pending is not None:
            self._pending.put(snapshot)

    @property
    def started(self) -> bool:
        return self._thread.ident is not None

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> BaseException | None:
        """Request teardown without allowing blocked display I/O to block the graph."""
        self._stop.set()
        interrupted: BaseException | None = None
        if not self.started:
            return None
        deadline = time.monotonic() + _REPAINT_JOIN_TIMEOUT
        while self._thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._thread.join(timeout=min(0.1, remaining))
            except (KeyboardInterrupt, SystemExit) as error:
                interrupted = interrupted or error
        return interrupted

    def _run(self) -> None:
        try:
            if self._pending is not None:
                self._run_plain_log()
                return
            interval = max(self.renderer.min_interval, _PLAIN_POLL_INTERVAL)
            while not self._stop.is_set():
                try:
                    self.renderer.paint(
                        self.snapshot,
                        now=time.monotonic(),
                        tick=self.tick,
                    )
                except (KeyboardInterrupt, SystemExit) as error:
                    self.cleanup_interrupt = error
                    self._stop.wait()
                    return
                except Exception as error:
                    self.error = error
                    self._stop.wait()
                    return
                self.tick += 1
                self._stop.wait(interval)
        finally:
            interrupt = _finalize_renderer(self.renderer, self.snapshot, self.tick)
            self.cleanup_interrupt = self.cleanup_interrupt or interrupt

    def _run_plain_log(self) -> None:
        assert self._pending is not None
        while not self._stop.is_set() or not self._pending.empty():
            try:
                snapshot = self._pending.get(timeout=_PLAIN_POLL_INTERVAL)
                self.renderer.paint(snapshot, now=time.monotonic(), tick=self.tick)
            except queue.Empty:
                continue
            except (KeyboardInterrupt, SystemExit) as error:
                self.cleanup_interrupt = error
                self._stop.wait()
                return
            except Exception as error:
                self.error = error
                self._stop.wait()
                return
            self.tick += 1


def _write_persistent_summary(
    renderer: AnsiAltScreen,
    snapshot: Snapshot,
    *,
    now: float,
) -> None:
    """Print the expanded final table after the alternate screen is restored."""
    width, _ = renderer.viewport()
    lines = render_lines(build_rows(snapshot, width, None, now=now))
    renderer.stream.write("\n".join(lines) + "\n")
    renderer.stream.flush()


def _finalize_renderer(
    renderer: Renderer,
    snapshot: Snapshot,
    tick: int,
) -> BaseException | None:
    """Force the final frame and close one renderer from its owning thread."""
    interrupted: BaseException | None = None
    now = time.monotonic()
    try:
        renderer.paint(snapshot, now=now, tick=tick, final=True)
    except (KeyboardInterrupt, SystemExit) as error:
        interrupted = error
    except Exception:
        pass

    # One signal may interrupt the restoration write itself. Retry enough to
    # restore the terminal while still bounding pathological renderer behavior.
    for _ in range(3):
        try:
            renderer.close()
            break
        except (KeyboardInterrupt, SystemExit) as error:
            interrupted = interrupted or error
        except Exception:
            break

    if isinstance(renderer, AnsiAltScreen):
        try:
            _write_persistent_summary(renderer, snapshot, now=now)
        except (KeyboardInterrupt, SystemExit) as error:
            interrupted = interrupted or error
        except Exception:
            pass
    return interrupted


class _ProgressSession:
    """Per-run dashboard bookkeeping: the model, the renderer, the painter thread.

    Exists so the sync and async runners differ in exactly one thing — whether
    items are pulled with ``for`` over ``graph.stream`` or ``async for`` over
    ``graph.astream``. Everything else (event normalization, ingestion, snapshot
    publication, teardown ordering, which interrupt wins) is shared, so the two
    paths cannot drift.

    The painter stays a thread in both modes. It never touches the event loop,
    and it is what advances ``tick`` for the spinner; a task doing blocking
    terminal writes on the loop would be strictly worse.
    """

    def __init__(
        self,
        graph_input: Any,
        *,
        subgraphs: bool = False,
        stream_mode: Sequence[str] | None = None,
        description: str = "Agent",
        node_kinds: Mapping[str, TaskKind] | None = None,
        show_branches: bool = False,
        target_groups: Any = None,
        group_hierarchy: Iterable[GroupNode] | None = None,
        show_note_counts: bool = False,
    ):
        self.subgraphs = subgraphs
        self.stream_mode = list(stream_mode or DEFAULT_STREAM_MODE)
        self.model = ProgressModel(
            description,
            time.monotonic(),
            target_groups=target_groups,
            group_hierarchy=group_hierarchy,
            graph_input=graph_input,
            node_kinds=node_kinds,
            show_note_counts=show_note_counts,
            include_input_group=show_branches,
        )
        self.renderer = _select_renderer(sys.stdout)
        self.painter = _RepaintLoop(self.renderer, self.model.snapshot())
        self.final_result: Any = None

    def start(self) -> None:
        try:
            self.painter.start()
        except Exception as error:
            # Progress is optional. A thread creation failure must not prevent
            # the graph itself from running; the caller will render teardown.
            self.painter.error = error

    def handle(self, raw_item: Any) -> ProgressEvent | None:
        """Ingest one raw stream item and repaint. Returns the normalized event
        so a streaming caller can consume it, or ``None`` when unusable."""
        event = normalize(raw_item, subgraphs=self.subgraphs)
        if event is None:
            return None
        self.model.ingest(event, time.monotonic())
        if event.kind == "values" and event.is_root:
            self.final_result = event.payload
        self.painter.publish(self.model.snapshot())
        return event

    def finish(self) -> Any:
        """Close out a successful run and return its last root state."""
        if self.final_result is None:
            raise RuntimeError("Graph produced no final state.")
        self.model.finish()
        self.painter.publish(self.model.snapshot())
        return self.final_result

    def fail(self, error: BaseException) -> None:
        self.model.fail(error)
        self.painter.publish(self.model.snapshot())

    def teardown(self, run_error: BaseException | None) -> None:
        """Stop the painter and restore the terminal. A cleanup interrupt is only
        raised when the run itself succeeded — a real graph error outranks it."""
        cleanup_interrupt: BaseException | None = None
        try:
            cleanup_interrupt = self.painter.stop()
        except (KeyboardInterrupt, SystemExit) as error:
            cleanup_interrupt = error
        except Exception:
            pass
        if not self.painter.started:
            cleanup_interrupt = cleanup_interrupt or _finalize_renderer(
                self.renderer,
                self.model.snapshot(),
                self.painter.tick,
            )
        elif not self.painter.is_alive:
            cleanup_interrupt = cleanup_interrupt or self.painter.cleanup_interrupt
        if run_error is None and cleanup_interrupt is not None:
            raise cleanup_interrupt


def run_with_progress(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    subgraphs: bool = False,
    stream_mode: Sequence[str] | None = None,
    description: str = "Agent",
    node_kinds: Mapping[str, TaskKind] | None = None,
    show_branches: bool = False,
    target_groups: Any = None,
    group_hierarchy: Iterable[GroupNode] | None = None,
    show_note_counts: bool = False,
) -> Any:
    """Run ``graph`` with live progress and return its last root state.

    ``show_branches`` preserves the existing public API and identifies a
    standalone extractor run; its requested variables are discovered from
    ``graph_input`` by :class:`ProgressModel`. ``group_hierarchy`` optionally
    restores nesting lost by the orchestrator's flattened planning groups.
    """
    session = _ProgressSession(
        graph_input,
        subgraphs=subgraphs,
        stream_mode=stream_mode,
        description=description,
        node_kinds=node_kinds,
        show_branches=show_branches,
        target_groups=target_groups,
        group_hierarchy=group_hierarchy,
        show_note_counts=show_note_counts,
    )
    run_error: BaseException | None = None
    try:
        session.start()
        for raw_item in graph.stream(
            graph_input,
            stream_mode=session.stream_mode,
            subgraphs=subgraphs,
        ):
            session.handle(raw_item)
        return session.finish()
    except BaseException as error:
        run_error = error
        session.fail(error)
        raise
    finally:
        session.teardown(run_error)


async def astream_with_progress(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    subgraphs: bool = False,
    stream_mode: Sequence[str] | None = None,
    description: str = "Agent",
    node_kinds: Mapping[str, TaskKind] | None = None,
    show_branches: bool = False,
    target_groups: Any = None,
    group_hierarchy: Iterable[GroupNode] | None = None,
    show_note_counts: bool = False,
) -> AsyncIterator[ProgressEvent]:
    """Drive ``graph.astream`` with live progress, yielding each normalized event.

    The dashboard is painted as a side effect, exactly as in the sync runner; the
    yielded events are what lets a caller act on results before the run ends.
    Drain it with :func:`arun_with_progress` when only the final state is wanted.
    """
    session = _ProgressSession(
        graph_input,
        subgraphs=subgraphs,
        stream_mode=stream_mode,
        description=description,
        node_kinds=node_kinds,
        show_branches=show_branches,
        target_groups=target_groups,
        group_hierarchy=group_hierarchy,
        show_note_counts=show_note_counts,
    )
    run_error: BaseException | None = None
    try:
        session.start()
        async for raw_item in graph.astream(
            graph_input,
            stream_mode=session.stream_mode,
            subgraphs=subgraphs,
        ):
            event = session.handle(raw_item)
            if event is not None:
                yield event
        session.finish()
    except BaseException as error:
        run_error = error
        session.fail(error)
        raise
    finally:
        session.teardown(run_error)


async def astream_events(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    subgraphs: bool = False,
    stream_mode: Sequence[str] | None = None,
) -> AsyncIterator[ProgressEvent]:
    """Normalized events from ``graph.astream``, with no dashboard attached.

    The progress-free half of :func:`astream_with_progress`, so a caller that
    wants incremental results inside a notebook cell (or a test) is not forced to
    paint a terminal display to get them.
    """
    async for raw_item in graph.astream(
        graph_input,
        stream_mode=list(stream_mode or DEFAULT_STREAM_MODE),
        subgraphs=subgraphs,
    ):
        event = normalize(raw_item, subgraphs=subgraphs)
        if event is not None:
            yield event


async def arun_with_progress(graph: CompiledStateGraph, graph_input: Any, **kwargs) -> Any:
    """Async twin of :func:`run_with_progress`: run to completion, return the last
    root state. A coroutine with no ``asyncio.run`` inside, so the caller owns the
    loop."""
    final_result: Any = None
    async for event in astream_with_progress(graph, graph_input, **kwargs):
        if event.kind == "values" and event.is_root:
            final_result = event.payload
    return final_result


__all__ = [
    "DEFAULT_STREAM_MODE",
    "arun_with_progress",
    "astream_events",
    "astream_with_progress",
    "run_with_progress",
]
