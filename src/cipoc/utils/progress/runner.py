"""LangGraph stream runner and progress-renderer selection."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, TextIO

from langgraph.graph.state import CompiledStateGraph

from cipoc.tools import GroupNode

from .events import normalize
from .layout import build_rows
from .model import ProgressModel, Snapshot, TaskKind
from .renderers import AnsiAltScreen, NotebookDisplay, PlainLog, Renderer, ansi_lines


_MIN_TTY_HEIGHT = 12
_PLAIN_POLL_INTERVAL = 0.05
_REPAINT_JOIN_TIMEOUT = 1.0


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
        self._hold_final_frame = False
        self._final_frame_ready = threading.Event()
        self._release_final_frame = threading.Event()
        self._final_frame_painted = False
        self._write_summary = True
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

    def stop(self, *, hold_final_frame: bool = False) -> BaseException | None:
        """Request teardown without allowing blocked display I/O to block the graph."""
        self._hold_final_frame = hold_final_frame
        self._stop.set()
        interrupted: BaseException | None = None
        if not self.started:
            return None
        deadline = time.monotonic() + _REPAINT_JOIN_TIMEOUT
        while self._thread.is_alive():
            if hold_final_frame and self._final_frame_ready.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._thread.join(timeout=min(0.1, remaining))
            except (KeyboardInterrupt, SystemExit) as error:
                interrupted = interrupted or error
        if hold_final_frame and not self.waiting_for_report:
            self._release_final_frame.set()
        return interrupted

    @property
    def waiting_for_report(self) -> bool:
        return (
            self._final_frame_ready.is_set()
            and self._final_frame_painted
            and self._thread.is_alive()
        )

    def release_final_frame(self, *, write_summary: bool = True) -> BaseException | None:
        """Allow a held final frame to restore the terminal and finish teardown."""
        self._write_summary = write_summary
        self._release_final_frame.set()
        interrupted: BaseException | None = None
        deadline = time.monotonic() + _REPAINT_JOIN_TIMEOUT
        while self._thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._thread.join(timeout=min(0.1, remaining))
            except (KeyboardInterrupt, SystemExit) as error:
                interrupted = interrupted or error
        return interrupted or self.cleanup_interrupt

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
            if self._hold_final_frame:
                now = time.monotonic()
                interrupt, painted = _paint_final_frame(
                    self.renderer,
                    self.snapshot,
                    self.tick,
                    report_prompt=True,
                    now=now,
                )
                self.cleanup_interrupt = self.cleanup_interrupt or interrupt
                self._final_frame_painted = painted
                self._final_frame_ready.set()
                if painted:
                    self._release_final_frame.wait()
                interrupt = _close_renderer(
                    self.renderer,
                    self.snapshot,
                    now=now,
                    write_summary=self._write_summary,
                )
            else:
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
    rows = build_rows(snapshot, width, None, now=now)
    lines = ansi_lines(rows, color=renderer.color)
    renderer.stream.write("\n".join(lines) + "\n")
    renderer.stream.flush()


def _paint_final_frame(
    renderer: Renderer,
    snapshot: Snapshot,
    tick: int,
    *,
    report_prompt: bool = False,
    now: float | None = None,
) -> tuple[BaseException | None, bool]:
    """Paint one final frame, reporting interruptions without leaking teardown."""
    now = time.monotonic() if now is None else now
    try:
        painted = renderer.paint(
            snapshot,
            now=now,
            tick=tick,
            final=True,
            report_prompt=report_prompt,
        )
        return None, painted
    except (KeyboardInterrupt, SystemExit) as error:
        return error, False
    except Exception:
        return None, False


def _close_renderer(
    renderer: Renderer,
    snapshot: Snapshot,
    *,
    now: float,
    write_summary: bool = True,
) -> BaseException | None:
    """Close a renderer and optionally print its persistent expanded report."""
    interrupted: BaseException | None = None

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

    if write_summary and isinstance(renderer, AnsiAltScreen):
        try:
            _write_persistent_summary(renderer, snapshot, now=now)
        except (KeyboardInterrupt, SystemExit) as error:
            interrupted = interrupted or error
        except Exception:
            pass
    return interrupted


def _finalize_renderer(
    renderer: Renderer,
    snapshot: Snapshot,
    tick: int,
) -> BaseException | None:
    """Force the final frame and close one renderer from its owning thread."""
    now = time.monotonic()
    interrupted, _ = _paint_final_frame(renderer, snapshot, tick, now=now)
    close_interrupt = _close_renderer(renderer, snapshot, now=now)
    return interrupted or close_interrupt


def run_with_progress(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    config: Mapping[str, Any] | None = None,
    subgraphs: bool = False,
    description: str = "Agent",
    node_kinds: Mapping[str, TaskKind] | None = None,
    show_branches: bool = False,
    target_groups: Any = None,
    group_hierarchy: Iterable[GroupNode] | None = None,
    show_note_counts: bool = False,
    pause_before_summary: bool = False,
) -> Any:
    """Run ``graph`` with live progress and return its last root state.

    ``show_branches`` preserves the existing public API and identifies a
    standalone extractor run; its requested variables are discovered from
    ``graph_input`` by :class:`ProgressModel`. ``group_hierarchy`` optionally
    restores nesting lost by the orchestrator's flattened planning groups.
    """
    started_at = time.monotonic()
    model = ProgressModel(
        description,
        started_at,
        target_groups=target_groups,
        group_hierarchy=group_hierarchy,
        graph_input=graph_input,
        node_kinds=node_kinds,
        show_note_counts=show_note_counts,
        include_input_group=show_branches,
    )
    renderer = _select_renderer(sys.stdout)
    painter = _RepaintLoop(renderer, model.snapshot())
    final_result: Any = None
    run_error: BaseException | None = None

    try:
        try:
            painter.start()
        except Exception as error:
            # Progress is optional. A thread creation failure must not prevent
            # the graph itself from running; the caller will render teardown.
            painter.error = error
        stream_kwargs = {
            "stream_mode": ["values", "tasks"],
            "subgraphs": subgraphs,
        }
        if config is not None:
            stream_kwargs["config"] = config
        for raw_item in graph.stream(graph_input, **stream_kwargs):
            event = normalize(raw_item, subgraphs=subgraphs)
            if event is None:
                continue
            model.ingest(event, time.monotonic())
            if event.kind == "values" and event.is_root:
                final_result = event.payload
            painter.publish(model.snapshot())

        if final_result is None:
            raise RuntimeError("Graph produced no final state.")
        model.finish()
        painter.publish(model.snapshot())
        return final_result
    except BaseException as error:
        run_error = error
        model.fail(error)
        painter.publish(model.snapshot())
        raise
    finally:
        cleanup_interrupt: BaseException | None = None
        pause_interrupt: BaseException | None = None
        hold_final_frame = (
            pause_before_summary
            and run_error is None
            and isinstance(renderer, AnsiAltScreen)
        )
        try:
            cleanup_interrupt = painter.stop(hold_final_frame=hold_final_frame)
        except (KeyboardInterrupt, SystemExit) as error:
            cleanup_interrupt = error
        except Exception:
            pass
        if not painter.started:
            cleanup_interrupt = cleanup_interrupt or _finalize_renderer(
                renderer,
                model.snapshot(),
                painter.tick,
            )
        elif hold_final_frame and painter.waiting_for_report:
            try:
                if cleanup_interrupt is None:
                    input()
            except EOFError:
                pass
            except (KeyboardInterrupt, SystemExit) as error:
                pause_interrupt = error
            finally:
                try:
                    cleanup_interrupt = cleanup_interrupt or painter.release_final_frame(
                        write_summary=pause_interrupt is None
                    )
                except (KeyboardInterrupt, SystemExit) as error:
                    cleanup_interrupt = cleanup_interrupt or error
                except Exception:
                    pass
        elif not painter.is_alive:
            cleanup_interrupt = cleanup_interrupt or painter.cleanup_interrupt
        if run_error is None:
            if pause_interrupt is not None:
                raise pause_interrupt
            if cleanup_interrupt is not None:
                raise cleanup_interrupt


__all__ = ["run_with_progress"]
