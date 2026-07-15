"""Live terminal progress display for LangGraph agent runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import sys
import time
from typing import Any, TextIO

from langgraph.graph.state import CompiledStateGraph


class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    MAIN = "\033[38;5;74m"
    WHITE = "\033[97m"
    GREY = "\033[90m"
    SUCCESS = "\033[38;5;33m"
    ERROR = "\033[38;5;208m"
    ACTIVE = "\033[36m"


@dataclass
class _Task:
    label: str
    started_at: float
    status: str = "active"
    finished_at: float | None = None
    error: str | None = None


def _task_label(namespace: tuple[str, ...], node_name: str) -> str:
    scope = [part.split(":", maxsplit=1)[0] for part in namespace]
    names = [*scope, node_name]
    return " > ".join(name.replace("_", " ").title() for name in names)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width == 1:
        return "…"
    return value[: width - 1] + "…"


class _ProgressDisplay:
    """Render a compact live dashboard without requiring a UI dependency."""

    def __init__(self, description: str, stream: TextIO | None = None):
        self.description = description
        self.stream = stream or sys.stdout
        self.started_at = time.monotonic()
        self.tasks: dict[str, _Task] = {}
        self._order: list[str] = []
        self._draw_count = 0
        self._rendered_lines = 0
        self._notebook = _in_notebook()
        self._interactive = self._notebook or bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self._color = self._interactive and "NO_COLOR" not in os.environ
        self._fatal_error: str | None = None

    def _style(self, text: str, *styles: str) -> str:
        if not self._color:
            return text
        return "".join(styles) + text + _Color.RESET

    def start(self, task_id: str, label: str) -> None:
        self.tasks[task_id] = _Task(label=label, started_at=time.monotonic())
        self._order.append(task_id)
        if self._interactive:
            self.draw()
        else:
            self._write(f"  > {label}")

    def finish(self, task_id: str, label: str, error: Any = None) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            task = _Task(label=label, started_at=time.monotonic())
            self.tasks[task_id] = task
            self._order.append(task_id)

        task.finished_at = time.monotonic()
        task.status = "error" if error is not None else "success"
        task.error = str(error) if error is not None else None
        if self._interactive:
            self.draw()
        else:
            marker = "x" if error is not None else "v"
            detail = f": {error}" if error is not None else ""
            self._write(f"  {marker} {label}{detail}")

    def fail(self, error: BaseException) -> None:
        self._fatal_error = str(error)
        now = time.monotonic()
        for task in self.tasks.values():
            if task.status == "active":
                task.status = "error"
                task.finished_at = now
                task.error = "Run stopped before this task completed."
        self.draw(final=True)

    def complete(self) -> None:
        self.draw(final=True)

    def _width(self) -> int:
        return max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))

    def _duration(self, task: _Task) -> str:
        end = task.finished_at or time.monotonic()
        elapsed = end - task.started_at
        return f"{elapsed:.1f}s" if elapsed >= 0.1 else "<0.1s"

    def _lane(self, width: int) -> str:
        visible_ids = self._order[-5:]
        parts: list[str] = []
        for task_id in visible_ids:
            task = self.tasks[task_id]
            if task.status == "success":
                icon, color = "✓", _Color.SUCCESS
            elif task.status == "error":
                icon, color = "✗", _Color.ERROR
            else:
                icon, color = "●", _Color.ACTIVE
            leaf_label = task.label.rsplit(" > ", maxsplit=1)[-1]
            parts.append(self._style(f"{icon} {_truncate(leaf_label, 20)}", color))

        separator = self._style("  →  ", _Color.GREY)
        lane = separator.join(parts)
        if len(self._order) > len(visible_ids):
            lane = self._style("…  ", _Color.GREY) + lane
        return "  " + lane

    def _render(self, final: bool) -> list[str]:
        width = self._width()
        elapsed = time.monotonic() - self.started_at
        tasks = list(self.tasks.values())
        succeeded = sum(task.status == "success" for task in tasks)
        failed = sum(task.status == "error" for task in tasks)
        active = sum(task.status == "active" for task in tasks)

        title = (
            f"  {self._style('CIPOC', _Color.BOLD, _Color.MAIN)}"
            f"{self._style(' / ' + self.description, _Color.GREY)}"
        )
        stats = [f"{succeeded} complete"]
        if failed:
            stats.append(self._style(f"{failed} failed", _Color.ERROR))
        if active and not final:
            stats.append(self._style(f"{active} running", _Color.ACTIVE))
        stats.append(f"{elapsed:.1f}s")

        pulse_width = min(24, max(12, width // 4))
        filled = min(succeeded + failed, pulse_width)
        if active and not final and filled < pulse_width:
            activity = "█" * filled + "▓" + "░" * (pulse_width - filled - 1)
        else:
            activity = "█" * filled + "░" * (pulse_width - filled)
        progress = (
            f"  {self._style(activity, _Color.MAIN)}"
            f"  {self._style('  '.join(stats), _Color.WHITE)}"
        )

        lines = ["", title, "", progress, "", self._lane(width), ""]
        lines.append("  " + self._style("━" * (width - 4), _Color.GREY))

        recent_complete = [task_id for task_id in self._order if self.tasks[task_id].status != "active"][-5:]
        active_ids = [task_id for task_id in self._order if self.tasks[task_id].status == "active"]
        visible_ids = recent_complete + active_ids

        if not visible_ids:
            lines.append(f"  {self._style('●', _Color.ACTIVE)} {self._style('Starting…', _Color.DIM)}")

        label_width = max(20, width - 20)
        for task_id in visible_ids:
            task = self.tasks[task_id]
            label = _truncate(task.label, label_width)
            if task.status == "success":
                icon = self._style("✓", _Color.SUCCESS)
                timing = self._style(self._duration(task), _Color.DIM)
            elif task.status == "error":
                icon = self._style("✗", _Color.ERROR)
                timing = self._style(self._duration(task), _Color.ERROR)
            else:
                icon = self._style("●", _Color.ACTIVE, _Color.BOLD)
                timing = self._style("running", _Color.ACTIVE)
            padded_label = f"{label:<{label_width}}"
            lines.append(f"  {icon} {self._style(padded_label, _Color.WHITE)}  {timing}")
            if task.error:
                lines.append(f"    {self._style('↳ ' + _truncate(task.error, width - 8), _Color.ERROR)}")

        if self._fatal_error:
            lines.extend(
                [
                    "",
                    f"  {self._style('✗ Run failed', _Color.BOLD, _Color.ERROR)}"
                    f"  {self._style(_truncate(self._fatal_error, width - 20), _Color.ERROR)}",
                ]
            )
        elif final:
            count = succeeded + failed
            step_word = "step" if count == 1 else "steps"
            lines.extend(
                [
                    "",
                    f"  {self._style('Done.', _Color.BOLD, _Color.SUCCESS)}"
                    f"  {count} {step_word} in {elapsed:.1f}s",
                ]
            )
        return lines

    def draw(self, *, final: bool = False) -> None:
        if not self._interactive:
            if self._draw_count == 0:
                self._write(f"CIPOC / {self.description}")
            if final:
                elapsed = time.monotonic() - self.started_at
                failed = sum(task.status == "error" for task in self.tasks.values())
                finished = sum(task.status != "active" for task in self.tasks.values())
                if self._fatal_error:
                    self._write(f"  failed after {elapsed:.1f}s: {self._fatal_error}")
                else:
                    suffix = f", {failed} failed" if failed else ""
                    self._write(f"  complete: {finished} steps{suffix} in {elapsed:.1f}s")
            self._draw_count += 1
            return

        lines = self._render(final)
        if self._notebook:
            from IPython.display import clear_output

            clear_output(wait=True)
            self._write("\n".join(lines))
        else:
            self._redraw_terminal(lines)
        self._draw_count += 1

    def _redraw_terminal(self, lines: list[str]) -> None:
        line_count = max(len(lines), self._rendered_lines)
        if self._rendered_lines:
            self.stream.write(f"\033[{self._rendered_lines}A")
        for index in range(line_count):
            line = lines[index] if index < len(lines) else ""
            self.stream.write(f"\r\033[2K{line}\n")
        self.stream.flush()
        self._rendered_lines = line_count

    def _write(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()


def run_with_progress(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    subgraphs: bool = False,
    description: str = "Agent",
) -> Any:
    """Run a LangGraph graph with live progress and return its final state.

    Set ``subgraphs=True`` to include nodes inside compiled subgraphs. This is
    useful for graphs such as the extractor's per-variable branches.
    """
    final_result: Any = None
    display = _ProgressDisplay(description)
    display.draw()

    try:
        stream = graph.stream(
            graph_input,
            stream_mode=["tasks", "values"],
            subgraphs=subgraphs,
        )

        for item in stream:
            if subgraphs:
                namespace, mode, event = item
            else:
                mode, event = item
                namespace = ()

            if mode == "values":
                if not namespace:
                    final_result = event
                continue

            label = _task_label(namespace, event["name"])
            task_id = event["id"]
            if "input" in event:
                display.start(task_id, label)
            else:
                display.finish(task_id, label, event.get("error"))
    except BaseException as error:
        display.fail(error)
        raise

    if final_result is None:
        error = RuntimeError("Graph produced no final state.")
        display.fail(error)
        raise error

    display.complete()
    return final_result


__all__ = ["run_with_progress"]
