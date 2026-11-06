"""I/O backends for progress snapshots.

The renderers deliberately share one small lifecycle: ``paint`` receives an
immutable snapshot and ``close`` releases any display resources. The live
renderers consume semantic rows from :mod:`.layout`; the plain renderer instead
emits snapshot deltas so redirected output remains a useful append-only log.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import time
from typing import Any, Callable, Mapping, Protocol, TextIO

from .layout import MAX_WIDTH, MIN_WIDTH, STATUS_DISPLAY, Row, build_rows, clock
from .model import (
    BranchSnapshot,
    NodeSnapshot,
    Snapshot,
    Stage,
    STAGE_LABELS,
    VariableSnapshot,
)


_ANSI_STYLES: Mapping[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[38;5;250m",
    "accent": "\033[38;5;74m",
    "active": "\033[1;36m",
    "ok": "\033[38;5;114m",
    "warn": "\033[38;5;208m",
    "err": "\033[38;5;220m",
    "llm": "\033[38;5;186m",
}

_HTML_STYLES: Mapping[str, str] = {
    "bold": "font-weight:700",
    "dim": "color:#64748b",
    "accent": "color:#2563eb",
    "active": "color:#0891b2;font-weight:700",
    "ok": "color:#15803d",
    "warn": "color:#b45309",
    "err": "color:#c2410c",
    "llm": "color:#a16207",
}

_RESET = "\033[0m"
_ENTER_ALT_SCREEN = "\033[?1049h"
_EXIT_ALT_SCREEN = "\033[?1049l"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_HOME = "\033[H"
_CLEAR_SCREEN = "\033[2J"
_CLEAR_LINE = "\033[2K"
_CLEAR_TO_END = "\033[J"
_CONTROL_SEQUENCE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


def _safe_text(value: Any) -> str:
    """Remove terminal controls and line breaks from untrusted display text."""
    text = _CONTROL_SEQUENCE.sub("", str(value))
    return "".join(
        character if character >= " " and character != "\x7f" else " "
        for character in text
    )


def _short(value: Any, width: int = 200) -> str:
    text = _safe_text(value)
    return text if len(text) <= width else text[: width - 1] + "…"


class Renderer(Protocol):
    """Common contract used by the stream runner."""

    min_interval: float

    def paint(
        self,
        snapshot: Snapshot,
        *,
        now: float | None = None,
        tick: int = 0,
        final: bool = False,
        report_prompt: bool = False,
    ) -> bool: ...

    def close(self) -> None: ...


class PlainLog:
    """Append one plain line per state transition.

    Repainting an unchanged snapshot is a no-op. This renderer never writes
    terminal escape sequences, even if a model-provided value contains them.
    """

    min_interval = 0.0

    def __init__(self, stream: TextIO):
        self.stream = stream
        self._previous: Snapshot | None = None

    def paint(
        self,
        snapshot: Snapshot,
        *,
        now: float | None = None,
        tick: int = 0,
        final: bool = False,
        report_prompt: bool = False,
    ) -> bool:
        del tick, final, report_prompt
        now = time.monotonic() if now is None else now
        lines = self._transitions(self._previous, snapshot, now)
        self._previous = snapshot
        if not lines:
            return False
        self.stream.write("".join(_safe_text(line) + "\n" for line in lines))
        self.stream.flush()
        return True

    def _transitions(
        self, previous: Snapshot | None, snapshot: Snapshot, now: float
    ) -> list[str]:
        lines: list[str] = []
        if previous is None:
            lines.append(f"CIPOC / {snapshot.description}")

        old_notes = previous.notes_done if previous is not None else 0
        if snapshot.notes_done != old_notes:
            lines.append(f"notes {snapshot.notes_done}/{snapshot.notes_total} scanned")

        old_groups = (
            {group.group_id: group for group in previous.groups}
            if previous is not None
            else {}
        )
        for group in snapshot.groups:
            old_group = old_groups.get(group.group_id)
            if old_group is not None and group.annotation != old_group.annotation:
                lines.append(f"group {group.name}: {group.annotation or 'eligible'}")

        old_branches = (
            {branch.key: branch for branch in previous.branches}
            if previous is not None
            else {}
        )
        branches = {branch.key: branch for branch in snapshot.branches}
        for key, branch in branches.items():
            if branch != old_branches.get(key):
                lines.append(self._branch_line(branch))
        for key, branch in old_branches.items():
            if key not in branches:
                lines.append(f"group {branch.label}: complete")

        old_variables = previous.variables if previous is not None else {}
        for item_id, variable in snapshot.variables.items():
            old_variable = old_variables.get(item_id)
            if old_variable == variable:
                continue
            if (
                old_variable is None
                and variable.status == "pending"
                and variable.stage is Stage.IDLE
            ):
                continue
            lines.append(self._variable_line(variable))

        old_nodes = (
            {node.name: node for node in previous.nodes} if previous is not None else {}
        )
        for node in snapshot.nodes:
            if node != old_nodes.get(node.name):
                lines.append(self._node_line(node))

        was_finished = previous.finished if previous is not None else False
        old_fatal = previous.fatal if previous is not None else None
        if snapshot.finished and (not was_finished or snapshot.fatal != old_fatal):
            lines.append(self._completion_line(snapshot, now))
        return lines

    @staticmethod
    def _branch_line(branch: BranchSnapshot) -> str:
        stage = STAGE_LABELS[branch.stage] or "queued"
        notes = ""
        if branch.note_count is not None:
            notes = f" · {branch.note_count} note{'s' if branch.note_count != 1 else ''}"
        return (
            f"group {branch.label}: {stage} · {branch.variables} "
            f"variable{'s' if branch.variables != 1 else ''}{notes}"
        )

    @staticmethod
    def _variable_line(variable: VariableSnapshot) -> str:
        prefix = f"{variable.item_id} {variable.name}: "
        if not variable.terminal:
            stage = STAGE_LABELS[variable.stage] or "pending"
            attempt = f"·{variable.attempt}" if variable.attempt > 1 else ""
            return prefix + stage + attempt

        glyph, word, _ = STATUS_DISPLAY.get(
            variable.status, ("?", variable.status, "")
        )
        line = prefix + f"{glyph} {word}"
        if variable.value is not None:
            line += f" = {_short(variable.value)}"
        elif variable.detail:
            line += f": {_short(variable.detail)}"
        if variable.flag:
            line += f" {variable.flag}"
        if variable.confidence:
            line += f" [{variable.confidence}]"
        return line

    @staticmethod
    def _node_line(node: NodeSnapshot) -> str:
        if node.state == "error":
            state = f"error ({node.errors}/{node.started})"
        elif node.state == "ok":
            state = f"complete ({node.done}/{node.started})"
        else:
            state = f"running ({node.done}/{node.started})"
        return f"node {node.name} [{node.kind}]: {state}"

    @staticmethod
    def _completion_line(snapshot: Snapshot, now: float) -> str:
        elapsed = clock(now - snapshot.started_at)
        if snapshot.fatal:
            return f"failed after {elapsed}: {_short(snapshot.fatal)}"
        if snapshot.mode == "compact":
            return f"complete in {elapsed}"
        line = (
            f"complete: {snapshot.terminal_variables}/{snapshot.total_variables} "
            f"variables in {elapsed}"
        )
        if snapshot.review_flags:
            line += f" · {snapshot.review_flags} flagged for review"
        if snapshot.stop:
            line += f" · {_short(snapshot.stop)}"
        return line

    def close(self) -> None:
        self.stream.flush()

    def __enter__(self) -> PlainLog:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AnsiAltScreen:
    """Full-screen ANSI renderer for interactive terminals."""

    min_interval = 0.1

    def __init__(
        self,
        stream: TextIO,
        *,
        color: bool | None = None,
        size_provider: Callable[[], os.terminal_size] | None = None,
    ):
        self.stream = stream
        self.color = "NO_COLOR" not in os.environ if color is None else color
        self._size_provider = size_provider or (
            lambda: shutil.get_terminal_size((100, 24))
        )
        self._entered = False
        self._closed = False
        self._last_paint = float("-inf")

    def enter(self) -> None:
        if self._entered or self._closed:
            return
        self.stream.write(_ENTER_ALT_SCREEN + _HIDE_CURSOR + _HOME + _CLEAR_SCREEN)
        self.stream.flush()
        self._entered = True

    def viewport(self) -> tuple[int, int]:
        try:
            size = self._size_provider()
            columns, lines = int(size.columns), int(size.lines)
        except (AttributeError, OSError, ValueError):
            columns, lines = 100, 24
        return max(MIN_WIDTH, min(columns, MAX_WIDTH)), max(1, lines)

    def paint(
        self,
        snapshot: Snapshot,
        *,
        now: float | None = None,
        tick: int = 0,
        final: bool = False,
        report_prompt: bool = False,
    ) -> bool:
        now = time.monotonic() if now is None else now
        if not final and now - self._last_paint < self.min_interval:
            return False
        self.enter()
        if self._closed:
            return False
        width, height = self.viewport()
        rows = build_rows(
            snapshot,
            width,
            height,
            now=now,
            tick=tick,
            report_prompt=report_prompt,
        )
        lines = [_ansi_row(row, color=self.color) for row in rows]
        frame = (
            _HOME
            + "\r\n".join(_CLEAR_LINE + line for line in lines)
            + _CLEAR_TO_END
        )
        self.stream.write(frame)
        self.stream.flush()
        self._last_paint = now
        return True

    def close(self) -> None:
        if self._closed:
            return
        if self._entered:
            self.stream.write(_RESET + _SHOW_CURSOR + _EXIT_ALT_SCREEN)
            self.stream.flush()
        self._closed = True

    def __enter__(self) -> AnsiAltScreen:
        self.enter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _ansi_row(row: Row, *, color: bool) -> str:
    parts: list[str] = []
    for cell in row.cells:
        text = _safe_text(cell.text)
        style = _ANSI_STYLES.get(cell.style) if color else None
        parts.append(f"{style}{text}{_RESET}" if style else text)
    return "".join(parts)


def ansi_lines(rows: list[Row], *, color: bool) -> list[str]:
    """Render semantic rows as safe ANSI text, or plain text when disabled."""
    return [_ansi_row(row, color=color) for row in rows]


class NotebookDisplay:
    """Update one IPython HTML display handle at no more than two frames/second."""

    min_interval = 0.5

    def __init__(
        self,
        *,
        width: int = 100,
        display_fn: Callable[..., Any] | None = None,
        html_factory: Callable[[str], Any] | None = None,
    ):
        if display_fn is None or html_factory is None:
            from IPython.display import HTML, display

            display_fn = display if display_fn is None else display_fn
            html_factory = HTML if html_factory is None else html_factory
        assert display_fn is not None and html_factory is not None
        self.width = max(MIN_WIDTH, min(width, MAX_WIDTH))
        self._display = display_fn
        self._html_factory = html_factory
        self._handle: Any = None
        self._last_paint = float("-inf")

    def paint(
        self,
        snapshot: Snapshot,
        *,
        now: float | None = None,
        tick: int = 0,
        final: bool = False,
        report_prompt: bool = False,
    ) -> bool:
        del report_prompt
        now = time.monotonic() if now is None else now
        if not final and now - self._last_paint < self.min_interval:
            return False
        rows = build_rows(snapshot, self.width, None, now=now, tick=tick)
        value = self._html_factory(_html_document(rows))
        if self._handle is None:
            self._handle = self._display(value, display_id=True)
        else:
            self._handle.update(value)
        self._last_paint = now
        return True

    def close(self) -> None:
        return None

    def __enter__(self) -> NotebookDisplay:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _html_document(rows: list[Row]) -> str:
    rendered_rows = []
    for row in rows:
        cells = []
        for cell in row.cells:
            text = html.escape(_safe_text(cell.text), quote=False)
            style = _HTML_STYLES.get(cell.style)
            cells.append(f'<span style="{style}">{text}</span>' if style else text)
        kind = html.escape(row.kind, quote=True)
        rendered_rows.append(f'<span data-kind="{kind}">{"".join(cells)}</span>')
    content = "\n".join(rendered_rows)
    return (
        '<div class="cipoc-progress-dashboard">'
        '<pre style="margin:0;overflow-x:auto;white-space:pre;line-height:1.35;'
        'font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;'
        'font-size:12px;color:#0f172a;background:#f8fafc;border:1px solid #cbd5e1;'
        'border-radius:6px;padding:12px">'
        f"{content}</pre></div>"
    )


__all__ = ["AnsiAltScreen", "NotebookDisplay", "PlainLog", "Renderer", "ansi_lines"]
