"""Pure layout: a ``Snapshot`` plus a viewport becomes a list of semantic rows.

No terminal awareness and no escape codes live here. Rows carry already-padded
cells tagged with a style *name*, which is what lets the ANSI and HTML renderers
share one layout: joining ``cell.text`` yields the plain-text line, and mapping
``cell.style`` yields the styled one.

Every glyph used is single-column in a monospace terminal, so ``len()`` is the
display width.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal

from .model import Snapshot, Stage, STAGE_LABELS, GroupSnapshot, VariableSnapshot


MIN_WIDTH = 60
MAX_WIDTH = 140

# Status glyph + word. Colour is additive only: the glyph and the word always
# carry the meaning on their own.
STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "pending": ("·", "pending", "dim"),
    "coding": ("·", "coding", "active"),
    "structured_data": ("▣", "struct", "ok"),
    "extracted": ("✔", "extracted", "ok"),
    "not_found": ("∅", "not found", "dim"),
    "not_applicable": ("⊗", "n/a", "dim"),
    "blocked": ("⊘", "blocked", "warn"),
    "error": ("✖", "error", "err"),
}

# Tally glyphs, in the order a collapsed group header lists them.
TALLY_GLYPHS = (
    ("extracted", "✔", "ok"),
    ("structured_data", "▣", "ok"),
    ("not_found", "∅", "dim"),
    ("not_applicable", "⊗", "dim"),
    ("blocked", "⊘", "warn"),
    ("error", "✖", "err"),
)

CONFIDENCE_METER = {"low": "░", "medium": "▒", "high": "▓", "max": "█"}

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

FILLED, EMPTY = "▰", "▱"
# The in-flight pipeline position pulses between two same-weight glyphs, so an
# animating row never changes width or visual density.
ACTIVE_FRAMES = ("◈", "◆")
_CONTROL_SEQUENCE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


@dataclass(frozen=True)
class Cell:
    text: str
    style: str = ""


@dataclass(frozen=True)
class Row:
    kind: str
    cells: tuple[Cell, ...]

    @property
    def text(self) -> str:
        return "".join(cell.text for cell in self.cells)


@dataclass(frozen=True)
class Columns:
    """Column widths for one viewport width.

    Note counts go first when space is tight (they are the optional column), the
    pipeline's word label second; the three pipeline glyphs and the status word
    are never dropped.
    """

    total: int
    name: int
    value: int
    pipeline: int
    notes: int
    item: int = 4
    status: int = 11
    gap: int = 2
    gutter: int = 2

    @property
    def show_notes(self) -> bool:
        return self.notes > 0

    @property
    def show_pipeline_label(self) -> bool:
        return self.pipeline > 3

    @property
    def annotation(self) -> int:
        """Group-header annotation field: spans the pipeline and notes columns."""
        return self.pipeline + self.gap + (self.notes + self.gap if self.show_notes else 0)


def columns_for(width: int, *, show_notes: bool) -> Columns:
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    notes = 5 if (show_notes and width >= 92) else 0
    pipeline = 15 if width >= 70 else 3
    fixed = 2 + 4 + pipeline + 11 + 4 * 2 + (notes + 2 if notes else 0)
    flex = width - fixed
    name = min(32, max(16, flex - 10))
    return Columns(
        total=width, name=name, value=max(8, flex - name), pipeline=pipeline, notes=notes
    )


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    text = _CONTROL_SEQUENCE.sub("", str(text))
    text = "".join(
        character if character >= " " and character != "\x7f" else " "
        for character in text
    )
    if len(text) <= width:
        return text
    return text[: width - 1] + "…" if width > 1 else "…"


def pad(text: str, width: int, align: Literal["left", "right"] = "left") -> str:
    text = clip(text, width)
    return text.rjust(width) if align == "right" else text.ljust(width)


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def bar(done: int, total: int, width: int) -> str:
    if width <= 0:
        return ""
    if total <= 0:
        return EMPTY * width
    filled = width if done >= total else int(width * done / total)
    return FILLED * filled + EMPTY * (width - filled)


# --- Cell builders ----------------------------------------------------------


def pipeline_cells(variable: VariableSnapshot, cols: Columns, tick: int) -> list[Cell]:
    """Three positions — retrieve, extract, validate — plus the active label.

    Terminal variables that never entered the pipeline (gated out, blocked,
    supplied as structured data) stay empty rather than claiming credit for work
    that did not happen.
    """
    if variable.status in {"not_applicable", "blocked", "structured_data"}:
        glyphs, label, style = EMPTY * 3, "", "dim"
    elif variable.terminal:
        glyphs, label, style = FILLED * 3, "", "dim"
    elif variable.stage is Stage.IDLE:
        glyphs, label, style = EMPTY * 3, "", "dim"
    else:
        position = int(variable.stage)
        active = ACTIVE_FRAMES[tick % len(ACTIVE_FRAMES)]
        glyphs = "".join(
            FILLED if index < position else active if index == position else EMPTY
            for index in range(1, 4)
        )
        label = STAGE_LABELS[variable.stage]
        if variable.attempt > 1:
            label = f"{label}·{variable.attempt}"
        style = "active"

    if not cols.show_pipeline_label:
        return [Cell(pad(glyphs, cols.pipeline), style)]
    return [
        Cell(glyphs, style),
        Cell(pad(f" {label}" if label else "", cols.pipeline - 3), "active"),
    ]


def status_cell(variable: VariableSnapshot, cols: Columns) -> Cell:
    key = variable.status
    if key == "pending" and variable.stage is not Stage.IDLE:
        key = "coding"
    glyph, word, style = STATUS_DISPLAY.get(key, ("?", key, ""))
    return Cell(pad(f"{glyph} {word}", cols.status), style)


def value_cells(variable: VariableSnapshot, cols: Columns) -> list[Cell]:
    """The coded value, with the review-flag and confidence markers trailing it.

    Eight facts have to fit six visual columns, so the last two characters are
    reserved: a flag marker (``!`` repairs exhausted, ``?`` low confidence) and a
    one-character confidence meter.
    """
    if variable.status in {"extracted", "structured_data"}:
        text, style = str(variable.value), "ok"
    elif variable.detail:
        text, style = variable.detail, ("err" if variable.status == "error" else "dim")
    elif variable.terminal:
        text, style = "—", "dim"
    else:
        text, style = "—", "dim"

    body = cols.value - 3
    cells = [Cell(pad(text, body), style)]
    cells.append(Cell(pad(variable.flag or "", 2, "right"), "warn"))
    cells.append(Cell(CONFIDENCE_METER.get(variable.confidence or "", " "), "dim"))
    return cells


def tally(snapshot: Snapshot, item_ids: Iterable[int]) -> list[Cell]:
    counts: dict[str, int] = {}
    for item_id in item_ids:
        status = snapshot.variables[item_id].status
        counts[status] = counts.get(status, 0) + 1
    cells: list[Cell] = []
    for status, glyph, style in TALLY_GLYPHS:
        if counts.get(status):
            cells.append(Cell(f"{glyph}{counts[status]} ", style))
    return cells


# --- Body blocks ------------------------------------------------------------


@dataclass
class _Block:
    """One top-level group and everything nested under it."""

    group: GroupSnapshot
    members: tuple[GroupSnapshot, ...]
    item_ids: tuple[int, ...]
    active: bool
    expanded_rows: int
    tier: int  # collapse priority; 0 = never collapse


def _blocks(snapshot: Snapshot) -> list[_Block]:
    blocks: list[_Block] = []
    for group in snapshot.groups:
        if group.depth:
            continue
        members = snapshot.descendants(group)
        item_ids = snapshot.group_item_ids(group)
        variables = [snapshot.variables[item_id] for item_id in item_ids]
        active = any(member.active for member in members)
        terminal = bool(variables) and all(variable.terminal for variable in variables)
        flagged = any(
            variable.flag or variable.status in {"error", "blocked"} for variable in variables
        )
        untouched = all(
            variable.status == "pending" and variable.stage is Stage.IDLE
            for variable in variables
        )
        if active:
            tier = 0
        elif terminal and not flagged:
            tier = 1
        elif terminal:
            tier = 2
        elif untouched:
            tier = 3
        else:
            tier = 4
        blocks.append(
            _Block(
                group=group,
                members=members,
                item_ids=item_ids,
                active=active,
                expanded_rows=sum(1 + len(member.item_ids) for member in members),
                tier=tier,
            )
        )
    return blocks


def _collapse(blocks: list[_Block], budget: int) -> set[str]:
    """Collapse whole groups until the body fits, cheapest information first.

    Fully-terminal clean groups go first, then terminal groups that still carry a
    flag (their header keeps the tally), then groups that have not started, and
    finally partially processed inactive groups. A group with a live branch is
    never collapsed — that is the one the reader is watching.
    """
    collapsed: set[str] = set()

    def height() -> int:
        return sum(1 if block.group.group_id in collapsed else block.expanded_rows for block in blocks)

    for tier in (1, 2, 3, 4):
        for block in blocks:
            if height() <= budget:
                return collapsed
            if block.tier == tier and block.tier != 0:
                collapsed.add(block.group.group_id)
    return collapsed


def _window(blocks: list[_Block], collapsed: set[str], budget: int) -> tuple[int, int]:
    """Return the best contiguous block range that fits the body budget.

    Scroll markers consume rows too. Prefer ranges containing active groups,
    then the range containing the first active group, then the most groups.
    There are only eight top-level groups in the current dashboard, so checking
    every range keeps this accounting simple and deterministic.
    """

    def rows(block: _Block) -> int:
        return 1 if block.group.group_id in collapsed else block.expanded_rows

    if not blocks:
        return 0, 0

    active = {index for index, block in enumerate(blocks) if block.active}
    anchor = min(active) if active else 0
    best: tuple[tuple[int, int, int, int, int], int, int] | None = None
    for start in range(len(blocks)):
        used = 1 if start else 0
        for end in range(start + 1, len(blocks) + 1):
            used += rows(blocks[end - 1])
            total = used + (1 if end < len(blocks) else 0)
            if total > budget:
                break
            active_count = sum(index in active for index in range(start, end))
            score = (
                active_count,
                int(start <= anchor < end),
                end - start,
                -abs((start + end - 1) - 2 * anchor),
                -start,
            )
            if best is None or score > best[0]:
                best = (score, start, end)

    if best is not None:
        return best[1], best[2]

    # A single expanded group can be taller than an extremely small viewport.
    # Keep its header and leading rows visible; ``build_rows`` clips the body to
    # the exact budget after adding the scroll markers.
    return anchor, anchor + 1


# --- Row builders -----------------------------------------------------------


def fit(cells: Iterable[Cell], width: int) -> tuple[list[Cell], int]:
    """Truncate a cell run to ``width``, returning it and the space left over."""
    kept: list[Cell] = []
    used = 0
    for cell in cells:
        if used >= width:
            break
        if used + len(cell.text) > width:
            kept.append(Cell(clip(cell.text, width - used), cell.style))
            used = width
            break
        kept.append(cell)
        used += len(cell.text)
    return kept, width - used


def _panel(width: int, title: str, right: str, lines: list[list[Cell]]) -> list[Row]:
    """Rounded box around the summary lines, with the title and clock inlaid."""
    head = clip(f"╭─ {title} ", width - 12)
    tail = f" {right} ─╮"
    rows = [
        Row(
            "chrome",
            (
                Cell(head, "accent"),
                Cell("─" * max(0, width - len(head) - len(tail)), "dim"),
                Cell(tail, "dim"),
            ),
        )
    ]
    for cells in lines:
        kept, spare = fit(cells, width - 4)
        rows.append(Row("panel", (Cell("│ ", "dim"), *kept, Cell(" " * spare + " │", "dim"))))
    rows.append(Row("chrome", (Cell("╰" + "─" * (width - 2) + "╯", "dim"),)))
    return rows


def _case_header(snapshot: Snapshot, cols: Columns, now: float) -> list[Row]:
    width = cols.total
    elapsed = clock(now - snapshot.started_at)
    title = f"CIPOC · {snapshot.description}"
    meter = 14 if width >= 92 else 8

    notes = f"{snapshot.notes_done}/{snapshot.notes_total}"
    groups = f"{snapshot.done_groups}/{snapshot.total_groups}"
    variables = f"{snapshot.terminal_variables}/{snapshot.total_variables}"

    line_one = [
        Cell("notes  ", "dim"),
        Cell(bar(snapshot.notes_done, snapshot.notes_total, meter), "accent"),
        Cell(f"  {notes} scanned", ""),
    ]
    if width >= 80:
        line_one += [
            Cell("      groups  ", "dim"),
            Cell(bar(snapshot.done_groups, snapshot.total_groups, meter // 2), "accent"),
            Cell(f"  {groups}", ""),
        ]

    line_two = [
        Cell("vars   ", "dim"),
        Cell(bar(snapshot.terminal_variables, snapshot.total_variables, meter), "accent"),
        Cell(f"  {variables}   ", ""),
        *tally(snapshot, snapshot.variables),
    ]
    active = len(snapshot.branches)
    if active and width >= 80:
        line_two.append(Cell(f"  {active} branch{'es' if active != 1 else ''} active", "active"))
    return _panel(width, title, elapsed, [line_one, line_two])


def _degraded_header(snapshot: Snapshot, cols: Columns, now: float) -> list[Row]:
    """Smaller chrome for standalone agents without case-level counters."""
    width = cols.total
    elapsed = clock(now - snapshot.started_at)
    meter = 14 if width >= 92 else 8
    title = f"CIPOC · {snapshot.description}"

    if snapshot.mode == "standalone":
        total = snapshot.total_variables
        noun = "variable" if total == 1 else "variables"
        title += f" · {total} {noun}"
        line = [
            Cell("vars   ", "dim"),
            Cell(bar(snapshot.terminal_variables, total, meter), "accent"),
            Cell(f"  {snapshot.terminal_variables}/{total}   ", ""),
            *tally(snapshot, snapshot.variables),
        ]
    else:
        total = snapshot.total_tasks
        completed = snapshot.completed_tasks
        progress = "starting" if not total and not snapshot.finished else f"{completed}/{total}"
        line = [
            Cell("steps  ", "dim"),
            Cell(bar(completed, total, meter), "accent"),
            Cell(f"  {progress}", ""),
        ]
        active = sum(node.state == "active" for node in snapshot.nodes)
        if active:
            line.append(Cell(f"   {active} running", "active"))
    return _panel(width, title, elapsed, [line])


def _column_titles(cols: Columns) -> list[Row]:
    cells = [
        Cell(" " * cols.gutter, ""),
        Cell(pad("ITEM", cols.item, "right") + " " * cols.gap, "dim"),
        Cell(pad("VARIABLE", cols.name) + " " * cols.gap, "dim"),
        Cell(pad("PIPELINE", cols.pipeline) + " " * cols.gap, "dim"),
    ]
    if cols.show_notes:
        cells.append(Cell(pad("NOTES", cols.notes, "right") + " " * cols.gap, "dim"))
    cells.append(Cell(pad("STATUS", cols.status) + " " * cols.gap, "dim"))
    cells.append(Cell(pad("VALUE", cols.value), "dim"))
    rule = [
        Cell(" " * cols.gutter, ""),
        Cell("─" * cols.item + " " * cols.gap, "dim"),
        Cell("─" * cols.name + " " * cols.gap, "dim"),
        Cell("─" * cols.pipeline + " " * cols.gap, "dim"),
    ]
    if cols.show_notes:
        rule.append(Cell("─" * cols.notes + " " * cols.gap, "dim"))
    rule.append(Cell("─" * cols.status + " " * cols.gap, "dim"))
    rule.append(Cell("─" * cols.value, "dim"))
    return [Row("columns", tuple(cells)), Row("rule", tuple(rule))]


def _group_row(
    snapshot: Snapshot, group: GroupSnapshot, cols: Columns, *, collapsed: bool, item_ids: tuple[int, ...]
) -> Row:
    done = sum(snapshot.variables[item_id].terminal for item_id in item_ids)
    marker = "▸" if collapsed else "▾"
    indent = "  " * group.depth
    name_width = cols.item + cols.gap + cols.name + cols.gap
    counts = f"{done}/{len(item_ids)}" if item_ids else ""

    if not item_ids:
        trailing: list[Cell] = []
    elif done == len(item_ids):
        trailing = tally(snapshot, item_ids)
    else:
        trailing = [Cell(bar(done, len(item_ids), min(cols.value, len(item_ids))), "accent")]
    trailing, spare = fit(trailing, cols.value)
    trailing.append(Cell(" " * spare, ""))

    return Row(
        "group",
        (
            Cell(f"{marker} " if item_ids or group.depth == 0 else "  ", "accent"),
            Cell(pad(indent + group.name, name_width), "bold"),
            Cell(pad(group.annotation, cols.annotation), "dim"),
            Cell(pad(counts, cols.status, "right") + " " * cols.gap, "dim"),
            *trailing,
        ),
    )


def _variable_row(
    variable: VariableSnapshot,
    cols: Columns,
    depth: int,
    tick: int,
    note_count: int | None,
) -> Row:
    cells = [
        Cell(" " * cols.gutter, ""),
        Cell(pad(str(variable.item_id), cols.item, "right") + " " * cols.gap, "dim"),
        Cell(pad("  " * depth + variable.name, cols.name) + " " * cols.gap, ""),
        *pipeline_cells(variable, cols, tick),
        Cell(" " * cols.gap, ""),
    ]
    if cols.show_notes:
        notes = "" if note_count is None else f"{note_count}n"
        cells.append(Cell(pad(notes, cols.notes, "right") + " " * cols.gap, "dim"))
    cells.append(status_cell(variable, cols))
    cells.append(Cell(" " * cols.gap, ""))
    cells.extend(value_cells(variable, cols))
    return Row("variable", tuple(cells))


def _footer(snapshot: Snapshot, cols: Columns, now: float) -> list[Row]:
    rows = [Row("rule", (Cell("─" * cols.total, "dim"),))]
    for branch in snapshot.branches:
        step = STAGE_LABELS[branch.stage] or "queued"
        notes = f" · {branch.note_count} notes" if branch.note_count is not None else ""
        detail = f" ⟳ {clip(branch.label, 24)}"
        body = f"  · {step:<8} · {branch.variables} vars{notes}"
        elapsed = clock(now - branch.started_at)
        content_width = max(1, cols.total - len(elapsed) - 1)
        content = clip(detail + body, content_width)
        pad_width = cols.total - len(content) - len(elapsed)
        rows.append(
            Row(
                "branch",
                (
                    Cell(content[: len(detail)], "active"),
                    Cell(content[len(detail) :], "dim"),
                    Cell(" " * pad_width, ""),
                    Cell(elapsed, "dim"),
                ),
            )
        )
    return rows


def _compact_rows(snapshot: Snapshot, cols: Columns, tick: int) -> list[Row]:
    """Node timeline for runs with no variable table (scanner, retriever)."""
    rows: list[Row] = []
    line: list[Cell] = []
    used = 0
    for node in snapshot.nodes:
        glyph, style = {
            "ok": ("✔", "ok"),
            "error": ("✖", "err"),
            "active": (SPINNER[tick % len(SPINNER)], "active"),
        }[node.state]
        kind = "◉" if node.kind == "llm" else "◆"
        count = f" {node.done}/{node.started}" if node.started > 1 else ""
        text = f" {glyph} {node.name}{count} {kind}  "
        if used + len(text) > cols.total - 2:
            rows.append(Row("nodes", tuple(line)))
            line, used = [], 0
        line.append(Cell(f" {glyph} ", style))
        line.append(Cell(f"{node.name}{count} ", ""))
        line.append(Cell(f"{kind}  ", "dim" if node.kind != "llm" else "llm"))
        used += len(text)
    if line:
        rows.append(Row("nodes", tuple(line)))
    return rows


def _status_row(
    snapshot: Snapshot,
    cols: Columns,
    now: float,
    *,
    report_prompt: bool = False,
) -> Row:
    if snapshot.fatal:
        return Row("status", (Cell(clip(f" ✖ {snapshot.fatal}", cols.total), "err"),))
    if snapshot.finished:
        if snapshot.mode == "compact":
            summary = f" ✔ complete in {clock(now - snapshot.started_at)}"
        else:
            summary = (
                f" ✔ {snapshot.terminal_variables}/{snapshot.total_variables} variables "
                f"in {clock(now - snapshot.started_at)}"
            )
        if snapshot.review_flags:
            summary += f" · {snapshot.review_flags} flagged for review"
        if snapshot.stop:
            summary += f" · {snapshot.stop}"
        if report_prompt:
            prompt = " · Press Enter to view report"
            summary = clip(summary, max(0, cols.total - len(prompt))) + prompt
        return Row("status", (Cell(clip(summary, cols.total), "ok"),))
    return Row("blank", (Cell("", ""),))


# --- Entry point ------------------------------------------------------------


def build_rows(
    snapshot: Snapshot,
    width: int,
    height: int | None = None,
    *,
    now: float | None = None,
    tick: int = 0,
    report_prompt: bool = False,
) -> list[Row]:
    """Render one frame. ``height=None`` means unbounded — nothing is collapsed.

    The notebook renderer and the persistent exit summary both pass ``None``:
    vertical space is free there, so all 52 variables stay visible.
    """
    now = snapshot.started_at if now is None else now
    cols = columns_for(width, show_notes=snapshot.show_note_counts)
    rows = (
        _case_header(snapshot, cols, now)
        if snapshot.mode == "case"
        else _degraded_header(snapshot, cols, now)
    )

    if snapshot.mode == "compact":
        node_rows = _compact_rows(snapshot, cols, tick)
        if height is not None:
            node_budget = max(0, height - len(rows) - 1)
            node_rows = node_rows[-node_budget:] if node_budget else []
        rows.extend(node_rows)
        rows.append(_status_row(snapshot, cols, now, report_prompt=report_prompt))
        return rows

    rows.extend(_column_titles(cols))
    blocks = _blocks(snapshot)

    footer_rows = len(snapshot.branches) + (1 if snapshot.branches else 0)
    node_rows = _compact_rows(snapshot, cols, tick) if snapshot.nodes else []
    if height is None:
        collapsed: set[str] = set()
        start, end = 0, len(blocks)
        budget = None
    else:
        chrome = len(rows) + footer_rows + len(node_rows) + 1
        budget = max(1, height - chrome)
        collapsed = _collapse(blocks, budget)
        start, end = _window(blocks, collapsed, budget)

    body_rows: list[Row] = []
    if start:
        body_rows.append(
            Row(
                "scroll",
                (Cell(f" ↑ {start} group{'s' if start != 1 else ''} above", "dim"),),
            )
        )
    for block in blocks[start:end]:
        is_collapsed = block.group.group_id in collapsed
        body_rows.append(
            _group_row(
                snapshot, block.group, cols, collapsed=is_collapsed, item_ids=block.item_ids
            )
        )
        if is_collapsed:
            continue
        for member in block.members:
            if member is not block.group:
                body_rows.append(
                    _group_row(
                        snapshot,
                        member,
                        cols,
                        collapsed=False,
                        item_ids=snapshot.group_item_ids(member),
                    )
                )
            for item_id in member.item_ids:
                body_rows.append(
                    _variable_row(
                        snapshot.variables[item_id],
                        cols,
                        member.depth,
                        tick,
                        member.note_count,
                    )
                )
    remaining = len(blocks) - end
    if remaining > 0:
        body_rows.append(
            Row("scroll", (Cell(f" ↓ {remaining} group{'s' if remaining != 1 else ''} below", "dim"),))
        )

    if budget is not None and len(body_rows) > budget:
        body_rows = body_rows[:budget]
        if remaining > 0 and body_rows:
            body_rows[-1] = Row(
                "scroll",
                (Cell(f" ↓ {remaining} group{'s' if remaining != 1 else ''} below", "dim"),),
            )
    rows.extend(body_rows)

    rows.extend(node_rows)
    if snapshot.branches:
        rows.extend(_footer(snapshot, cols, now))
    rows.append(_status_row(snapshot, cols, now, report_prompt=report_prompt))
    return rows


def render_lines(rows: Iterable[Row]) -> list[str]:
    """Plain text, no styling — the exit summary and the piped fallback."""
    return [row.text.rstrip() for row in rows]


__all__ = [
    "Cell",
    "Columns",
    "MAX_WIDTH",
    "MIN_WIDTH",
    "Row",
    "build_rows",
    "clip",
    "clock",
    "columns_for",
    "render_lines",
]
