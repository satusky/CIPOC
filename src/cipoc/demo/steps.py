"""Phase 2 — group the event stream into presenter steps (pause boundaries).

A demo is presented one *step* at a time: the presenter reveals a step, explains
it, then advances. A :class:`Step` is a contiguous span of the ordered
:class:`~cipoc.demo.events.DemoEvent` stream whose end is a natural place to
pause. Replaying the trace up to a step's ``end_seq`` leaves
:class:`~cipoc.demo.state.DemoState` showing exactly the state to talk about at
that step — so ``next``/``prev``/``goto`` are pure index moves over this list.

**Boundary rule.** A new step opens at every *root-level* orchestrator task
(``task_start`` with an empty namespace) — ``initialize``, ``scan_notes``, each
``note_branch``, ``characterize_corpus``, ``plan_extraction``, each
``extract_branch``, ``finalize_case``, and so on. Every event nested beneath a
root task (subagent task/values events and the LLM calls captured inside them)
falls into that root task's step, because their namespaces are nested under it.
This gives one linear, deterministic partition even though subagents fan out.
Two exceptions keep the partition readable: consecutive instances of a collapsed
fan-out node share one step (see :data:`FANOUT_COLLAPSE_NODES`), and a pair of
root tasks describing a single decision shares one step (``check_state`` then
``plan_extraction``).

The leading ``run_start`` and any state that arrives before the first root task
form an intro step so the whole stream is covered with no gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .events import DemoEvent
from .mapping import map_node_id


# Root orchestrator node -> a human step title. Nodes that repeat (one per note
# or per variable group) get an instance counter appended by :func:`build_steps`.
_STEP_TITLES: dict[str, str] = {
    "initialize": "Initialize case",
    "scan_notes": "Scan notes",
    "note_branch": "Characterize note",
    "characterize_corpus": "Characterize corpus",
    "check_state": "Check state",
    "plan_extraction": "Plan extraction",
    "extract_branch": "Extract group",
    "merge_and_update": "Merge results",
    "finalize_case": "Finalize case",
}

# Nodes that fan out (many instances per run) — their step titles get a counter.
_COUNTED_NODES = frozenset({"extract_branch"})

# Every node that fans out into parallel instances worth tracking *individually*
# (:class:`~cipoc.demo.state.InstanceDetail`), at any namespace depth:
# ``note_branch`` runs at the root, ``variable_branch`` two levels down inside
# an extractor sub-agent. Each instance keeps its own input, result, model calls
# and validation attempts so a card can be drawn per note / per variable instead
# of the last instance overwriting the shared map node's detail.
FANOUT_INSTANCE_NODES = frozenset({"note_branch", "variable_branch"})

# The separate, step-level question: which fan-outs collapse into a *single*
# presenter step. ``note_branch`` instances run interleaved, so giving each its
# own step leaves the early ones as empty "active" shells while the last
# swallows every note's work. ``variable_branch`` is deliberately absent — its
# instances live inside their group's ``extract_branch`` step.
FANOUT_COLLAPSE_NODES = frozenset({"note_branch"})

# Title for a collapsed fan-out step (plural — it covers every instance).
_FANOUT_STEP_TITLES: dict[str, str] = {"note_branch": "Characterize notes"}

# Consecutive root tasks that describe one thing and so share a step. Keyed
# ``(previous node, incoming node)``; the incoming task extends the open step
# rather than starting a new one. ``check_state`` decides whether groups remain
# and ``plan_extraction`` says which; ``scan_notes`` is the fan-out that hands
# every note to the ``note_branch`` instances that characterize them.
_MERGE_WITH_PREVIOUS: dict[tuple[str, str], str] = {
    ("check_state", "plan_extraction"): "Check state & plan extraction",
    ("scan_notes", "note_branch"): "Scan & characterize notes",
}


@dataclass(frozen=True)
class Step:
    """One presenter pause unit: a labeled span of the event stream.

    ``start_seq``/``end_seq`` are inclusive event ``seq`` bounds. ``map_node_id``
    is the conceptual map node this step centers on (``None`` for the intro).
    """

    index: int
    title: str
    subtitle: str
    node: str
    map_node_id: str | None
    agent: str | None
    start_seq: int
    end_seq: int
    fanout: bool = False
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "subtitle": self.subtitle,
            "node": self.node,
            "map_node_id": self.map_node_id,
            "agent": self.agent,
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "fanout": self.fanout,
            "task_id": self.task_id,
        }


@dataclass
class _Pending:
    """The step currently being accumulated, before its end is known."""

    title: str
    subtitle: str
    node: str
    map_node_id: str | None
    agent: str | None
    start_seq: int
    fanout: bool = False
    task_id: str = ""


def _is_root_task_start(event: DemoEvent) -> bool:
    return event.type == "task_start" and not event.namespace


def _subtitle(node: str, payload: Any) -> str:
    """Best-effort human subtitle from a root task's input payload."""
    if not isinstance(payload, Mapping):
        return ""
    if node == "extract_branch":
        requested = payload.get("requested_variables") or {}
        if isinstance(requested, Mapping):
            return str(requested.get("name") or requested.get("group_id") or "")
    if node == "note_branch":
        note_id = payload.get("note_id")
        note_type = payload.get("note_type")
        if note_type and note_id is not None:
            return f"{note_type} #{note_id}"
        if note_id is not None:
            return f"Note #{note_id}"
    return ""


def build_steps(events: Iterable[DemoEvent]) -> list[Step]:
    """Partition ``events`` into ordered presenter :class:`Step`\\ s.

    Every event lands in exactly one step, and the steps' ``[start_seq, end_seq]``
    ranges tile the stream with no gaps, so a cursor can move by whole steps.
    """
    events = list(events)
    if not events:
        return []

    steps: list[Step] = []
    counters: dict[str, int] = {}

    # Leading events before the first root task (run_start, the initial corpus
    # values) form an intro step — but only if any actually precede it, so a stream
    # that opens on a root task gets no empty intro.
    first_root = next(
        (i for i, event in enumerate(events) if _is_root_task_start(event)), len(events)
    )
    pending: _Pending | None = (
        _Pending("Run start", "", "", None, None, events[0].seq) if first_root > 0 else None
    )
    prev_seq = events[0].seq

    def close(end_seq: int) -> None:
        if pending is None:
            return
        steps.append(
            Step(
                index=len(steps),
                title=pending.title,
                subtitle=pending.subtitle,
                node=pending.node,
                map_node_id=pending.map_node_id,
                agent=pending.agent,
                start_seq=pending.start_seq,
                end_seq=end_seq,
                fanout=pending.fanout,
                task_id=pending.task_id,
            )
        )

    for event in events:
        if _is_root_task_start(event):
            node = event.node
            # A collapsed fan-out node folds every consecutive same-node instance
            # into the step opened by its first instance (they run interleaved, so
            # one step holds them all, rendered as per-instance cards).
            if pending is not None and node in FANOUT_COLLAPSE_NODES and pending.node == node:
                prev_seq = event.seq
                continue
            # A follow-on task that continues the open step's decision extends it
            # in place (retitled, and re-centered on the incoming node's map block)
            # instead of opening a second step for the same thing.
            merged_title = (
                _MERGE_WITH_PREVIOUS.get((pending.node, node)) if pending is not None else None
            )
            if merged_title is not None:
                pending.title = merged_title
                pending.node = node
                pending.map_node_id = event.map_node_id or map_node_id(node, event.namespace)
                pending.task_id = event.task_id
                # Merging *into* a collapsed fan-out makes the extended step a
                # fan-out step, so its remaining instances collapse into it too
                # (the branch above) and Panel 2 renders per-instance cards.
                pending.fanout = node in FANOUT_COLLAPSE_NODES
                prev_seq = event.seq
                continue
            close(prev_seq)
            if node in FANOUT_COLLAPSE_NODES:
                title = _FANOUT_STEP_TITLES.get(node, _STEP_TITLES.get(node, node or "Step"))
                subtitle = ""
                fanout = True
            else:
                title = _STEP_TITLES.get(node, node or "Step")
                if node in _COUNTED_NODES:
                    counters[node] = counters.get(node, 0) + 1
                    title = f"{title} {counters[node]}"
                subtitle = _subtitle(node, event.payload)
                fanout = False
            pending = _Pending(
                title=title,
                subtitle=subtitle,
                node=node,
                map_node_id=event.map_node_id or map_node_id(node, event.namespace),
                agent=event.agent,
                start_seq=event.seq,
                fanout=fanout,
                task_id=event.task_id,
            )
        prev_seq = event.seq

    close(prev_seq)
    return steps


__all__ = ["Step", "build_steps", "FANOUT_COLLAPSE_NODES", "FANOUT_INSTANCE_NODES"]
