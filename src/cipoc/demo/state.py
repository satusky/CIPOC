"""Phase 2 — fold the merged ``DemoEvent`` stream into presentable dashboard state.

:class:`DemoState` is the demo's counterpart to
:class:`cipoc.utils.progress.model.ProgressModel`: it ingests the ordered
:class:`~cipoc.demo.events.DemoEvent` stream (live or replayed from a trace) and
produces an immutable :class:`DemoSnapshot` describing the run at a point in time.
It does three things the terminal model does not:

* **Reuses ``ProgressModel``** for the variable overview (Panel 3). The demo
  trace stores JSON, so root ``values`` payloads are re-hydrated into a real
  :class:`~cipoc.agents.orchestrator.CaseState` before being fed to the model —
  the eligibility predicates it runs (``corpus_gate_passes`` / ``site_applies``)
  need typed models, not dicts, and this is what makes the replayed variable
  table identical to a live run's.
* **Tracks map activity** (Panel 1): which conceptual ``agent_system.json`` nodes
  are executing right now, how many concurrent tasks each has (the fan-out
  multiplicity badge), and which have run at all (traversed).
* **Accumulates per-node detail** (Panel 2 raw material): for each map node, the
  latest task input/result and the LLM calls captured inside it, plus the latest
  root ``CaseState`` for corpus-level detail. The rich per-component views are
  Phase 4; this keeps the material addressable by map node id.

A :class:`DemoState` is fed one event at a time via :meth:`ingest` and read via
:meth:`snapshot`, so a server can rebuild state up to any cursor position by
replaying a prefix of the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace
from typing import Any, Iterable, Mapping

from cipoc.agents.orchestrator import CaseState
from cipoc.utils.progress.events import ProgressEvent
from cipoc.utils.progress.model import ProgressModel, Snapshot, TaskKind

from .events import DemoEvent
from .steps import FANOUT_COLLAPSE_NODES


@dataclass(frozen=True)
class NodeDetail:
    """The latest material that flowed through one conceptual map node.

    ``count`` is the number of task instances that have started for this node
    (fan-out multiplicity); ``active`` is how many have not yet finished. The
    ``input``/``result`` are the JSON payloads of the most recent task instance,
    and ``llm_calls`` are the model captures correlated to this node so far.
    """

    map_node_id: str
    node: str
    agent: str | None
    count: int = 0
    active: int = 0
    errors: int = 0
    started_t: float | None = None
    finished_t: float | None = None
    input: Any = None
    result: Any = None
    error: Any = None
    llm_calls: tuple[dict[str, Any], ...] = ()

    @property
    def status(self) -> str:
        if self.errors:
            return "error"
        if self.active > 0:
            return "active"
        return "done" if self.count else "idle"


@dataclass(frozen=True)
class InstanceDetail:
    """One fan-out instance's own material — e.g. a single note's characterization.

    A fan-out node such as ``note_branch`` runs many times in parallel, all
    sharing one conceptual map node. Folding them onto a single
    :class:`NodeDetail` (keyed by map node) collapses every instance's input,
    result, and LLM calls together — the last write wins and the calls pile up.
    So each instance is *also* tracked here, keyed by its root scope
    ``"<node>:<task_id>"``, and every event nested beneath that scope (the
    sub-agent's ``values`` snapshots and captured LLM calls) is attributed back
    to it. ``index`` is the 1-based fan-out order; ``label`` is a human tag drawn
    from the instance's input (e.g. ``"pathology #12"``).
    """

    key: str
    node: str
    map_node_id: str
    agent: str | None
    index: int
    label: str = ""
    active: int = 1
    errors: int = 0
    started_t: float | None = None
    finished_t: float | None = None
    input: Any = None
    result: Any = None
    error: Any = None
    llm_calls: tuple[dict[str, Any], ...] = ()

    @property
    def status(self) -> str:
        if self.errors:
            return "error"
        return "active" if self.active > 0 else "done"


@dataclass(frozen=True)
class DemoSnapshot:
    """An immutable read of the demo run at one point in the stream.

    Combines the reused variable-table :class:`Snapshot` (Panel 3) with the map
    activity (Panel 1) and per-node detail (Panel 2). Serialized to JSON by
    :meth:`to_dict` for the server API.
    """

    seq: int
    t: float
    finished: bool
    current_map_node: str | None
    current_agent: str | None
    active_map_nodes: tuple[str, ...]
    visited_map_nodes: tuple[str, ...]
    node_multiplicity: Mapping[str, int]
    details: Mapping[str, NodeDetail]
    instances: Mapping[str, InstanceDetail]
    progress: Snapshot | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t": round(self.t, 6),
            "finished": self.finished,
            "current_map_node": self.current_map_node,
            "current_agent": self.current_agent,
            "active_map_nodes": list(self.active_map_nodes),
            "visited_map_nodes": list(self.visited_map_nodes),
            "node_multiplicity": dict(self.node_multiplicity),
            "details": {key: _detail_to_dict(value) for key, value in self.details.items()},
            "instances": {key: _instance_to_dict(value) for key, value in self.instances.items()},
            "progress": _progress_to_dict(self.progress),
        }


class DemoState:
    """Fold ``DemoEvent``s into a :class:`DemoSnapshot`.

    The internal :class:`ProgressModel` is built lazily on the first root
    ``values`` event that carries a plan (``target_variables``); events seen
    before then are buffered and replayed into the model once it exists, so the
    variable table is correct regardless of when the plan lands in the stream.
    ``target_groups`` / ``group_hierarchy`` may be supplied up front (live mode
    has them from the agent) to seed the model before the plan is streamed.
    """

    def __init__(
        self,
        *,
        description: str = "CIPOC extraction",
        target_groups: Iterable[Any] | None = None,
        group_hierarchy: Iterable[Any] | None = None,
        graph_input: Any = None,
        node_kinds: Mapping[str, TaskKind] | None = None,
    ) -> None:
        self._description = description
        self._node_kinds = node_kinds
        self._group_hierarchy = list(group_hierarchy) if group_hierarchy is not None else None

        self._model: ProgressModel | None = None
        self._buffer: list[tuple[ProgressEvent, float]] = []
        if target_groups is not None:
            self._build_model(list(target_groups), graph_input)

        self._seq = 0
        self._t = 0.0
        self._finished = False

        # Map activity (Panel 1).
        self._active: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        self._visited: list[str] = []
        self._current_node: str | None = None
        self._current_agent: str | None = None

        # Per-node detail (Panel 2) and latest case state.
        self._details: dict[str, NodeDetail] = {}
        self._case: Any = None

        # Per-instance detail for fan-out nodes (Panel 2), keyed by root scope
        # ``"<node>:<task_id>"``; ``_instance_order`` preserves fan-out order.
        self._instances: dict[str, InstanceDetail] = {}
        self._instance_order: list[str] = []

    # --- Model construction ----------------------------------------------

    def _build_model(self, target_groups: list[Any], graph_input: Any) -> None:
        """Create the ``ProgressModel`` and drain any buffered events into it."""
        self._model = ProgressModel(
            self._description,
            0.0,
            target_groups=target_groups,
            group_hierarchy=self._group_hierarchy,
            graph_input=graph_input,
            node_kinds=self._node_kinds,
            include_input_group=False,
        )
        for event, now in self._buffer:
            self._model.ingest(event, now)
        self._buffer.clear()

    # --- Ingest -----------------------------------------------------------

    def ingest(self, event: DemoEvent) -> None:
        """Fold one ``DemoEvent`` into the running state."""
        self._seq = event.seq
        self._t = event.t

        if event.type == "run_start":
            return
        if event.type == "run_end":
            self.finish()
            return

        # Attribute the event to its fan-out instance (if any) before the
        # per-map-node folding below; the two views are independent.
        scope = _instance_scope(event)
        if scope is not None:
            self._ingest_instance(event, scope)

        if event.type == "llm_call":
            self._ingest_llm(event)
            return
        if event.type == "values":
            self._ingest_values(event)
            return
        if event.type == "task_start":
            self._ingest_task_start(event)
        elif event.type == "task_end":
            self._ingest_task_end(event)

        progress_event = _to_progress_event(event)
        if progress_event is not None and self._model is not None:
            self._model.ingest(progress_event, event.t)
        elif progress_event is not None:
            self._buffer.append((progress_event, event.t))

    def _ingest_values(self, event: DemoEvent) -> None:
        if not event.namespace:  # root CaseState
            case = _hydrate_case(event.payload)
            self._case = case
            if self._model is None and case is not None and case.target_variables:
                self._build_model(
                    list(case.target_variables),
                    {"note_corpus": case.note_corpus},
                )
            progress_event = ProgressEvent(kind="values", namespace=(), payload=case)
            if self._model is not None:
                self._model.ingest(progress_event, event.t)
            else:
                self._buffer.append((progress_event, event.t))

    def _ingest_task_start(self, event: DemoEvent) -> None:
        self._current_agent = event.agent
        map_id = event.map_node_id
        if map_id is None:
            return
        self._current_node = map_id
        self._active[map_id] = self._active.get(map_id, 0) + 1
        self._counts[map_id] = self._counts.get(map_id, 0) + 1
        if map_id not in self._visited:
            self._visited.append(map_id)

        existing = self._details.get(map_id)
        self._details[map_id] = NodeDetail(
            map_node_id=map_id,
            node=event.node,
            agent=event.agent,
            count=self._counts[map_id],
            active=self._active[map_id],
            errors=existing.errors if existing else 0,
            started_t=existing.started_t if existing and existing.started_t is not None else event.t,
            finished_t=None,
            input=event.payload,
            result=None,
            error=None,
            llm_calls=existing.llm_calls if existing else (),
        )

    def _ingest_task_end(self, event: DemoEvent) -> None:
        map_id = event.map_node_id
        if map_id is None:
            return
        self._active[map_id] = max(0, self._active.get(map_id, 0) - 1)
        existing = self._details.get(map_id)
        errors = (existing.errors if existing else 0) + (1 if event.error else 0)
        self._details[map_id] = NodeDetail(
            map_node_id=map_id,
            node=event.node,
            agent=event.agent,
            count=self._counts.get(map_id, existing.count if existing else 0),
            active=self._active[map_id],
            errors=errors,
            started_t=existing.started_t if existing else None,
            finished_t=event.t,
            input=existing.input if existing else None,
            result=event.payload,
            error=event.error,
            llm_calls=existing.llm_calls if existing else (),
        )

    def _ingest_llm(self, event: DemoEvent) -> None:
        map_id = event.map_node_id
        if map_id is None:
            return
        existing = self._details.get(map_id)
        call = event.payload if isinstance(event.payload, dict) else {"response": event.payload}
        if existing is None:
            self._details[map_id] = NodeDetail(
                map_node_id=map_id,
                node=event.node,
                agent=event.agent,
                llm_calls=(call,),
            )
        else:
            self._details[map_id] = _replace_calls(existing, (*existing.llm_calls, call))

    def _ingest_instance(self, event: DemoEvent, scope: str) -> None:
        """Fold one event into its fan-out instance (keyed by root scope).

        The instance is opened by the root fan-out ``task_start`` (which carries
        the instance input, e.g. the note) and closed by that node's matching
        ``task_end``. In between, the sub-agent's nested ``values`` snapshots
        accumulate into the instance ``result`` (the running characterization)
        and any captured LLM calls are appended.
        """
        inst = self._instances.get(scope)
        if inst is None:
            node = scope.split(":", 1)[0]
            inst = InstanceDetail(
                key=scope,
                node=node,
                map_node_id=event.map_node_id or "",
                agent=event.agent,
                index=len(self._instance_order) + 1,
                started_t=event.t,
            )
            self._instances[scope] = inst
            self._instance_order.append(scope)

        is_root = not event.namespace and event.node == inst.node
        if event.type == "task_start" and is_root:
            inst = replace(
                inst,
                input=event.payload,
                label=_instance_label(inst.node, event.payload),
                map_node_id=event.map_node_id or inst.map_node_id,
                agent=event.agent or inst.agent,
            )
        elif event.type == "task_end" and is_root:
            inst = replace(
                inst,
                active=0,
                finished_t=event.t,
                errors=inst.errors + (1 if event.error else 0),
                error=event.error if event.error is not None else inst.error,
            )
        elif event.type == "values":
            inst = replace(inst, result=_merge_result(inst.result, event.payload))
        elif event.type == "llm_call":
            call = event.payload if isinstance(event.payload, dict) else {"response": event.payload}
            inst = replace(inst, llm_calls=(*inst.llm_calls, call))
        self._instances[scope] = inst

    # --- Lifecycle --------------------------------------------------------

    def finish(self) -> None:
        self._finished = True
        self._current_node = None
        if self._model is not None:
            self._model.finish()

    # --- Read -------------------------------------------------------------

    def snapshot(self) -> DemoSnapshot:
        active = tuple(node for node, n in self._active.items() if n > 0)
        multiplicity = {node: n for node, n in self._active.items() if n > 0}
        return DemoSnapshot(
            seq=self._seq,
            t=self._t,
            finished=self._finished,
            current_map_node=None if self._finished else self._current_node,
            current_agent=self._current_agent,
            active_map_nodes=active,
            visited_map_nodes=tuple(self._visited),
            node_multiplicity=multiplicity,
            details=dict(self._details),
            instances={key: self._instances[key] for key in self._instance_order},
            progress=self._model.snapshot() if self._model is not None else None,
        )

    @property
    def latest_case(self) -> CaseState | None:
        """The most recent root ``CaseState`` seen, for corpus-level detail.

        Exposed separately from :meth:`snapshot` because the full case state is
        large and mostly duplicates the variable table; the server serves it from
        its own endpoint rather than embedding it in every snapshot read.
        """
        return self._case


# --- Helpers -------------------------------------------------------------


def replay(events: Iterable[DemoEvent], **kwargs: Any) -> DemoState:
    """Build a :class:`DemoState` by ingesting ``events`` in order."""
    state = DemoState(**kwargs)
    for event in events:
        state.ingest(event)
    return state


def _to_progress_event(event: DemoEvent) -> ProgressEvent | None:
    """Reconstruct a ``ProgressEvent`` from a task ``DemoEvent`` (dict payloads).

    ``values`` events are handled separately (they need root hydration), so this
    only covers ``task_start`` / ``task_end``. Task payloads stay as dicts:
    ``ProgressModel`` reads them through the dict-tolerant ``field`` accessor.
    """
    if event.type == "task_start":
        return ProgressEvent(
            kind="task_start",
            namespace=event.namespace,
            node=event.node,
            task_id=event.task_id,
            payload=event.payload,
        )
    if event.type == "task_end":
        return ProgressEvent(
            kind="task_end",
            namespace=event.namespace,
            node=event.node,
            task_id=event.task_id,
            payload=event.payload,
            error=event.error,
        )
    return None


def _instance_scope(event: DemoEvent) -> str | None:
    """The fan-out instance scope this event belongs to, or ``None``.

    A root fan-out ``task_start``/``task_end`` (empty namespace, node in
    :data:`~cipoc.demo.steps.FANOUT_COLLAPSE_NODES`) owns scope
    ``"<node>:<task_id>"``; every nested event carries that scope as the first
    element of its namespace.
    """
    ns = event.namespace
    if ns:
        head = ns[0]
        node = head.split(":", 1)[0]
        return head if node in FANOUT_COLLAPSE_NODES else None
    if event.type in ("task_start", "task_end") and event.node in FANOUT_COLLAPSE_NODES:
        return f"{event.node}:{event.task_id}"
    return None


def _instance_label(node: str, payload: Any) -> str:
    """Human tag for a fan-out instance drawn from its input payload."""
    if not isinstance(payload, Mapping):
        return ""
    note_id = payload.get("note_id")
    note_type = payload.get("note_type")
    if note_type and note_id is not None:
        return f"{note_type} #{note_id}"
    if note_id is not None:
        return f"Note #{note_id}"
    return ""


def _merge_result(prior: Any, payload: Any) -> Any:
    """Accumulate nested ``values`` snapshots into one running result dict.

    Sub-agent ``values`` arrive as progressively fuller partial states; merging
    keeps the union so an instance's result carries every field it has produced
    so far (summary, then concepts, then cancer mentions, …).
    """
    if isinstance(prior, Mapping) and isinstance(payload, Mapping):
        return {**prior, **payload}
    return payload


def _hydrate_case(payload: Any) -> CaseState | None:
    """Re-hydrate a root ``values`` JSON payload into a ``CaseState``.

    Returns ``None`` if the payload cannot be validated (e.g. a partial trace);
    the caller falls back to leaving the model unfed for that event rather than
    aborting the whole replay.
    """
    if not isinstance(payload, Mapping):
        return None
    try:
        return CaseState.model_validate(payload)
    except Exception:
        return None


def _replace_calls(detail: NodeDetail, calls: tuple[dict[str, Any], ...]) -> NodeDetail:
    return NodeDetail(
        map_node_id=detail.map_node_id,
        node=detail.node,
        agent=detail.agent,
        count=detail.count,
        active=detail.active,
        errors=detail.errors,
        started_t=detail.started_t,
        finished_t=detail.finished_t,
        input=detail.input,
        result=detail.result,
        error=detail.error,
        llm_calls=calls,
    )


def _detail_to_dict(detail: NodeDetail) -> dict[str, Any]:
    return {
        "map_node_id": detail.map_node_id,
        "node": detail.node,
        "agent": detail.agent,
        "status": detail.status,
        "count": detail.count,
        "active": detail.active,
        "errors": detail.errors,
        "started_t": detail.started_t,
        "finished_t": detail.finished_t,
        "input": detail.input,
        "result": detail.result,
        "error": detail.error,
        "llm_calls": list(detail.llm_calls),
    }


def _instance_to_dict(instance: InstanceDetail) -> dict[str, Any]:
    return {
        "key": instance.key,
        "node": instance.node,
        "map_node_id": instance.map_node_id,
        "agent": instance.agent,
        "index": instance.index,
        "label": instance.label,
        "status": instance.status,
        "active": instance.active,
        "errors": instance.errors,
        "started_t": instance.started_t,
        "finished_t": instance.finished_t,
        "input": instance.input,
        "result": instance.result,
        "error": instance.error,
        "llm_calls": list(instance.llm_calls),
    }


def _progress_to_dict(progress: Snapshot | None) -> dict[str, Any] | None:
    """Serialize the reused variable-table snapshot into the frontend contract."""
    if progress is None:
        return None
    return {
        "description": progress.description,
        "mode": progress.mode,
        "finished": progress.finished,
        "fatal": progress.fatal,
        "notes_total": progress.notes_total,
        "notes_done": progress.notes_done,
        "review_flags": progress.review_flags,
        "counts": dict(progress.counts),
        "totals": {
            "variables": progress.total_variables,
            "terminal": progress.terminal_variables,
            "groups": progress.total_groups,
            "done_groups": progress.done_groups,
        },
        "groups": [
            {
                "group_id": group.group_id,
                "name": group.name,
                "depth": group.depth,
                "annotation": group.annotation,
                "active": group.active,
                "stage": group.stage.name.lower(),
                "note_count": group.note_count,
                "item_ids": list(group.item_ids),
            }
            for group in progress.groups
        ],
        "variables": [
            {
                "item_id": variable.item_id,
                "name": variable.name,
                "group_id": variable.group_id,
                "stage": variable.stage.name.lower(),
                "attempt": variable.attempt,
                "status": variable.status,
                "value": variable.value,
                "detail": variable.detail,
                "confidence": variable.confidence,
                "flag": variable.flag,
                "terminal": variable.terminal,
            }
            for variable in progress.variables.values()
        ],
        "branches": [
            {
                "key": branch.key,
                "label": branch.label,
                "stage": branch.stage.name.lower(),
                "variables": branch.variables,
                "note_count": branch.note_count,
            }
            for branch in progress.branches
        ],
    }


__all__ = ["DemoState", "DemoSnapshot", "NodeDetail", "InstanceDetail", "replay"]
