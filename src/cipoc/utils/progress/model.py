"""The dashboard's state: fold ``ProgressEvent``s into an immutable ``Snapshot``.

Pure bookkeeping — no terminal awareness, no I/O, no clock reading beyond
recording monotonic start times. ``ingest`` is deliberately cheap because a
fan-out-heavy orchestrator run duplexes hundreds of nested-graph events into one
stream; painting is somebody else's job, and reads the snapshot instead.

Two sources feed the same variable table, and they are not equally trusted:

* Root ``values`` carries the durable ``CaseVariableResult`` per item — the only
  authority on terminal status, coded value, confidence, and validity.
* Task events carry the *transient* pipeline position (retrieve → extract →
  validate, plus the repair-attempt counter), which no state channel records.

So a variable's status always comes from ``values`` and its stage from tasks,
and a terminal status wins over any stage still in flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace
from enum import IntEnum
from typing import Any, Iterable, Literal, Mapping

from cipoc.models import VariableStatus
from cipoc.models.case import TERMINAL_VARIABLE_STATUSES
from cipoc.tools import GroupNode, corpus_gate_passes, site_applies

from .events import ProgressEvent, enum_value, field


TaskKind = Literal["llm", "deterministic", "container"]

# Which graph nodes cost an LLM call. Drives the compact timeline's kind marker
# and keeps container nodes (which only fan out) out of step counts.
DEFAULT_NODE_KINDS: dict[str, TaskKind] = {
    "initialize": "deterministic",
    "load_notes": "deterministic",
    "extract_group_values": "llm",
    "variable_branch": "container",
    "extract_individual_value": "llm",
    "validate_extraction": "deterministic",
    "repair_invalid_extraction": "llm",
    "complete_variable": "container",
    "merge_variable_results": "deterministic",
    "summarize_note": "llm",
    "detect_concepts": "llm",
    "get_cancer_mentions": "llm",
    "scan_notes": "deterministic",
    "note_branch": "container",
    "characterize_corpus": "deterministic",
    "check_state": "deterministic",
    "plan_extraction": "deterministic",
    "extract_branch": "container",
    "retrieve_notes": "llm",
    "extract": "container",
    "merge_and_update": "deterministic",
    "finalize_case": "deterministic",
    "identify_relevant_notes": "llm",
}

# Corpus gates and site restrictions get a short label so a group header can say
# *why* it is (or is not) running without eating the whole row.
GATE_LABELS = {
    "metastasis_present": "mets",
    "treatment_present": "treatment",
    "lymph_nodes_removed": "nodes",
}

_TERMINAL_STATUSES = {status.value for status in TERMINAL_VARIABLE_STATUSES}


class Stage(IntEnum):
    """Position in the three-step pipeline a variable's group walks."""

    IDLE = 0
    RETRIEVE = 1
    EXTRACT = 2
    VALIDATE = 3
    DONE = 4


STAGE_LABELS = {
    Stage.IDLE: "",
    Stage.RETRIEVE: "retrieve",
    Stage.EXTRACT: "extract",
    Stage.VALIDATE: "validate",
    Stage.DONE: "",
}


@dataclass(frozen=True)
class VariableSnapshot:
    item_id: int
    name: str
    group_id: str
    stage: Stage = Stage.IDLE
    attempt: int = 0
    status: str = VariableStatus.PENDING.value
    value: str | None = None
    detail: str | None = None
    confidence: str | None = None
    flag: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True)
class GroupSnapshot:
    group_id: str
    name: str
    depth: int
    annotation: str = ""
    item_ids: tuple[int, ...] = ()
    active: bool = False
    stage: Stage = Stage.IDLE
    note_count: int | None = None


@dataclass(frozen=True)
class BranchSnapshot:
    key: str
    label: str
    stage: Stage
    variables: int
    note_count: int | None
    started_at: float


@dataclass(frozen=True)
class NodeSnapshot:
    """One node of the compact timeline, aggregated over its fan-out tasks."""

    name: str
    kind: TaskKind
    started: int
    done: int
    errors: int
    started_at: float
    finished_at: float | None = None

    @property
    def state(self) -> Literal["active", "ok", "error"]:
        if self.errors:
            return "error"
        return "ok" if self.done >= self.started else "active"


@dataclass(frozen=True)
class Snapshot:
    """An immutable read of the run, safe to hand to a repaint thread.

    Timestamps stay absolute (monotonic) rather than pre-computed elapsed times,
    so a renderer painting between two events still shows a moving clock.
    """

    description: str
    started_at: float
    groups: tuple[GroupSnapshot, ...] = ()
    variables: Mapping[int, VariableSnapshot] = dc_field(default_factory=dict)
    branches: tuple[BranchSnapshot, ...] = ()
    nodes: tuple[NodeSnapshot, ...] = ()
    notes_total: int = 0
    notes_done: int = 0
    counts: Mapping[str, int] = dc_field(default_factory=dict)
    fatal: str | None = None
    finished: bool = False
    show_note_counts: bool = False
    review_flags: int = 0
    standalone: bool = False

    @property
    def mode(self) -> Literal["case", "standalone", "compact"]:
        if not self.variables:
            return "compact"
        return "standalone" if self.standalone else "case"

    @property
    def total_tasks(self) -> int:
        return sum(node.started for node in self.nodes)

    @property
    def completed_tasks(self) -> int:
        return sum(min(node.done, node.started) for node in self.nodes)

    @property
    def total_variables(self) -> int:
        return len(self.variables)

    @property
    def terminal_variables(self) -> int:
        return sum(variable.terminal for variable in self.variables.values())

    @property
    def total_groups(self) -> int:
        return sum(1 for group in self.groups if group.item_ids)

    @property
    def done_groups(self) -> int:
        return sum(
            1
            for group in self.groups
            if group.item_ids
            and all(self.variables[item_id].terminal for item_id in group.item_ids)
        )

    def descendants(self, group: GroupSnapshot) -> tuple[GroupSnapshot, ...]:
        """``group`` plus the sub-groups nested under it, in display order."""
        index = self.groups.index(group)
        nested = [group]
        for candidate in self.groups[index + 1 :]:
            if candidate.depth <= group.depth:
                break
            nested.append(candidate)
        return tuple(nested)

    def group_item_ids(self, group: GroupSnapshot) -> tuple[int, ...]:
        """Every item under ``group``, including its sub-groups' items."""
        return tuple(
            item_id for node in self.descendants(group) for item_id in node.item_ids
        )


def _annotation(group: Any) -> str:
    """Static header annotation explaining what governs a group's eligibility."""
    if group is None:
        return ""
    applies_to = field(group, "applies_to")
    if applies_to is not None:
        sites = list(field(applies_to, "gross_primary_sites", []) or [])
        sites += list(field(applies_to, "histology_families", []) or [])
        if sites:
            return "site:" + "/".join(sites[:2])
    gates = field(group, "gate") or []
    if gates:
        return "gate:" + "+".join(
            GATE_LABELS.get(enum_value(gate), str(enum_value(gate))) for gate in gates
        )
    if field(group, "stage") == "initial":
        return "initial"
    return ""


class ProgressModel:
    """Folds the event stream into the dashboard state."""

    def __init__(
        self,
        description: str,
        started_at: float,
        *,
        target_groups: Iterable[Any] | None = None,
        group_hierarchy: Iterable[GroupNode] | None = None,
        graph_input: Any = None,
        node_kinds: Mapping[str, TaskKind] | None = None,
        show_note_counts: bool = False,
        include_input_group: bool = True,
    ):
        self.description = description
        self.started_at = started_at
        self.show_note_counts = show_note_counts
        self._node_kinds = {**DEFAULT_NODE_KINDS, **(node_kinds or {})}

        self._configs: dict[str, Any] = {
            str(field(group, "group_id") or index): group
            for index, group in enumerate(target_groups or [])
        }
        self._names = {
            int(field(variable, "item_id")): str(
                field(variable, "name") or f"Item {field(variable, 'item_id')}"
            )
            for group in (target_groups or [])
            for variable in (field(group, "variables", []) or [])
        }

        self._groups: dict[str, GroupSnapshot] = {}
        self._order: list[str] = []
        self._variables: dict[int, VariableSnapshot] = {}
        self._standalone = False
        self._build_tree(
            target_groups,
            group_hierarchy,
            graph_input if include_input_group else None,
        )

        # Namespace prefix -> the group / variable every nested event belongs to.
        self._group_scopes: dict[tuple[str, ...], str] = {}
        self._variable_scopes: dict[tuple[str, ...], int] = {}
        self._branches: dict[tuple[str, ...], BranchSnapshot] = {}
        self._nodes: dict[str, NodeSnapshot] = {}
        # The node timeline is the fallback view for runs with no variable table
        # and the second half of a standalone extractor run's display. Case runs
        # stay focused on their variable groups, even if configured with one.
        self._track_nodes = self._standalone or not self._variables
        self._note_tasks: set[str] = set()
        note_corpus = field(graph_input, "note_corpus", None)
        if note_corpus is None:
            note_corpus = field(graph_input, "notes", [])
        self._notes_total = len(note_corpus or {})
        self._notes_done = 0
        self._fatal: str | None = None
        self._finished = False
        self._review_flags = 0
        self._report_seen = False

    # --- Construction -----------------------------------------------------

    def _build_tree(
        self,
        target_groups: Iterable[Any] | None,
        group_hierarchy: Iterable[GroupNode] | None,
        graph_input: Any,
    ) -> None:
        """Seed the display tree from the richest source available.

        The config hierarchy is preferred because it keeps sub-groups nested the
        way a reader recognizes them; ``load_variable_groups`` deliberately
        flattens those into peers. Failing that, the flat plan groups are used,
        and failing *that* a standalone run's requested variables become a single
        synthetic group.
        """
        nodes = list(group_hierarchy or [])
        if nodes:
            depths = {None: -1}
            for node in nodes:
                depth = depths[node.parent_id] + 1
                depths[node.group_id] = depth
                self._add_group(
                    node.group_id, node.name, depth, node.item_ids
                )
            return

        groups = list(target_groups or [])
        if groups:
            for index, group in enumerate(groups):
                group_id = str(field(group, "group_id") or index)
                item_ids = tuple(
                    int(field(variable, "item_id"))
                    for variable in (field(group, "variables", []) or [])
                )
                self._add_group(
                    group_id, str(field(group, "name") or group_id), 0, item_ids
                )
            return

        requested = field(graph_input, "requested_variables")
        variables = field(requested, "variables", []) or []
        if not variables:
            return
        for variable in variables:
            item_id = int(field(variable, "item_id"))
            self._names.setdefault(
                item_id, str(field(variable, "name") or f"Item {item_id}")
            )
        group_id = str(field(requested, "group_id") or "requested")
        self._configs.setdefault(group_id, requested)
        self._standalone = True
        self._add_group(
            group_id,
            str(field(requested, "name") or self.description),
            0,
            tuple(int(field(variable, "item_id")) for variable in variables),
        )
        notes = field(graph_input, "notes", None)
        if notes is not None:
            self._groups[group_id] = replace(
                self._groups[group_id], note_count=len(notes)
            )

    def _add_group(
        self, group_id: str, name: str, depth: int, item_ids: tuple[int, ...]
    ) -> None:
        self._groups[group_id] = GroupSnapshot(
            group_id=group_id,
            name=name,
            depth=depth,
            annotation=_annotation(self._configs.get(group_id)),
            item_ids=item_ids,
        )
        self._order.append(group_id)
        for item_id in item_ids:
            self._variables[item_id] = VariableSnapshot(
                item_id=item_id,
                name=self._names.get(item_id, f"Item {item_id}"),
                group_id=group_id,
            )

    # --- Ingest -----------------------------------------------------------

    def ingest(self, event: ProgressEvent, now: float) -> None:
        """Fold one event in. ``updates`` events are deliberately ignored: they
        carry a node's own write, which is redundant with — and less trustworthy
        than — the root ``values`` this model treats as authoritative. They ride
        the same stream only because ``astream_results`` consumes them.
        """
        if event.kind == "values":
            if event.is_root:
                self._ingest_values(event.payload)
        elif event.kind == "task_start":
            self._ingest_start(event, now)
        elif event.kind == "task_end":
            self._ingest_end(event, now)

    def _ingest_start(self, event: ProgressEvent, now: float) -> None:
        node = event.node
        if self._track_nodes and event.is_root:
            existing = self._nodes.get(node)
            self._nodes[node] = (
                replace(existing, started=existing.started + 1, finished_at=None)
                if existing is not None
                else NodeSnapshot(
                    name=node,
                    kind=self._node_kinds.get(node, "deterministic"),
                    started=1,
                    done=0,
                    errors=0,
                    started_at=now,
                )
            )

        if node == "note_branch":
            self._note_tasks.add(event.task_id)
            return

        if node == "extract_branch":
            group_id = str(field(field(event.payload, "requested_variables"), "group_id") or "")
            if group_id in self._groups:
                self._group_scopes[event.scope] = group_id
                self._start_branch(group_id, event.scope, now)
            return

        group_id = self._resolve(self._group_scopes, event.namespace)
        item_id = self._resolve(self._variable_scopes, event.namespace)

        if node == "variable_branch":
            task = field(event.payload, "task")
            variable = field(task, "variable")
            if variable is not None:
                item_id = int(field(variable, "item_id"))
                self._variable_scopes[event.scope] = item_id

        stage = {
            "retrieve_notes": Stage.RETRIEVE,
            "extract": Stage.EXTRACT,
            "extract_group_values": Stage.EXTRACT,
            "variable_branch": Stage.EXTRACT,
            "extract_individual_value": Stage.EXTRACT,
            "repair_invalid_extraction": Stage.EXTRACT,
            "validate_extraction": Stage.VALIDATE,
        }.get(node)
        if stage is None:
            return

        attempt = int(field(field(event.payload, "task"), "extraction_attempts", 0) or 0)
        if item_id is not None:
            self._advance_variable(item_id, stage, attempt)
            group_id = group_id or self._variables[item_id].group_id
        elif group_id is not None:
            for member in self._groups[group_id].item_ids:
                self._advance_variable(member, stage, attempt)
        if group_id is not None:
            self._advance_group(group_id, stage)

    def _ingest_end(self, event: ProgressEvent, now: float) -> None:
        node = event.node
        if self._track_nodes and event.is_root:
            existing = self._nodes.get(node)
            if existing is not None:
                self._nodes[node] = replace(
                    existing,
                    done=existing.done + 1,
                    errors=existing.errors + (event.error is not None),
                    finished_at=now,
                )

        if node == "note_branch":
            self._note_tasks.discard(event.task_id)
            self._notes_done += 1
            return

        if node == "retrieve_notes":
            group_id = self._resolve(self._group_scopes, event.namespace)
            note_ids = field(event.payload, "retrieved_note_ids", []) or []
            if group_id is not None:
                self._groups[group_id] = replace(
                    self._groups[group_id], note_count=len(note_ids)
                )
                self._retime_branch(group_id, note_count=len(note_ids))
            return

        if node == "extract_branch":
            group_id = self._group_scopes.get(event.scope)
            if group_id is not None:
                self._branches.pop(event.scope, None)
                self._groups[group_id] = replace(
                    self._groups[group_id], active=False, stage=Stage.DONE
                )

    def _ingest_values(self, values: Any) -> None:
        note_corpus = field(values, "note_corpus", None)
        if note_corpus:
            self._notes_total = max(self._notes_total, len(note_corpus))

        self._fatal = field(values, "fatal_blocker") or self._fatal
        report = field(values, "report")
        if report is not None:
            self._report_seen = True
            self._review_flags = len(field(report, "flags", []) or [])

        self._apply_eligibility(values)

        results = field(values, "variable_results", {}) or {}
        if isinstance(results, Mapping):
            result_items = results.items()
        else:
            result_items = (
                (field(result, "item_id"), _standalone_result(result))
                for result in results
            )
        for key, result in result_items:
            if key is None:
                continue
            item_id = int(key)
            current = self._variables.get(item_id)
            if current is None:
                continue
            self._variables[item_id] = _apply_result(current, result)

        extracted = field(field(values, "extracted_values"), "variables", []) or []
        for result in extracted:
            item_id = int(field(result, "item_id"))
            current = self._variables.get(item_id)
            if current is not None:
                self._variables[item_id] = _apply_result(
                    current, _standalone_result(result)
                )

    def _apply_eligibility(self, values: Any) -> None:
        """Mark each gate/site annotation with the verdict the planner reached.

        The predicates are the same pure functions ``plan_extraction`` uses, so
        the dashboard never guesses: once corpus characterization has produced
        descriptors, a group's ✓/✗ is exactly the planner's own answer.
        """
        descriptors = field(values, "note_corpus_descriptors")
        if descriptors is None:
            return
        facts = field(values, "case_facts")
        for group_id, group in self._groups.items():
            config = self._configs.get(group_id)
            if config is None or not group.annotation:
                continue
            base = group.annotation.split(" ")[0]
            if base.startswith("site:"):
                passes = site_applies(field(config, "applies_to"), facts)
            elif base.startswith("gate:"):
                passes = corpus_gate_passes(field(config, "gate"), descriptors)
            else:
                continue
            self._groups[group_id] = replace(
                group, annotation=f"{base} {'✓' if passes else '✗'}"
            )

    # --- Mutation helpers -------------------------------------------------

    def _advance_variable(self, item_id: int, stage: Stage, attempt: int) -> None:
        variable = self._variables.get(item_id)
        if variable is None or variable.terminal:
            return
        self._variables[item_id] = replace(variable, stage=stage, attempt=attempt)

    def _advance_group(self, group_id: str, stage: Stage) -> None:
        group = self._groups[group_id]
        if group.stage is not stage or not group.active:
            self._groups[group_id] = replace(group, stage=stage, active=True)
        for key, branch in self._branches.items():
            if branch.key == group_id and branch.stage is not stage:
                self._branches[key] = replace(branch, stage=stage)

    def _start_branch(self, group_id: str, scope: tuple[str, ...], now: float) -> None:
        group = self._groups[group_id]
        pending = [
            item_id
            for item_id in group.item_ids
            if not self._variables[item_id].terminal
        ]
        self._groups[group_id] = replace(group, active=True, stage=Stage.RETRIEVE)
        self._branches[scope] = BranchSnapshot(
            key=group_id,
            label=group.name,
            stage=Stage.RETRIEVE,
            variables=len(pending) or len(group.item_ids),
            note_count=None,
            started_at=now,
        )
        for item_id in pending:
            self._advance_variable(item_id, Stage.RETRIEVE, 0)

    def _retime_branch(self, group_id: str, *, note_count: int) -> None:
        for key, branch in self._branches.items():
            if branch.key == group_id:
                self._branches[key] = replace(branch, note_count=note_count)

    @staticmethod
    def _resolve(scopes: Mapping[tuple[str, ...], Any], namespace: tuple[str, ...]) -> Any:
        """Longest namespace prefix bound to a group/variable, if any."""
        for length in range(len(namespace), 0, -1):
            found = scopes.get(namespace[:length])
            if found is not None:
                return found
        return None

    # --- Lifecycle --------------------------------------------------------

    def fail(self, error: BaseException) -> None:
        self._fatal = str(error) or error.__class__.__name__
        self._finished = True

    def finish(self) -> None:
        self._finished = True

    # --- Read -------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        counts: dict[str, int] = {}
        variables: dict[int, VariableSnapshot] = {}
        for item_id, variable in self._variables.items():
            if self._finished and not variable.terminal:
                variable = replace(variable, stage=Stage.IDLE, attempt=0)
            variables[item_id] = variable
            counts[variable.status] = counts.get(variable.status, 0) + 1
        review_flags = (
            self._review_flags
            if self._report_seen
            else sum(variable.flag is not None for variable in variables.values())
        )
        groups = tuple(self._groups[group_id] for group_id in self._order)
        if self._finished and self._standalone:
            groups = tuple(
                replace(
                    group,
                    active=False,
                    stage=(
                        Stage.DONE
                        if group.item_ids
                        and all(variables[item_id].terminal for item_id in group.item_ids)
                        else group.stage
                    ),
                )
                for group in groups
            )
        return Snapshot(
            description=self.description,
            started_at=self.started_at,
            groups=groups,
            variables=variables,
            branches=() if self._finished else tuple(self._branches.values()),
            nodes=tuple(self._nodes.values()),
            notes_total=self._notes_total,
            notes_done=self._notes_done,
            counts=counts,
            fatal=self._fatal,
            finished=self._finished,
            show_note_counts=self.show_note_counts,
            review_flags=review_flags,
            standalone=self._standalone,
        )


def _apply_result(variable: VariableSnapshot, result: Any) -> VariableSnapshot:
    """Overlay one durable ``CaseVariableResult`` onto a variable row."""
    status = str(enum_value(field(result, "status", VariableStatus.PENDING.value)))
    if status == VariableStatus.PENDING.value:
        return variable

    extraction = field(result, "extraction")
    confidence = enum_value(field(extraction, "presence_confidence"))
    is_valid = field(extraction, "is_valid", True)
    value = field(result, "value")

    if status == VariableStatus.BLOCKED.value:
        blockers = field(result, "blocking_item_ids", []) or []
        detail = "←" + ",".join(str(item) for item in blockers[:3]) if blockers else field(result, "reason")
    elif status in {VariableStatus.ERROR.value, VariableStatus.NOT_APPLICABLE.value}:
        detail = field(result, "reason")
    else:
        detail = None

    flag = None
    if extraction is not None and not is_valid:
        flag = "!"
    elif value is not None and confidence == "low":
        flag = "?"

    return replace(
        variable,
        stage=Stage.DONE,
        attempt=0,
        status=status,
        value=None if value is None else str(value),
        detail=detail,
        confidence=None if confidence is None else str(confidence),
        flag=flag,
    )


def _standalone_result(extraction: Any) -> dict[str, Any]:
    """Give a validated extractor output the case-result shape used above."""
    is_valid = bool(field(extraction, "is_valid", True))
    value = field(extraction, "value")
    if not is_valid:
        status = VariableStatus.ERROR.value
        errors = field(extraction, "validation_errors", []) or []
        reason = "; ".join(str(error) for error in errors) or "Extraction failed validation."
    elif value is None:
        status = VariableStatus.NOT_FOUND.value
        reason = None
    else:
        status = VariableStatus.EXTRACTED.value
        reason = None
    return {
        "status": status,
        "value": value,
        "reason": reason,
        "extraction": extraction,
    }


__all__ = [
    "DEFAULT_NODE_KINDS",
    "BranchSnapshot",
    "GroupSnapshot",
    "NodeSnapshot",
    "ProgressModel",
    "Snapshot",
    "Stage",
    "STAGE_LABELS",
    "TaskKind",
    "VariableSnapshot",
]
