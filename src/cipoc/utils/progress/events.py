"""Normalization of raw LangGraph stream items into ``ProgressEvent``.

A graph streamed with ``stream_mode=["values", "tasks"]`` and ``subgraphs=True``
yields ``(namespace, mode, payload)`` tuples; without ``subgraphs`` it yields
``(mode, payload)``. Task payloads are the debug records from
``langgraph.pregel.debug`` — ``{"id", "name", "input", "triggers"}`` on start and
``{"id", "name", "result", "error", "interrupts"}`` on end — so a start is told
from an end by the presence of ``input``.

Namespaces are the binding mechanism the dashboard depends on: a child task's
namespace is ``parent_namespace + (f"{node}:{task_id}",)``, so the segment a
task-start contributes identifies every event nested beneath it. That holds even
for compiled graphs invoked with a bare ``.invoke()`` inside a node body, because
such a graph inherits the parent's stream.

``updates`` is normalized too, for consumers that need a node's own state write
the moment it happens rather than at the next root ``values`` — the orchestrator's
incremental result stream is the one caller. Its payload is the raw
``{node: state_update}`` mapping LangGraph emits. ``custom`` is deliberately not
supported: a graph that requests it builds its own writer instead of inheriting
the parent's, which breaks nested-subagent events entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


EventKind = Literal["task_start", "task_end", "values", "updates"]


@dataclass(frozen=True)
class ProgressEvent:
    """One normalized observation from a running graph."""

    kind: EventKind
    namespace: tuple[str, ...]
    node: str = ""
    task_id: str = ""
    payload: Any = None
    error: Any = None

    @property
    def scope(self) -> tuple[str, ...]:
        """The namespace this task's own children will be nested under."""
        return (*self.namespace, f"{self.node}:{self.task_id}")

    @property
    def is_root(self) -> bool:
        return not self.namespace

    @property
    def writes(self) -> tuple[tuple[str, Any], ...]:
        """``(node, state_update)`` pairs carried by an ``updates`` event.

        Empty for every other kind, so a consumer can iterate unconditionally.
        LangGraph emits one node per item in practice, but the payload is a
        mapping and nothing guarantees that, so this yields all of them.
        """
        if self.kind != "updates" or not isinstance(self.payload, Mapping):
            return ()
        return tuple(self.payload.items())


def normalize(item: Any, *, subgraphs: bool) -> ProgressEvent | None:
    """Turn one raw stream item into a ``ProgressEvent`` (``None`` if unusable)."""
    if subgraphs:
        namespace, mode, payload = item
        namespace = tuple(namespace)
    else:
        mode, payload = item
        namespace = ()

    if mode == "values":
        return ProgressEvent(kind="values", namespace=namespace, payload=payload)
    if mode == "updates":
        if not isinstance(payload, Mapping):
            return None
        nodes = list(payload)
        return ProgressEvent(
            kind="updates",
            namespace=namespace,
            # A convenience for the common single-node item; read ``writes`` when
            # the mapping may carry more than one.
            node=str(nodes[0]) if len(nodes) == 1 else "",
            payload=payload,
        )
    if mode != "tasks" or not isinstance(payload, Mapping):
        return None

    node = str(payload.get("name", ""))
    task_id = str(payload.get("id", ""))
    if "input" in payload:
        return ProgressEvent(
            kind="task_start",
            namespace=namespace,
            node=node,
            task_id=task_id,
            payload=payload["input"],
        )
    return ProgressEvent(
        kind="task_end",
        namespace=namespace,
        node=node,
        task_id=task_id,
        payload=payload.get("result"),
        error=payload.get("error"),
    )


def field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a Pydantic model or a mapping.

    Task payloads arrive as real models in the orchestrator and as plain dicts in
    hand-built or replayed graphs; every reader here has to tolerate both.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def enum_value(value: Any) -> Any:
    """Unwrap ``Enum.value`` so a status compares equal whether or not it is parsed."""
    return getattr(value, "value", value)


__all__ = ["EventKind", "ProgressEvent", "normalize", "field", "enum_value"]
