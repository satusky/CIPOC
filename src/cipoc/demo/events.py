"""The demo's wire format: one ordered, serializable observation per line.

A :class:`DemoEvent` is the single unit that both trace recording (Phase 1) and
the live UI (Phase 2+) consume, so recording and replay drive an identical
pipeline. Events are produced by merging the two taps described in the package
docstring:

* Tap 1 (graph stream) yields ``task_start`` / ``task_end`` / ``values`` events,
  built from :class:`cipoc.utils.progress.events.ProgressEvent`.
* Tap 2 (LLM callback) yields ``llm_call`` events carrying an :class:`LLMCall`.

A :class:`DemoEvent` is intentionally a plain, JSON-round-trippable record: the
``namespace`` is a tuple in memory and a list on disk, and ``payload`` holds only
JSON-safe data (callers are responsible for reducing Pydantic models to
``model_dump(mode="json")`` before constructing an event). Keeping the schema
dumb here is what lets the trace file be the contract between the recorder and
every downstream consumer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# ``run_start`` / ``run_end`` bracket the stream; ``step_boundary`` is reserved
# for Phase 2 (presenter pauses) so traces recorded now stay forward-compatible.
DemoEventType = Literal[
    "run_start",
    "task_start",
    "task_end",
    "values",
    "llm_call",
    "step_boundary",
    "run_end",
]


@dataclass(frozen=True)
class LLMCall:
    """One captured model interaction, correlated to the graph node that made it.

    All fields are JSON-safe. ``prompt_messages`` is a list of
    ``{"role", "content"}`` dicts; ``response`` is the model's text/serialized
    structured output; ``reasoning`` is the reasoning summary when the endpoint
    returns one; ``usage`` is the token-usage mapping when available.
    """

    node: str
    namespace: tuple[str, ...]
    run_id: str
    parent_run_id: str | None = None
    model: str | None = None
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    response: Any = None
    usage: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["namespace"] = list(self.namespace)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMCall":
        data = dict(data)
        data["namespace"] = tuple(data.get("namespace", ()))
        return cls(**data)


@dataclass(frozen=True)
class DemoEvent:
    """One ordered observation in a demo run.

    ``seq`` is a monotonic index assigned by the producer; ``t`` is elapsed
    seconds since ``run_start``. ``map_node_id`` is the resolved
    ``agent_system.json`` node ID (``None`` for events with no node, e.g.
    ``values``). ``agent`` is the owning subagent inferred from ``namespace``.

    ``task_id`` is the LangGraph task id of the node this event reports on. It is
    empty for events with no node (``values``/``run_*``). Preserving it is what
    lets a replay reconstruct a node's own scope — ``namespace + (f"{node}:
    {task_id}",)`` — which is the key nested-subgraph events resolve against, so
    :class:`~cipoc.demo.state.DemoState` can drive a ``ProgressModel`` off a trace
    exactly as a live run does.
    """

    seq: int
    t: float
    type: DemoEventType
    node: str = ""
    task_id: str = ""
    namespace: tuple[str, ...] = ()
    map_node_id: str | None = None
    agent: str | None = None
    payload: Any = None
    error: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t": round(self.t, 6),
            "type": self.type,
            "node": self.node,
            "task_id": self.task_id,
            "namespace": list(self.namespace),
            "map_node_id": self.map_node_id,
            "agent": self.agent,
            "payload": self.payload,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemoEvent":
        return cls(
            seq=int(data["seq"]),
            t=float(data.get("t", 0.0)),
            type=data["type"],
            node=data.get("node", ""),
            task_id=data.get("task_id", ""),
            namespace=tuple(data.get("namespace", ())),
            map_node_id=data.get("map_node_id"),
            agent=data.get("agent"),
            payload=data.get("payload"),
            error=data.get("error"),
        )


__all__ = ["DemoEventType", "LLMCall", "DemoEvent"]
