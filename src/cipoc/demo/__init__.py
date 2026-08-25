"""Demo visualization package for the CIPOC pipeline.

Additive, isolated tooling that turns a single orchestrator run into a
web-dashboard demo revealing the decision-making under the hood. Nothing here is
imported by the production pipeline; the core agents, orchestrator, LLM layer,
and terminal progress display are untouched.

The demo taps the compiled LangGraph in two complementary places and merges them
into one ordered, serializable :class:`~cipoc.demo.events.DemoEvent` stream:

* **Tap 1** — the existing ``graph.stream(stream_mode=["values","tasks"],
  subgraphs=True)`` topology/state stream, normalized by
  :func:`cipoc.utils.progress.events.normalize`.
* **Tap 2** — a LangChain callback handler
  (:class:`~cipoc.demo.capture.LLMCaptureHandler`) attached via the stream config,
  capturing raw prompts, reasoning, responses, and token usage per graph node.

See ``planning/demo_visualization_plan.md`` for the full design.
"""

from __future__ import annotations

from .capture import LLMCaptureHandler
from .events import DemoEvent, DemoEventType, LLMCall
from .mapping import infer_agent, map_node_id, unmapped_runtime_nodes
from .serialize import to_jsonable
from .state import DemoSnapshot, DemoState, NodeDetail, replay
from .steps import Step, build_steps
from .stream import merge_events, record_demo_trace, run_demo_stream
from .trace import TraceWriter, iter_trace, read_trace, write_trace

# NOTE: ``server`` and ``__main__`` are intentionally NOT imported here — they
# pull in the demo-only ``fastapi``/``uvicorn`` extras, and the rest of the demo
# package (record/replay/state) must stay importable without them installed.

__all__ = [
    "DemoEvent",
    "DemoEventType",
    "LLMCall",
    "LLMCaptureHandler",
    "infer_agent",
    "map_node_id",
    "unmapped_runtime_nodes",
    "to_jsonable",
    "DemoState",
    "DemoSnapshot",
    "NodeDetail",
    "replay",
    "Step",
    "build_steps",
    "merge_events",
    "run_demo_stream",
    "record_demo_trace",
    "TraceWriter",
    "iter_trace",
    "read_trace",
    "write_trace",
]
