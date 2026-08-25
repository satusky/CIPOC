"""Bridge runtime LangGraph node names to conceptual ``agent_system.json`` IDs.

The live event stream (Tap 1) reports the *runtime* node names wired into the
compiled graphs — the ``add_node("...")`` names enumerated in
``cipoc.utils.progress.model.DEFAULT_NODE_KINDS``. The animated map (Panel 1)
uses the *conceptual* node IDs authored in
``src/cipoc/agents/visualization/agent_system.json``. The two vocabularies
overlap but are not identical, so this module is the single translation layer.

Two subtleties it handles:

* The bare node name ``initialize`` exists in all four agents (orchestrator,
  scanner, retriever, extractor). It is disambiguated by the *namespace*: a
  scanner/retriever/extractor subgraph runs under a namespace segment naming the
  orchestrator node that invoked it (``note_branch`` / ``retrieve_notes`` /
  ``extract``), so :func:`infer_agent` recovers the owning agent from there.
* Orchestrator container nodes that merely fan out or delegate map onto the
  corresponding conceptual marker (e.g. ``scan_notes`` -> ``fan_out_notes``,
  ``extract`` -> ``extractor_initialize``).

A test (Phase 0) asserts every ``DEFAULT_NODE_KINDS`` name resolves to a real
``agent_system.json`` node ID, so the map cannot silently drift from the graph.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from cipoc.utils.progress.model import DEFAULT_NODE_KINDS


AGENT_SYSTEM_JSON = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "visualization"
    / "agent_system.json"
)

# The namespace segment (a runtime orchestrator node) that owns each subagent's
# nested execution. Read by :func:`infer_agent` — the deepest match wins, so an
# extractor call nested under an extract branch resolves to ``extractor`` even
# though the branch also passed through ``note_branch``-free orchestrator nodes.
_NAMESPACE_OWNER: dict[str, str] = {
    "extract": "extractor",
    "retrieve_notes": "retriever",
    "note_branch": "scanner",
}

# Runtime node name -> conceptual map node ID, for every name that is NOT the
# agent-ambiguous bare ``initialize`` (handled separately below).
_FLAT: dict[str, str] = {
    # Orchestrator root flow.
    "scan_notes": "fan_out_notes",
    "note_branch": "scanner_initialize",
    "characterize_corpus": "characterize_corpus",
    "check_state": "check_state",
    "plan_extraction": "plan_extraction",
    "extract_branch": "fan_out_groups",
    "retrieve_notes": "hard_filter_notes",
    "extract": "extractor_initialize",
    "merge_and_update": "merge_and_update",
    "finalize_case": "finalize_case",
    # Note-scanner subagent.
    "summarize_note": "scanner_summarize_note",
    "detect_concepts": "scanner_detect_concepts",
    "get_cancer_mentions": "scanner_get_cancer_mentions",
    # Note-retriever subagent.
    "identify_relevant_notes": "retriever_identify_relevant_notes",
    # Extractor subagent.
    "load_notes": "extractor_load_notes",
    "extract_group_values": "extractor_extract_group_values",
    "variable_branch": "fan_out_variables",
    "extract_individual_value": "extractor_extract_individual_value",
    "validate_extraction": "extractor_validate_extraction",
    "repair_invalid_extraction": "extractor_repair_invalid_extraction",
    "complete_variable": "extractor_complete_variable",
    "merge_variable_results": "merge_variable_results",
}

# The bare ``initialize`` node, resolved by owning agent.
_INITIALIZE_BY_AGENT: dict[str, str] = {
    "orchestrator": "initialize_case",
    "scanner": "scanner_initialize",
    "retriever": "retriever_initialize",
    "extractor": "extractor_initialize",
}

# Second bridge: the fine ``agent_system.json`` IDs above collapse onto the coarse
# blocks of the simplified *overview* flowchart (``agent_flowcharts.json``) that
# Panel 1 draws. Snapshots report the fine IDs; the map highlights the coarse
# block that contains them. A whole subagent (many fine nodes) lights up as its
# one block. Every value here must be a node ID in the overview chart, and every
# fine ID that :func:`map_node_id` can emit must appear as a key — the demo web
# test asserts both so the animated map cannot drift from the graph.
_OVERVIEW_BLOCK: dict[str, str] = {
    "initialize_case": "initialize_case",
    "fan_out_notes": "scanner_agent_block",
    "scanner_initialize": "scanner_agent_block",
    "scanner_summarize_note": "scanner_agent_block",
    "scanner_detect_concepts": "scanner_agent_block",
    "scanner_get_cancer_mentions": "scanner_agent_block",
    "characterize_corpus": "characterize_corpus",
    "check_state": "eligible_groups_gate",
    "plan_extraction": "eligible_groups_gate",
    "fan_out_groups": "retriever_agent_block",
    "hard_filter_notes": "retriever_agent_block",
    "retriever_initialize": "retriever_agent_block",
    "retriever_identify_relevant_notes": "retriever_agent_block",
    "extractor_initialize": "extractor_agent_block",
    "extractor_load_notes": "extractor_agent_block",
    "extractor_extract_group_values": "extractor_agent_block",
    "fan_out_variables": "extractor_agent_block",
    "extractor_extract_individual_value": "extractor_agent_block",
    "extractor_validate_extraction": "extractor_agent_block",
    "extractor_repair_invalid_extraction": "extractor_agent_block",
    "extractor_complete_variable": "extractor_agent_block",
    "merge_variable_results": "extractor_agent_block",
    "merge_and_update": "update_case",
    "finalize_case": "finalize_case",
}


def overview_block_for(map_id: str) -> str | None:
    """Collapse a fine ``agent_system.json`` node ID onto its overview block."""
    return _OVERVIEW_BLOCK.get(map_id)


def overview_block_map() -> dict[str, str]:
    """The full fine-ID -> overview-block mapping (copy), for the frontend."""
    return dict(_OVERVIEW_BLOCK)


def _segment_node(segment: str) -> str:
    """A namespace segment is ``f"{node}:{task_id}"`` — return the node part."""
    return segment.split(":", 1)[0]


def infer_agent(namespace: Iterable[str]) -> str:
    """Return the owning subagent for a namespace, or ``"orchestrator"``.

    Checks the namespace for the orchestrator node that delegates to each
    subagent. Extractor is checked first because an extractor call is nested
    below the extract branch and should win over any shallower marker.
    """
    nodes = {_segment_node(segment) for segment in namespace}
    for marker, agent in _NAMESPACE_OWNER.items():
        if marker in nodes:
            return agent
    return "orchestrator"


def map_node_id(node_name: str, namespace: Iterable[str] = ()) -> str | None:
    """Resolve a runtime node name (+ namespace) to a map node ID.

    Returns ``None`` for names with no conceptual counterpart (unknown nodes),
    so callers can decide whether to warn. The bare ``initialize`` node is
    disambiguated by the agent inferred from ``namespace``.
    """
    if node_name == "initialize":
        return _INITIALIZE_BY_AGENT[infer_agent(namespace)]
    return _FLAT.get(node_name)


@lru_cache(maxsize=1)
def map_node_ids() -> frozenset[str]:
    """Every node ID present in ``agent_system.json`` (cached)."""
    data = json.loads(AGENT_SYSTEM_JSON.read_text())
    return frozenset(
        node["data"]["id"] for node in data["elements"]["nodes"]
    )


def unmapped_runtime_nodes() -> dict[str, str | None]:
    """Runtime nodes that fail to resolve to a real map node ID.

    Returns a ``{runtime_node: resolved_or_None}`` dict of offenders — empty when
    the mapping fully covers ``DEFAULT_NODE_KINDS``. This is what the coverage
    test asserts is empty, and what surfaces drift when a graph node is added.
    """
    valid = map_node_ids()
    offenders: dict[str, str | None] = {}
    for node_name in DEFAULT_NODE_KINDS:
        # ``initialize`` is resolved per agent; check all four owners.
        candidates = (
            _INITIALIZE_BY_AGENT.values()
            if node_name == "initialize"
            else [map_node_id(node_name)]
        )
        for resolved in candidates:
            if resolved not in valid:
                offenders[node_name] = resolved
                break
    return offenders


__all__ = [
    "AGENT_SYSTEM_JSON",
    "infer_agent",
    "map_node_id",
    "map_node_ids",
    "unmapped_runtime_nodes",
    "overview_block_for",
    "overview_block_map",
]
