"""Generate the unified CIPOC execution graph as Cytoscape.js JSON.

The output contains no fixed coordinates. A graph library owns layout and
rendering while this script remains the source of truth for graph semantics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "cipoc"
    / "agents"
    / "visualization"
    / "agent_system.json"
)
DEFAULT_HTML_OUTPUT = DEFAULT_OUTPUT.with_suffix(".html")
CYTOSCAPE_VERSION = "3.33.1"
CYTOSCAPE_URL = (
    f"https://cdn.jsdelivr.net/npm/cytoscape@{CYTOSCAPE_VERSION}/dist/cytoscape.min.js"
)
CYTOSCAPE_START_MARKER = "/* CYTOSCAPE_BUNDLE_START */"
CYTOSCAPE_END_MARKER = "/* CYTOSCAPE_BUNDLE_END */"

# UNC Chapel Hill brand palette. These are the canonical agent hues: they are
# baked into the generated JSON, served as ``metadata.agent_colors``, and written
# onto the demo's ``:root`` at runtime, so they override the fallbacks in
# ``demo/web/styles.css``. Change them here, then regenerate the JSON.
#
# Four hues held as far apart as the palette allows. The retriever cannot be pink
# or amber — the gate disc runs retrieve -> extract -> open/shut on one shape, and
# those are the error and in-progress colours — which leaves Basin Slate.
AGENT_COLORS = {
    "orchestrator": "#13294B",  # Navy, PMS 2767
    "scanner": "#00A5AD",  # Tile Teal, PMS 7466
    "retriever": "#4F758B",  # Basin Slate, PMS 5405
    "extractor": "#4B9CD3",  # Carolina Blue, PMS 542
}

# Label ink on the pale node fills: Navy, 14.7:1 on white.
INK = "#13294B"

# Node-kind fills, all breakdowns of the primary palette plus one Sunburst wash
# to mark the steps that call an LLM.
KIND_COLORS = {
    "endpoint": "#13294B",  # Navy
    "deterministic": "#F1F3F6",  # Navy at 8%
    "llm": "#FFF4CC",  # Sunburst Yellow at 15%
    "decision": "#FFFFFF",
    "fanout": "#E4F1F9",  # Carolina Blue at 15%
    "convergence": "#E4F1F9",
}


def build_graph() -> dict[str, Any]:
    """Build one connected graph spanning the orchestrator and all subagents."""
    nodes: list[dict[str, dict[str, Any]]] = []
    edges: list[dict[str, dict[str, Any]]] = []

    def node(
        node_id: str,
        label: str,
        agent: str,
        kind: str,
        *,
        detail: str = "",
        multiplicity: str = "once",
        implementation: str = "langgraph_node",
    ) -> None:
        nodes.append(
            {
                "data": {
                    "id": node_id,
                    "label": label,
                    "agent": agent,
                    "kind": kind,
                    "detail": detail,
                    "multiplicity": multiplicity,
                    "implementation": implementation,
                }
            }
        )

    def edge(
        source: str,
        target: str,
        agent: str,
        *,
        label: str = "",
        kind: str = "flow",
        multiplicity: str = "once",
    ) -> None:
        edges.append(
            {
                "data": {
                    "id": f"edge_{len(edges) + 1:02d}",
                    "source": source,
                    "target": target,
                    "label": label,
                    "agent": agent,
                    "kind": kind,
                    "multiplicity": multiplicity,
                }
            }
        )

    # Case setup and note-scanner fan-out.
    node("case_start", "START", "orchestrator", "endpoint", implementation="virtual_endpoint")
    node(
        "initialize_case",
        "Initialize case",
        "orchestrator",
        "deterministic",
        detail="Seed targets, structured values, pending results, and case facts",
    )
    node(
        "fan_out_notes",
        "Fan out notes",
        "orchestrator",
        "fanout",
        detail="Send one scanner branch per clinical note",
        implementation="fanout_marker",
    )
    node(
        "scanner_initialize",
        "Initialize note scan",
        "scanner",
        "deterministic",
        multiplicity="per_note",
    )
    node(
        "scanner_summarize_note",
        "Summarize note",
        "scanner",
        "llm",
        multiplicity="per_note",
    )
    node(
        "scanner_detect_concepts",
        "Detect concepts",
        "scanner",
        "llm",
        multiplicity="per_note",
    )
    node(
        "scanner_cancer_gate",
        "Cancer evidence?",
        "scanner",
        "decision",
        multiplicity="per_note",
        implementation="router",
    )
    node(
        "scanner_get_cancer_mentions",
        "Extract cancer mentions",
        "scanner",
        "llm",
        multiplicity="per_note",
    )
    node(
        "scanner_note_complete",
        "Note scan complete",
        "scanner",
        "convergence",
        detail="Return one ProcessedClinicalNote",
        multiplicity="per_note",
        implementation="convergence_marker",
    )
    node(
        "characterize_corpus",
        "Characterize corpus",
        "orchestrator",
        "deterministic",
        detail="Merge scans and build corpus descriptors and note digests",
    )

    edge("case_start", "initialize_case", "orchestrator")
    edge("initialize_case", "fan_out_notes", "orchestrator")
    edge(
        "fan_out_notes",
        "scanner_initialize",
        "orchestrator",
        label="one branch per note",
        kind="fanout",
        multiplicity="per_note",
    )
    edge("scanner_initialize", "scanner_summarize_note", "scanner", multiplicity="per_note")
    edge("scanner_summarize_note", "scanner_detect_concepts", "scanner", multiplicity="per_note")
    edge(
        "scanner_detect_concepts",
        "scanner_cancer_gate",
        "scanner",
        multiplicity="per_note",
    )
    edge(
        "scanner_cancer_gate",
        "scanner_get_cancer_mentions",
        "scanner",
        label="yes",
        kind="conditional",
        multiplicity="per_note",
    )
    edge(
        "scanner_cancer_gate",
        "scanner_note_complete",
        "scanner",
        label="no",
        kind="conditional",
        multiplicity="per_note",
    )
    edge(
        "scanner_get_cancer_mentions",
        "scanner_note_complete",
        "scanner",
        multiplicity="per_note",
    )
    edge(
        "scanner_note_complete",
        "characterize_corpus",
        "orchestrator",
        label="join all note branches",
        kind="convergence",
    )

    # Orchestrator planning loop and per-group note selection.
    node(
        "check_state",
        "Outstanding work?",
        "orchestrator",
        "decision",
        implementation="langgraph_node_and_router",
    )
    node(
        "plan_extraction",
        "Plan extraction",
        "orchestrator",
        "deterministic",
        detail="Find eligible groups and scope their pending variables",
    )
    node(
        "resolve_leftovers",
        "Resolve leftovers",
        "orchestrator",
        "deterministic",
        detail="Mark gated or blocked pending variables terminal",
        implementation="expanded_operation",
    )
    node(
        "fan_out_groups",
        "Fan out eligible groups",
        "orchestrator",
        "fanout",
        implementation="fanout_marker",
    )
    node(
        "hard_filter_notes",
        "Apply hard note filter",
        "orchestrator",
        "deterministic",
        multiplicity="per_group",
        implementation="expanded_operation",
    )
    node(
        "hard_filter_gate",
        "Any notes remain?",
        "orchestrator",
        "decision",
        multiplicity="per_group",
        implementation="expanded_operation",
    )
    node(
        "retriever_initialize",
        "Initialize retrieval",
        "retriever",
        "deterministic",
        multiplicity="per_group",
    )
    node(
        "retriever_identify_relevant_notes",
        "Identify relevant notes",
        "retriever",
        "llm",
        multiplicity="per_group",
    )
    node(
        "relevant_notes_gate",
        "Relevant notes selected?",
        "orchestrator",
        "decision",
        multiplicity="per_group",
        implementation="expanded_operation",
    )
    node(
        "record_not_found",
        "Record not found",
        "orchestrator",
        "deterministic",
        detail="Skip extraction and create clean misses for the group",
        multiplicity="per_group",
        implementation="expanded_operation",
    )

    edge("characterize_corpus", "check_state", "orchestrator")
    edge(
        "check_state",
        "plan_extraction",
        "orchestrator",
        label="work remains and no fatal blocker",
        kind="conditional",
    )
    edge(
        "plan_extraction",
        "fan_out_groups",
        "orchestrator",
        label="eligible groups",
        kind="conditional",
    )
    edge(
        "plan_extraction",
        "resolve_leftovers",
        "orchestrator",
        label="none eligible",
        kind="conditional",
    )
    edge(
        "resolve_leftovers",
        "check_state",
        "orchestrator",
        label="recheck terminal state",
        kind="loop",
    )
    edge(
        "fan_out_groups",
        "hard_filter_notes",
        "orchestrator",
        label="one branch per group",
        kind="fanout",
        multiplicity="per_group",
    )
    edge("hard_filter_notes", "hard_filter_gate", "orchestrator", multiplicity="per_group")
    edge(
        "hard_filter_gate",
        "record_not_found",
        "orchestrator",
        label="no",
        kind="conditional",
        multiplicity="per_group",
    )
    edge(
        "hard_filter_gate",
        "retriever_initialize",
        "retriever",
        label="yes",
        kind="conditional",
        multiplicity="per_group",
    )
    edge(
        "retriever_initialize",
        "retriever_identify_relevant_notes",
        "retriever",
        multiplicity="per_group",
    )
    edge(
        "retriever_identify_relevant_notes",
        "relevant_notes_gate",
        "retriever",
        multiplicity="per_group",
    )
    edge(
        "relevant_notes_gate",
        "record_not_found",
        "orchestrator",
        label="no",
        kind="conditional",
        multiplicity="per_group",
    )

    # Extractor group/individual paths and bounded repair loop.
    node(
        "extractor_initialize",
        "Initialize extractor",
        "extractor",
        "deterministic",
        multiplicity="per_group",
    )
    node(
        "extractor_load_notes",
        "Load selected notes",
        "extractor",
        "deterministic",
        multiplicity="per_group",
    )
    node(
        "extractor_mode_gate",
        "Extraction mode",
        "extractor",
        "decision",
        multiplicity="per_group",
        implementation="router",
    )
    node(
        "extractor_extract_group_values",
        "Extract group values",
        "extractor",
        "llm",
        multiplicity="per_group",
    )
    node(
        "fan_out_variables",
        "Fan out variables",
        "extractor",
        "fanout",
        implementation="fanout_marker",
        multiplicity="per_group",
    )
    node(
        "variable_entry_gate",
        "Candidate available?",
        "extractor",
        "decision",
        detail="Group mode validates its shared candidate; individual mode extracts first",
        multiplicity="per_variable",
        implementation="router",
    )
    node(
        "extractor_extract_individual_value",
        "Extract individual value",
        "extractor",
        "llm",
        multiplicity="per_variable",
    )
    node(
        "extractor_validate_extraction",
        "Validate extraction",
        "extractor",
        "deterministic",
        multiplicity="per_variable",
    )
    node(
        "validation_gate",
        "Valid or attempts exhausted?",
        "extractor",
        "decision",
        multiplicity="per_variable",
        implementation="router",
    )
    node(
        "extractor_repair_invalid_extraction",
        "Repair invalid extraction",
        "extractor",
        "llm",
        multiplicity="per_variable_attempt",
    )
    node(
        "extractor_complete_variable",
        "Complete variable",
        "extractor",
        "deterministic",
        multiplicity="per_variable",
    )
    node(
        "merge_variable_results",
        "Merge variable results",
        "extractor",
        "convergence",
        multiplicity="per_group",
    )
    node(
        "map_case_results",
        "Map case results",
        "orchestrator",
        "deterministic",
        detail="Convert validated extractor output to per-item case results",
        multiplicity="per_group",
        implementation="expanded_operation",
    )
    node(
        "group_complete",
        "Group branch complete",
        "orchestrator",
        "convergence",
        multiplicity="per_group",
        implementation="convergence_marker",
    )
    node(
        "merge_and_update",
        "Merge groups and update facts",
        "orchestrator",
        "deterministic",
        detail="Join group branches and derive newly coded case facts",
    )

    edge(
        "relevant_notes_gate",
        "extractor_initialize",
        "extractor",
        label="yes",
        kind="conditional",
        multiplicity="per_group",
    )
    edge("extractor_initialize", "extractor_load_notes", "extractor", multiplicity="per_group")
    edge("extractor_load_notes", "extractor_mode_gate", "extractor", multiplicity="per_group")
    edge(
        "extractor_mode_gate",
        "extractor_extract_group_values",
        "extractor",
        label="group",
        kind="conditional",
        multiplicity="per_group",
    )
    edge(
        "extractor_mode_gate",
        "fan_out_variables",
        "extractor",
        label="individual",
        kind="conditional",
        multiplicity="per_group",
    )
    edge(
        "extractor_extract_group_values",
        "fan_out_variables",
        "extractor",
        label="shared candidates",
        multiplicity="per_group",
    )
    edge(
        "fan_out_variables",
        "variable_entry_gate",
        "extractor",
        label="one branch per variable",
        kind="fanout",
        multiplicity="per_variable",
    )
    edge(
        "variable_entry_gate",
        "extractor_extract_individual_value",
        "extractor",
        label="individual mode",
        kind="conditional",
        multiplicity="per_variable",
    )
    edge(
        "variable_entry_gate",
        "extractor_validate_extraction",
        "extractor",
        label="group candidate",
        kind="conditional",
        multiplicity="per_variable",
    )
    edge(
        "extractor_extract_individual_value",
        "extractor_validate_extraction",
        "extractor",
        multiplicity="per_variable",
    )
    edge(
        "extractor_validate_extraction",
        "validation_gate",
        "extractor",
        multiplicity="per_variable",
    )
    edge(
        "validation_gate",
        "extractor_repair_invalid_extraction",
        "extractor",
        label="invalid and attempts remain",
        kind="conditional",
        multiplicity="per_variable",
    )
    edge(
        "extractor_repair_invalid_extraction",
        "extractor_validate_extraction",
        "extractor",
        label="retry (max 3 attempts)",
        kind="loop",
        multiplicity="per_variable_attempt",
    )
    edge(
        "validation_gate",
        "extractor_complete_variable",
        "extractor",
        label="valid or attempts exhausted",
        kind="conditional",
        multiplicity="per_variable",
    )
    edge(
        "extractor_complete_variable",
        "merge_variable_results",
        "extractor",
        label="join all variable branches",
        kind="convergence",
        multiplicity="per_group",
    )
    edge("merge_variable_results", "map_case_results", "orchestrator", multiplicity="per_group")
    edge("map_case_results", "group_complete", "orchestrator", multiplicity="per_group")
    edge("record_not_found", "group_complete", "orchestrator", multiplicity="per_group")
    edge(
        "group_complete",
        "merge_and_update",
        "orchestrator",
        label="join all group branches",
        kind="convergence",
    )
    edge(
        "merge_and_update",
        "check_state",
        "orchestrator",
        label="plan next pass",
        kind="loop",
    )

    # Finalization path.
    node(
        "finalize_case",
        "Finalize case",
        "orchestrator",
        "deterministic",
        detail="Build the review report and durable Case",
    )
    node("case_end", "END", "orchestrator", "endpoint", implementation="virtual_endpoint")

    edge(
        "check_state",
        "finalize_case",
        "orchestrator",
        label="nothing outstanding or fatal blocker",
        kind="conditional",
    )
    edge("finalize_case", "case_end", "orchestrator")

    graph = {
        "format": "cytoscape-elements-v1",
        "directed": True,
        "metadata": {
            "title": "CIPOC unified agent execution graph",
            "description": (
                "One connected execution graph integrating deterministic orchestration, "
                "note scanning, note retrieval, extraction, validation, and repair."
            ),
            "generated_by": "scripts/generate_agent_system_graph.py",
            "entry_node": "case_start",
            "exit_node": "case_end",
            "agent_colors": AGENT_COLORS,
            "kind_colors": KIND_COLORS,
            "multiplicity": {
                "once": "Runs once at that point in the case flow",
                "per_note": "Runs concurrently for each note",
                "per_group": "Runs concurrently for each eligible variable group",
                "per_variable": "Runs concurrently for each variable",
                "per_variable_attempt": "May repeat within the bounded variable repair loop",
            },
        },
        "elements": {"nodes": nodes, "edges": edges},
        "style": cytoscape_style(),
        "layout": {
            "name": "breadthfirst",
            "directed": True,
            "circle": False,
            "spacingFactor": 1.15,
            "roots": "#case_start",
        },
    }
    validate_graph(graph)
    return graph


def cytoscape_style() -> list[dict[str, Any]]:
    """Return styles that Cytoscape.js can consume without translation."""
    style: list[dict[str, Any]] = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": KIND_COLORS["deterministic"],
                "border-width": 3,
                "border-color": AGENT_COLORS["orchestrator"],
                "color": INK,
                "font-size": 13,
                "text-wrap": "wrap",
                "text-max-width": 150,
                "text-valign": "center",
                "text-halign": "center",
                "width": 170,
                "height": 58,
                "shape": "roundrectangle",
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "width": 2.5,
                "line-color": AGENT_COLORS["orchestrator"],
                "target-arrow-color": AGENT_COLORS["orchestrator"],
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "font-size": 10,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.9,
                "text-background-padding": 2,
            },
        },
        {
            "selector": 'node[kind = "endpoint"]',
            "style": {"background-color": KIND_COLORS["endpoint"], "color": "#ffffff", "shape": "ellipse"},
        },
        {
            "selector": 'node[kind = "llm"]',
            "style": {"background-color": KIND_COLORS["llm"]},
        },
        {
            "selector": 'node[kind = "decision"]',
            "style": {"background-color": KIND_COLORS["decision"], "shape": "diamond", "width": 145, "height": 80},
        },
        {
            "selector": 'node[kind = "fanout"], node[kind = "convergence"]',
            "style": {"background-color": KIND_COLORS["fanout"], "shape": "hexagon"},
        },
        {
            "selector": 'edge[kind = "conditional"]',
            "style": {"line-style": "dashed"},
        },
        {
            "selector": 'edge[kind = "loop"]',
            "style": {"line-style": "dotted", "curve-style": "unbundled-bezier"},
        },
    ]
    for agent, color in AGENT_COLORS.items():
        style.extend(
            [
                {
                    "selector": f'node[agent = "{agent}"]',
                    "style": {"border-color": color},
                },
                {
                    "selector": f'edge[agent = "{agent}"]',
                    "style": {"line-color": color, "target-arrow-color": color},
                },
            ]
        )
    return style


def validate_graph(graph: dict[str, Any]) -> None:
    """Reject dangling, duplicate, or disconnected graph definitions."""
    node_data = [element["data"] for element in graph["elements"]["nodes"]]
    edge_data = [element["data"] for element in graph["elements"]["edges"]]
    node_ids = [node["id"] for node in node_data]
    edge_ids = [edge["id"] for edge in edge_data]

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Graph contains duplicate node IDs.")
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Graph contains duplicate edge IDs.")

    known_nodes = set(node_ids)
    for edge_data_item in edge_data:
        for endpoint in ("source", "target"):
            if edge_data_item[endpoint] not in known_nodes:
                raise ValueError(
                    f"Edge {edge_data_item['id']} has unknown {endpoint} "
                    f"{edge_data_item[endpoint]!r}."
                )

    entry = graph["metadata"]["entry_node"]
    exit_node = graph["metadata"]["exit_node"]
    if entry not in known_nodes or exit_node not in known_nodes:
        raise ValueError("Graph entry or exit node is missing.")

    adjacency: dict[str, set[str]] = defaultdict(set)
    reverse_adjacency: dict[str, set[str]] = defaultdict(set)
    for edge_data_item in edge_data:
        source = edge_data_item["source"]
        target = edge_data_item["target"]
        adjacency[source].add(target)
        reverse_adjacency[target].add(source)

    reachable_from_entry = _reachable(entry, adjacency)
    can_reach_exit = _reachable(exit_node, reverse_adjacency)
    unreachable = known_nodes - reachable_from_entry
    dead_ends = known_nodes - can_reach_exit
    if unreachable:
        raise ValueError(f"Nodes unreachable from entry: {sorted(unreachable)}")
    if dead_ends:
        raise ValueError(f"Nodes that cannot reach exit: {sorted(dead_ends)}")


def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adjacency[current] - visited)
    return visited


def serialize_graph(graph: dict[str, Any]) -> str:
    return json.dumps(graph, indent=2) + "\n"


def load_cytoscape_source(
    source_path: Path | None,
    existing_html_path: Path,
) -> str:
    """Load Cytoscape from a supplied file, an existing viewer, or the pinned CDN."""
    if source_path is not None:
        return source_path.read_text()

    if existing_html_path.is_file():
        existing_html = existing_html_path.read_text()
        start = existing_html.find(CYTOSCAPE_START_MARKER)
        end = existing_html.find(CYTOSCAPE_END_MARKER)
        if start != -1 and end != -1 and start < end:
            return existing_html[start + len(CYTOSCAPE_START_MARKER) : end].strip()

    request = Request(CYTOSCAPE_URL, headers={"User-Agent": "CIPOC graph generator"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except (OSError, URLError) as exc:
        raise SystemExit(
            "Could not obtain Cytoscape.js. Supply a local bundle with "
            "--cytoscape-js or regenerate while the pinned CDN is reachable."
        ) from exc


def build_html(graph: dict[str, Any], cytoscape_source: str) -> str:
    """Build a self-contained interactive viewer with no runtime dependencies."""
    graph_json = json.dumps(graph, separators=(",", ":")).replace("</", "<\\/")
    safe_cytoscape_source = cytoscape_source.strip().replace("</script", "<\\/script")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIPOC unified agent execution graph</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172033;
      background: #f7f8fb;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { display: grid; grid-template-columns: 300px minmax(0, 1fr); }
    aside {
      z-index: 2;
      display: flex;
      min-height: 0;
      flex-direction: column;
      gap: 20px;
      padding: 24px 20px;
      overflow: auto;
      border-right: 1px solid #d9dee8;
      background: #ffffff;
      box-shadow: 8px 0 24px rgb(24 32 51 / 6%);
    }
    h1 { margin: 0; font-size: 20px; line-height: 1.2; letter-spacing: -0.3px; }
    .subtitle { margin: 7px 0 0; color: #667085; font-size: 13px; line-height: 1.45; }
    h2 {
      margin: 0 0 9px;
      color: #7a8498;
      font-size: 11px;
      letter-spacing: 1.25px;
      text-transform: uppercase;
    }
    .legend { display: grid; gap: 9px; }
    .legend-row { display: flex; align-items: center; gap: 9px; font-size: 12px; }
    .swatch { width: 24px; height: 6px; border-radius: 999px; background: var(--swatch); }
    .fill { width: 24px; height: 18px; border: 1px solid #a8b0bf; border-radius: 5px; background: var(--swatch); }
    .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .display-controls { display: grid; gap: 12px; }
    .range-control { display: grid; grid-template-columns: 1fr auto; gap: 5px 10px; align-items: center; }
    .range-control label, .select-control label { color: #475467; font-size: 12px; font-weight: 600; }
    .range-control output { color: #667085; font-size: 11px; font-variant-numeric: tabular-nums; }
    .range-control input { grid-column: 1 / -1; width: 100%; accent-color: #6d5bd0; }
    .select-control { display: grid; gap: 6px; }
    select {
      width: 100%;
      padding: 7px 9px;
      border: 1px solid #c7cdd8;
      border-radius: 7px;
      color: #263248;
      background: #ffffff;
      font: inherit;
      font-size: 12px;
    }
    .toggle { display: flex; align-items: center; gap: 8px; color: #475467; font-size: 12px; font-weight: 600; }
    .toggle input { width: 15px; height: 15px; margin: 0; accent-color: #6d5bd0; }
    button {
      appearance: none;
      padding: 8px 10px;
      border: 1px solid #c7cdd8;
      border-radius: 7px;
      color: #263248;
      background: #ffffff;
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { border-color: #6d5bd0; background: #f4f2ff; }
    #selection {
      min-height: 104px;
      padding: 12px;
      border: 1px solid #e0e4eb;
      border-radius: 9px;
      background: #f8f9fb;
      font-size: 12px;
      line-height: 1.45;
    }
    #selection strong { display: block; margin-bottom: 5px; font-size: 13px; }
    .tag {
      display: inline-block;
      margin: 7px 5px 0 0;
      padding: 2px 6px;
      border-radius: 999px;
      color: #475467;
      background: #e9edf3;
      font-size: 10px;
    }
    .hint { margin-top: auto; color: #7a8498; font-size: 11px; line-height: 1.45; }
    main { position: relative; min-width: 0; min-height: 0; }
    #graph { position: absolute; inset: 0; background: #f7f8fb; }
    #status {
      position: absolute;
      right: 14px;
      bottom: 12px;
      padding: 5px 8px;
      border: 1px solid #d9dee8;
      border-radius: 6px;
      color: #667085;
      background: rgb(255 255 255 / 88%);
      font-size: 11px;
      pointer-events: none;
    }
    @media (max-width: 760px) {
      body { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
      aside { max-height: 300px; padding: 14px 16px; border-right: 0; border-bottom: 1px solid #d9dee8; }
      aside .desktop-only, .hint { display: none; }
      .button-row { display: flex; }
    }
  </style>
</head>
<body>
  <aside>
    <header>
      <h1>CIPOC agent system</h1>
      <p class="subtitle">One executable flow across orchestration, scanning, retrieval, and extraction.</p>
    </header>
    <section>
      <h2>Controls</h2>
      <div class="button-row">
        <button id="fit" type="button">Fit graph</button>
        <button id="relayout" type="button">Re-layout</button>
      </div>
    </section>
    <section>
      <h2>Display</h2>
      <div class="display-controls">
        <div class="range-control">
          <label for="node-font-size">Node text</label>
          <output id="node-font-size-value" for="node-font-size">13 px</output>
          <input id="node-font-size" type="range" min="8" max="24" step="1" value="13">
        </div>
        <div class="range-control">
          <label for="edge-font-size">Edge text</label>
          <output id="edge-font-size-value" for="edge-font-size">10 px</output>
          <input id="edge-font-size" type="range" min="7" max="20" step="1" value="10">
        </div>
        <div class="range-control">
          <label for="node-scale">Node size</label>
          <output id="node-scale-value" for="node-scale">100%</output>
          <input id="node-scale" type="range" min="70" max="160" step="5" value="100">
        </div>
        <div class="range-control">
          <label for="layout-spacing">Layout spacing</label>
          <output id="layout-spacing-value" for="layout-spacing">115%</output>
          <input id="layout-spacing" type="range" min="70" max="200" step="5" value="115">
        </div>
        <div class="select-control">
          <label for="font-family">Font family</label>
          <select id="font-family">
            <option value="Inter, ui-sans-serif, system-ui, sans-serif">System sans</option>
            <option value="Georgia, ui-serif, serif">Serif</option>
            <option value="ui-monospace, SFMono-Regular, Menlo, monospace">Monospace</option>
          </select>
        </div>
        <label class="toggle"><input id="show-edge-labels" type="checkbox" checked>Show edge labels</label>
        <button id="reset-display" type="button">Reset display</button>
      </div>
    </section>
    <section class="desktop-only">
      <h2>Agent</h2>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="--swatch:#6d5bd0"></span>Orchestrator</div>
        <div class="legend-row"><span class="swatch" style="--swatch:#008c7a"></span>Note scanner</div>
        <div class="legend-row"><span class="swatch" style="--swatch:#d16b22"></span>Note retriever</div>
        <div class="legend-row"><span class="swatch" style="--swatch:#1473e6"></span>Extractor</div>
      </div>
    </section>
    <section class="desktop-only">
      <h2>Node type</h2>
      <div class="legend">
        <div class="legend-row"><span class="fill" style="--swatch:#eef2f7"></span>Deterministic</div>
        <div class="legend-row"><span class="fill" style="--swatch:#fff1c7"></span>LLM call</div>
        <div class="legend-row"><span class="fill" style="--swatch:#ffffff"></span>Decision</div>
        <div class="legend-row"><span class="fill" style="--swatch:#e8e4ff"></span>Fan-out / convergence</div>
      </div>
    </section>
    <section class="desktop-only">
      <h2>Selection</h2>
      <div id="selection">Select a node or edge to inspect its graph data.</div>
    </section>
    <p class="hint">Scroll to zoom, drag the canvas to pan, and drag nodes to adjust the generated layout.</p>
  </aside>
  <main>
    <div id="graph" role="img" aria-label="Interactive CIPOC unified execution graph"></div>
    <div id="status"></div>
  </main>

  <script type="application/json" id="graph-data">__GRAPH_JSON__</script>
  <script>
__CYTOSCAPE_START_MARKER__
__CYTOSCAPE_SOURCE__
__CYTOSCAPE_END_MARKER__
  </script>
  <script>
    const graph = JSON.parse(document.getElementById("graph-data").textContent);
    const cy = cytoscape({
      container: document.getElementById("graph"),
      elements: graph.elements,
      style: graph.style,
      layout: { ...graph.layout, animate: false, padding: 48 },
      minZoom: 0.08,
      maxZoom: 2.5,
      wheelSensitivity: 0.18
    });

    const selection = document.getElementById("selection");
    const status = document.getElementById("status");
    const displayControls = {
      nodeFontSize: document.getElementById("node-font-size"),
      edgeFontSize: document.getElementById("edge-font-size"),
      nodeScale: document.getElementById("node-scale"),
      layoutSpacing: document.getElementById("layout-spacing"),
      fontFamily: document.getElementById("font-family"),
      showEdgeLabels: document.getElementById("show-edge-labels")
    };
    const displayDefaults = {
      nodeFontSize: "13",
      edgeFontSize: "10",
      nodeScale: "100",
      layoutSpacing: "115",
      fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      showEdgeLabels: true
    };
    status.textContent = `${cy.nodes().length} nodes / ${cy.edges().length} edges`;

    function fitGraph() {
      cy.animate({ fit: { eles: cy.elements(), padding: 44 }, duration: 250 });
    }

    function runLayout() {
      const spacingFactor = Number(displayControls.layoutSpacing.value) / 100;
      cy.layout({ ...graph.layout, spacingFactor, animate: false, padding: 48 }).run();
      fitGraph();
    }

    function updateControlLabels() {
      document.getElementById("node-font-size-value").value = `${displayControls.nodeFontSize.value} px`;
      document.getElementById("edge-font-size-value").value = `${displayControls.edgeFontSize.value} px`;
      document.getElementById("node-scale-value").value = `${displayControls.nodeScale.value}%`;
      document.getElementById("layout-spacing-value").value = `${displayControls.layoutSpacing.value}%`;
    }

    function applyDisplayStyles() {
      const nodeScale = Number(displayControls.nodeScale.value) / 100;
      cy.batch(() => {
        cy.nodes().style({
          "font-size": `${displayControls.nodeFontSize.value}px`,
          "font-family": displayControls.fontFamily.value,
          "width": node => (node.data("kind") === "decision" ? 145 : 170) * nodeScale,
          "height": node => (node.data("kind") === "decision" ? 80 : 58) * nodeScale
        });
        cy.edges().style({
          "font-size": `${displayControls.edgeFontSize.value}px`,
          "font-family": displayControls.fontFamily.value,
          "label": displayControls.showEdgeLabels.checked ? "data(label)" : ""
        });
      });
      updateControlLabels();
    }

    function resetDisplay() {
      for (const [name, value] of Object.entries(displayDefaults)) {
        const control = displayControls[name];
        if (control.type === "checkbox") control.checked = value;
        else control.value = value;
      }
      applyDisplayStyles();
      runLayout();
    }

    function tag(value) {
      return value ? `<span class="tag">${value}</span>` : "";
    }

    function escapeHtml(value) {
      const element = document.createElement("span");
      element.textContent = value == null ? "" : String(value);
      return element.innerHTML;
    }

    cy.on("tap", "node", event => {
      const data = event.target.data();
      selection.innerHTML = `
        <strong>${escapeHtml(data.label)}</strong>
        ${escapeHtml(data.detail || "No additional detail.")}
        <div>${tag(escapeHtml(data.agent))}${tag(escapeHtml(data.kind))}${tag(escapeHtml(data.multiplicity))}</div>`;
    });

    cy.on("tap", "edge", event => {
      const data = event.target.data();
      const source = cy.getElementById(data.source).data("label");
      const target = cy.getElementById(data.target).data("label");
      selection.innerHTML = `
        <strong>${escapeHtml(source)} &rarr; ${escapeHtml(target)}</strong>
        ${escapeHtml(data.label || "Unconditional flow")}
        <div>${tag(escapeHtml(data.agent))}${tag(escapeHtml(data.kind))}${tag(escapeHtml(data.multiplicity))}</div>`;
    });

    document.getElementById("fit").addEventListener("click", fitGraph);
    document.getElementById("relayout").addEventListener("click", runLayout);
    document.getElementById("reset-display").addEventListener("click", resetDisplay);
    displayControls.nodeFontSize.addEventListener("input", applyDisplayStyles);
    displayControls.edgeFontSize.addEventListener("input", applyDisplayStyles);
    displayControls.nodeScale.addEventListener("input", applyDisplayStyles);
    displayControls.nodeScale.addEventListener("change", runLayout);
    displayControls.layoutSpacing.addEventListener("input", updateControlLabels);
    displayControls.layoutSpacing.addEventListener("change", runLayout);
    displayControls.fontFamily.addEventListener("change", applyDisplayStyles);
    displayControls.showEdgeLabels.addEventListener("change", applyDisplayStyles);
    window.addEventListener("resize", () => cy.resize());
    applyDisplayStyles();
    requestAnimationFrame(fitGraph);
  </script>
</body>
</html>
"""
    return (
        template.replace("__GRAPH_JSON__", graph_json)
        .replace("__CYTOSCAPE_START_MARKER__", CYTOSCAPE_START_MARKER)
        .replace("__CYTOSCAPE_SOURCE__", safe_cytoscape_source)
        .replace("__CYTOSCAPE_END_MARKER__", CYTOSCAPE_END_MARKER)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(REPOSITORY_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the JSON or HTML output is missing or stale.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
        help=f"Standalone viewer path (default: {DEFAULT_HTML_OUTPUT.relative_to(REPOSITORY_ROOT)})",
    )
    parser.add_argument(
        "--cytoscape-js",
        type=Path,
        default=None,
        help="Optional local Cytoscape.js bundle to embed instead of using the existing HTML or pinned CDN.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate or check only the graph JSON.",
    )
    args = parser.parse_args()

    graph = build_graph()
    serialized = serialize_graph(graph)
    html = None
    if not args.json_only:
        cytoscape_source = load_cytoscape_source(args.cytoscape_js, args.html_output)
        html = build_html(graph, cytoscape_source)

    if args.check:
        if not args.output.is_file() or args.output.read_text() != serialized:
            raise SystemExit(f"Graph JSON is stale or missing: {args.output}")
        if html is not None and (
            not args.html_output.is_file() or args.html_output.read_text() != html
        ):
            raise SystemExit(f"Graph HTML is stale or missing: {args.html_output}")
        print(f"Graph outputs are current: {args.output}, {args.html_output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(f"Wrote unified graph to {args.output}")
    if html is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(html)
        print(f"Wrote standalone viewer to {args.html_output}")


if __name__ == "__main__":
    main()
