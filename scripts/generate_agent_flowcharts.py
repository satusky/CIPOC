"""Generate compact, conceptual CIPOC workflow diagrams.

The detailed unified graph remains the source of truth. These diagrams retain
only the major phases and decisions needed to understand the system at a glance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .generate_agent_system_graph import (
        AGENT_COLORS,
        CYTOSCAPE_END_MARKER,
        CYTOSCAPE_START_MARKER,
        DEFAULT_HTML_OUTPUT as UNIFIED_HTML_OUTPUT,
        INK,
        KIND_COLORS,
        build_graph,
        load_cytoscape_source,
        validate_graph,
    )
else:
    from generate_agent_system_graph import (
        AGENT_COLORS,
        CYTOSCAPE_END_MARKER,
        CYTOSCAPE_START_MARKER,
        DEFAULT_HTML_OUTPUT as UNIFIED_HTML_OUTPUT,
        INK,
        KIND_COLORS,
        build_graph,
        load_cytoscape_source,
        validate_graph,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "cipoc"
    / "agents"
    / "visualization"
    / "agent_flowcharts.json"
)
DEFAULT_HTML_OUTPUT = DEFAULT_OUTPUT.with_suffix(".html")

def _expand_fanout_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render each logical fan-out as three dashed lanes with shared endpoints."""
    expanded: list[dict[str, Any]] = []
    for edge in edges:
        data = edge["data"]
        if data.get("kind") != "fanout":
            expanded.append({"data": dict(data)})
            continue

        for lane in ("left", "center", "right"):
            lane_data = dict(data)
            lane_data["fanout_lane"] = lane
            if lane != "center":
                lane_data["label"] = ""
            expanded.append({"data": lane_data})

    for index, edge in enumerate(expanded, start=1):
        edge["data"]["id"] = f"edge_{index:02d}"
    return expanded


def _node(
    node_id: str,
    label: str,
    agent: str,
    kind: str = "deterministic",
    *,
    detail: str = "",
    multiplicity: str = "once",
) -> dict[str, Any]:
    return {
        "data": {
            "id": node_id,
            "label": label,
            "agent": agent,
            "kind": kind,
            "detail": detail,
            "multiplicity": multiplicity,
            "implementation": "conceptual_stage",
        }
    }


def _edge(
    source: str,
    target: str,
    agent: str,
    *,
    label: str = "",
    kind: str = "flow",
    multiplicity: str = "once",
) -> dict[str, Any]:
    return {
        "data": {
            "source": source,
            "target": target,
            "label": label,
            "agent": agent,
            "kind": kind,
            "multiplicity": multiplicity,
        }
    }


def _overview_chart() -> dict[str, Any]:
    nodes = [
        _node("case_start", "START", "orchestrator", "endpoint"),
        _node(
            "initialize_case",
            "Initialize case",
            "orchestrator",
            detail="Load notes, requested variables, structured values, and known facts",
        ),
        _node(
            "scanner_agent_block",
            "Note Scanner Agent",
            "scanner",
            "subagent",
            detail="Scan and summarize each clinical note",
            multiplicity="per_note",
        ),
        _node(
            "characterize_corpus",
            "Characterize corpus",
            "orchestrator",
            detail="Combine note scans into case-level evidence and note digests",
        ),
        _node(
            "eligible_groups_gate",
            "Eligible groups remain?",
            "orchestrator",
            "decision",
            detail="Apply corpus gates, site rules, and variable dependencies",
        ),
        _node(
            "retriever_agent_block",
            "Note Retriever Agent",
            "retriever",
            "subagent",
            detail="Select the notes relevant to each eligible variable group",
            multiplicity="per_group",
        ),
        _node(
            "relevant_notes_gate",
            "Relevant notes?",
            "orchestrator",
            "decision",
            detail="Skip extraction when no useful notes survive selection",
            multiplicity="per_group",
        ),
        _node(
            "extractor_agent_block",
            "Extractor Agent",
            "extractor",
            "subagent",
            detail="Extract, validate, and repair values for the group",
            multiplicity="per_group",
        ),
        _node(
            "update_case",
            "Update case results and facts",
            "orchestrator",
            detail="Record extracted values or clean not-found results, then update facts",
        ),
        _node(
            "finalize_case",
            "Finalize case",
            "orchestrator",
            detail="Build the durable case and review report",
        ),
        _node("case_end", "END", "orchestrator", "endpoint"),
    ]
    edges = [
        _edge("case_start", "initialize_case", "orchestrator"),
        _edge(
            "initialize_case",
            "scanner_agent_block",
            "orchestrator",
            label="one branch per note",
            kind="fanout",
            multiplicity="per_note",
        ),
        _edge("scanner_agent_block", "characterize_corpus", "orchestrator"),
        _edge("characterize_corpus", "eligible_groups_gate", "orchestrator"),
        _edge(
            "eligible_groups_gate",
            "retriever_agent_block",
            "orchestrator",
            label="yes, one branch per group",
            kind="fanout",
            multiplicity="per_group",
        ),
        _edge(
            "eligible_groups_gate",
            "finalize_case",
            "orchestrator",
            label="no",
            kind="conditional",
        ),
        _edge("retriever_agent_block", "relevant_notes_gate", "retriever", multiplicity="per_group"),
        _edge(
            "relevant_notes_gate",
            "extractor_agent_block",
            "extractor",
            label="yes",
            kind="conditional",
            multiplicity="per_group",
        ),
        _edge(
            "relevant_notes_gate",
            "update_case",
            "orchestrator",
            label="no: not found",
            kind="conditional",
            multiplicity="per_group",
        ),
        _edge("extractor_agent_block", "update_case", "extractor", multiplicity="per_group"),
        _edge(
            "update_case",
            "eligible_groups_gate",
            "orchestrator",
            label="plan next pass",
            kind="loop",
        ),
        _edge("finalize_case", "case_end", "orchestrator"),
    ]
    return _chart(
        "overview",
        "Overall Workflow",
        "Major orchestration phases and decisions; implementation-level nodes are omitted.",
        nodes,
        _expand_fanout_edges(edges),
        "case_start",
        "case_end",
    )


def _scanner_chart() -> dict[str, Any]:
    nodes = [
        _node("scanner_start", "START", "scanner", "endpoint"),
        _node("scanner_summarize_note", "Summarize note", "scanner", "llm"),
        _node("scanner_detect_concepts", "Detect clinical concepts", "scanner", "llm"),
        _node("scanner_cancer_gate", "Cancer evidence?", "scanner", "decision"),
        _node("scanner_get_cancer_mentions", "Extract cancer mentions", "scanner", "llm"),
        _node(
            "scanner_note_complete",
            "Return processed note",
            "scanner",
            detail="Return the summary, concepts, temporality, and evidence",
        ),
        _node("scanner_end", "END", "scanner", "endpoint"),
    ]
    edges = [
        _edge("scanner_start", "scanner_summarize_note", "scanner"),
        _edge("scanner_summarize_note", "scanner_detect_concepts", "scanner"),
        _edge("scanner_detect_concepts", "scanner_cancer_gate", "scanner"),
        _edge(
            "scanner_cancer_gate",
            "scanner_get_cancer_mentions",
            "scanner",
            label="yes",
            kind="conditional",
        ),
        _edge(
            "scanner_cancer_gate",
            "scanner_note_complete",
            "scanner",
            label="no",
            kind="conditional",
        ),
        _edge("scanner_get_cancer_mentions", "scanner_note_complete", "scanner"),
        _edge("scanner_note_complete", "scanner_end", "scanner"),
    ]
    return _chart(
        "scanner",
        "Note Scanning",
        "One streamlined pass over each clinical note.",
        nodes,
        _expand_fanout_edges(edges),
        "scanner_start",
        "scanner_end",
    )


def _retrieval_extraction_chart() -> dict[str, Any]:
    nodes = [
        _node("pipeline_start", "START", "orchestrator", "endpoint"),
        _node(
            "filter_notes",
            "Apply note filters",
            "orchestrator",
            detail="Apply deterministic date, type, and concept filters",
            multiplicity="per_group",
        ),
        _node("candidate_notes_gate", "Candidate notes?", "orchestrator", "decision", multiplicity="per_group"),
        _node(
            "retriever_pipeline_block",
            "Note Retriever Agent",
            "retriever",
            "subagent",
            detail="Rank the candidate notes for the requested variable group",
            multiplicity="per_group",
        ),
        _node("selected_notes_gate", "Relevant notes selected?", "orchestrator", "decision", multiplicity="per_group"),
        _node(
            "extractor_pipeline_block",
            "Extractor Agent",
            "extractor",
            "subagent",
            detail="Extract candidate values using scoped coding guidance",
            multiplicity="per_group",
        ),
        _node("validate_values", "Validate values", "extractor", detail="Check values against allowed NAACCR codes", multiplicity="per_group"),
        _node("values_valid_gate", "Valid or attempts exhausted?", "extractor", "decision", multiplicity="per_group"),
        _node("repair_values", "Repair invalid values", "extractor", "llm", multiplicity="per_group"),
        _node("not_found", "Return not found", "orchestrator", detail="Create clean misses without running extraction", multiplicity="per_group"),
        _node("return_results", "Return group results", "extractor", multiplicity="per_group"),
        _node("pipeline_end", "END", "orchestrator", "endpoint"),
    ]
    edges = [
        _edge("pipeline_start", "filter_notes", "orchestrator"),
        _edge("filter_notes", "candidate_notes_gate", "orchestrator"),
        _edge("candidate_notes_gate", "retriever_pipeline_block", "retriever", label="yes", kind="conditional"),
        _edge("candidate_notes_gate", "not_found", "orchestrator", label="no", kind="conditional"),
        _edge("retriever_pipeline_block", "selected_notes_gate", "retriever"),
        _edge("selected_notes_gate", "extractor_pipeline_block", "extractor", label="yes", kind="conditional"),
        _edge("selected_notes_gate", "not_found", "orchestrator", label="no", kind="conditional"),
        _edge("extractor_pipeline_block", "validate_values", "extractor"),
        _edge("validate_values", "values_valid_gate", "extractor"),
        _edge("values_valid_gate", "return_results", "extractor", label="yes", kind="conditional"),
        _edge("values_valid_gate", "repair_values", "extractor", label="no", kind="conditional"),
        _edge("repair_values", "validate_values", "extractor", label="retry", kind="loop"),
        _edge("not_found", "pipeline_end", "orchestrator"),
        _edge("return_results", "pipeline_end", "extractor"),
    ]
    return _chart(
        "retrieval_extraction",
        "Note Retrieval and Extraction",
        "Candidate-note selection, value extraction, validation, and bounded repair.",
        nodes,
        _expand_fanout_edges(edges),
        "pipeline_start",
        "pipeline_end",
    )


def _chart(
    chart_id: str,
    title: str,
    description: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    entry_node: str,
    exit_node: str,
) -> dict[str, Any]:
    chart = {
        "id": chart_id,
        "title": title,
        "description": description,
        "metadata": {"entry_node": entry_node, "exit_node": exit_node},
        "elements": {"nodes": nodes, "edges": edges},
        "layout": {
            "name": "breadthfirst",
            "directed": True,
            "circle": False,
            "spacingFactor": 0.95,
            "roots": f"#{entry_node}",
        },
    }
    validate_graph(
        {
            "metadata": chart["metadata"],
            "elements": chart["elements"],
        }
    )
    return chart


def flowchart_style() -> list[dict[str, Any]]:
    style: list[dict[str, Any]] = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": KIND_COLORS["deterministic"],
                "border-width": 3,
                "border-color": AGENT_COLORS["orchestrator"],
                "color": INK,
                "font-size": 11,
                "text-wrap": "wrap",
                "text-max-width": 130,
                "text-valign": "center",
                "text-halign": "center",
                "width": 145,
                "height": 48,
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
                "font-size": 9,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.92,
                "text-background-padding": 2,
            },
        },
        {
            "selector": 'node[kind = "endpoint"]',
            "style": {
                "background-color": KIND_COLORS["endpoint"],
                "color": "#ffffff",
                "shape": "ellipse",
                "width": 78,
                "height": 34,
            },
        },
        {
            "selector": 'node[kind = "llm"]',
            "style": {"background-color": KIND_COLORS["llm"]},
        },
        {
            "selector": 'node[kind = "decision"]',
            "style": {
                "background-color": KIND_COLORS["decision"],
                "shape": "diamond",
                "width": 125,
                "height": 70,
            },
        },
        {
            "selector": 'node[kind = "fanout"], node[kind = "convergence"]',
            "style": {
                "background-color": KIND_COLORS["fanout"],
                "shape": "hexagon",
            },
        },
        {
            "selector": 'node[kind = "subagent"]',
            "style": {
                "shape": "roundrectangle",
                "width": 190,
                "height": 62,
                "color": "#ffffff",
                "font-size": 13,
                "font-weight": 700,
                "border-width": 0,
            },
        },
        {
            "selector": 'edge[kind = "conditional"]',
            "style": {"line-style": "dashed"},
        },
        {
            "selector": 'edge[kind = "loop"]',
            "style": {"line-style": "dotted", "curve-style": "unbundled-bezier"},
        },
        {
            "selector": 'edge[kind = "fanout"]',
            "style": {
                "line-style": "dashed",
                "curve-style": "unbundled-bezier",
                "control-point-weights": 0.5,
            },
        },
        {
            "selector": 'edge[fanout_lane = "left"]',
            "style": {"control-point-distances": -18},
        },
        {
            "selector": 'edge[fanout_lane = "center"]',
            "style": {"control-point-distances": 0},
        },
        {
            "selector": 'edge[fanout_lane = "right"]',
            "style": {"control-point-distances": 18},
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
                    "selector": f'node[kind = "subagent"][agent = "{agent}"]',
                    "style": {"background-color": color},
                },
                {
                    "selector": f'edge[agent = "{agent}"]',
                    "style": {"line-color": color, "target-arrow-color": color},
                },
            ]
        )
    return style


def build_flowcharts() -> dict[str, Any]:
    # Building the detailed graph first keeps this presentation coupled to the
    # validated source topology even though it intentionally aggregates nodes.
    unified = build_graph()
    source_node_ids = {
        node["data"]["id"] for node in unified["elements"]["nodes"]
    }
    represented_source_nodes = {
        "case_start",
        "initialize_case",
        "characterize_corpus",
        "relevant_notes_gate",
        "finalize_case",
        "case_end",
        "scanner_summarize_note",
        "scanner_detect_concepts",
        "scanner_cancer_gate",
        "scanner_get_cancer_mentions",
        "scanner_note_complete",
        "extractor_validate_extraction",
        "validation_gate",
        "extractor_repair_invalid_extraction",
    }
    missing = represented_source_nodes - source_node_ids
    if missing:
        raise ValueError(f"Simplified flowcharts reference missing source nodes: {sorted(missing)}")

    charts = [
        _overview_chart(),
        _scanner_chart(),
        _retrieval_extraction_chart(),
    ]
    return {
        "format": "cipoc-agent-flowcharts-v2",
        "metadata": {
            "title": "CIPOC compact workflow diagrams",
            "description": (
                "Three conceptual flowcharts covering overall orchestration, note "
                "scanning, and the combined retrieval/extraction pipeline."
            ),
            "generated_by": "scripts/generate_agent_flowcharts.py",
            "agent_colors": AGENT_COLORS,
            "kind_colors": KIND_COLORS,
        },
        "style": flowchart_style(),
        "charts": charts,
    }


def serialize_flowcharts(flowcharts: dict[str, Any]) -> str:
    return json.dumps(flowcharts, indent=2) + "\n"


def build_html(flowcharts: dict[str, Any], cytoscape_source: str) -> str:
    data_json = json.dumps(flowcharts, separators=(",", ":")).replace("</", "<\\/")
    safe_cytoscape_source = cytoscape_source.strip().replace("</script", "<\\/script")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIPOC compact workflow diagrams</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172033;
      background: #eef1f6;
    }
    * { box-sizing: border-box; }
    html, body { min-width: 100%; min-height: 100%; margin: 0; }
    body { padding: 12px; }
    .page-header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      max-width: 1800px;
      margin: 0 auto 10px;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: -0.35px; }
    .subtitle { max-width: 760px; margin: 3px 0 0; color: #667085; font-size: 11px; line-height: 1.35; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; }
    button {
      appearance: none;
      padding: 6px 9px;
      border: 1px solid #c7cdd8;
      border-radius: 7px;
      color: #263248;
      background: #ffffff;
      font: inherit;
      font-size: 11px;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { border-color: #6d5bd0; background: #f4f2ff; }
    .toggle { display: flex; align-items: center; gap: 5px; color: #475467; font-size: 11px; font-weight: 600; }
    .toggle input { width: 14px; height: 14px; margin: 0; accent-color: #6d5bd0; }
    .charts {
      display: grid;
      grid-template-columns: minmax(340px, 4fr) minmax(560px, 7fr);
      gap: 10px;
      max-width: 1800px;
      margin: 0 auto;
    }
    .chart-card {
      min-width: 0;
      overflow: hidden;
      border: 1px solid #d7dce6;
      border-radius: 9px;
      background: #ffffff;
      box-shadow: 0 4px 14px rgb(24 32 51 / 6%);
    }
    .chart-card[data-chart="overview"] { grid-column: 1 / -1; }
    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 11px;
      border-bottom: 1px solid #e5e8ee;
    }
    .chart-actions { display: flex; flex: 0 0 auto; gap: 6px; }
    .chart-title { margin: 0; font-size: 14px; }
    .chart-description { margin: 2px 0 0; color: #7a8498; font-size: 10px; }
    .canvas { height: 350px; background: #fbfcfe; }
    .chart-card[data-chart="overview"] .canvas { height: 360px; }
    .chart-footer {
      min-height: 28px;
      padding: 6px 11px;
      border-top: 1px solid #e5e8ee;
      color: #667085;
      font-size: 10px;
      line-height: 1.35;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; max-width: 1800px; margin: 8px auto 0; color: #667085; font-size: 10px; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .swatch { width: 20px; height: 6px; border-radius: 999px; background: var(--color); }
    @media (max-width: 900px) {
      body { padding: 10px; }
      .page-header { align-items: flex-start; flex-direction: column; }
      .toolbar { justify-content: flex-start; }
      .charts { grid-template-columns: 1fr; }
      .chart-card[data-chart="overview"] { grid-column: auto; }
      .canvas, .chart-card[data-chart="overview"] .canvas { height: 540px; }
    }
  </style>
</head>
<body>
  <header class="page-header">
    <div>
      <h1>CIPOC workflow overview</h1>
      <p class="subtitle">A compact view of the major phases and decisions. Detailed implementation nodes are intentionally omitted.</p>
    </div>
    <div class="toolbar">
      <button id="fit-all" type="button">Fit all</button>
      <button id="relayout-all" type="button">Re-layout all</button>
      <label class="toggle"><input id="show-edge-labels" type="checkbox" checked>Show edge labels</label>
    </div>
  </header>
  <main id="charts" class="charts"></main>
  <footer class="legend">
    <span class="legend-item"><span class="swatch" style="--color:#6d5bd0"></span>Orchestrator</span>
    <span class="legend-item"><span class="swatch" style="--color:#008c7a"></span>Note scanner</span>
    <span class="legend-item"><span class="swatch" style="--color:#d16b22"></span>Note retriever</span>
    <span class="legend-item"><span class="swatch" style="--color:#1473e6"></span>Extractor</span>
    <span>Triple dashed line = concurrent fan-out</span>
    <span>Dotted line = loop</span>
  </footer>

  <script type="application/json" id="flowchart-data">__FLOWCHART_DATA__</script>
  <script>
__CYTOSCAPE_START_MARKER__
__CYTOSCAPE_SOURCE__
__CYTOSCAPE_END_MARKER__
  </script>
  <script>
    const documentData = JSON.parse(document.getElementById("flowchart-data").textContent);
    const chartsElement = document.getElementById("charts");
    const instances = [];

    function fit(instance) {
      instance.cy.animate({ fit: { eles: instance.cy.elements(), padding: 22 }, duration: 180 });
    }

    function runLayout(instance) {
      instance.cy.layout({ ...instance.chart.layout, animate: false, padding: 24 }).run();
      fit(instance);
    }

    function savePng(instance) {
      const blob = instance.cy.png({
        output: "blob",
        full: true,
        scale: 2,
        bg: "#fbfcfe"
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `cipoc-${instance.chart.id.replaceAll("_", "-")}.png`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    for (const chart of documentData.charts) {
      const card = document.createElement("article");
      card.className = "chart-card";
      card.dataset.chart = chart.id;

      const header = document.createElement("header");
      header.className = "chart-header";
      const headingGroup = document.createElement("div");
      const title = document.createElement("h2");
      title.className = "chart-title";
      title.textContent = chart.title;
      const description = document.createElement("p");
      description.className = "chart-description";
      description.textContent = chart.description;
      headingGroup.append(title, description);
      const fitButton = document.createElement("button");
      fitButton.type = "button";
      fitButton.textContent = "Fit chart";
      const saveButton = document.createElement("button");
      saveButton.type = "button";
      saveButton.textContent = "Save PNG";
      const actions = document.createElement("div");
      actions.className = "chart-actions";
      actions.append(fitButton, saveButton);
      header.append(headingGroup, actions);

      const canvas = document.createElement("div");
      canvas.className = "canvas";
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", `${chart.title} flowchart`);
      const footer = document.createElement("div");
      footer.className = "chart-footer";
      footer.textContent = `${chart.elements.nodes.length} nodes / ${chart.elements.edges.length} rendered edges. Select a node or edge for details.`;
      card.append(header, canvas, footer);
      chartsElement.append(card);

      const cy = cytoscape({
        container: canvas,
        elements: chart.elements,
        style: documentData.style,
        layout: { ...chart.layout, animate: false, padding: 24 },
        minZoom: 0.12,
        maxZoom: 2.5,
        wheelSensitivity: 0.18
      });
      const instance = { chart, cy, footer };
      instances.push(instance);
      fitButton.addEventListener("click", () => fit(instance));
      saveButton.addEventListener("click", () => savePng(instance));

      cy.on("tap", "node", event => {
        const data = event.target.data();
        footer.textContent = `${data.label}: ${data.detail || "No additional detail."}`;
      });
      cy.on("tap", "edge", event => {
        const data = event.target.data();
        const source = cy.getElementById(data.source).data("label");
        const target = cy.getElementById(data.target).data("label");
        footer.textContent = `${source} to ${target}: ${data.label || data.kind || "flow"}`;
      });
      cy.ready(() => fit(instance));
    }

    document.getElementById("fit-all").addEventListener("click", () => instances.forEach(fit));
    document.getElementById("relayout-all").addEventListener("click", () => instances.forEach(runLayout));
    document.getElementById("show-edge-labels").addEventListener("change", event => {
      for (const { cy } of instances) {
        cy.edges().style("label", event.target.checked ? "data(label)" : "");
      }
    });
    window.addEventListener("resize", () => instances.forEach(({ cy }) => cy.resize()));
  </script>
</body>
</html>
"""
    return (
        template.replace("__FLOWCHART_DATA__", data_json)
        .replace("__CYTOSCAPE_START_MARKER__", CYTOSCAPE_START_MARKER)
        .replace("__CYTOSCAPE_SOURCE__", safe_cytoscape_source)
        .replace("__CYTOSCAPE_END_MARKER__", CYTOSCAPE_END_MARKER)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html-output", type=Path, default=DEFAULT_HTML_OUTPUT)
    parser.add_argument("--cytoscape-js", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when either generated output is missing or stale.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate or check only the flowchart JSON.",
    )
    args = parser.parse_args()

    flowcharts = build_flowcharts()
    serialized = serialize_flowcharts(flowcharts)
    html = None
    if not args.json_only:
        bundle_source = args.html_output if args.html_output.is_file() else UNIFIED_HTML_OUTPUT
        cytoscape_source = load_cytoscape_source(args.cytoscape_js, bundle_source)
        html = build_html(flowcharts, cytoscape_source)

    if args.check:
        if not args.output.is_file() or args.output.read_text() != serialized:
            raise SystemExit(f"Flowchart JSON is stale or missing: {args.output}")
        if html is not None and (
            not args.html_output.is_file() or args.html_output.read_text() != html
        ):
            raise SystemExit(f"Flowchart HTML is stale or missing: {args.html_output}")
        print(f"Flowchart outputs are current: {args.output}, {args.html_output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(f"Wrote separate flowcharts to {args.output}")
    if html is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(html)
        print(f"Wrote standalone flowchart viewer to {args.html_output}")


if __name__ == "__main__":
    main()
