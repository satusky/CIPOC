/*
 * CIPOC extraction demo — frontend (Phase 3).
 *
 * Renders three panels from the demo server's SSE cursor stream:
 *   1. Workflow map   — the overview flowchart, with the current node haloed,
 *                       visited nodes lit, and traversed edges highlighted.
 *   2. Current step   — the model I/O and task input/result for every map node
 *                       touched during the presenter's current step.
 *   3. Variables      — the reused ProgressModel variable table, grouped.
 *
 * The server is the single source of truth: every control (Prev/Next/goto/play)
 * POSTs to the server, which broadcasts the new cursor over SSE; the UI only
 * ever redraws in response to an SSE `cursor` message, so all viewers stay in
 * lockstep with the presenter.
 */

"use strict";

const AUTOPLAY_MS = 3500;

/*
 * Map the fine-grained agent_system.json node IDs that snapshots report onto the
 * coarse "overview" flowchart blocks that Panel 1 draws. (mapping.py bridges the
 * runtime graph -> agent_system IDs; this bridges those -> the simplified map.)
 *
 * This is the fallback; at runtime it is replaced by the authoritative map the
 * server sends in /api/graph.coarse_map (see buildMap), which a Python test keeps
 * in sync with the graph.
 */
let COARSE = {
  initialize_case: "initialize_case",
  fan_out_notes: "scanner_agent_block",
  scanner_initialize: "scanner_agent_block",
  scanner_summarize_note: "scanner_agent_block",
  scanner_detect_concepts: "scanner_agent_block",
  scanner_get_cancer_mentions: "scanner_agent_block",
  characterize_corpus: "characterize_corpus",
  check_state: "eligible_groups_gate",
  plan_extraction: "eligible_groups_gate",
  fan_out_groups: "retriever_agent_block",
  hard_filter_notes: "retriever_agent_block",
  retriever_initialize: "retriever_agent_block",
  retriever_identify_relevant_notes: "retriever_agent_block",
  extractor_initialize: "extractor_agent_block",
  extractor_load_notes: "extractor_agent_block",
  extractor_extract_group_values: "extractor_agent_block",
  fan_out_variables: "extractor_agent_block",
  extractor_extract_individual_value: "extractor_agent_block",
  extractor_validate_extraction: "extractor_agent_block",
  extractor_repair_invalid_extraction: "extractor_agent_block",
  extractor_complete_variable: "extractor_agent_block",
  merge_variable_results: "extractor_agent_block",
  merge_and_update: "update_case",
  finalize_case: "finalize_case",
};

// Inverse: coarse block -> the fine node IDs beneath it (for click-to-focus).
let COARSE_MEMBERS = {};
function rebuildCoarseMembers() {
  COARSE_MEMBERS = {};
  for (const [fine, block] of Object.entries(COARSE)) {
    (COARSE_MEMBERS[block] = COARSE_MEMBERS[block] || []).push(fine);
  }
}
rebuildCoarseMembers();

const AGENTS = ["orchestrator", "scanner", "retriever", "extractor"];

// Human section titles for the fine map-node IDs Panel 2 renders. Falls back to
// the raw ID, so a node added to the graph shows up rather than disappearing.
const NODE_TITLES = {
  initialize_case: "Initialize case",
  fan_out_notes: "Fan out notes",
  scanner_initialize: "Scanner init",
  scanner_summarize_note: "Note summary",
  scanner_detect_concepts: "Concept detection",
  scanner_get_cancer_mentions: "Cancer mentions",
  characterize_corpus: "Corpus characterization",
  check_state: "Check state",
  plan_extraction: "Extraction plan",
  fan_out_groups: "Fan out groups",
  hard_filter_notes: "Hard filter",
  retriever_initialize: "Retriever init",
  retriever_identify_relevant_notes: "Relevant notes",
  extractor_initialize: "Extractor init",
  extractor_load_notes: "Load notes",
  extractor_extract_group_values: "Group extraction",
  fan_out_variables: "Fan out variables",
  extractor_extract_individual_value: "Individual extraction",
  extractor_validate_extraction: "Validation",
  extractor_repair_invalid_extraction: "Repair",
  extractor_complete_variable: "Complete variable",
  merge_variable_results: "Merge variable results",
  merge_and_update: "Update case",
  finalize_case: "Finalize case",
  relevant_notes_gate: "Relevant notes?",
  eligible_groups_gate: "Groups remain?",
};

// Structural plumbing — initialization, fan-outs, loaders, logic gates. They
// carry no decision worth reading, and their raw payloads (a whole CaseState, a
// whole note corpus) bury the components that do. Rendered as one compact strip
// with no input/result dropdowns; an *erroring* one is promoted back to a full
// card so a failure is never swallowed.
const MINOR_NODES = new Set([
  "initialize_case",
  "fan_out_notes",
  "scanner_initialize",
  "retriever_initialize",
  "extractor_initialize",
  "extractor_load_notes",
  "hard_filter_notes",
  "fan_out_groups",
  "fan_out_variables",
  "relevant_notes_gate",
  "merge_variable_results",
]);

// Nodes whose view is subsumed by another node's when both land in one step.
// ``check_state`` and ``plan_extraction`` now share a step (steps.py merges
// them) and both render the same eligibility gate off snapshot.progress, so the
// gate is drawn once, by the node that names it.
const SUBSUMED_BY = { check_state: "plan_extraction" };

// Runtime node -> what that pass of the extractor's inner loop did.
const ATTEMPT_LABELS = {
  extract_individual_value: "individual extraction",
  validate_extraction: "validation",
  repair_invalid_extraction: "repair",
};

// Layout: default share of the left column given to the Variables panel, and
// where a presenter's dragged size is remembered across reloads.
const VARS_H_KEY = "cipoc.demo.varsHeight";
const VARS_MIN = 120;
const MAP_MIN = 200;

// --- runtime state -------------------------------------------------------
const els = {};
let cy = null;
let steps = [];          // static step list (GET /api/steps)
let events = [];         // full event list (GET /api/events) for per-step nodes
let numSteps = 0;
let lastView = null;     // most recent SSE cursor view
let focusBlock = null;   // manual Panel-2 focus (coarse block id) or null
let autoTimer = null;    // presenter-side auto-play interval
let notesById = {};      // note_id -> {note_type, date, content} for span highlighting
let notesFetching = false;

// -------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheEls();
  wireControls();
  wireSplitter();

  const [meta, graph] = await Promise.all([
    getJSON("/api/meta"),
    getJSON("/api/graph"),
    loadStatic(),
  ]);

  applyMeta(meta);
  buildMap(graph);
  buildStepSelect();
  ensureNotes(); // raw note text for inline evidence-span highlighting
  openStream();
}

// Fetch the note corpus (note_id -> content) once for evidence highlighting.
// Content is immutable once a note is scanned, so a single fetch serves every
// cursor; re-fetched only if a referenced note is still missing (live mode).
async function ensureNotes() {
  if (notesFetching) return;
  notesFetching = true;
  try {
    const notes = await getJSON("/api/notes");
    if (notes && Object.keys(notes).length) {
      notesById = notes;
      if (lastView) renderDetail(lastView); // re-render now that text is available
    }
  } catch {
    /* notes are best-effort; highlighting degrades to quotes only */
  } finally {
    notesFetching = false;
  }
}

function cacheEls() {
  els.title = document.getElementById("run-title");
  els.sub = document.getElementById("run-sub");
  els.modeBadge = document.getElementById("mode-badge");
  els.prev = document.getElementById("btn-prev");
  els.next = document.getElementById("btn-next");
  els.play = document.getElementById("btn-play");
  els.stepSelect = document.getElementById("step-select");
  els.counter = document.getElementById("step-counter");
  els.legend = document.getElementById("agent-legend");
  els.detail = document.getElementById("detail");
  els.detailNode = document.getElementById("detail-node");
  els.vars = document.getElementById("vars");
  els.varsSummary = document.getElementById("vars-summary");
}

async function loadStatic() {
  [steps, events] = await Promise.all([
    getJSON("/api/steps"),
    getJSON("/api/events"),
  ]);
}

function applyMeta(meta) {
  numSteps = meta.num_steps || 0;
  els.title.textContent = meta.description || "CIPOC extraction";
  els.sub.textContent =
    `${meta.num_steps} steps · ${meta.num_events} events`;
  els.modeBadge.textContent = meta.mode;
  els.modeBadge.classList.toggle("live", meta.mode === "live");
}

// --- Panel 1: workflow map ----------------------------------------------

// Hand-authored positions matching the reference flowchart: a horizontal top
// row (Initialize → Scanner → Characterize) feeding a central decision column
// (Groups? → Retriever → Relevant? → Extractor) that LOOPS back up the left
// side through Update case to the gate — so the per-group extraction loop reads
// as an actual cycle. The two "no" branches exit sideways (→ Finalize, →
// Update). START/END endpoints are hidden in the demo (Initialize/Finalize read
// as the entry/terminal, as in the reference). Keyed by node id.
const MAP_POS = {
  initialize_case:       { x: 130, y: 60 },
  scanner_agent_block:   { x: 410, y: 60 },
  characterize_corpus:   { x: 640, y: 60 },
  eligible_groups_gate:  { x: 640, y: 205 },   // "Groups remain?" — top of the loop
  finalize_case:         { x: 960, y: 205 },   // "no" exit → terminal
  retriever_agent_block: { x: 640, y: 350 },
  relevant_notes_gate:   { x: 640, y: 495 },
  update_case:           { x: 210, y: 495 },   // left side of the loop
  extractor_agent_block: { x: 640, y: 640 },
};

// Endpoints the demo map omits (the reference starts at Initialize, ends at
// Finalize); their edges are dropped with them.
const MAP_HIDE = new Set(["case_start", "case_end"]);

// Lighten a hex color toward white (soft node fills that keep the agent hue).
function tint(hex, amt) {
  const c = String(hex).replace("#", "");
  if (c.length < 6) return hex;
  const r = parseInt(c.slice(0, 2), 16);
  const g = parseInt(c.slice(2, 4), 16);
  const b = parseInt(c.slice(4, 6), 16);
  const mix = (x) => Math.round(x + (255 - x) * amt);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

// Demo stylesheet for the map, built from the chart's agent colors. Uniform
// rounded blocks with a soft agent tint + agent-colored border; diamonds for the
// decision gates; Finalize colored as the terminal. Edges are orthogonal (taxi,
// like the reference): forward flow into an agent is blue, the two "no" exits are
// red, and the per-group loop-back is an emphasized cycle so it can't be missed.
function mapStyle(agentColors) {
  const BLUE = "#4a86d8", RED = "#e0564b", LOOP = "#6b5bd0", GRAY = "#9aa3b2";
  const perAgent = Object.entries(agentColors).map(([agent, color]) => ({
    selector: `node[agent="${agent}"]`,
    style: { "border-color": color, "background-color": tint(color, 0.86) },
  }));
  return [
    {
      selector: "node",
      style: {
        shape: "round-rectangle",
        width: 150, height: 50,
        "background-color": "#ffffff",
        "border-width": 2, "border-color": "#cbd0da",
        label: "data(label)", "text-wrap": "wrap", "text-max-width": 126,
        "text-valign": "center", "text-halign": "center",
        "font-size": 11.5, "font-weight": 600, color: "#2b3040",
      },
    },
    { selector: 'node[kind="subagent"]', style: { width: 168, height: 58, "border-width": 3, "font-size": 12.5, "font-weight": 700 } },
    { selector: 'node[kind="decision"]', style: { shape: "round-diamond", width: 150, height: 108, "background-color": "#ffffff", "font-size": 11 } },
    ...perAgent,
    // Finalize reads as the terminal (as in the reference): soft red.
    { selector: 'node[id="finalize_case"]', style: { "background-color": "#f6dede", "border-color": RED, color: "#8a2c2c" } },

    // Orthogonal connectors, thin gray by default.
    {
      selector: "edge",
      style: {
        width: 2, "line-color": GRAY, "curve-style": "taxi",
        "taxi-turn": "50%", "taxi-turn-min-distance": 8,
        "target-arrow-shape": "triangle", "target-arrow-color": GRAY, "arrow-scale": 1.1,
      },
    },
    // Forward flow that enters an agent block: blue (fan-out branches + the
    // "yes" edge into the extractor).
    { selector: 'edge[kind="fanout"]', style: { "line-color": BLUE, "target-arrow-color": BLUE } },
    { selector: 'edge[label="yes"]', style: { "line-color": BLUE, "target-arrow-color": BLUE } },
    // The two "no" branches exit the loop sideways: red.
    { selector: 'edge[label="no"]', style: { "line-color": RED, "target-arrow-color": RED, width: 2.5 } },
    { selector: 'edge[label="no: not found"]', style: { "line-color": RED, "target-arrow-color": RED, width: 2.5 } },
    // Extractor → Update case: leave horizontally so it runs along the bottom
    // then up the left side (the loop's lower-left corner).
    { selector: 'edge[source="extractor_agent_block"]', style: { "taxi-direction": "horizontal" } },
    // The per-group loop-back (Update case → gate): emphasized cycle — thick,
    // colored, routed up the left and across the top.
    {
      selector: 'edge[kind="loop"]',
      style: {
        "line-color": LOOP, "target-arrow-color": LOOP, width: 3,
        "taxi-direction": "vertical", "taxi-turn": "90%",
      },
    },
    // Edge captions (yes / no / loop / "one branch per …") on a small white chip.
    {
      selector: "edge[label]",
      style: {
        label: "data(label)", "font-size": 9.5, "font-weight": 600, color: "#5b6270",
        "text-background-color": "#ffffff", "text-background-opacity": 0.92,
        "text-background-padding": 2.5, "text-background-shape": "round-rectangle",
        "text-rotation": "none",
      },
    },
  ];
}

function buildMap(graph) {
  if (graph.coarse_map && Object.keys(graph.coarse_map).length) {
    COARSE = graph.coarse_map;
    rebuildCoarseMembers();
  }
  const agentColors = graph.agent_colors || {};
  applyAgentColors(agentColors);
  renderLegend(agentColors);

  // Hand-authored layout matching the reference flowchart (see MAP_POS). Endpoint
  // blocks are hidden; each fan-out is drawn as ONE labeled edge (the ×N badge on
  // the target node already conveys the multiplicity), so the map reads as a clean
  // cyclic flowchart. Falls back to breadthfirst if the chart grows nodes we have
  // no position for.
  const rawNodes = graph.elements.nodes.filter((n) => !MAP_HIDE.has(n.data.id));
  const havePreset = rawNodes.every((n) => MAP_POS[n.data.id]);
  const nodes = rawNodes.map((n) => ({
    data: { ...n.data, baseLabel: n.data.label },
    ...(MAP_POS[n.data.id] ? { position: { ...MAP_POS[n.data.id] } } : {}),
  }));
  const edges = graph.elements.edges
    .filter((e) => !MAP_HIDE.has(e.data.source) && !MAP_HIDE.has(e.data.target))
    // Collapse fan-out triples: keep only the center (labeled) lane.
    .filter((e) => e.data.kind !== "fanout" || e.data.fanout_lane === "center")
    .map((e) => ({ data: { ...e.data } }));

  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: { nodes, edges },
    // Demo-only stylesheet (built from the chart's agent colors) + the authored
    // preset positions; the shared chart JSON is untouched. Run-state overlays
    // (visited / current / active / traversed) layer on top.
    style: [...mapStyle(agentColors), ...stateStyles()],
    layout: havePreset
      ? { name: "preset", fit: true, padding: 24 }
      : graph.layout || { name: "breadthfirst", directed: true },
    wheelSensitivity: 0.2,
    minZoom: 0.3,
    maxZoom: 2.5,
  });

  // The map lives in a flex panel that may not have its final size when
  // Cytoscape initializes, so re-fit once laid out and whenever it resizes.
  const fit = () => { cy.resize(); cy.fit(undefined, 24); };
  cy.one("layoutstop", fit);
  requestAnimationFrame(fit);
  const container = document.getElementById("cy");
  if (window.ResizeObserver) new ResizeObserver(fit).observe(container);

  // Click a block to pin Panel 2 to it; click empty space to unpin.
  cy.on("tap", "node", (evt) => {
    focusBlock = evt.target.id();
    if (lastView) renderDetail(lastView);
  });
  cy.on("tap", (evt) => {
    if (evt.target === cy) {
      focusBlock = null;
      if (lastView) renderDetail(lastView);
    }
  });
}

function applyAgentColors(colors) {
  const root = document.documentElement.style;
  for (const a of AGENTS) if (colors[a]) root.setProperty(`--${a}`, colors[a]);
}

function stateStyles() {
  return [
    {
      selector: "node",
      style: {
        "transition-property": "opacity, border-width, background-color",
        "transition-duration": "150ms",
      },
    },
    { selector: ".dim", style: { opacity: 0.32 } },
    { selector: "node.visited", style: { opacity: 1 } },
    {
      selector: "node.current",
      style: {
        "border-width": 5,
        "overlay-color": "#ffb020",
        "overlay-opacity": 0.28,
        "overlay-padding": 9,
        "z-index": 20,
      },
    },
    {
      selector: "node.active",
      style: { "overlay-color": "#ffb020", "overlay-opacity": 0.22, "overlay-padding": 7 },
    },
    { selector: "edge.dim", style: { opacity: 0.16 } },
    // Traversed edges just light up (full opacity + raised) — the run-state must
    // NOT recolor them, so the semantic edge colors (blue forward / red exit /
    // purple loop) stay readable along the walked path.
    { selector: "edge.traversed", style: { opacity: 1, "z-index": 15 } },
  ];
}

function renderLegend(colors) {
  els.legend.innerHTML = AGENTS.map(
    (a) =>
      `<span class="lg"><span class="sw" style="background:${colors[a] || "#6d5bd0"}"></span>${a}</span>`
  ).join("");
}

function coarse(fineId) {
  return fineId ? COARSE[fineId] || null : null;
}

function updateMap(snapshot, step) {
  if (!cy) return;
  const visitedCoarse = new Set();
  for (const fine of snapshot.visited_map_nodes || []) {
    const c = coarse(fine);
    if (c) visitedCoarse.add(c);
  }
  // Older recorded traces mapped the extract wrapper directly onto the
  // extractor. The wrapper is the implicit relevant-notes decision, so recover
  // that conceptual visit without requiring persisted traces to be rewritten.
  if (events.some((event) =>
    event.seq <= snapshot.seq && event.type === "task_start" && event.node === "extract"
  )) {
    visitedCoarse.add("relevant_notes_gate");
  }
  visitedCoarse.add("case_start");
  if (snapshot.finished) visitedCoarse.add("case_end");

  const activeCoarse = new Set();
  for (const fine of snapshot.active_map_nodes || []) {
    const c = coarse(fine);
    if (c) activeCoarse.add(c);
  }
  const currentCoarse = coarse(step && step.map_node_id);

  // Fan-out multiplicity badges: max instance count among member fine nodes.
  const badge = {};
  for (const [block, members] of Object.entries(COARSE_MEMBERS)) {
    let n = 0;
    for (const m of members) {
      const d = snapshot.details[m];
      if (d && d.count > n) n = d.count;
    }
    if (n > 1) badge[block] = n;
  }

  cy.batch(() => {
    cy.nodes().forEach((node) => {
      const id = node.id();
      node.removeClass("visited current active dim");
      if (visitedCoarse.has(id)) node.addClass("visited");
      else node.addClass("dim");
      if (activeCoarse.has(id)) node.addClass("active");
      if (id === currentCoarse) node.addClass("current");
      const base = node.data("baseLabel");
      node.data("label", badge[id] ? `${base}  ×${badge[id]}` : base);
    });
    cy.edges().forEach((edge) => {
      edge.removeClass("traversed dim");
      const s = edge.source().id();
      const t = edge.target().id();
      if (visitedCoarse.has(s) && visitedCoarse.has(t)) edge.addClass("traversed");
      else edge.addClass("dim");
    });
  });
}

// --- Panel 2: current-step detail ---------------------------------------

// Ordered, de-duplicated fine map-node IDs touched within a step's seq range.
function stepNodeIds(step) {
  if (!step) return [];
  const seen = new Set();
  const out = [];
  for (const ev of events) {
    if (ev.seq < step.start_seq || ev.seq > step.end_seq) continue;
    if (!ev.map_node_id || seen.has(ev.map_node_id)) continue;
    seen.add(ev.map_node_id);
    out.push(ev.map_node_id);
  }
  return out;
}

function nodeTitle(id) {
  return NODE_TITLES[id] || id;
}

function detailHeadline(title, subtitle, agent, extra) {
  return `<div class="detail-headline">
      <h3 class="detail-title">${esc(title)}</h3>
      <div class="detail-meta">
        ${subtitle ? `<span class="muted">${esc(subtitle)}</span>` : ""}
        ${agent ? `<span class="chip agent-${agent}">${esc(agent)}</span>` : ""}
        ${extra || ""}
      </div>
    </div>`;
}

function renderDetail(view) {
  const snap = view.snapshot;
  const step = view.step;

  if (!focusBlock && step) {
    // A collapsed fan-out step (e.g. "Characterize notes") shows one card per
    // instance instead of one merged card per map node.
    if (step.fanout) {
      renderFanoutDetail(step, snap);
      return;
    }
    // A group's extraction step is likewise per-instance — one card per
    // variable, not one per node of the extractor's inner loop.
    if (step.node === "extract_branch") {
      renderExtractDetail(step, snap);
      return;
    }
  }

  let title, subtitle, agent, nodeIds;
  // A pinned block is an explicit "show me everything about this component"
  // request, so it keeps the full cards (raw payloads included).
  const pinned = Boolean(focusBlock);
  if (pinned) {
    title = blockLabel(focusBlock);
    subtitle = "pinned component";
    agent = blockAgent(focusBlock);
    nodeIds = (COARSE_MEMBERS[focusBlock] || []).filter((id) => snap.details[id]);
  } else if (step) {
    title = step.title;
    subtitle = step.subtitle || "";
    agent = step.agent;
    // Primary node first, then any other fine nodes touched this step.
    const touched = stepNodeIds(step);
    const primary = step.map_node_id;
    nodeIds = primary && !touched.includes(primary) ? [primary, ...touched] : touched;
    const present = nodeIds.filter((id) => snap.details[id]);
    nodeIds = present.filter((id) => !present.includes(SUBSUMED_BY[id]));
  } else {
    title = "Run start";
    subtitle = "";
    agent = null;
    nodeIds = [];
  }

  els.detailNode.textContent = step && step.map_node_id ? step.map_node_id : "";

  const parts = [
    detailHeadline(
      title,
      subtitle,
      agent,
      pinned ? `<span class="chip">click background to unpin</span>` : ""
    ),
  ];

  if (nodeIds.length === 0) {
    parts.push(`<p class="empty">No model activity captured for this step.</p>`);
  } else if (pinned) {
    for (const id of nodeIds) parts.push(renderNodeDetail(id, snap.details[id], snap, true));
  } else {
    parts.push(
      renderTimeline(nodeIds, snap, (id) => renderNodeDetail(id, snap.details[id], snap, false))
    );
  }
  els.detail.innerHTML = parts.join("");
}

function isMinor(id, detail) {
  return MINOR_NODES.has(id) && !(detail && detail.error);
}

// A structural node as one quiet row *in place* — the step reads top-to-bottom
// as the order things actually happened, with the plumbing shrunk rather than
// swept into a header strip. No card, no raw payload dropdowns, and no ×N:
// NodeDetail.count is cumulative over the whole run, so it would read as this
// step's fan-out and be wrong on the second group. The map node's badge
// (Panel 1) is where multiplicity belongs.
const MINOR_GLYPH = { done: "✓", active: "◐", error: "✗", invalid: "⚠", idle: "·" };

function renderMinorRow(id, snap) {
  const status = (snap.details[id] || {}).status || "idle";
  return `<div class="minor-row st-${esc(status)}">
    <span class="minor-glyph">${MINOR_GLYPH[status] || "·"}</span>
    <span class="minor-name">${esc(nodeTitle(id))}</span>
    <span class="minor-id">${esc(id)}</span>
  </div>`;
}

// Walk a step's map nodes in the order they were touched, rendering each as
// either a quiet row (plumbing) or a full section (``renderMajor``). ``inject``
// maps a node id to extra HTML emitted right after it, which is how the
// per-variable cards land at the point the fan-out actually happened.
function renderTimeline(ids, snap, renderMajor, inject) {
  const parts = [];
  for (const id of ids) {
    parts.push(isMinor(id, snap.details[id]) ? renderMinorRow(id, snap) : renderMajor(id));
    if (inject && inject[id]) parts.push(inject[id]);
  }
  return parts.join("");
}

// A section header that reads as a *separator* — an agent-colored rule with the
// component's name — rather than one more pill in a row of pills. The fine node
// ID stays visible but de-emphasized, and the status pill is the only pill left.
function nodeHead(id, agent, status, extra) {
  return `<header class="node-head agent-${agent || "orchestrator"}">
    <span class="node-head-title">${esc(nodeTitle(id))}</span>
    <span class="node-head-id">${esc(id)}</span>
    ${extra || ""}
    ${status ? `<span class="status-pill status-${esc(status)}">${esc(status)}</span>` : ""}
  </header>`;
}

// ``raw`` adds the "Task input"/"Task result" payload dropdowns; step rendering
// leaves them off (they drown out everything else) and pinning turns them on.
function renderNodeDetail(id, detail, snap, raw) {
  const status = detail.status || "idle";
  // Cumulative over the whole run, so it only makes sense on the pinned
  // component view — inside a step it reads as this step's fan-out.
  const count =
    raw && detail.count > 1 ? `<span class="summary-tag">×${detail.count} instances</span>` : "";
  const headline = componentHeadline(id, detail, snap);
  const calls = (detail.llm_calls || []).map(renderLLMCall).join("");

  return `<section class="node-detail">
    ${nodeHead(id, detail.agent, status, count)}
    ${headline}
    ${calls || ""}
    ${raw ? collapsible("Task input", detail.input, "input") : ""}
    ${raw ? collapsible("Task result", detail.result, "result") : ""}
    ${
      detail.error
        ? `<pre class="code">${esc(fmt(detail.error))}</pre>`
        : ""
    }
  </section>`;
}

// --- Panel 2: a group's extraction step ---------------------------------
// One card per variable (fed by the per-variable fan-out instances in
// state.py), preceded by the retriever's verdict and the single group-level
// model call. The merged group result is deliberately omitted: it is just the
// variable cards concatenated.
function renderExtractDetail(step, snap) {
  const details = snap.details || {};
  const prefix = `extract_branch:${step.task_id}/`;
  const vars = Object.values(snap.instances || {})
    .filter((i) => i.node === "variable_branch" && i.key.startsWith(prefix))
    .sort((a, b) => a.index - b.index);
  const settled = vars.filter((i) => i.status !== "active").length;

  els.detailNode.textContent = step.map_node_id || "";

  const counts = vars.length
    ? `${vars.length} variable${vars.length === 1 ? "" : "s"} · ${settled} extracted`
    : "";
  const subtitle = [step.subtitle, counts].filter(Boolean).join(" · ");
  const parts = [detailHeadline(step.title, subtitle, step.agent, "")];

  // The inner loop's own nodes are replaced wholesale by the variable cards.
  const perVariable = new Set([
    "extractor_validate_extraction",
    "extractor_repair_invalid_extraction",
    "extractor_complete_variable",
    "extractor_extract_individual_value",
  ]);
  const touched = stepNodeIds(step).filter((id) => details[id] && !perVariable.has(id));

  // The variable cards belong where the fan-out happened; if that node never
  // reported (an older trace), fall back to the end of the timeline.
  const cards = vars.length
    ? vars.map(renderVariableDetail).join("")
    : `<p class="empty">No variables extracted for this group${
        details.retriever_identify_relevant_notes
          ? " — see the retriever verdict above."
          : "."
      }</p>`;
  const anchor = touched.includes("fan_out_variables")
    ? "fan_out_variables"
    : touched[touched.length - 1];

  parts.push(
    renderTimeline(touched, snap, (id) => renderNodeDetail(id, details[id], snap, false), {
      [anchor]: cards,
    })
  );
  els.detail.innerHTML = parts.join("");
}

// One variable, start to finish: its coded value with evidence, its final
// validation errors in plain sight, and the earlier attempts tucked away.
function renderVariableDetail(inst) {
  const status = inst.status || "active";
  const result = inst.result && typeof inst.result === "object" ? inst.result : {};
  const results = Array.isArray(result.variable_results) ? result.variable_results : [];
  const final = results[0];
  const task = (inst.input && inst.input.task) || {};
  const itemId = final ? final.item_id : (task.variable || {}).item_id;
  const calls = (inst.llm_calls || []).map(renderLLMCall).join("");

  return `<section class="node-detail variable status-${esc(status)}">
    <header class="node-head agent-${inst.agent || "extractor"}">
      <span class="node-head-title">${esc(inst.label || inst.key)}</span>
      <span class="node-head-id">${itemId == null ? "" : `#${esc(itemId)}`}</span>
      <span class="status-pill status-${esc(status)}">${esc(status)}</span>
    </header>
    ${final ? extractionRow(final, null) : pendingSlot("Extraction")}
    ${renderAttempts(inst.attempts || [])}
    ${calls || ""}
    ${inst.error ? `<pre class="code">${esc(fmt(inst.error))}</pre>` : ""}
  </section>`;
}

// The extractor's inner loop, collapsed. A single clean pass says nothing worth
// a dropdown, so only a genuine retry opens one — the final verdict and its
// errors are already visible on the card above.
function renderAttempts(attempts) {
  if (attempts.length <= 1) return "";
  // Only a validation pass carries a verdict. An extract/repair pass clears the
  // flag before re-validating, so counting it as a failure would double-count.
  const checks = attempts.filter((a) => a.node === "validate_extraction");
  const failed = checks.filter((a) => a.is_valid === false).length;
  const rows = attempts
    .map((a) => {
      const check = a.node === "validate_extraction";
      const errs = (a.validation_errors || [])
        .map((e) => `<li>${esc(e)}</li>`)
        .join("");
      const verdict = !check
        ? "→ new candidate"
        : a.is_valid
        ? "✓ valid"
        : "✗ invalid";
      return `<div class="attempt ${!check ? "" : a.is_valid ? "ok" : "bad"}">
        <div class="attempt-head">
          <b>Attempt ${esc(a.attempt == null ? "?" : a.attempt)}</b>
          <span class="muted">${esc(ATTEMPT_LABELS[a.node] || a.node || "")}</span>
          <span class="attempt-verdict">${verdict}</span>
        </div>
        ${errs ? `<ul class="val-errors">${errs}</ul>` : ""}
      </div>`;
    })
    .join("");
  return `<details class="block">
    <summary>Validation attempts<span class="summary-tag">${checks.length} check${
    checks.length === 1 ? "" : "s"
  } · ${failed} failed</span></summary>
    <div class="block-body">${rows}</div>
  </details>`;
}

// A fan-out step: render each parallel instance (e.g. each characterized note)
// as its own card, in fan-out order, so every note's summary/concepts/mentions
// and its own model calls stay grouped with that note.
function renderFanoutDetail(step, snap) {
  const insts = Object.values(snap.instances || {})
    .filter((i) => i.node === step.node)
    .sort((a, b) => a.index - b.index);
  const done = insts.filter((i) => i.status === "done").length;

  const parts = [];
  parts.push(`<div class="detail-headline">
      <h3 class="detail-title">${esc(step.title)}</h3>
      <div class="detail-meta">
        <span class="muted">${insts.length} note${insts.length === 1 ? "" : "s"}${
    insts.length ? ` · ${done} characterized` : ""
  }</span>
        ${step.agent ? `<span class="chip agent-${step.agent}">${esc(step.agent)}</span>` : ""}
      </div>
    </div>`);

  els.detailNode.textContent = step.map_node_id || "";

  if (!insts.length) {
    parts.push(`<p class="empty">No notes characterized yet for this step.</p>`);
  } else {
    for (const inst of insts) parts.push(renderInstanceDetail(inst));
  }
  els.detail.innerHTML = parts.join("");
}

function renderInstanceDetail(inst) {
  const status = inst.status || (inst.active ? "active" : "done");
  const r = inst.result && typeof inst.result === "object" ? inst.result : {};
  const calls = inst.llm_calls || [];
  // Anything not claimed by one of the note's three sub-steps still surfaces,
  // so a new scanner node never silently loses its model call.
  const other = calls
    .filter((c) => !NOTE_SLOT_NODES.has(c.node))
    .map(renderLLMCall)
    .join("");

  return `<section class="node-detail instance status-${status}">
    <header class="node-head agent-${inst.agent || "scanner"}">
      <span class="node-head-title">${esc(inst.label || inst.key)}</span>
      <span class="node-head-id"></span>
      <span class="status-pill status-${status}">${status}</span>
    </header>
    ${viewNote(r, inst.input, calls)}
    ${other}
    ${collapsible("Result", inst.result, "result")}
    ${inst.error ? `<pre class="code">${esc(fmt(inst.error))}</pre>` : ""}
  </section>`;
}

// The scanner sub-step behind each of a note card's three slots. Its captured
// model call belongs *inside* that slot — the call is what produced the slot's
// content, so reading them apart makes the reader correlate by hand.
const NOTE_SLOT_NODES = new Set([
  "summarize_note",
  "detect_concepts",
  "get_cancer_mentions",
]);

function callsFor(calls, node) {
  return (calls || [])
    .filter((c) => c.node === node)
    .map(renderLLMCall)
    .join("");
}

// One note's characterization as three fixed slots — summary, detected concepts,
// and cancer mentions — each holding the model call that produced it. Rendering
// all three always makes the card a stable skeleton: in live mode each slot
// shows a "pending…" placeholder until its part of the result streams in (keyed
// on the field's presence, since the scanner sub-agent's values arrive summary →
// concepts → mentions), then fills in place. (componentHeadline only dispatches
// one view; a note has all three, so compose them here.)
function viewNote(r, input, calls) {
  const summary =
    "summary" in r
      ? viewSummary(r, { note: input }, callsFor(calls, "summarize_note"))
      : pendingSlot("Note summary");
  const concepts =
    "concepts" in r
      ? viewConcepts(r.concepts || {}, callsFor(calls, "detect_concepts"))
      : pendingSlot("Concepts detected");
  let mentions;
  const mentionCalls = callsFor(calls, "get_cancer_mentions");
  if ("cancer_mentions" in r) {
    const m = r.cancer_mentions || [];
    mentions = m.length
      ? viewCancerMentions(m, r.cancer_status, mentionCalls)
      : emptySlot("Cancer mentions", "none found", mentionCalls);
  } else {
    mentions = pendingSlot("Cancer mentions");
  }
  return summary + concepts + mentions;
}

function pendingSlot(label) {
  return `<div class="headline-fact pending"><b>${esc(label)}</b>
    <span class="pending-dot">pending…</span></div>`;
}

function emptySlot(label, note, extra) {
  return `<div class="headline-fact"><b>${esc(label)}</b>
    <span class="muted" style="font-size:.78rem"> ${esc(note)}</span>${extra || ""}</div>`;
}

/*
 * Panel-2 component views (Phase 4). Each conceptual pipeline component gets a
 * purpose-built "headline" summarizing what it decided — the note scanner's
 * characterization, the retriever's kept/dropped verdict, the extractor's coded
 * values with inline evidence — instead of a raw payload dump (which stays
 * available in the collapsible Task input/result blocks below the headline).
 *
 * Dispatch is by result/input *shape* rather than node name so a node rename in
 * the graph never silently blanks a view.
 */
function componentHeadline(id, detail, snap) {
  const r = detail.result && typeof detail.result === "object" ? detail.result : {};
  const inp = detail.input && typeof detail.input === "object" ? detail.input : {};

  // The run's closing summary is drawn from the variable table rather than the
  // node's own thin result ({"report": {...}}), which says nothing on its own.
  if (detail.node === "finalize_case") return viewFinalSummary(r, inp, snap);
  if (r.note_corpus_descriptors) return viewCorpus(r);
  // Updating the case is exactly the step where the facts change, so show them.
  if (r.case_facts && /merge|update/.test(detail.node || ""))
    return viewCaseFacts(r.case_facts, "Case facts updated");
  if (Array.isArray(r.cancer_mentions) && r.cancer_mentions.length)
    return viewCancerMentions(r.cancer_mentions, r.cancer_status);
  if (r.concepts && !r.note_corpus_descriptors) return viewConcepts(r.concepts);
  if (typeof r.summary === "string" && r.summary) return viewSummary(r, inp);
  if (Array.isArray(r.relevant_note_ids)) return viewRetriever(r, snap);
  if (Array.isArray(r.variable_results) && r.variable_results.length)
    return viewExtractions(r.variable_results, extractionLabel(detail.node));
  if (r.extracted_values && Array.isArray(r.extracted_values.variables))
    return viewExtractions(r.extracted_values.variables, "Merged group results");
  // Case-level result dict: only meaningful where the case is actually updated —
  // on structural fan-out/init nodes it just echoes Panel 3, so leave it to the
  // collapsible payload there.
  if (
    r.variable_results &&
    typeof r.variable_results === "object" &&
    /merge|update/.test(detail.node || "")
  )
    return viewExtractions(
      Object.values(r.variable_results).map((cr) => ({ ...cr.extraction, item_id: cr.item_id, value: cr.value, status: cr.status })),
      "Case values updated"
    );
  if (r.task && r.task.candidate) return viewValidateRepair(r.task);
  if ((detail.node === "plan_extraction" || detail.node === "check_state") && snap && snap.progress)
    return viewGate(snap.progress);
  return "";
}

// --- scanner: single-note characterization ------------------------------
// Each view takes a trailing ``extra`` so the model call that produced it can be
// nested inside the slot rather than pooled at the bottom of the card.
function viewSummary(r, inp, extra) {
  const note = inp.note || {};
  const flags = Array.isArray(r.flags) && r.flags.length
    ? `<div class="chips">${r.flags.map((f) => `<span class="tag-chip">${esc(f)}</span>`).join("")}</div>`
    : "";
  const head = note.note_type
    ? `<div class="muted" style="font-size:.74rem">${esc(note.note_type)}${note.date ? " · " + esc(note.date) : ""}${note.note_id != null ? " · #" + esc(note.note_id) : ""}</div>`
    : "";
  return `<div class="headline-fact">
    <b>Note summary</b>${head}
    <p class="summary-text">${esc(r.summary)}</p>
    ${flags}
    ${extra || ""}
  </div>`;
}

function viewConcepts(concepts, extra) {
  const chips = Object.entries(concepts)
    .map(([name, c]) => {
      const present = c && c.presence;
      const conf = c && c.confidence ? ` <span class="conf">${esc(c.confidence)}</span>` : "";
      return `<span class="concept-chip ${present ? "present" : "absent"}">${present ? "✓" : "○"} ${esc(name)}${present ? conf : ""}</span>`;
    })
    .join("");
  return `<div class="headline-fact"><b>Concepts detected</b><div class="chips">${chips}</div>${
    extra || ""
  }</div>`;
}

function viewCancerMentions(mentions, statuses, extra) {
  const cards = mentions
    .map((m) => {
      const meta = [
        m.status ? `<span class="tag-chip">${esc(m.status)}</span>` : "",
        m.affected_tissue ? `<span class="tag-chip">${esc(m.affected_tissue)}</span>` : "",
        m.metastasis ? `<span class="tag-chip warn">metastasis</span>` : "",
        m.confidence ? `<span class="conf">${esc(m.confidence)}</span>` : "",
      ].join(" ");
      return `<div class="mention-card"><div class="detail-meta">${meta}</div>${renderEvidence(m.evidence)}</div>`;
    })
    .join("");
  const st = Array.isArray(statuses) && statuses.length
    ? `<span class="muted" style="font-size:.74rem"> (${statuses.map(esc).join(", ")})</span>`
    : "";
  return `<div class="headline-fact"><b>Cancer mentions${st}</b>${cards}${extra || ""}</div>`;
}

// --- characterize: corpus-level descriptors -----------------------------
// The corpus descriptors and the case facts derived from them are two different
// answers, so they get two containers rather than one with a sub-heading.
function viewCorpus(r) {
  const d = r.note_corpus_descriptors || {};
  const rows = [];
  if (d.note_count != null) rows.push(stat("Notes", d.note_count));
  if (Array.isArray(d.date_range) && d.date_range.length === 2)
    rows.push(stat("Date range", `${d.date_range[0]} → ${d.date_range[1]}`));
  if (d.types) rows.push(stat("Types", (Array.isArray(d.types) ? d.types : Object.keys(d.types)).map(esc).join(", ")));
  const tissues = d.affected_tissues
    ? Object.entries(d.affected_tissues)
        .map(([s, ts]) => `${(Array.isArray(ts) ? ts : Object.keys(ts)).join(", ")} (${s})`)
        .join("; ")
    : "";
  if (tissues) rows.push(stat("Affected tissue", tissues));
  return `<div class="headline-fact">
    <b>Corpus characterization</b>
    <div class="stat-grid">${rows.join("")}</div>
  </div>` + viewCaseFacts(r.case_facts, "Case facts");
}

// The case-level facts the orchestrator carries between steps. Facts that are
// still unresolved are shown as "—" rather than dropped, so the same rows appear
// every time the case is updated and a value filling in is visible as a change.
function viewCaseFacts(facts, label) {
  if (!facts || typeof facts !== "object") return "";
  const entries = Object.entries(facts);
  if (!entries.length) return "";
  const rows = entries
    .map(([k, v]) =>
      stat(k.replace(/_/g, " "), v == null || v === "" ? "—" : fmt(v))
    )
    .join("");
  const known = entries.filter(([, v]) => v != null && v !== "").length;
  return `<div class="headline-fact">
    <b>${esc(label)}<span class="conf"> ${known}/${entries.length} known</span></b>
    <div class="stat-grid">${rows}</div>
  </div>`;
}

// --- finalize: what the run actually produced ---------------------------
// finalize_case's own result is just the report envelope, so the summary is
// built from the variable table (Panel 3's source): how much of the plan was
// answered, how it broke down by status, and the coded values themselves.
function viewFinalSummary(r, inp, snap) {
  const prog = (snap && snap.progress) || null;
  const report = r.report && typeof r.report === "object" ? r.report : {};
  const flags = Array.isArray(report.flags) ? report.flags : [];
  // finalize only adds the report envelope, so the case it was handed *is* the
  // final case — which is where the settled facts live.
  const facts = viewCaseFacts(inp.case_facts, "Final case facts");

  if (!prog) {
    return `<div class="headline-fact"><b>Case finalized</b>
      <div class="muted" style="font-size:.78rem">No variable plan was produced.</div></div>${facts}`;
  }

  const t = prog.totals || {};
  const rows = [
    stat("Variables", `${t.terminal ?? 0} of ${t.variables ?? 0} resolved`),
    stat("Groups", `${t.done_groups ?? 0} of ${t.groups ?? 0} complete`),
    stat("Notes", `${prog.notes_done ?? 0} of ${prog.notes_total ?? 0} characterized`),
  ].join("");

  // One chip per outcome (extracted / not found / blocked / …), straight from
  // the model's own tally so a new status never needs a code change here.
  const counts = prog.counts || {};
  const chips = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([status, n]) =>
        `<span class="gate-chip ${
          status === "extracted" ? "eligible" : "blocked"
        }">${esc(status.replace(/_/g, " "))} <span class="conf">${n}</span></span>`
    )
    .join("");

  const values = (prog.variables || [])
    .map(
      (v) => `<tr>
        <td class="vt-name">${esc(v.name)}</td>
        <td class="vt-value">${v.value == null || v.value === "" ? "—" : esc(fmt(v.value))}</td>
        <td class="vt-status st-${esc(v.status)}">${esc(v.status || v.stage)}</td>
      </tr>`
    )
    .join("");

  const flagBlock = flags.length
    ? `<div class="headline-fact"><b>Review flags<span class="conf"> ${flags.length}</span></b>
        <ul>${flags.map((f) => `<li>${esc(fmt(f))}</li>`).join("")}</ul></div>`
    : "";
  const fatal = prog.fatal
    ? `<div class="val-errors">⚠ ${esc(fmt(prog.fatal))}</div>`
    : "";

  return `<div class="headline-fact">
    <b>Case finalized</b>
    <div class="stat-grid">${rows}</div>
    ${chips ? `<div class="chips">${chips}</div>` : ""}
    ${prog.review_flags ? `<div class="val-errors">⚑ ${prog.review_flags} variable(s) flagged for review</div>` : ""}
    ${fatal}
  </div>
  ${values ? `<div class="headline-fact"><b>Final values</b><table class="vartable">${values}</table></div>` : ""}
  ${facts}
  ${flagBlock}`;
}

// --- plan / gate: which variable groups are eligible --------------------
function viewGate(prog) {
  const groups = prog.groups || [];
  if (!groups.length) return "";
  const chip = (g) => {
    const blocked = g.stage === "blocked" || g.stage === "skipped";
    return `<span class="gate-chip ${blocked ? "blocked" : "eligible"}">${blocked ? "✗" : "✓"} ${esc(g.name || g.group_id)}<span class="conf"> ${esc(g.stage)}</span></span>`;
  };
  return `<div class="headline-fact">
    <b>Extraction plan</b>
    <div class="muted" style="font-size:.74rem">${groups.length} group(s) — eligible groups run; gated/blocked groups are skipped.</div>
    <div class="chips">${groups.map(chip).join("")}</div>
  </div>`;
}

// --- retriever: kept vs dropped candidate notes -------------------------
function viewRetriever(r, snap) {
  const kept = new Set((r.relevant_note_ids || []).map(String));
  // Candidate pool: the hard-filter output for this group if we have it, else
  // every scanned note. Dropped = candidates the retriever did not keep.
  let candidates = null;
  const hf = snap && snap.details && snap.details.hard_filter_notes;
  if (hf && hf.result && Array.isArray(hf.result.retrieved_note_ids)) {
    candidates = hf.result.retrieved_note_ids.map(String);
  } else if (Object.keys(notesById).length) {
    candidates = Object.keys(notesById);
  }
  const keptChips = [...kept].map((i) => `<span class="note-chip kept">✓ #${esc(i)}</span>`).join(" ");
  let droppedChips = "";
  if (candidates) {
    const dropped = candidates.filter((c) => !kept.has(c));
    droppedChips = dropped.map((i) => `<span class="note-chip dropped">✗ #${esc(i)}</span>`).join(" ");
  }
  return `<div class="headline-fact">
    <b>Relevant notes selected</b>
    <div class="chips">
      ${kept.size ? keptChips : `<span class="dropped">none — extraction skipped for this group</span>`}
      ${droppedChips}
    </div>
    ${candidates ? `<div class="muted" style="font-size:.72rem">${kept.size} kept${candidates.length ? ` of ${candidates.length} candidate(s)` : ""}</div>` : ""}
  </div>`;
}

// --- extractor: coded values, repair loop, inline evidence --------------
// Distinct label per extractor node so a step that touches several (group
// extraction → per-variable completion → merge) reads as a sequence, not a
// repeated "Extracted values" block.
function extractionLabel(node) {
  return (
    {
      extract_group_values: "Group extraction",
      fan_out_variables: "Group extraction",
      complete_variable: "Variable completed",
      merge_variable_results: "Merged group results",
    }[node] || "Extracted values"
  );
}

// One coded value: verdict, value, confidence, repair-loop badge, explanation,
// validation errors, and its evidence spans highlighted inline in the note.
// ``label`` names the variable; pass null inside a per-variable card, whose
// header already says which variable this is.
function extractionRow(v, label) {
  const invalid = v.is_valid === false;
  const ok = invalid ? "✗" : "✓";
  const conf = v.presence_confidence
    ? `<span class="conf">(${esc(v.presence_confidence)})</span>`
    : "";
  const attempts = v.extraction_attempts;
  const repair =
    attempts > 1 || invalid
      ? `<span class="repair-badge ${invalid ? "failed" : "repaired"}">${invalid ? "repair exhausted" : "repaired"} · ${esc(attempts != null ? attempts : "?")} attempt${attempts === 1 ? "" : "s"}</span>`
      : "";
  const val = v.value === null || v.value === undefined ? "—" : esc(fmt(v.value));
  const errs = Array.isArray(v.validation_errors) && v.validation_errors.length
    ? `<div class="val-errors">${v.validation_errors.map((e) => `⚠ ${esc(e)}`).join("<br>")}</div>`
    : "";
  const expl = v.explanation ? `<div class="expl">${esc(v.explanation)}</div>` : "";
  return `<div class="extraction ${invalid ? "invalid" : ""}">
    <div class="extraction-head">
      <span class="ext-ok">${ok}</span>
      ${label === null ? "" : `<b>${esc(label === undefined ? String(v.item_id) : label)}</b>`}
      <span class="ext-val">= <span class="vt-value">${val}</span></span>
      ${conf}
      ${repair}
    </div>
    ${expl}
    ${errs}
    ${renderEvidence(v.spans)}
  </div>`;
}

function viewExtractions(vars, label) {
  const rows = vars.map((v) => extractionRow(v)).join("");
  return `<div class="headline-fact"><b>${esc(label || "Extracted values")}</b>${rows}</div>`;
}

// One extraction under validation/repair (the extractor's inner loop).
function viewValidateRepair(task) {
  const c = task.candidate || {};
  const invalid = task.is_valid === false;
  const merged = { ...c, is_valid: task.is_valid, validation_errors: task.validation_errors, extraction_attempts: task.extraction_attempts };
  const label = invalid ? "Repairing invalid extraction" : "Validating extraction";
  const name = task.variable && task.variable.name ? ` — ${esc(task.variable.name)}` : "";
  return `<div class="headline-fact"><b>${esc(label + name)}</b>${extractionRow(merged)}</div>`;
}

// --- evidence spans, highlighted inline in the source note text ---------
function renderEvidence(spans) {
  if (!Array.isArray(spans) || !spans.length) return "";
  const byNote = {};
  const quotes = [];
  for (const s of spans) {
    if (!s || s.text == null) continue;
    quotes.push(`<span class="evidence-quote">“${esc(trunc(s.text, 130))}”</span>`);
    const nid = String(s.note_id);
    (byNote[nid] = byNote[nid] || []).push(s.text);
  }
  if (!quotes.length) return "";
  let needFetch = false;
  const noteBlocks = Object.entries(byNote)
    .map(([nid, texts]) => {
      const note = notesById[nid];
      if (!note || !note.content) {
        needFetch = true;
        return "";
      }
      return `<details class="block evidence-note">
        <summary>📄 In note #${esc(nid)} <span class="muted">${esc(note.note_type || "")}</span></summary>
        <div class="block-body"><div class="note-text">${highlightContent(note.content, texts)}</div></div>
      </details>`;
    })
    .join("");
  if (needFetch) ensureNotes();
  return `<div class="evidence">
    <div class="evidence-label">Evidence</div>
    <div class="evidence-quotes">${quotes.join(" ")}</div>
    ${noteBlocks}
  </div>`;
}

// Escape note text and wrap every verbatim span occurrence in <mark>. Ranges are
// found on the raw text, merged to handle overlap, then spliced with escaping so
// the marks land on exactly the evidence and nothing is double-escaped.
function highlightContent(content, texts) {
  content = String(content);
  const ranges = [];
  for (const t of texts) {
    if (!t) continue;
    let from = 0;
    let idx;
    while ((idx = content.indexOf(t, from)) !== -1) {
      ranges.push([idx, idx + t.length]);
      from = idx + t.length;
    }
  }
  if (!ranges.length) return esc(content);
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([r[0], r[1]]);
  }
  let out = "";
  let pos = 0;
  for (const [s, e] of merged) {
    out += esc(content.slice(pos, s));
    out += `<mark>${esc(content.slice(s, e))}</mark>`;
    pos = e;
  }
  return out + esc(content.slice(pos));
}

function stat(label, value) {
  return `<div class="stat"><span class="stat-k">${esc(label)}</span><span class="stat-v">${esc(value)}</span></div>`;
}

// --- model calls ---------------------------------------------------------

// A chat-bubble-with-spark mark, inline so the page stays self-contained and the
// icon inherits the surrounding text color instead of a fixed emoji palette.
const CALL_ICON = `<svg class="call-icon" viewBox="0 0 16 16" aria-hidden="true">
  <path d="M2.4 2.2h11.2v8.1H7.2L3.9 13v-2.7H2.4z" fill="none" stroke="currentColor"
        stroke-width="1.3" stroke-linejoin="round"/>
  <path d="M8 4.1l.85 1.9 1.9.85-1.9.85L8 9.6l-.85-1.9-1.9-.85 1.9-.85z" fill="currentColor"/>
</svg>`;

// Roles worth their own color. Anything unrecognized falls back to a neutral
// bubble rather than being forced into one of these.
const ROLE_CLASSES = new Set([
  "system", "human", "user", "ai", "assistant", "tool", "reasoning", "response", "error",
]);

function roleClass(role) {
  const key = String(role || "").toLowerCase();
  return ROLE_CLASSES.has(key) ? key : "other";
}

function msgBubble(role, value) {
  return `<div class="msg role-${roleClass(role)}">
    <div class="role">${esc(role)}</div>
    ${codeBlock(value)}
  </div>`;
}

function renderLLMCall(call) {
  const usage = call.usage || {};
  const tok =
    usage.total_tokens != null
      ? `${usage.total_tokens} tok (${usage.input_tokens ?? "?"} in / ${usage.output_tokens ?? "?"} out)`
      : "";
  const messages = (call.prompt_messages || [])
    .map((m) => msgBubble(m.role || "message", m.content))
    .join("");
  const reasoning = call.reasoning ? msgBubble("reasoning", call.reasoning) : "";
  const response = msgBubble("response", call.response);
  const err = call.error ? msgBubble("error", call.error) : "";

  return `<details class="block llm-call">
    <summary><span class="llm-head">
      ${CALL_ICON} Model call <span class="muted">${esc(call.model || "")}</span>
      <span class="tok">${tok}</span></span></summary>
    <div class="block-body">${messages}${reasoning}${response}${err}</div>
  </details>`;
}

function collapsible(label, value, tag) {
  if (value === null || value === undefined) return "";
  const empty = typeof value === "object" && Object.keys(value).length === 0;
  return `<details class="block">
    <summary>${esc(label)}${empty ? `<span class="summary-tag">empty</span>` : ""}</summary>
    <div class="block-body">${codeBlock(value)}</div>
  </details>`;
}

// --- JSON rendering ------------------------------------------------------
// Prompts and structured responses are mostly JSON. Rendered with
// JSON.stringify's fixed indent, one short object per line costs four lines and
// the bubbles grow without bound; rendered flat, nothing is readable. So:
// pretty-print only the containers that are actually too wide, then colorize.

function codeBlock(value) {
  if (value && typeof value === "object") {
    return `<pre class="code plain json">${highlightJSON(compactJSON(value, 0, 0))}</pre>`;
  }
  if (typeof value === "string") {
    return `<pre class="code plain json">${renderMixed(value)}</pre>`;
  }
  return `<pre class="code plain">${esc(fmt(value))}</pre>`;
}

// Prompts are rarely pure JSON: they are a line of prose ("Clinical notes:")
// followed by one or more JSON documents. Walk the text, pretty-printing each
// embedded document in place and leaving the prose around it alone.
function renderMixed(text) {
  let out = "";
  let i = 0;
  while (i < text.length) {
    const start = jsonStart(text, i);
    if (start < 0) return out + esc(text.slice(i));
    const end = matchBalanced(text, start);
    let parsed;
    if (end > start) {
      try {
        parsed = JSON.parse(text.slice(start, end));
      } catch {
        /* a brace that only looked like a document */
      }
    }
    if (parsed === undefined) {
      out += esc(text.slice(i, start + 1));
      i = start + 1;
      continue;
    }
    out += esc(text.slice(i, start)) + highlightJSON(compactJSON(parsed, 0, 0));
    i = end;
  }
  return out;
}

// First `{`/`[` at or after ``from`` that opens a line — a brace mid-sentence is
// prose (or a format placeholder), not the start of a document.
function jsonStart(text, from) {
  for (let i = from; i < text.length; i++) {
    const c = text[i];
    if (c !== "{" && c !== "[") continue;
    let j = i - 1;
    while (j >= 0 && (text[j] === " " || text[j] === "\t")) j--;
    if (j < 0 || text[j] === "\n") return i;
  }
  return -1;
}

// Index just past the bracket matching the one at ``start``, or -1. Skips over
// string literals so a brace inside note text cannot unbalance the scan.
function matchBalanced(text, start) {
  const open = text[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') inString = true;
    else if (c === open) depth++;
    else if (c === close && --depth === 0) return i + 1;
  }
  return -1;
}

// Serialize with indentation only where a container does not fit on one line,
// so leaves like {"note_id": 50, "text": "…"} stay single lines. The width is
// tuned to the detail panel so lines rarely wrap.
//
// ``indent`` and ``column`` are deliberately separate. ``indent`` is the nesting
// depth and only ever grows by 2 — it is what the padding is built from.
// ``column`` is where this value actually starts on its line (further right,
// because a key like `"presence_confidence": ` precedes it) and is used *only*
// to decide whether the flat form still fits. Feeding the column back in as the
// indent is what makes every key name push its children further right, so a few
// levels of nesting march off the side of the panel.
const JSON_WIDTH = 58;

// The one-line form, spaced like the expanded form so a collapsed leaf sitting
// next to an expanded sibling doesn't read as a different notation.
function flatJSON(value) {
  if (value === undefined) return "null";
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(flatJSON).join(", ")}]`;
  const entries = Object.entries(value).map(
    ([key, item]) => `${JSON.stringify(key)}: ${flatJSON(item)}`
  );
  return `{${entries.join(", ")}}`;
}

function compactJSON(value, indent, column) {
  const flat = flatJSON(value);
  if (value === null || typeof value !== "object" || flat.length + column <= JSON_WIDTH) {
    return flat;
  }
  const pad = " ".repeat(indent + 2);
  const close = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const items = value.map((item) => pad + compactJSON(item, indent + 2, indent + 2));
    return `[\n${items.join(",\n")}\n${close}]`;
  }
  const entries = Object.entries(value);
  if (!entries.length) return "{}";
  const rows = entries.map(([key, item]) => {
    const label = JSON.stringify(key);
    return `${pad}${label}: ${compactJSON(item, indent + 2, indent + 2 + label.length + 2)}`;
  });
  return `{\n${rows.join(",\n")}\n${close}}`;
}

// Colorize keys, strings, numbers and literals. Tokenizes the *raw* text and
// escapes each piece as it is emitted, so nothing is double-escaped and no
// markup can be smuggled in through a string value.
function highlightJSON(text) {
  const token = /("(?:\\.|[^"\\])*")(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
  let out = "";
  let last = 0;
  let match;
  while ((match = token.exec(text)) !== null) {
    out += esc(text.slice(last, match.index));
    const [whole, string, colon] = match;
    if (string && colon) out += `<span class="j-key">${esc(string)}</span>${esc(colon)}`;
    else if (string) out += `<span class="j-str">${esc(whole)}</span>`;
    else if (/^(?:true|false|null)$/.test(whole)) out += `<span class="j-lit">${esc(whole)}</span>`;
    else out += `<span class="j-num">${esc(whole)}</span>`;
    last = token.lastIndex;
  }
  return out + esc(text.slice(last));
}

function blockLabel(block) {
  const n = cy && cy.getElementById(block);
  return n && n.nonempty() ? n.data("baseLabel") : block;
}
function blockAgent(block) {
  const n = cy && cy.getElementById(block);
  return n && n.nonempty() ? n.data("agent") : null;
}

// --- Panel 3: variable overview -----------------------------------------
function renderVars(snapshot) {
  const prog = snapshot.progress;
  if (!prog) {
    els.varsSummary.textContent = "";
    els.vars.innerHTML = `<p class="empty">The extraction plan has not been produced yet.</p>`;
    return;
  }
  const t = prog.totals;
  els.varsSummary.innerHTML =
    `<span>Variables <b>${t.terminal}/${t.variables}</b></span>` +
    `<span>Groups <b>${t.done_groups}/${t.groups}</b></span>` +
    `<span>Notes <b>${prog.notes_done}/${prog.notes_total}</b></span>` +
    (prog.review_flags ? `<span class="flag">⚑ ${prog.review_flags} flag(s)</span>` : "");

  const byGroup = {};
  for (const v of prog.variables) (byGroup[v.group_id] = byGroup[v.group_id] || []).push(v);

  const groups = prog.groups.length
    ? prog.groups
    : Object.keys(byGroup).map((id) => ({ group_id: id, name: id, stage: "pending" }));

  const html = groups
    .map((g) => {
      const vars = byGroup[g.group_id] || [];
      const rows = vars
        .map(
          (v) => `<tr>
            <td class="vt-name">${esc(v.name)}${v.flag ? ` <span class="flag" title="${esc(v.flag)}">⚑</span>` : ""}</td>
            <td class="vt-value">${v.value == null || v.value === "" ? "—" : esc(fmt(v.value))}
              ${v.confidence ? `<span class="conf"> · ${esc(v.confidence)}</span>` : ""}</td>
            <td class="vt-status st-${esc(v.status)}">${esc(v.status || v.stage)}</td>
          </tr>`
        )
        .join("");
      return `<div class="vargroup">
        <div class="vargroup-head">
          <span>${esc(g.name || g.group_id)}</span>
          <span class="stage-badge stage-${esc(g.stage)}">${esc(g.stage)}</span>
          <span class="gcount">${vars.length} var${vars.length === 1 ? "" : "s"}</span>
        </div>
        ${rows ? `<table class="vartable">${rows}</table>` : `<p class="muted" style="padding:.3rem .5rem;font-size:.78rem">No variables yet.</p>`}
      </div>`;
    })
    .join("");
  els.vars.innerHTML = html || `<p class="empty">No variable groups planned.</p>`;
}

// --- controls ------------------------------------------------------------
function wireControls() {
  els.prev.addEventListener("click", () => post("/api/prev"));
  els.next.addEventListener("click", () => post("/api/next"));
  els.play.addEventListener("click", togglePlay);
  els.stepSelect.addEventListener("change", (e) => post(`/api/goto/${e.target.value}`));

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { e.preventDefault(); post("/api/next"); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); post("/api/prev"); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  });
}

// Drag the boundary between the workflow map and the variables panel. The size
// is written as a pixel `--vars-h` on the grid; the map's ResizeObserver re-fits
// Cytoscape as it changes, so the flowchart redraws live during the drag.
function wireSplitter() {
  const handle = document.getElementById("row-split");
  const grid = document.querySelector(".grid");
  if (!handle || !grid) return;

  const setHeight = (px) => {
    const max = grid.clientHeight - MAP_MIN;
    const clamped = Math.max(VARS_MIN, Math.min(px, Math.max(VARS_MIN, max)));
    grid.style.setProperty("--vars-h", `${Math.round(clamped)}px`);
    return clamped;
  };

  const saved = Number(localStorage.getItem(VARS_H_KEY));
  if (saved > 0) setHeight(saved);

  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add("row-resizing");

    const onMove = (ev) => {
      // The variables panel runs from the pointer to the bottom of the grid.
      setHeight(grid.getBoundingClientRect().bottom - ev.clientY);
    };
    const onUp = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("row-resizing");
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
      const current = grid.style.getPropertyValue("--vars-h");
      if (current) localStorage.setItem(VARS_H_KEY, String(parseFloat(current)));
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });

  handle.addEventListener("dblclick", () => {
    grid.style.removeProperty("--vars-h");
    localStorage.removeItem(VARS_H_KEY);
  });
}

function buildStepSelect() {
  els.stepSelect.innerHTML = steps
    .map(
      (s) =>
        `<option value="${s.index}">${s.index + 1}. ${esc(s.title)}${s.subtitle ? " — " + esc(s.subtitle) : ""}</option>`
    )
    .join("");
}

function togglePlay() {
  if (!lastView) return;
  if (lastView.playing) {
    stopAutoplay();
    post("/api/pause");
  } else {
    post("/api/play").then(startAutoplay);
  }
}

function startAutoplay() {
  stopAutoplay();
  autoTimer = setInterval(() => {
    if (lastView && lastView.at_end) { stopAutoplay(); return; }
    post("/api/next");
  }, AUTOPLAY_MS);
}
function stopAutoplay() {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
}

function updateControls(view) {
  els.prev.disabled = view.cursor <= 0;
  els.next.disabled = !!view.at_end;
  els.play.disabled = !!view.at_end && !view.playing;
  els.play.textContent = view.playing ? "⏸ Pause" : "▶ Play";
  els.play.classList.toggle("ctl-primary", true);
  if (els.stepSelect.value !== String(view.cursor)) els.stepSelect.value = String(view.cursor);
  const total = numSteps || steps.length;
  els.counter.textContent = total ? `Step ${view.cursor + 1} / ${total}` : "–";
  // A remote pause (or reaching the end) must stop our local timer.
  if (!view.playing) stopAutoplay();
}

// --- SSE -----------------------------------------------------------------
function openStream() {
  const es = new EventSource("/api/stream");
  es.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === "cursor") applyView(msg);
    else if (msg.type === "live") applyLive(msg);
  };
  es.onerror = () => { /* EventSource auto-reconnects */ };
}

// Live mode: the graph produced an event. Refresh the step list on growth so
// controls (Next / at_end) reflect it, then re-fetch the authoritative cursor
// view. Only the in-progress *frontier* step's snapshot changes as events
// stream in, so panels are re-rendered just for a presenter parked on that
// frontier — letting per-note results fill in live — while an earlier step the
// presenter has scrubbed back to stays frozen (its expanded payloads and scroll
// position aren't disturbed). Growth also self-heals auto-play stuck at the end.
async function applyLive(msg) {
  if (msg.num_steps && msg.num_steps !== numSteps) {
    numSteps = msg.num_steps;
    try {
      steps = await getJSON("/api/steps");
      events = await getJSON("/api/events");
      buildStepSelect();
    } catch { /* keep the stale list; the next tick refreshes */ }
  }
  els.sub.textContent =
    `${msg.num_steps} steps · ${msg.num_events} events${msg.done ? " · complete" : " · running"}`;

  let view;
  try { view = await getJSON("/api/cursor"); } catch { return; /* transient */ }

  if (view.cursor >= numSteps - 1) {
    // On the live frontier: its snapshot is still growing — redraw everything.
    applyView(view);
  } else {
    // Behind the frontier on a frozen step: refresh controls only.
    updateControls(view);
    if (view.playing && !view.at_end && !autoTimer) startAutoplay();
  }
}

function applyView(view) {
  const stepChanged = !lastView || lastView.cursor !== view.cursor;
  lastView = view;
  if (stepChanged) focusBlock = null; // new step clears any pinned component
  updateControls(view);
  updateMap(view.snapshot, view.step);
  renderDetail(view);
  renderVars(view.snapshot);
  // Self-heal auto-play: if still playing and more steps are now available
  // (live growth), make sure the timer is running.
  if (view.playing && !view.at_end && !autoTimer) startAutoplay();
}

// --- helpers -------------------------------------------------------------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}
function post(url) {
  return fetch(url, { method: "POST" }).catch(() => {});
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}
function trunc(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
