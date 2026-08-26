/*
 * CIPOC extraction demo — frontend (Phase 3).
 *
 * Renders three panels from the demo server's SSE cursor stream:
 *   1. Workflow map   — this run's fan-out/fan-in graph: one node per note, per
 *                       variable group's gate, and per variable, animated
 *                       through each step's own span of the trace.
 *   2. Current step   — what the components decided during the presenter's
 *                       current step, per note / per group / per variable.
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
  els.mapReplay = document.getElementById("btn-replay-step");
  els.mapScrub = document.getElementById("map-scrub");
  els.mapTip = document.getElementById("map-tip");
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

/* --- Panel 1: workflow map ----------------------------------------------
 *
 * The map is derived from *this run*, not from the static overview chart: the
 * things that fan out get one node each, so the drawing is a picture of what
 * actually happened (see `shitty_cipoc_drawing.png`).
 *
 *                     [ Initialize case ]
 *                      ╱      │      ╲            one edge per note
 *                   (n50)   (n51)   (n52) …
 *                      ╲      │      ╱
 *                   [ Characterize corpus ]
 *                             │
 *      ┌───────────► [ Plan extraction ] ──────► [ Finalize case ]
 *      │              ╱      │      ╲            one edge per group
 *      │           (gate)  (gate)  (gate) …      ✓ open / ✗ shut / … pending
 *      │           ╱ │ ╲   ╱ │ ╲   ╱ │ ╲         one edge per variable
 *      │         (v)(v)(v)(v)(v)(v)(v)(v) …
 *      │           ╲ │ ╱   ╲ │ ╱   ╲ │ ╱
 *      └─────────── [ Update case ]
 *
 * The per-group retriever and extractor are deliberately *not* nodes — eight
 * groups would add sixteen boxes carrying no information the gate disc cannot.
 * The group's pipeline stage rides on its gate instead (retriever-orange while
 * selecting notes, extractor-blue while extracting, then settled).
 */

// The orchestrator root nodes that are phases of the extraction loop. Each one
// keeps its own busy window, so the case can name the phase that is running.
const CASE_REST = "";
const CASE_PHASE_NODES = { check_state: "Check state", plan_extraction: "Plan extraction", merge_and_update: "Update case" };

// The orchestrator stages that stay plain slabs, with the coarse overview block
// each one stands for (see mapping.py::overview_block_map).
const STAGES = [
  { id: "stage:initialize", label: "Initialize case", block: "initialize_case" },
  { id: "stage:corpus", label: "Characterize corpus", block: "characterize_corpus" },
  { id: "stage:finalize", label: "Finalize case", block: "finalize_case" },
];

// The middle of the drawing is the case itself, and the extraction loop is two
// poles inside it: work is dispatched from Plan and comes back to Update. That
// is a structural separation of the two directions, so the fan-out and the
// fan-in never share a corridor and no edge has to loop back around the band.
const CASE_ID = "stage:case";
const POLES = [
  { id: "pole:plan", label: "Plan", block: "eligible_groups_gate" },
  { id: "pole:update", label: "Update", block: "update_case" },
];

// Coarse overview block -> what now represents it on the drawing. The Python
// side (mapping.py) is untouched and still owns runtime-node -> block; this is
// only the last hop, block -> drawn element, and it is what keeps click-to-pin
// and the "current stage" highlight working against the new topology.
const BLOCK_TO_MAP = {
  initialize_case: "stage:initialize",
  characterize_corpus: "stage:corpus",
  eligible_groups_gate: "pole:plan",
  update_case: "pole:update",
  finalize_case: "stage:finalize",
  scanner_agent_block: "band:notes",
  retriever_agent_block: "band:groups",
  extractor_agent_block: "band:groups",
  relevant_notes_gate: "band:groups",
};

/* Layout geometry, in Cytoscape model units.
 *
 * Model units, not pixels: everything here — including font sizes — is
 * multiplied by cy.zoom(), and zoom is whatever cy.fit() settles on. So the size
 * text ends up on screen is `font-size × zoom`, and growing a node trades
 * directly against zoom. That is why the type here is large and the padding
 * mean: computeLayout picks the packing that maximises zoom, and the type scale
 * spends the winnings.
 */
const GEO = {
  slabW: 190, slabH: 48,
  poleW: 112, poleH: 38, poleGap: 26, casePad: 50,
  noteD: 38, noteGapX: 66, noteGapY: 48,
  // The notes sit *below* the strip's slabs, never on their baseline: level with
  // them every fan edge is collinear and seven parallel notes read as a chain.
  noteDrop: 56,
  gateD: 50, gateGap: 16, varD: 20, varGapX: 30, varGapY: 28,
  clusterGap: 40,
  rowGap: 78,
  stripGap: 58,           // between the strip's slabs and the notes between them
};

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

/* "Light clinical" stylesheet.
 *
 * Stage slabs are elevated white cards with a colored rule along the top edge;
 * instance nodes are white discs with a thick agent-colored ring that fills in
 * as they complete; active edges carry a travelling dash.
 *
 * Two Cytoscape-specific notes: it has no per-side borders, so the slab's top
 * rule is the first stop of a vertical gradient; and it dropped `shadow-*` in
 * v3, so the card elevation is a low-opacity `underlay-*` halo instead.
 */
const MAP_COLORS = { ok: "#1a7f52", warn: "#c98a12", err: "#c0392b", line: "#c9d2e3", muted: "#9aa3b2" };

function mapStyle(agentColors) {
  const { ok, warn, err, line, muted } = MAP_COLORS;
  const agent = (name, fallback) => agentColors[name] || fallback;
  const ORCH = agent("orchestrator", "#6d5bd0");
  const SCAN = agent("scanner", "#008c7a");
  const EXTR = agent("extractor", "#1473e6");
  const RETR = agent("retriever", "#d16b22");

  // A white card with `color` as a rule across its top edge.
  const ruled = (color) => ({
    "background-fill": "linear-gradient",
    "background-gradient-direction": "to-bottom",
    "background-gradient-stop-colors": `${color} ${color} #ffffff #ffffff`,
    "background-gradient-stop-positions": "0% 8% 8% 100%",
  });

  return [
    // --- stage slabs ---
    {
      selector: "node.slab",
      style: {
        shape: "round-rectangle",
        width: GEO.slabW, height: GEO.slabH,
        ...ruled(ORCH),
        "border-width": 1, "border-color": line,
        // Stands in for a drop shadow (see the note above).
        "underlay-color": "#172033", "underlay-opacity": 0.1, "underlay-padding": 2,
        label: "data(label)", "text-wrap": "wrap", "text-max-width": GEO.slabW - 22,
        "text-valign": "center", "text-halign": "center",
        "text-margin-y": 2,
        "font-size": 15, "font-weight": 700, color: "#2b3040",
      },
    },
    { selector: 'node.slab[id="stage:finalize"]', style: ruled(err) },

    // --- the case, and the two poles of the loop inside it ---
    //
    // A compound parent, like the group clusters, so it sizes itself around the
    // poles. Its label is the one text on the drawing that moves: `Case`, plus
    // the phase of the loop that is running.
    {
      selector: "node.case",
      style: {
        shape: "round-rectangle",
        ...ruled(ORCH),
        "border-width": 1, "border-color": "#b9c4d8",
        "underlay-color": "#172033", "underlay-opacity": 0.14, "underlay-padding": 3,
        // The label lives *inside* the top padding band, unlike the group
        // clusters' labels: the spine lands on this node's top edge, and a
        // label sitting above the boundary would be written through it.
        padding: GEO.casePad,
        label: "data(label)", "text-wrap": "wrap", "text-max-width": 240,
        // `text-valign: top` anchors the block's *bottom* to the top edge, so
        // the margin has to clear the whole two-line block plus the top rule.
        "text-valign": "top", "text-halign": "center", "text-margin-y": 44,
        "font-size": 14.5, "font-weight": 700, color: "#2b3040",
      },
    },
    // The poles are slabs, so they inherit every node.slab.st-* state rule.
    {
      selector: "node.slab.pole",
      style: {
        width: GEO.poleW, height: GEO.poleH,
        "background-fill": "solid", "background-color": "#ffffff",
        "font-size": 13.5, "text-margin-y": 0,
      },
    },

    // --- group cluster (compound parent carries the group's name) ---
    {
      selector: "node.cluster",
      style: {
        shape: "round-rectangle",
        "background-color": "#f4f7fc", "background-opacity": 0.9,
        "border-width": 1, "border-color": "#e3e9f4",
        padding: 10,
        label: "data(label)", "text-valign": "top", "text-halign": "center",
        "text-margin-y": -3, "text-wrap": "wrap", "text-max-width": 170,
        "font-size": 12.5, "font-weight": 700, color: "#5b6270",
      },
    },
    { selector: "node.cluster.gate-shut", style: { "background-color": "#faf1f0", "border-color": "#f0d9d6" } },

    // --- instance discs: white with an agent-colored ring ---
    {
      selector: "node.disc",
      style: {
        shape: "ellipse",
        "background-color": "#ffffff",
        "border-width": 3, "border-color": muted, "border-opacity": 1,
        label: "data(label)", "text-valign": "center", "text-halign": "center",
        "font-size": 11, "font-weight": 700, color: "#4a5468",
      },
    },
    { selector: "node.disc.note", style: { width: GEO.noteD, height: GEO.noteD, "border-color": SCAN } },
    { selector: "node.disc.var", style: { width: GEO.varD, height: GEO.varD, "border-color": EXTR, "font-size": 0 } },
    {
      selector: "node.disc.gate",
      style: {
        width: GEO.gateD, height: GEO.gateD,
        "border-color": warn, "font-size": 19, color: warn,
      },
    },

    // --- edges: hairline by default, no arrowheads on the fan-out lines ---
    {
      selector: "edge",
      style: {
        width: 1.4, "line-color": line, "curve-style": "straight",
        "target-arrow-shape": "none", opacity: 0.45,
      },
    },
    // Straight, not taxi: the strip puts Corpus at the far right and the case
    // below centre, so a taxi dogleg wandered down past Finalize on its way in.
    {
      selector: "edge.spine",
      style: {
        width: 2, "curve-style": "straight",
        "target-arrow-shape": "triangle", "target-arrow-color": line, "arrow-scale": 0.9,
      },
    },
    // Dispatch leaves the Plan pole and results arrive at the Update pole, so
    // the two directions are separated by the drawing itself and neither edge
    // needs an endpoint offset to stay out of the other's corridor.
    //
    // The group's return arc still sags outward — unbundled-bezier bows
    // perpendicular to the source->target line, which for a fan converging
    // upward means "away from the case" — so a second-row cluster sweeps around
    // the first row rather than through it.
    {
      selector: "edge.grp-out",
      style: {
        "curve-style": "unbundled-bezier",
        "control-point-distances": 48, "control-point-weights": 0.5,
        "target-arrow-shape": "triangle", "target-arrow-color": EXTR, "arrow-scale": 0.8,
        width: 1.8,
      },
    },
    // Out and back between a gate and its variable are two edges on one pair, so
    // they bow apart into a lobe instead of lying on top of each other.
    {
      selector: "edge.var-in",
      style: { "curve-style": "unbundled-bezier", "control-point-distances": -7, "control-point-weights": 0.5 },
    },
    {
      selector: "edge.var-out",
      style: {
        "curve-style": "unbundled-bezier",
        "control-point-distances": -7, "control-point-weights": 0.5,
        "target-arrow-shape": "triangle", "target-arrow-color": EXTR, "arrow-scale": 0.5,
      },
    },
    { selector: "edge.to-scanner", style: { "line-color": SCAN } },
    { selector: "edge.to-retriever", style: { "line-color": RETR } },
    { selector: "edge.to-extractor", style: { "line-color": EXTR } },
    { selector: "edge.exit", style: { "line-color": err, "target-arrow-color": err } },
    { selector: "node.disc.gate.pipe-retrieve", style: { "border-color": RETR, color: RETR } },
    { selector: "node.disc.gate.pipe-extract", style: { "border-color": EXTR, color: EXTR } },
    { selector: "node.disc.gate.gate-open", style: { "border-color": ok, color: ok, "background-color": tint(ok, 0.9) } },
    { selector: "node.disc.gate.gate-shut", style: { "border-color": err, color: err, "background-color": tint(err, 0.9) } },
    { selector: "node.disc.gate.gate-skipped", style: { "border-color": muted, color: muted } },
  ];
}

/* --- the run's shape, and when each part of it ran ------------------------
 *
 * `mapIndex` is built once per event-list load and answers "what was this node
 * doing at trace-time t". It is what lets the map animate *inside* a step
 * without any server round-trip: `/api/events` is already fetched in full, so
 * every fan-out instance's start/end is known client-side.
 *
 * `windows` maps a node key to the intervals it was busy. Slabs can run more
 * than once (the planner runs once per pass), hence a list rather than a pair.
 */
let mapIndex = {
  windows: new Map(), notes: [], groups: new Map(), groupByTask: new Map(),
  passStarts: [], verdictT: Infinity, tMax: 0,
};

function addWindow(windows, key, t0, t1) {
  if (!windows.has(key)) windows.set(key, []);
  windows.get(key).push({ t0, t1 });
}

function buildMapIndex() {
  const windows = new Map();
  const notes = [];
  const groups = new Map();          // group_id -> {id, name, variables:[…]}
  const openByTask = new Map();      // task_id -> {key, t0}
  const groupByTask = new Map();     // extract_branch task_id -> group_id
  const passStarts = [];             // each plan_extraction start: a loop turn
  let verdictT = Infinity;
  let tMax = 0;

  const open = (taskId, key, t) => openByTask.set(taskId, { key, t0: t });
  const close = (taskId, t) => {
    const entry = openByTask.get(taskId);
    if (!entry) return;
    openByTask.delete(taskId);
    addWindow(windows, entry.key, entry.t0, t);
  };

  for (const ev of events) {
    tMax = Math.max(tMax, ev.t || 0);
    const payload = ev.payload && typeof ev.payload === "object" ? ev.payload : {};

    // The corpus descriptors are what the planner's gate predicates read, so
    // this is the moment every gate verdict becomes knowable.
    if (ev.type === "task_end" && ev.node === "characterize_corpus") {
      verdictT = Math.min(verdictT, ev.t);
    }

    if (ev.type !== "task_start" && ev.type !== "task_end") continue;

    // Stage slabs and the two case poles, via the coarse block the runtime node
    // belongs to. Both kinds are drawn boxes with their own busy state; only the
    // `band:` entries are not.
    const slab = BLOCK_TO_MAP[coarse(ev.map_node_id)];
    if (slab && !slab.startsWith("band:")) {
      if (ev.type === "task_start") open(`slab/${ev.task_id}`, slab, ev.t);
      else close(`slab/${ev.task_id}`, ev.t);
    }

    // The case stands for several root nodes at once, so each keeps its own
    // window — that is what lets the case's label name the running phase.
    if (!ev.namespace.length && CASE_PHASE_NODES[ev.node]) {
      if (ev.type === "task_start") {
        open(`phase/${ev.task_id}`, `phase:${ev.node}`, ev.t);
        if (ev.node === "plan_extraction") passStarts.push(ev.t);
      } else close(`phase/${ev.task_id}`, ev.t);
    }

    if (ev.node === "note_branch" && !ev.namespace.length) {
      const key = `note:${payload.note_id}`;
      if (ev.type === "task_start") {
        notes.push({ id: key, noteId: payload.note_id, type: payload.note_type || "" });
        open(`note/${ev.task_id}`, key, ev.t);
      } else close(`note/${ev.task_id}`, ev.t);
      continue;
    }

    if (ev.node === "extract_branch" && !ev.namespace.length) {
      const requested = payload.requested_variables || {};
      const gid = requested.group_id;
      if (ev.type === "task_start") {
        groupByTask.set(ev.task_id, gid);
        if (gid && !groups.has(gid)) {
          groups.set(gid, {
            id: gid,
            name: requested.name || gid,
            variables: (requested.variables || []).map((v) => ({
              itemId: v.item_id,
              name: v.name || `Item ${v.item_id}`,
            })),
          });
        }
        open(`grp/${ev.task_id}`, `grp:${gid}`, ev.t);
      } else close(`grp/${ev.task_id}`, ev.t);
      continue;
    }

    // Which pipeline stage a group's gate disc should show while it runs.
    if (ev.node === "retrieve_notes" || ev.node === "extract") {
      const gid = groupByTask.get((ev.namespace[0] || "").split(":")[1]);
      if (!gid) continue;
      const key = ev.node === "retrieve_notes" ? `grpret:${gid}` : `grpext:${gid}`;
      if (ev.type === "task_start") open(`pipe/${ev.task_id}`, key, ev.t);
      else close(`pipe/${ev.task_id}`, ev.t);
      continue;
    }

    if (ev.node === "variable_branch") {
      // Only the *start* payload names the variable — the end payload carries
      // the result. Closing has to key off the task id alone, or the window
      // never closes and the variable reads as running for the rest of the run.
      if (ev.type === "task_start") {
        const itemId = ((payload.task || {}).variable || {}).item_id;
        if (itemId != null) open(`var/${ev.task_id}`, `var:${itemId}`, ev.t);
      } else close(`var/${ev.task_id}`, ev.t);
      continue;
    }
  }

  // Anything still open at the end of the stream (a live run, or a crash) stays
  // busy to the end rather than vanishing.
  for (const [, entry] of openByTask) addWindow(windows, entry.key, entry.t0, Infinity);

  mapIndex = { windows, notes, groups, groupByTask, passStarts, verdictT, tMax };
  return mapIndex;
}

// "idle" | "active" | "done" for a node key at trace-time t.
function stateAt(key, t) {
  const windows = mapIndex.windows.get(key);
  if (!windows) return "idle";
  let seen = false;
  for (const w of windows) {
    if (t >= w.t0 && t < w.t1) return "active";
    if (t >= w.t1) seen = true;
  }
  return seen ? "done" : "idle";
}

// The start of the loop turn `t` falls in. Used to scope the return arcs to the
// pass that is actually reporting back, rather than lighting every variable the
// run has ever finished each time the case is updated.
function passStartBefore(t) {
  let start = 0;
  for (const s of mapIndex.passStarts) if (s <= t) start = s;
  return start;
}

function endedSince(key, t, since) {
  return (mapIndex.windows.get(key) || []).some((w) => w.t1 <= t && w.t1 >= since);
}

function anyGroupActive(t) {
  for (const gid of mapIndex.groups.keys()) if (stateAt(`grp:${gid}`, t) === "active") return true;
  return false;
}

// Which phase of the loop is running at `t` — the second line of the case's
// label, and "" at rest. Dispatch is checked before planning because a pass runs
// *inside* the loop turn the planner opened. Extraction is the reason this lives
// on the case rather than on a pole: it belongs to neither.
function casePhase(t) {
  if (stateAt("phase:merge_and_update", t) === "active") return CASE_PHASE_NODES.merge_and_update;
  if (anyGroupActive(t)) return "Extracting…";
  for (const node of ["plan_extraction", "check_state"]) {
    if (stateAt(`phase:${node}`, t) === "active") return CASE_PHASE_NODES[node];
  }
  return CASE_REST;
}

// A planning check has run by `t`, so the planner's verdicts are now things it
// has *decided* rather than things merely computable. Gate lines hang off this.
function planChecked(t) {
  return mapIndex.passStarts.length > 0 && mapIndex.passStarts[0] <= t;
}

/* --- the drawing -------------------------------------------------------- */

// Node/edge elements for this run. Groups and their variables come from the
// snapshot's plan when there is one (it is authoritative and present from
// Initialize onward); the index fills in for a cursor that has not reached the
// plan yet, so the map's shape never changes shape mid-run.
function buildMapModel(snapshot) {
  const nodes = [];
  const edges = [];
  const add = (data, classes) => nodes.push({ data, classes });
  const link = (source, target, classes, data) =>
    edges.push({ data: { id: `${source}->${target}`, source, target, ...(data || {}) }, classes });

  for (const stage of STAGES) add({ id: stage.id, label: stage.label, block: stage.block }, "slab");

  // The case, and the two poles of the extraction loop inside it. The container
  // carries no `block`, so a tap on it falls through and leaves Panel 2's pin
  // alone; the poles are the click targets.
  add({ id: CASE_ID, label: "Case" }, "case");
  for (const pole of POLES) {
    add({ id: pole.id, parent: CASE_ID, label: pole.label, title: pole.label, block: pole.block }, "slab pole");
  }

  for (const note of mapIndex.notes) {
    add({ id: note.id, label: `#${note.noteId}`, title: `${note.type} #${note.noteId}`.trim(), block: "scanner_agent_block" }, "disc note");
    link("stage:initialize", note.id, "fan to-scanner note-in");
    link(note.id, "stage:corpus", "fan to-scanner note-out");
  }

  for (const group of planGroups(snapshot)) {
    const cluster = `grp:${group.id}`;
    const gate = `gate:${group.id}`;
    add({ id: cluster, label: group.name, block: "retriever_agent_block" }, "cluster");
    add({ id: gate, parent: cluster, label: "", title: group.annotation || group.name, group: group.id, block: "retriever_agent_block" }, "disc gate");
    link("pole:plan", gate, "fan to-retriever gate-in", { group: group.id });
    for (const variable of group.variables) {
      const id = `var:${variable.itemId}`;
      add({ id, parent: cluster, label: "", title: variable.name, group: group.id, block: "extractor_agent_block" }, "disc var");
      link(gate, id, "fan to-extractor var-in", { group: group.id });
      // Results come back out of the variable — but to the gate, which then
      // reports the group to the Update pole. Thirty-two arcs converging on one
      // box is the same wall of ink as the old bottom fan-in, just upside down;
      // one arc per group is legible, and it is what actually happens (the
      // group's variables are merged before the case is updated).
      link(id, gate, "fan to-extractor var-out", { group: group.id });
    }
    // The loop closes on the case: dispatched from Plan, returned to Update.
    link(gate, "pole:update", "fan to-extractor grp-out", { group: group.id });
  }

  // Onto the container, not the Plan pole: it stops at the case's top edge and
  // so stays clear of the label sitting in the padding band below it.
  link("stage:corpus", CASE_ID, "spine");
  // Sourced from the container, so it leaves the case's right boundary rather
  // than piercing it from a pole on the far side.
  link(CASE_ID, "stage:finalize", "spine exit");

  return { nodes, edges };
}

// The planned groups, each with its variables — snapshot first, index as backup.
function planGroups(snapshot) {
  const progress = snapshot && snapshot.progress;
  if (progress && progress.groups && progress.groups.length) {
    const byGroup = {};
    for (const v of progress.variables || []) {
      (byGroup[v.group_id] = byGroup[v.group_id] || []).push({ itemId: v.item_id, name: v.name });
    }
    return progress.groups.map((g) => ({
      id: g.group_id,
      name: g.name || g.group_id,
      annotation: g.annotation || "",
      variables: byGroup[g.group_id] || [],
    }));
  }
  return [...mapIndex.groups.values()].map((g) => ({ ...g, annotation: "" }));
}

/* --- layout ---------------------------------------------------------------
 *
 * The drawing's shape has to track the panel's, or cy.fit() throws half the
 * panel away: a 1.4:1 drawing in a 2.3:1 panel scales to the height and leaves
 * the sides empty, which is what made every label too small to read.
 *
 * So there is no fixed arrangement. `packLayout` lays the run out for a given
 * (bandCols, varCols, noteCols), and `computeLayout` tries them all and keeps
 * whichever renders *largest* in the container we actually have. Scoring on the
 * achievable zoom rather than on some target aspect means the thing being
 * optimised is exactly the thing that was wrong.
 */
const FIT_PAD = 12;
const CLUSTER_PAD = 10;   // node.cluster's `padding`, which Cytoscape adds around the children
const BAND_ROW_GAP = 56;  // between band rows: a cluster's label hangs above its box
const FINALIZE_GAP = 70;

// The container, or a sane guess: on first paint the flex panel has no size yet.
function viewportBox() {
  const w = cy ? cy.width() : 0;
  const h = cy ? cy.height() : 0;
  return { w: w > 40 ? w : 1100, h: h > 40 ? h : 500 };
}

// Group the model's variables under their cluster once, so the search below can
// re-pack a few hundred times without re-filtering the node list every pass.
function modelParts(model) {
  const varsBy = new Map();
  for (const n of model.nodes) {
    if (!n.classes.includes("var") || !n.data.parent) continue;
    if (!varsBy.has(n.data.parent)) varsBy.set(n.data.parent, []);
    varsBy.get(n.data.parent).push(n.data.id);
  }
  const clusters = model.nodes
    .filter((n) => n.classes === "cluster")
    .map((n) => ({ id: n.data.id, gate: n.data.id.replace("grp:", "gate:"), vars: varsBy.get(n.data.id) || [] }));
  return { clusters, notes: mapIndex.notes };
}

// One candidate arrangement: a scanner strip, the case row, then the group band.
// Pure — same parts + opts always give the same positions and extent.
function packLayout(parts, opts) {
  const { bandCols, varCols, noteCols } = opts;
  const pos = {};

  // --- measure each cluster: the gate at the left, its variables in a grid to
  // the right of it. Stacking the gate above cost `gateD + 26` of height on
  // every cluster, and height is what binds in a panel this wide — sideways the
  // gate costs nothing at all until a group has fewer than two rows of
  // variables. It also reads like the case's poles: dispatch flows rightward.
  const sized = parts.clusters.map((c) => {
    const cols = Math.min(varCols, Math.max(c.vars.length, 1));
    const rows = Math.ceil(c.vars.length / cols) || 1;
    return {
      ...c, cols, rows,
      w: GEO.gateD + GEO.gateGap + cols * GEO.varGapX + CLUSTER_PAD * 2,
      h: Math.max(GEO.gateD, rows * GEO.varGapY) + CLUSTER_PAD * 2,
    };
  });

  // --- pack the clusters into rows of `bandCols`
  const bandRows = [];
  for (let i = 0; i < sized.length; i += bandCols) {
    const items = sized.slice(i, i + bandCols);
    bandRows.push({
      items,
      w: items.reduce((a, s) => a + s.w, 0) + GEO.clusterGap * (items.length - 1),
      h: Math.max(0, ...items.map((s) => s.h)),
    });
  }
  const bandW = Math.max(0, ...bandRows.map((r) => r.w));
  const bandH = bandRows.reduce((a, r, i) => a + r.h + (i ? BAND_ROW_GAP : 0), 0);

  // --- the strip: Initialize and Corpus at the ends, notes in a grid between
  const notes = parts.notes;
  const perRow = Math.max(1, Math.min(noteCols, notes.length || 1));
  const noteRows = Math.ceil(notes.length / perRow) || 1;
  const notesW = (perRow - 1) * GEO.noteGapX + GEO.noteD;
  const notesH = (noteRows - 1) * GEO.noteGapY + GEO.noteD;
  const stripW = GEO.slabW * 2 + GEO.stripGap * 2 + notesW;
  const slabCy = GEO.slabH / 2;
  const notesCy = slabCy + GEO.noteDrop;
  const stripH = notesCy + notesH / 2;

  // --- the case row, with Finalize hanging off its right
  const caseW = GEO.poleW * 2 + GEO.poleGap + GEO.casePad * 2;
  const caseH = GEO.poleH + GEO.casePad * 2;

  // Finalize makes the case row asymmetric, so the two sides are measured apart.
  const left = Math.max(stripW, bandW, caseW) / 2;
  const right = Math.max(stripW / 2, bandW / 2, caseW / 2 + FINALIZE_GAP + GEO.slabW);
  const cx = left;

  let y = slabCy;
  pos["stage:initialize"] = { x: cx - stripW / 2 + GEO.slabW / 2, y };
  pos["stage:corpus"] = { x: cx + stripW / 2 - GEO.slabW / 2, y };
  const noteY0 = notesCy - notesH / 2 + GEO.noteD / 2;
  notes.forEach((note, i) => {
    const row = Math.floor(i / perRow);
    const inRow = Math.min(perRow, notes.length - row * perRow);
    pos[note.id] = {
      x: cx - ((inRow - 1) * GEO.noteGapX) / 2 + (i % perRow) * GEO.noteGapX,
      y: noteY0 + row * GEO.noteGapY,
    };
  });

  // Only the poles get positions — the case container is a compound parent and
  // sizes itself around them.
  y = stripH + GEO.rowGap + caseH / 2;
  const poleDX = (GEO.poleW + GEO.poleGap) / 2;
  pos["pole:plan"] = { x: cx - poleDX, y };
  pos["pole:update"] = { x: cx + poleDX, y };
  pos["stage:finalize"] = { x: cx + caseW / 2 + FINALIZE_GAP + GEO.slabW / 2, y };

  let by = stripH + GEO.rowGap + caseH + GEO.rowGap;
  for (const band of bandRows) {
    let x = cx - band.w / 2;
    for (const s of band.items) {
      const midY = by + s.h / 2;
      pos[s.gate] = { x: x + CLUSTER_PAD + GEO.gateD / 2, y: midY };
      const varsX = x + CLUSTER_PAD + GEO.gateD + GEO.gateGap + GEO.varGapX / 2;
      const varsY = midY - ((s.rows - 1) * GEO.varGapY) / 2;
      s.vars.forEach((id, i) => {
        pos[id] = {
          x: varsX + (i % s.cols) * GEO.varGapX,
          y: varsY + Math.floor(i / s.cols) * GEO.varGapY,
        };
      });
      x += s.w + GEO.clusterGap;
    }
    by += band.h + BAND_ROW_GAP;
  }

  return {
    pos,
    w: left + right,
    h: stripH + GEO.rowGap + caseH + (bandH ? GEO.rowGap + bandH : 0),
  };
}

// Column counts worth trying for `n` items: every value while that is cheap,
// thinning out beyond. `n` itself is always included — capping below it would
// hide the single-row packing, which is the flattest one there is and the only
// answer when the panel is very wide.
function colCandidates(n) {
  const out = new Set([1, n]);
  for (let i = 2; i <= Math.min(n, 16); i++) out.add(i);
  for (let i = 20; i < n; i += 4) out.add(i);
  return [...out].filter((v) => v >= 1).sort((a, b) => a - b);
}

// Which packing renders largest here. `layoutKey` is what refitMap watches: if
// the panel changes shape enough to change the winner, the band re-packs.
let layoutKey = "";

function computeLayout(model) {
  const parts = modelParts(model);
  const view = viewportBox();
  const maxVars = Math.max(1, ...parts.clusters.map((c) => c.vars.length));
  const bandChoices = colCandidates(Math.max(1, parts.clusters.length));
  const noteChoices = colCandidates(Math.max(1, parts.notes.length));
  let best = null;

  for (const bandCols of bandChoices) {
    for (let varCols = 1; varCols <= Math.min(maxVars, 16); varCols++) {
      for (const noteCols of noteChoices) {
        const packed = packLayout(parts, { bandCols, varCols, noteCols });
        const scale = Math.min(
          (view.w - FIT_PAD * 2) / packed.w,
          (view.h - FIT_PAD * 2) / packed.h,
        );
        const area = packed.w * packed.h;
        const better = !best
          || scale > best.scale + 1e-6
          || (scale > best.scale - 1e-6 && area < best.area);
        if (better) best = { packed, scale, area, key: `${bandCols}/${varCols}/${noteCols}` };
      }
    }
  }

  layoutKey = best.key;
  return best.packed.pos;
}

function buildMap(graph) {
  if (graph.coarse_map && Object.keys(graph.coarse_map).length) {
    COARSE = graph.coarse_map;
    rebuildCoarseMembers();
  }
  const agentColors = graph.agent_colors || {};
  applyAgentColors(agentColors);

  buildMapIndex();
  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: { nodes: [], edges: [] },
    style: [...mapStyle(agentColors), ...stateStyles(agentColors)],
    layout: { name: "preset" },
    wheelSensitivity: 0.2,
    minZoom: 0.2,
    maxZoom: 2.5,
  });

  // The map lives in a flex panel that may not have its final size when
  // Cytoscape initializes, and the presenter can drag the splitter at any time.
  // Since the packing is *chosen against* the container, a resize may want a
  // different one — so re-pack, not just re-fit.
  const container = document.getElementById("cy");
  if (window.ResizeObserver) new ResizeObserver(refitMap).observe(container);

  // Click a slab or a disc to pin Panel 2 to the component behind it; click
  // empty space to unpin. Discs resolve through their own `block`, so clicking
  // a note pins the scanner and clicking a variable pins the extractor.
  cy.on("tap", "node", (evt) => {
    const block = evt.target.data("block");
    if (!block) return;
    focusBlock = block;
    if (lastView) renderDetail(lastView);
  });
  cy.on("tap", (evt) => {
    if (evt.target === cy) {
      focusBlock = null;
      if (lastView) renderDetail(lastView);
    }
  });
  wireMapTooltip();
}

function fitMap() {
  if (!cy) return;
  cy.resize();
  cy.fit(undefined, FIT_PAD);
}

// The panel changed shape. Re-run the search; if it picks a different packing,
// move the nodes and repaint the frame we were on — replaying `lastRenderT`
// rather than the step end, or a resize mid-animation would jump the map to
// some other point in the run.
let refitTimer = null;

function refitMap() {
  if (refitTimer !== null) clearTimeout(refitTimer);
  refitTimer = setTimeout(() => {
    refitTimer = null;
    if (!cy) return;
    if (lastMapModel) {
      const before = layoutKey;
      const pos = computeLayout(lastMapModel);
      if (layoutKey !== before) {
        cy.batch(() => {
          for (const id of Object.keys(pos)) {
            const node = cy.getElementById(id);
            if (node.nonempty()) node.position({ ...pos[id] });
          }
        });
        if (lastView) renderMapAt(lastRenderT, lastView.snapshot, lastView.step);
      }
    }
    fitMap();
  }, 120);
}

// Rebuild the drawing when the run's shape changes (first snapshot, or a live
// run growing). `shapeKey` keeps a redraw from firing on every cursor move,
// which would reset pan/zoom and kill the animation.
let mapShapeKey = "";
// Kept so a resize can re-pack without rebuilding the elements.
let lastMapModel = null;

function syncMap(snapshot) {
  if (!cy) return;
  const model = buildMapModel(snapshot);
  const key = model.nodes.map((n) => n.data.id).join("|");
  if (key === mapShapeKey) return;
  mapShapeKey = key;
  lastMapModel = model;

  const pos = computeLayout(model);
  cy.elements().remove();
  cy.add(model.nodes.map((n) => ({
    group: "nodes",
    data: n.data,
    classes: n.classes,
    ...(pos[n.data.id] ? { position: { ...pos[n.data.id] } } : {}),
  })));
  cy.add(model.edges.map((e) => ({ group: "edges", data: e.data, classes: e.classes })));
  // Fit now (the panel may still be sizing) and again next frame, once it has.
  fitMap();
  requestAnimationFrame(fitMap);
}

function applyAgentColors(colors) {
  const root = document.documentElement.style;
  for (const a of AGENTS) if (colors[a]) root.setProperty(`--${a}`, colors[a]);
}

// Run-state overlay: how a node/edge looks at a point in the run. A disc's ring
// is faint before it runs, haloed while it runs, and filled once done — the
// "○ empty → ◎ half → ● filled" progression.
function stateStyles(agentColors) {
  const { warn, err } = MAP_COLORS;
  const scanner = (agentColors || {}).scanner || "#008c7a";
  const extractor = (agentColors || {}).extractor || "#1473e6";
  return [
    {
      selector: "node, edge",
      style: {
        "transition-property": "opacity, border-opacity, background-color, width",
        "transition-duration": "160ms",
      },
    },

    { selector: "node.disc.st-idle", style: { "border-opacity": 0.3, opacity: 0.55 } },
    {
      selector: "node.disc.st-active",
      style: {
        opacity: 1, "border-opacity": 1,
        "underlay-color": warn, "underlay-opacity": 0.35, "underlay-padding": 7,
        "z-index": 20,
      },
    },
    // Done is *filled*, not merely un-dimmed: at the size these discs end up on
    // screen a pale tint is indistinguishable from idle, and telling finished
    // work from pending work at a glance is the whole job.
    { selector: "node.disc.st-done", style: { opacity: 1, "border-opacity": 1 } },
    { selector: "node.disc.note.st-done", style: { "background-color": tint(scanner, 0.42) } },
    { selector: "node.disc.var.st-done", style: { "background-color": tint(extractor, 0.42) } },
    // A variable the run settled without a value — visibly reached, but empty.
    { selector: "node.disc.var.st-empty", style: { opacity: 1, "border-opacity": 0.55, "background-color": "#ffffff" } },
    { selector: "node.disc.var.st-flagged", style: { "border-color": err, "background-color": tint(err, 0.9) } },

    // Everything behind a shut gate stays dark for the whole run: it never ran,
    // and showing it as merely "not yet" would be a lie.
    { selector: ".blocked", style: { opacity: 0.16 } },
    { selector: "node.cluster.blocked", style: { opacity: 0.5 } },
    // A ruled-out group still gets its wire from Plan — that is what the ✗
    // hangs on — so it has to be faint but actually visible.
    { selector: "edge.blocked", style: { opacity: 0.34 } },

    { selector: "node.slab.st-idle", style: { opacity: 0.45 } },
    { selector: "node.slab.st-done", style: { opacity: 1 } },
    {
      selector: "node.slab.st-active",
      style: { opacity: 1, "underlay-color": warn, "underlay-opacity": 0.3, "underlay-padding": 5 },
    },
    {
      selector: "node.current",
      style: { "border-width": 2, "border-color": warn, "z-index": 25 },
    },

    { selector: "edge.st-idle", style: { opacity: 0.28 } },
    { selector: "edge.st-done", style: { opacity: 0.95, width: 2 } },
    // Nineteen variables converging on Update case is a lot of ink; the filled
    // discs already carry the state, so a walked fan edge only has to show the
    // path exists. The spine keeps its weight.
    { selector: "edge.fan.st-done", style: { opacity: 0.5, width: 1.3 } },
    // A 12-variable group is 24 lobes off one gate. Once walked they only have
    // to show the path existed — the filled discs carry the state — so they get
    // out of the way until something flows along them again.
    { selector: "edge.var-in.st-done, edge.var-out.st-done", style: { opacity: 0.3, width: 1 } },
    // The travelling dash: `line-dash-offset` is stepped by dashLoop().
    {
      selector: "edge.flowing",
      style: {
        opacity: 1, width: 2.6,
        "line-style": "dashed", "line-dash-pattern": [7, 5],
        "z-index": 18,
      },
    },

    // An edge with nothing to say yet. Last in the sheet so it beats
    // `edge.flowing`'s opacity on a specificity tie.
    //
    // Opacity, *not* `display: none`: display would drop the edge out of
    // cy.fit()'s bounds, so the viewport would lurch every time one appeared
    // mid-animation. Opacity keeps the bounds fixed and picks up the 160ms
    // transition above, so edges fade in as the run reaches them. `events: no`
    // keeps an invisible edge from catching the hover tooltip.
    { selector: ".undrawn", style: { opacity: 0, events: "no" } },
  ];
}

// One shared rAF loop drives every travelling dash, and stops itself when no
// edge is flowing so an idle map costs nothing.
let dashOffset = 0;
let dashRaf = null;

function dashLoop() {
  const flowing = cy && cy.edges(".flowing");
  if (!flowing || flowing.length === 0) {
    dashRaf = null;
    return;
  }
  dashOffset = (dashOffset - 0.9) % 24;
  flowing.style("line-dash-offset", dashOffset);
  dashRaf = requestAnimationFrame(dashLoop);
}

function startDashLoop() {
  if (dashRaf === null) dashRaf = requestAnimationFrame(dashLoop);
}

function coarse(fineId) {
  return fineId ? COARSE[fineId] || null : null;
}

// A gate's verdict at trace-time t. Undecided until corpus characterization
// produces the descriptors the planner's predicates read; after that the
// annotation the model already computed *is* the verdict, so the map never
// second-guesses the planner.
function gateVerdict(annotation, t) {
  const gated = /^(gate:|site:)/.test(annotation || "");
  if (gated && t < mapIndex.verdictT) return "pending";
  if (/✗/.test(annotation || "")) return "shut";
  return "open";
}

const GATE_GLYPH = { open: "✓", shut: "✗", pending: "?", skipped: "–" };

// The trace-time of the frame currently painted, for refitMap to replay.
let lastRenderT = 0;

// Assign every node/edge its class for trace-time `t`. Pure: the same t always
// produces the same drawing, so scrubbing and animating share one code path.
function renderMapAt(t, snapshot, step) {
  if (!cy) return;
  lastRenderT = t;   // so a resize can repaint this frame, not the step's end
  const progress = (snapshot && snapshot.progress) || {};
  const byItem = {};
  for (const v of progress.variables || []) byItem[v.item_id] = v;
  const annotations = {};
  for (const g of progress.groups || []) annotations[g.group_id] = g.annotation || "";

  const currentBlock = BLOCK_TO_MAP[coarse(step && step.map_node_id)] || null;
  // An extraction pass is *about* the groups it fanned out over, so highlight
  // those clusters rather than the whole band — the step's coarse block can
  // only name "the extractor".
  const currentGroups = new Set();
  if (step && step.node === "extract_branch") {
    for (const taskId of stepTaskIds(step)) {
      const gid = mapIndex.groupByTask.get(taskId);
      if (gid) currentGroups.add(gid);
    }
  }
  const blocked = new Set();

  cy.batch(() => {
    // Gates first: their verdict decides whether anything behind them may light.
    cy.nodes(".gate").forEach((node) => {
      const gid = node.data("group");
      const annotation = annotations[gid] || "";
      let verdict = gateVerdict(annotation, t);
      const retrieving = stateAt(`grpret:${gid}`, t);
      const extracting = stateAt(`grpext:${gid}`, t);
      const ran = stateAt(`grp:${gid}`, t);

      // An open group the retriever found no notes for is skipped, not failed.
      if (verdict === "open" && ran === "done" && !hasAnyVariableRun(gid, t)) {
        verdict = "skipped";
      }
      if (verdict === "shut") blocked.add(gid);

      node.removeClass("gate-open gate-shut gate-pending gate-skipped pipe-retrieve pipe-extract st-idle st-active st-done");
      node.addClass(`gate-${verdict}`);
      node.addClass(ran === "idle" ? "st-idle" : ran === "active" ? "st-active" : "st-done");
      if (retrieving === "active") node.addClass("pipe-retrieve");
      else if (extracting === "active") node.addClass("pipe-extract");
      node.data("label", GATE_GLYPH[verdict] || "");
      node.data("title", annotation || node.data("title"));
    });

    cy.nodes(".cluster").forEach((node) => {
      const gid = node.id().replace("grp:", "");
      node.toggleClass("blocked", blocked.has(gid));
      node.toggleClass("gate-shut", blocked.has(gid));
      node.toggleClass("current", currentGroups.has(gid));
    });

    cy.nodes(".note").forEach((node) => setState(node, stateAt(node.id(), t)));

    cy.nodes(".var").forEach((node) => {
      const gid = node.data("group");
      if (blocked.has(gid)) {
        node.removeClass("st-idle st-active st-done st-empty st-flagged").addClass("blocked");
        return;
      }
      node.removeClass("blocked");
      const state = stateAt(node.id(), t);
      setState(node, state);
      // A settled variable with no value reads as reached-but-empty rather than
      // as another filled dot.
      const result = byItem[Number(node.id().slice(4))];
      const empty = state === "done" && result && (result.value == null || result.value === "");
      node.toggleClass("st-empty", Boolean(empty));
      node.toggleClass("st-flagged", Boolean(result && result.flag));
      if (empty) node.removeClass("st-done");
    });

    cy.nodes(".slab").forEach((node) => {
      setState(node, stateAt(node.id(), t));
      node.toggleClass("current", node.id() === currentBlock);
    });

    // The case is the one label on the drawing that moves.
    const phase = casePhase(t);
    const caseNode = cy.getElementById(CASE_ID);
    if (caseNode.nonempty()) {
      caseNode.data("label", phase ? `Case\n${phase}` : "Case");
      const poles = ["pole:plan", "pole:update"].map((id) => stateAt(id, t));
      setState(caseNode, phase ? "active" : poles.some((s) => s !== "idle") ? "done" : "idle");
    }

    const checked = planChecked(t);
    cy.edges().forEach((edge) => {
      edge.removeClass("st-idle st-done flowing blocked undrawn");
      const gid = edge.data("group");
      if (gid && blocked.has(gid)) {
        // The planner decided this group, so its gate line is drawn — dim, but
        // there, because the ✗ needs a wire to hang on. Everything behind the
        // gate never ran, so it is never drawn at all.
        edge.addClass(edge.hasClass("gate-in") && checked ? "blocked" : "undrawn");
        return;
      }
      edge.addClass(edgeState(edge, t));
    });
  });
  startDashLoop();
}

function setState(node, state) {
  node.removeClass("st-idle st-active st-done");
  node.addClass(`st-${state}`);
}

function hasAnyVariableRun(groupId, t) {
  return cy
    .nodes(`.var[group = "${groupId}"]`)
    .some((node) => stateAt(node.id(), t) !== "idle");
}

// An edge *into* something lights the moment that thing starts and stays lit;
// an edge *out of* it lights when it finishes. That is the whole point of the
// per-instance nodes: you can see work being handed out and handed back.
//
// An edge that has nothing to say yet is `undrawn` rather than merely dim: a
// hundred hairlines showing the run's final wiring before any of it has
// happened is a grey web the lit edges have to fight through. Every rule below
// is monotone in `t`, so "once drawn, stays drawn" needs no extra bookkeeping —
// the wiring accumulates as the run explains itself.
function edgeState(edge, t) {
  if (edge.hasClass("note-in") || edge.hasClass("var-in")) {
    const state = stateAt(edge.target().id(), t);
    if (state === "active") return "flowing";
    return state === "done" ? "st-done" : "undrawn";
  }
  if (edge.hasClass("note-out")) {
    return stateAt(edge.source().id(), t) === "done" ? "st-done" : "undrawn";
  }
  // A variable hands its result back to its gate the moment it settles, and
  // keeps flowing while the rest of the group is still working.
  if (edge.hasClass("var-out")) {
    if (stateAt(edge.source().id(), t) !== "done") return "undrawn";
    return stateAt(`grp:${edge.data("group")}`, t) === "active" ? "flowing" : "st-done";
  }
  // The group's arc into the Update pole is the picture of the case being
  // updated, so it runs while it is — but only for the pass that is reporting
  // back, or every group the run has ever finished would light on every turn.
  if (edge.hasClass("grp-out")) {
    const gid = edge.data("group");
    const ran = stateAt(`grp:${gid}`, t);
    if (ran !== "done") return "undrawn";
    const updating = stateAt("phase:merge_and_update", t) === "active";
    return updating && endedSince(`grp:${gid}`, t, passStartBefore(t)) ? "flowing" : "st-done";
  }
  // A gate is wired up by the planning check that *decides* it — dispatched
  // here, or (in the blocked branch of renderMapAt) definitively ruled out. A
  // group that only becomes eligible on a later pass stays unwired until then.
  if (edge.hasClass("gate-in")) {
    const ran = stateAt(`grp:${edge.data("group")}`, t);
    if (ran === "active") return "flowing";
    return ran === "done" ? "st-done" : "undrawn";
  }
  // The exit has to read its *target*: its source is the case container, which
  // is a compound with no busy window of its own and so is forever "idle".
  if (edge.hasClass("exit")) {
    const state = stateAt("stage:finalize", t);
    if (state === "active") return "flowing";
    return state === "done" ? "st-done" : "undrawn";
  }
  // The backbone — the one edge that is always drawn. It lands on the case
  // container, so like the exit it has to read a pole rather than its endpoint.
  if (edge.hasClass("spine")) {
    if (stateAt("pole:plan", t) === "active") return "flowing";
    return stateAt(edge.source().id(), t) === "done" ? "st-done" : "st-idle";
  }
  // Every edge buildMapModel creates is classed, so this is unreachable — and
  // an edge nobody claimed has, by definition, nothing to say.
  return "undrawn";
}

/* --- animating a step ----------------------------------------------------
 *
 * At a step *boundary* the work in that step is already finished — replaying
 * the scan step would just show seven green notes. So revealing a step replays
 * its own span of the trace over a fixed wall-clock budget, mapping the
 * recorded [t0, t1] onto it so relative durations still read.
 */
const STEP_ANIM_MS = 2500;
let stepAnim = null;
let stepFallback = null;

function stepSpan(step) {
  let t0 = Infinity;
  let t1 = -Infinity;
  for (const ev of events) {
    if (ev.seq < step.start_seq || ev.seq > step.end_seq) continue;
    t0 = Math.min(t0, ev.t);
    t1 = Math.max(t1, ev.t);
  }
  return t0 <= t1 ? [t0, t1] : null;
}

// The trace time a step is finished at — everything up to and including it has
// happened, and nothing after it has. Not the same as the whole run's end.
function stepEndT(step) {
  let t = 0;
  for (const ev of events) {
    if (ev.seq > step.end_seq) break;
    t = Math.max(t, ev.t);
  }
  return t;
}

// Show a step's settled end state with no animation.
function settleStep(step, snapshot) {
  stopStepAnim();
  renderMapAt(step ? stepEndT(step) : 0, snapshot, step);
  setStepProgress(1);
}

function stopStepAnim() {
  if (stepAnim !== null) cancelAnimationFrame(stepAnim);
  if (stepFallback !== null) clearTimeout(stepFallback);
  stepAnim = null;
  stepFallback = null;
}

function playStep(step, snapshot) {
  stopStepAnim();
  const span = step && stepSpan(step);
  if (!span) {
    settleStep(step, snapshot);
    return;
  }
  const [t0, t1] = span;
  // Paint the opening frame synchronously. requestAnimationFrame does not fire
  // while the tab is hidden, and leaving the map unpainted until the first
  // frame arrives means a backgrounded tab shows an empty graph forever.
  renderMapAt(t0, snapshot, step);
  setStepProgress(0);

  const started = performance.now();
  const frame = (now) => {
    const p = Math.min(1, (now - started) / STEP_ANIM_MS);
    renderMapAt(t0 + (t1 - t0) * p, snapshot, step);
    setStepProgress(p);
    stepAnim = p < 1 ? requestAnimationFrame(frame) : null;
    if (stepAnim === null) stopStepAnim();
  };
  stepAnim = requestAnimationFrame(frame);
  // …and for the same reason, guarantee the map reaches the step's settled
  // state even if the frames never come.
  stepFallback = setTimeout(() => {
    stopStepAnim();
    settleStep(step, snapshot);
  }, STEP_ANIM_MS + 400);
}

// Scrub to a fraction of the current step without animating.
function seekStep(fraction) {
  stopStepAnim();
  if (!lastView || !lastView.step) return;
  const span = stepSpan(lastView.step);
  if (!span) return;
  renderMapAt(span[0] + (span[1] - span[0]) * fraction, lastView.snapshot, lastView.step);
  setStepProgress(fraction);
}

function setStepProgress(fraction) {
  if (els.mapScrub && document.activeElement !== els.mapScrub) {
    els.mapScrub.value = String(Math.round(fraction * 1000));
  }
}

// Hover a disc for the name behind it — the discs themselves are deliberately
// unlabeled (43 labels would be unreadable), so this is how a presenter answers
// "which variable is that one?".
function wireMapTooltip() {
  const tip = els.mapTip;
  if (!tip) return;
  cy.on("mouseover", "node.disc", (evt) => {
    const title = evt.target.data("title");
    if (!title) return;
    tip.textContent = title;
    tip.hidden = false;
  });
  cy.on("mousemove", "node.disc", (evt) => {
    const box = cy.container().getBoundingClientRect();
    const point = evt.renderedPosition || { x: 0, y: 0 };
    tip.style.left = `${Math.min(point.x + 12, box.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${Math.max(point.y - 30, 4)}px`;
  });
  cy.on("mouseout", "node.disc", () => { tip.hidden = true; });
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
    // An extraction pass is per-group and then per-variable, so it is checked
    // before the generic fan-out path (it is also a collapsed fan-out step).
    if (step.node === "extract_branch") {
      renderExtractDetail(step, snap);
      return;
    }
    // A collapsed fan-out step (e.g. "Characterize notes") shows one card per
    // instance instead of one merged card per map node.
    if (step.fanout) {
      renderFanoutDetail(step, snap);
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

// --- Panel 2: an extraction pass ----------------------------------------
// A pass fans out over every eligible group at once, so this renders one
// section per group — its retriever verdict, its single group-level model call,
// then one card per variable (fed by the per-group and per-variable fan-out
// instances in state.py). The merged group result is deliberately omitted: it
// is just the variable cards concatenated.
function renderExtractDetail(step, snap) {
  const instances = Object.values(snap.instances || {});
  const groups = instances
    .filter((i) => i.node === "extract_branch" && withinStep(i, step))
    .sort((a, b) => a.index - b.index);

  const varsFor = (group) =>
    instances
      .filter((i) => i.node === "variable_branch" && i.key.startsWith(`${group.key}/`))
      .sort((a, b) => a.index - b.index);

  els.detailNode.textContent = step.map_node_id || "";

  const total = groups.reduce((n, g) => n + varsFor(g).length, 0);
  const counts = [
    groups.length ? `${groups.length} group${groups.length === 1 ? "" : "s"}` : "",
    total ? `${total} variable${total === 1 ? "" : "s"}` : "",
  ].filter(Boolean).join(" · ");
  const parts = [detailHeadline(step.title, [step.subtitle, counts].filter(Boolean).join(" · "), step.agent, "")];

  if (!groups.length) {
    parts.push(`<p class="empty">No group extraction captured for this step.</p>`);
  } else {
    for (const group of groups) parts.push(renderGroupDetail(group, varsFor(group), snap));
  }
  els.detail.innerHTML = parts.join("");
}

// A fan-out instance belongs to the step whose seq range opened it. The keys
// carry no seq, so match on the task id the step's own events name.
function withinStep(instance, step) {
  const taskId = instance.key.split(":")[1];
  return stepTaskIds(step).has(taskId);
}

function stepTaskIds(step) {
  const ids = new Set();
  if (!step) return ids;
  for (const ev of events) {
    if (ev.seq < step.start_seq || ev.seq > step.end_seq) continue;
    if (ev.type === "task_start" && !ev.namespace.length) ids.add(ev.task_id);
  }
  return ids;
}

// One group within the pass: what the retriever kept, the one model call that
// produced every candidate, then the variables themselves.
function renderGroupDetail(group, vars, snap) {
  const result = group.result && typeof group.result === "object" ? group.result : {};
  const settled = vars.filter((i) => i.status !== "active").length;
  const calls = (group.llm_calls || []).map(renderLLMCall).join("");
  const retrieval = Array.isArray(result.relevant_note_ids)
    ? viewRetriever(result, snap)
    : "";
  const cards = vars.length
    ? vars.map(renderVariableDetail).join("")
    : `<p class="empty">No variables extracted${
        retrieval ? " — the retriever kept no notes for this group." : "."
      }</p>`;

  return `<section class="node-detail group">
    <header class="node-head agent-${group.agent || "orchestrator"}">
      <span class="node-head-title">${esc(group.label || group.key)}</span>
      <span class="node-head-id">${vars.length ? `${settled}/${vars.length}` : ""}</span>
      <span class="status-pill status-${esc(group.status)}">${esc(group.status)}</span>
    </header>
    ${retrieval}
    ${calls}
    ${cards}
    ${group.error ? `<pre class="code">${esc(fmt(group.error))}</pre>` : ""}
  </section>`;
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
  els.mapReplay.addEventListener("click", () => {
    if (lastView) playStep(lastView.step, lastView.snapshot);
  });
  els.mapScrub.addEventListener("input", (e) => seekStep(Number(e.target.value) / 1000));

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
  syncMap(view.snapshot);
  // A step the presenter has just arrived at replays its own span; scrubbing
  // back to one already seen jumps to its settled end state instead of
  // re-animating work they have already watched.
  if (stepChanged) playStep(view.step, view.snapshot);
  else settleStep(view.step, view.snapshot);
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
