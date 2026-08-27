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
// ``check_state`` is merged into a step by steps.py twice over: with the plan
// that follows it, where both render the same eligibility gate off
// snapshot.progress, and — when it is the final check that finds nothing left —
// with the finalization it triggers, where it is a verbatim repeat of the check
// the presenter has just finished talking about. Either way the check has
// nothing of its own to say, so it is drawn once, by the node that names it.
const SUBSUMED_BY = { check_state: ["plan_extraction", "finalize_case"] };

// Runtime node -> what that pass of the extractor's inner loop did.
const ATTEMPT_LABELS = {
  extract_individual_value: "individual extraction",
  validate_extraction: "validation",
  repair_invalid_extraction: "repair",
};

// Layout: default share of the left column given to the Variables panel, and
// where a presenter's dragged size is remembered across reloads.
const VARS_H_KEY = "cipoc.demo.varsHeight";
const VARS_OPEN_KEY = "cipoc.demo.varsOpen";
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
  wireVarsPane();

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
  watchCards(els.detail);
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

// One box at the top of the drawing for the whole front of the pipeline: the
// notes the scanner reads *and* the characterization drawn from them. They were
// two boxes and an arrow, but the arrow only ever said "and then", and the notes
// are what the characterization is made of — so they live inside it.
const CORPUS_ID = "stage:corpus";

// The corpus box is shaped like an ungated group — a marker disc at the left and
// its discs in a grid beside it — so that the front of the pipeline reads as a
// peer of the boxes in the band rather than as its own kind of object. It has no
// gate to show, so the marker carries a note glyph instead of a verdict and is
// never given a `gate-*` class.
const CORPUS_MARK_ID = "stage:corpus:mark";
// The note mark is *drawn*, not typed, because no character could satisfy both
// halves of what it has to be:
//
//   - Monochrome, so it takes the scanner's hue and dims with the rest of the
//     map. That rules out PAGE FACING UP and every other emoji: they render
//     from the colour font and ignore the fill colour they are given.
//   - Present in a font the machine actually has. That rules out the literal
//     page glyphs which *are* monochrome — U+1F5CF PAGE, U+1F5CB EMPTY DOCUMENT
//     and their neighbours in the Wingdings-derived stretch of Miscellaneous
//     Symbols and Pictographs — because macOS carries them in *LastResort* and
//     nowhere else, i.e. they draw the tofu box. U+1F5CF was the mark here and
//     that is exactly what it drew. The vendored faces cannot rescue a
//     codepoint either: both are latin subsets, so anything symbolic falls
//     through to whatever the system happens to have.
//
// A path has neither problem, and on an airgapped target that is worth more
// than the few lines it costs: a drawing renders identically on a machine whose
// font set we never get to inspect. The geometry lives in one place and is
// wrapped two ways below, so the map and panel 2 cannot drift apart.
const NOTE_PAGE_PATHS =
  `<path d="M3.2 1.6h6.3l3.3 3.3v9.5H3.2z" fill="none" stroke="currentColor"
         stroke-width="1.3" stroke-linejoin="round"/>` +
  `<path d="M9.5 1.6v3.3h3.3" fill="none" stroke="currentColor"
         stroke-width="1.3" stroke-linejoin="round"/>` +
  `<path d="M5.6 7.8h4.8M5.6 10.2h4.8M5.6 12.6h3.2" fill="none" stroke="currentColor"
         stroke-width="1.2" stroke-linecap="round"/>`;

// Panel 2's copy: inline, so `currentColor` resolves against `.note-glyph` and
// the icon follows --scanner-ink the way the text beside it follows its own.
const NOTE_ICON = `<svg class="note-glyph" viewBox="0 0 16 16" aria-hidden="true">${NOTE_PAGE_PATHS}</svg>`;

// The map's copy. A node background-image is fetched as an image and so has no
// CSS context to resolve `currentColor` against — the hue has to be baked in.
// Safe to bake because the marker's colour is fixed (`theme.scannerInk`, never
// restyled by a state class); the dimming it *does* get is node `opacity`,
// which applies to the image like any other node content. `xmlns` is required
// for a standalone SVG document, and encodeURIComponent is not optional: the
// `#` of the hex colour would otherwise open a fragment and truncate the URI.
const noteIconUri = (color) =>
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">` +
      NOTE_PAGE_PATHS.replace(/currentColor/g, color) +
      `</svg>`,
  );

// The middle of the drawing is the case, and it is *one box*. Splitting it into
// Plan and Update containers spent the box's whole interior on two small labels
// saying what the case's own label can say on its own — so the box is plain now,
// work is dispatched from it and returns to it, and the label gets the room.
const CASE_ID = "stage:case";

// The orchestrator root nodes that are phases of the case's life. Each keeps its
// own busy window, so the case can name the phase that is running — and name it
// in the colour of whatever is flowing along the lines at that moment, so the
// label and the lit edges always agree about whose work this is.
const CASE_REST = "Case";
// Each tone is the colour of whatever is on the lines during that phase: the
// planner's own verdicts are the orchestrator's, dispatch is the retriever's,
// results coming back are the extractor's. Finalizing is orchestrator, not err —
// nothing flows by then, and pink means "ruled out" everywhere else on the map.
const CASE_PHASE_NODES = {
  initialize_case: { label: "Initializing", tone: "orch" },
  check_state: { label: "Checking state", tone: "orch" },
  plan_extraction: { label: "Planning extraction", tone: "orch" },
  merge_and_update: { label: "Updating case", tone: "extr" },
  finalize_case: { label: "Finalizing", tone: "orch" },
};
// Extraction belongs to no root node — it is the fan-out running — so it is not
// in the table above, but it is the phase the case spends most of its time in.
// Retriever, because the lines up during a pass are its dispatch to the gates.
const CASE_EXTRACTING = { label: "Extracting…", tone: "retr" };

// Coarse overview block -> what now represents it on the drawing. The Python
// side (mapping.py) is untouched and still owns runtime-node -> block; this is
// only the last hop, block -> drawn element, and it is what keeps click-to-pin
// and the "current stage" highlight working against the new topology.
//
// Four blocks land on the case: initializing it, planning against it, updating
// it and finalizing it are all things that happen *to the case*, and with the
// poles gone the case's label is where they are told apart.
const BLOCK_TO_MAP = {
  initialize_case: CASE_ID,
  characterize_corpus: CORPUS_ID,
  eligible_groups_gate: CASE_ID,
  update_case: CASE_ID,
  finalize_case: CASE_ID,
  scanner_agent_block: CORPUS_ID,
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
  caseW: 264, caseH: 100,
  noteD: 34, noteGapX: 46, noteGapY: 46,
  gateD: 50, gateGap: 16, varD: 20, varGapX: 30, varGapY: 28,
  clusterGap: 40,
  rowGap: 78,
  // The corpus box sits closer to the case than the band does. One line runs
  // between them and nothing else competes for the corridor, whereas the band
  // below fans twenty arcs into the case's underside and needs the room.
  headGap: 44,
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
 * Two kinds of box — the case, and the containers that hold work in progress —
 * plus instance discs: white with a thick agent-colored ring that fills in as
 * they complete. Active edges carry a travelling dash.
 *
 * Two Cytoscape-specific notes: it has no per-side borders, so a box's top rule
 * is the first stop of a vertical gradient; and it dropped `shadow-*` in v3, so
 * the card elevation is a low-opacity `underlay-*` halo instead.
 */
/* styles.css is the single source of truth for colour; the map reads the same
 * custom properties the panels do rather than keeping its own copy. It used to
 * keep one, and the two had already drifted — the map's amber was #c98a12 while
 * --warn was #b8860b. Read after applyAgentColors() has written the
 * server-supplied agent hues onto :root, so those win here too.
 *
 * On `err`: it is a magenta-leaning pink rather than a brick red, so
 * the ✓/✗ pair and the pass/fail verdict lines stay apart under red-green colour
 * blindness — green and pink separate on the blue axis, which deuteranopia
 * leaves intact. The glyphs carry the same distinction in shape, so colour is
 * never load-bearing.
 */
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    ok: v("--ok"), warn: v("--warn"), err: v("--err"), errSoft: v("--err-soft"),
    line: v("--line-strong"), lineSoft: v("--line"), muted: v("--miss"),
    navy: v("--navy"), ink: v("--ink"), inkMuted: v("--muted"),
    panel: v("--panel"), sunk: v("--panel-sunk"),
    orchestrator: v("--orchestrator"), scanner: v("--scanner"),
    scannerInk: v("--scanner-ink"),
    retriever: v("--retriever"), extractor: v("--extractor"),
    // Canvas labels do not inherit CSS, so the family is set per selector below.
    font: v("--font-sans"),
  };
}

function mapStyle(theme) {
  const { ok, warn, err, line, muted, font } = theme;
  const ORCH = theme.orchestrator;
  const SCAN = theme.scanner;
  const EXTR = theme.extractor;
  const RETR = theme.retriever;

  // A white card with `color` as a rule across its top edge.
  const ruled = (color) => ({
    "background-fill": "linear-gradient",
    "background-gradient-direction": "to-bottom",
    "background-gradient-stop-colors": `${color} ${color} ${theme.panel} ${theme.panel}`,
    "background-gradient-stop-positions": "0% 8% 8% 100%",
  });

  return [
    // --- the case ---
    //
    // A plain box with nothing inside it but its own label, which is the one
    // text on the drawing that moves: `Case`, plus the phase that is running.
    // Emptying it out is what lets that label be read from the back of a room.
    {
      selector: "node.case",
      style: {
        shape: "round-rectangle",
        width: GEO.caseW, height: GEO.caseH,
        ...ruled(ORCH),
        "border-width": 1, "border-color": tint(theme.navy, 0.72),
        "underlay-color": theme.navy, "underlay-opacity": 0.14, "underlay-padding": 3,
        label: "data(label)", "text-wrap": "wrap", "text-max-width": GEO.caseW - 26,
        "text-valign": "center", "text-halign": "center", "text-margin-y": 3,
        "font-family": font,
        "font-size": 21, "font-weight": 700, color: theme.ink, "line-height": 1.25,
      },
    },
    // The phase is named in the colour of what is flowing while it runs: the
    // dispatch out to the gates is the retriever's, the results coming back are
    // the extractor's. Label and top rule both take it, so the case agrees with
    // whichever lines are lit rather than merely sitting between them.
    { selector: "node.case.tone-retr", style: { color: RETR, ...ruled(RETR) } },
    { selector: "node.case.tone-extr", style: { color: EXTR, ...ruled(EXTR) } },

    // --- group cluster (compound parent carries the group's name) ---
    {
      selector: "node.cluster",
      style: {
        shape: "round-rectangle",
        "background-color": theme.sunk, "background-opacity": 0.9,
        "border-width": 1, "border-color": theme.lineSoft,
        padding: 10,
        label: "data(label)", "text-valign": "top", "text-halign": "center",
        "text-margin-y": -3, "text-wrap": "wrap", "text-max-width": 170,
        "font-family": font,
        "font-size": 12.5, "font-weight": 700, color: theme.inkMuted,
      },
    },
    // A cluster's border carries its gate's verdict, so live work and ruled-out
    // work are told apart by weight and not only by the glyph on one small disc:
    // a group that passed — or that passed and is still waiting its turn — is
    // drawn in a heavier line than one the planner has already discarded.
    { selector: "node.cluster.gate-open", style: { "border-width": 2.5, "border-color": tint(ok, 0.5) } },
    { selector: "node.cluster.gate-ungated", style: { "border-width": 2.5, "border-color": tint(ORCH, 0.55) } },
    { selector: "node.cluster.gate-shut", style: { "background-color": theme.errSoft, "border-color": tint(err, 0.72) } },
    // The front of the pipeline is a cluster too, and now says so: same fill,
    // same label, same 2.5 border an ungated group wears — in the scanner's hue
    // rather than the orchestrator's, because everything in this box is the
    // scanner's work. It was a teal-tinted, elevated, larger-labelled box before,
    // which made the one thing structurally identical to a group read as a
    // different kind of object entirely.
    {
      selector: "node.cluster.corpus",
      style: {
        "border-color": tint(SCAN, 0.55), "border-width": 2.5,
        "text-max-width": 320,
      },
    },

    // --- instance discs: white with an agent-colored ring ---
    {
      selector: "node.disc",
      style: {
        shape: "ellipse",
        "background-color": theme.panel,
        "border-width": 3, "border-color": muted, "border-opacity": 1,
        label: "data(label)", "text-valign": "center", "text-halign": "center",
        "font-family": font,
        "font-size": 11, "font-weight": 700, color: theme.inkMuted,
      },
    },
    { selector: "node.disc.note", style: { width: GEO.noteD, height: GEO.noteD, "border-color": SCAN } },
    { selector: "node.disc.var", style: { width: GEO.varD, height: GEO.varD, "border-color": EXTR, "font-size": 0 } },
    // The verdict glyph is the smallest thing on the map carrying the biggest
    // meaning, and ✓ / ✗ come from a fallback symbol font that ignores
    // font-weight — so each is outlined in its own colour to give it weight.
    // *Narrowly*, though: at 1.8 the two strokes of a ✗ ran together and the
    // icon read as a blob rather than as a mark.
    {
      selector: "node.disc.gate",
      style: {
        width: GEO.gateD, height: GEO.gateD,
        "border-color": warn, color: warn,
        "font-size": 24, "text-outline-width": 0.9,
        "text-outline-color": warn, "text-outline-opacity": 1,
      },
    },
    // The corpus marker: a gate-sized disc in the same slot, carrying a note
    // glyph rather than a verdict. Filled the way an ungated gate is, so the box
    // opens on a marked disc like every box in the band does — in the scanner's
    // hue, and a size larger than the note discs it introduces.
    {
      selector: "node.disc.mark",
      style: {
        width: GEO.gateD, height: GEO.gateD,
        "background-color": tint(SCAN, 0.92),
        // Ring in the scanner's hue so the marker belongs to the note discs
        // beside it; the page in the darker ink, because rules this fine need
        // the weight that #00A5AD at 2.7:1 does not give them.
        "border-color": SCAN,
        // Drawn, not lettered, so the disc carries no label at all — `label: ""`
        // rather than the `font-size: 0` a var disc uses, because this node has
        // no label data to suppress. 48% of a 50px disc puts the page at 24px,
        // a size larger than the note discs it introduces, and well inside the
        // ~35px square the circle inscribes, so the ellipse never clips it.
        label: "",
        "background-image": noteIconUri(theme.scannerInk),
        "background-fit": "none",
        "background-width": "48%", "background-height": "48%",
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
    // Dispatch and return now share one pair of endpoints — case to gate and
    // gate back to case — so they have to bow apart or they lie on top of each
    // other. `unbundled-bezier` bows perpendicular to the source->target line,
    // and that line is reversed between the two, so the *same* sign puts them on
    // opposite sides. The sag also means a second-row cluster's arc sweeps
    // around the first row rather than through it.
    {
      selector: "edge.gate-in",
      style: {
        "curve-style": "unbundled-bezier",
        "control-point-distances": 44, "control-point-weights": 0.5,
        "target-arrow-shape": "triangle", "arrow-scale": 0.8,
        width: 1.8,
      },
    },
    {
      selector: "edge.grp-out",
      style: {
        "curve-style": "unbundled-bezier",
        "control-point-distances": 44, "control-point-weights": 0.5,
        "target-arrow-shape": "triangle", "target-arrow-color": EXTR, "arrow-scale": 0.8,
        width: 1.8,
      },
    },
    { selector: "edge.to-scanner", style: { "line-color": SCAN } },
    { selector: "edge.to-extractor", style: { "line-color": EXTR } },
    // Each verdict re-colours the glyph, so it has to re-colour the outline with
    // it or the fattening reverts to the amber of the pending state.
    { selector: "node.disc.gate.pipe-retrieve", style: { "border-color": RETR, color: RETR, "text-outline-color": RETR } },
    { selector: "node.disc.gate.pipe-extract", style: { "border-color": EXTR, color: EXTR, "text-outline-color": EXTR } },
    { selector: "node.disc.gate.gate-open", style: { "border-color": ok, color: ok, "text-outline-color": ok, "background-color": tint(ok, 0.9) } },
    { selector: "node.disc.gate.gate-shut", style: { "border-color": err, color: err, "text-outline-color": err, "background-color": tint(err, 0.9) } },
    { selector: "node.disc.gate.gate-skipped", style: { "border-color": muted, color: muted, "text-outline-color": muted } },
    // A group with no gate predicate at all never passed anything, so it does
    // not get the ✓ that a gated group earns — it gets an arrow, in the
    // orchestrator's colour, for work that goes straight through. A plain ↓,
    // set a couple of units larger than the marks above because its head is a
    // fraction of its shaft and is the first thing to go at disc size.
    {
      selector: "node.disc.gate.gate-ungated",
      style: {
        "border-color": ORCH, color: ORCH, "text-outline-color": ORCH,
        "background-color": tint(ORCH, 0.92),
        "font-size": 26,
      },
    },
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

    // A verdict lands when the planner *reaches* it, not when it first became
    // computable. Corpus characterization produces the descriptors the gate
    // predicates read, but flipping the glyphs there put the ✓/✗ on screen a
    // whole step before the check that decides them — the lines reaching out to
    // the gates arrived to find the answer already written.
    if (ev.type === "task_end" && !ev.namespace.length && ev.map_node_id === "plan_extraction") {
      verdictT = Math.min(verdictT, ev.t);
    }

    if (ev.type !== "task_start" && ev.type !== "task_end") continue;

    // The two top-level boxes — the corpus and the case — via the coarse block
    // the runtime node belongs to. Several blocks land on each, so each keeps a
    // list of windows; only the `band:` entries are not drawn boxes at all.
    const box = BLOCK_TO_MAP[coarse(ev.map_node_id)];
    if (box && !box.startsWith("band:")) {
      if (ev.type === "task_start") open(`box/${ev.task_id}`, box, ev.t);
      else close(`box/${ev.task_id}`, ev.t);
    }

    // The case stands for several root nodes at once, so each keeps its own
    // window — that is what lets the case's label name the running phase.
    //
    // Keyed on `map_node_id`, not `ev.node`: the orchestrator's initialize node
    // is called plain `initialize` at runtime and only the map id disambiguates
    // it from the subagents' (mapping.py::map_node_id).
    const phase = ev.map_node_id;
    if (!ev.namespace.length && CASE_PHASE_NODES[phase]) {
      if (ev.type === "task_start") {
        open(`phase/${ev.task_id}`, `phase:${phase}`, ev.t);
        if (phase === "plan_extraction") passStarts.push(ev.t);
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

// Which phase the case is in at `t` — the second line of its label and the tone
// of both — or null at rest. Dispatch is checked before planning because a pass
// runs *inside* the loop turn the planner opened; the two ends of the run come
// last because they never overlap anything.
function casePhase(t) {
  if (stateAt("phase:merge_and_update", t) === "active") return CASE_PHASE_NODES.merge_and_update;
  if (anyGroupActive(t)) return CASE_EXTRACTING;
  for (const node of ["plan_extraction", "check_state", "finalize_case", "initialize_case"]) {
    if (stateAt(`phase:${node}`, t) === "active") return CASE_PHASE_NODES[node];
  }
  return null;
}

// The phase a *step* is about, for the frame it settles on. These orchestrator
// nodes are near-instantaneous, so at a step's own end its window has already
// closed and the window rule above finds nothing — the label would drop back to
// "Case" the moment the presenter stopped scrubbing, on the very step whose name
// they are standing there saying out loud.
function stepPhase(step) {
  if (!step) return null;
  if (step.node === "extract_branch") return CASE_EXTRACTING;
  return CASE_PHASE_NODES[step.map_node_id] || null;
}

// The planner has *reached* its first set of verdicts by `t`, so a gate's ✓/✗ is
// something it decided rather than something merely computable. Every gate glyph
// and every verdict line hangs off this.
function planChecked(t) {
  return t >= mapIndex.verdictT;
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

  // The case. Initializing, planning, updating and finalizing all land here, so
  // it carries the planner's block for click-to-pin and says the rest in its
  // label — there is no Finalize box any more either, just a label that reads
  // "Finalizing" when the run gets there.
  add({ id: CASE_ID, label: CASE_REST, block: "eligible_groups_gate" }, "case");

  // The front of the pipeline: one container of note discs that fill as the
  // scanner reads them, and which *is* the corpus characterization drawn from
  // them. Structurally identical to a group and its variables — no edges in or
  // out of the individual notes, because the box already says what they are.
  add({ id: CORPUS_ID, label: "Scan & characterize notes", block: "characterize_corpus" }, "cluster corpus");
  add({ id: CORPUS_MARK_ID, parent: CORPUS_ID, title: "Clinical notes", block: "scanner_agent_block" }, "disc mark");
  for (const note of mapIndex.notes) {
    add({ id: note.id, parent: CORPUS_ID, label: `#${note.noteId}`, title: `${note.type} #${note.noteId}`.trim(), block: "scanner_agent_block" }, "disc note");
  }

  for (const group of planGroups(snapshot)) {
    const cluster = `grp:${group.id}`;
    const gate = `gate:${group.id}`;
    add({ id: cluster, label: group.name, block: "retriever_agent_block" }, "cluster");
    add({ id: gate, parent: cluster, label: "", title: group.annotation || group.name, group: group.id, block: "retriever_agent_block" }, "disc gate");
    // The group's two lines are its whole story: dispatched from the case, and
    // reported back to it. Its variables fill in place — a line per variable to
    // and from the gate they already sit beside adds nothing but ink.
    //
    // No colour class here: this one edge is the planner's verdict during the
    // check and the retriever's dispatch during the pass, so renderMapAt gives
    // it a `wire-*` class per frame rather than it being fixed at build.
    link(CASE_ID, gate, "fan gate-in", { group: group.id });
    for (const variable of group.variables) {
      const id = `var:${variable.itemId}`;
      add({ id, parent: cluster, label: "", title: variable.name, group: group.id, block: "extractor_agent_block" }, "disc var");
    }
    link(gate, CASE_ID, "fan to-extractor grp-out", { group: group.id });
  }

  link(CORPUS_ID, CASE_ID, "spine");

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

  // --- the corpus box: a marker disc at the left and note discs in a grid to
  // the right of it — measured exactly the way a cluster is, because it is one.
  // Only the discs get positions; the container is a compound parent and sizes
  // itself around them.
  const notes = parts.notes;
  const perRow = Math.max(1, Math.min(noteCols, notes.length || 1));
  const noteRows = Math.ceil(notes.length / perRow) || 1;
  const stripW = GEO.gateD + GEO.gateGap + perRow * GEO.noteGapX + CLUSTER_PAD * 2;
  const stripH = Math.max(GEO.gateD, noteRows * GEO.noteGapY) + CLUSTER_PAD * 2;

  // Nothing hangs off either side any more, so the drawing is symmetric about
  // `cx` and one half-width describes it.
  const half = Math.max(stripW, bandW, GEO.caseW) / 2;
  const cx = half;

  const stripX0 = cx - stripW / 2;
  pos[CORPUS_MARK_ID] = { x: stripX0 + CLUSTER_PAD + GEO.gateD / 2, y: stripH / 2 };
  const noteX0 = stripX0 + CLUSTER_PAD + GEO.gateD + GEO.gateGap + GEO.noteGapX / 2;
  const noteY0 = stripH / 2 - ((noteRows - 1) * GEO.noteGapY) / 2;
  notes.forEach((note, i) => {
    pos[note.id] = {
      x: noteX0 + (i % perRow) * GEO.noteGapX,
      y: noteY0 + Math.floor(i / perRow) * GEO.noteGapY,
    };
  });

  pos[CASE_ID] = { x: cx, y: stripH + GEO.headGap + GEO.caseH / 2 };

  let by = stripH + GEO.headGap + GEO.caseH + GEO.rowGap;
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
    w: half * 2,
    h: stripH + GEO.headGap + GEO.caseH + (bandH ? GEO.rowGap + bandH : 0),
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
  // Server-supplied agent hues land on :root first, so readTheme() picks them up
  // along with every other token from styles.css.
  applyAgentColors(graph.agent_colors || {});
  const theme = readTheme();

  buildMapIndex();
  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: { nodes: [], edges: [] },
    style: [...mapStyle(theme), ...stateStyles(theme)],
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

  // Click a box or a disc to pin Panel 2 to the component behind it; click
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
    // First, before anything reads the viewport: cy.width()/height() are cached
    // and keep reporting the old size until this runs, so the packing search
    // below would score every candidate against the panel we no longer have.
    cy.resize();
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
function stateStyles(theme) {
  const { ok, warn, err, line } = theme;
  const { scanner, extractor, retriever, orchestrator: orch } = theme;
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
    // A note being read and a note already read were both scanner teal, a shade
    // apart — at disc size that is no distinction at all. In flight is amber
    // now, the colour the map already uses for work in progress, so the strip
    // reads as amber turning teal rather than as teal turning slightly darker.
    //
    // The halo also shrinks: six or seven notes run at once and pack tightly, so
    // the generic 7-unit active halo above merged them into one tan bar.
    {
      selector: "node.disc.note.st-active",
      style: {
        "underlay-padding": 2, "underlay-opacity": 0.16,
        "background-color": tint(warn, 0.55), "border-color": warn,
      },
    },
    // The corpus marker is an icon naming the box, not a unit of work, so it
    // does not pulse: the generic active halo put a 7-unit amber block at the
    // head of the strip, competing with the notes that are actually being read.
    // It simply comes up out of the dim when the box is live.
    { selector: "node.disc.mark.st-active", style: { opacity: 1, "border-opacity": 1, "underlay-opacity": 0 } },
    // Done is *filled*, not merely un-dimmed: at the size these discs end up on
    // screen a pale tint is indistinguishable from idle, and telling finished
    // work from pending work at a glance is the whole job.
    { selector: "node.disc.st-done", style: { opacity: 1, "border-opacity": 1 } },
    { selector: "node.disc.note.st-done", style: { "background-color": tint(scanner, 0.42) } },
    { selector: "node.disc.var.st-done", style: { "background-color": tint(extractor, 0.42) } },
    // A variable the run settled without a value — visibly reached, but empty.
    { selector: "node.disc.var.st-empty", style: { opacity: 1, "border-opacity": 0.55, "background-color": theme.panel } },
    { selector: "node.disc.var.st-flagged", style: { "border-color": err, "background-color": tint(err, 0.9) } },

    // Everything behind a shut gate stays dark for the whole run: it never ran,
    // and showing it as merely "not yet" would be a lie.
    { selector: ".blocked", style: { opacity: 0.16 } },
    { selector: "node.cluster.blocked", style: { opacity: 0.5 } },

    // The case never dims all the way: it is the subject of the whole drawing
    // and its label has to stay readable even before the run touches it.
    { selector: "node.case.st-idle", style: { opacity: 0.7 } },
    { selector: "node.case.st-done", style: { opacity: 1 } },
    {
      selector: "node.case.st-active",
      style: { opacity: 1, "underlay-color": warn, "underlay-opacity": 0.3, "underlay-padding": 6 },
    },
    // The corpus box lights across both jobs it stands for: the scanner reading
    // the notes, and the characterization drawn from them.
    { selector: "node.cluster.corpus.st-idle", style: { opacity: 0.6 } },
    // Border only, no halo: the discs inside carry their own, and two tan bands
    // stacked read as one filled block rather than as work in progress.
    {
      selector: "node.cluster.corpus.st-active",
      style: { "border-width": 2.5, "border-color": scanner },
    },
    {
      selector: "node.current",
      style: { "border-width": 2, "border-color": warn, "z-index": 25 },
    },

    { selector: "edge.st-idle", style: { opacity: 0.28 } },
    { selector: "edge.st-done", style: { opacity: 0.95, width: 2 } },

    // What the case-to-gate line means at this moment. It is the planner's
    // verdict during a check and the retriever's dispatch during a pass, so its
    // colour is assigned per frame rather than fixed when the edge is built.
    // Each matches the thing it is about: the gate glyph it lands on, or the
    // agent doing the work.
    // Named for the verdict, so `wire-${verdict}` in edgeState needs no table.
    { selector: "edge.wire-pending", style: { "line-color": line, "target-arrow-color": line } },
    { selector: "edge.wire-open", style: { "line-color": ok, "target-arrow-color": ok } },
    { selector: "edge.wire-skipped", style: { "line-color": ok, "target-arrow-color": ok } },
    { selector: "edge.wire-shut", style: { "line-color": err, "target-arrow-color": err } },
    { selector: "edge.wire-ungated", style: { "line-color": orch, "target-arrow-color": orch } },
    { selector: "edge.wire-dispatch", style: { "line-color": retriever, "target-arrow-color": retriever } },

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

// A gate's verdict at trace-time t. A group with no `gate:`/`site:` predicate has
// no gate to pass and so gets no verdict at all — showing it the same ✓ as a
// group that actually cleared a corpus gate claims a check that never ran. The
// rest stay undecided until the planner reaches them, and then the annotation
// the model already computed *is* the verdict: the map never second-guesses it.
// The verdict an annotation encodes, with no regard for when it becomes visible.
// Shared with Panel 2's extraction plan, so the two panels cannot disagree about
// which groups the planner ruled in.
function annotationVerdict(annotation) {
  if (!/^(gate:|site:)/.test(annotation || "")) return "ungated";
  return /✗/.test(annotation) ? "shut" : "open";
}

function gateVerdict(annotation, t) {
  const verdict = annotationVerdict(annotation);
  if (verdict === "ungated") return "ungated";
  return planChecked(t) ? verdict : "pending";
}

const GATE_GLYPH = { open: "✓", shut: "✗", pending: "?", skipped: "–", ungated: "↓" };

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

  // Which phase of the run is on screen. Lines are scoped to the step that owns
  // them rather than accumulating: the planner's verdicts show during the check,
  // the retriever's dispatch during the pass, the extractor's results during the
  // merge, and nothing lingers afterwards. Wiring that stays up past its moment
  // is a static diagram of the run drawn over the part of it that is moving.
  const stepNode = (step && step.node) || "";
  const ctx = {
    verdicts: new Map(),
    currentGroups,
    deciding: stepNode === "check_state" || stepNode === "plan_extraction",
    dispatching: stepNode === "extract_branch",
    returning: stepNode === "merge_and_update",
  };

  cy.batch(() => {
    // Gates first: their verdict decides whether anything behind them may light.
    cy.nodes(".gate").forEach((node) => {
      const gid = node.data("group");
      const annotation = annotations[gid] || "";
      let verdict = gateVerdict(annotation, t);
      const retrieving = stateAt(`grpret:${gid}`, t);
      const extracting = stateAt(`grpext:${gid}`, t);
      const ran = stateAt(`grp:${gid}`, t);

      // A group the retriever found no notes for is skipped, not failed.
      if (verdict !== "shut" && verdict !== "pending" && ran === "done" && !hasAnyVariableRun(gid, t)) {
        verdict = "skipped";
      }
      if (verdict === "shut") blocked.add(gid);
      ctx.verdicts.set(gid, verdict);

      node.removeClass("gate-open gate-shut gate-pending gate-skipped gate-ungated pipe-retrieve pipe-extract st-idle st-active st-done");
      node.addClass(`gate-${verdict}`);
      node.addClass(ran === "idle" ? "st-idle" : ran === "active" ? "st-active" : "st-done");
      if (retrieving === "active") node.addClass("pipe-retrieve");
      else if (extracting === "active") node.addClass("pipe-extract");
      node.data("label", GATE_GLYPH[verdict] || "");
      node.data("title", annotation || node.data("title"));
    });

    // The cluster wears its gate's verdict too, so the band reads at a glance:
    // heavier borders are groups still in play, faint pink ones are discarded.
    cy.nodes(".cluster").not(".corpus").forEach((node) => {
      const gid = node.id().replace("grp:", "");
      const verdict = ctx.verdicts.get(gid) || "pending";
      node.removeClass("gate-open gate-shut gate-pending gate-skipped gate-ungated");
      node.addClass(`gate-${verdict}`);
      node.toggleClass("blocked", blocked.has(gid));
      node.toggleClass("current", currentGroups.has(gid));
    });

    cy.nodes(".note").forEach((node) => setState(node, stateAt(node.id(), t)));

    // The corpus box carries both jobs it now stands for — the scanner reading
    // the notes and the characterization drawn from them — so it stays lit
    // across the pair while the discs inside it fill one by one.
    const corpusNode = cy.getElementById(CORPUS_ID);
    if (corpusNode.nonempty()) {
      setState(corpusNode, stateAt(CORPUS_ID, t));
      corpusNode.toggleClass("current", currentBlock === CORPUS_ID);
    }
    // The marker stands for the box, so it lights with it rather than with any
    // one note — the same way an ungated gate lights with its group.
    const corpusMark = cy.getElementById(CORPUS_MARK_ID);
    if (corpusMark.nonempty()) setState(corpusMark, stateAt(CORPUS_ID, t));

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

    // The case is the one label on the drawing that moves — and the one that
    // changes colour, to agree with whichever lines are carrying data right now.
    const phase = casePhase(t) || stepPhase(step);
    const caseNode = cy.getElementById(CASE_ID);
    if (caseNode.nonempty()) {
      caseNode.data("label", phase ? `Case\n${phase.label}` : CASE_REST);
      caseNode.removeClass("tone-orch tone-retr tone-extr tone-err");
      caseNode.addClass(`tone-${phase ? phase.tone : "orch"}`);
      // Naming a phase *is* the case being busy, so the halo and the label agree
      // — including during extraction, which is work the case is waiting on
      // rather than work it is doing and so has no window of its own.
      setState(caseNode, phase ? "active" : stateAt(CASE_ID, t));
      caseNode.toggleClass("current", currentBlock === CASE_ID);
    }

    cy.edges().forEach((edge) => {
      edge.removeClass(
        "st-idle st-done flowing undrawn " +
        "wire-pending wire-open wire-shut wire-skipped wire-ungated wire-dispatch"
      );
      edge.addClass(edgeState(edge, t, ctx));
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

// An edge belongs to one phase of the run and is drawn only while that phase is
// the step on screen. Lines that persisted past their moment turned the map into
// a static diagram of the whole run drawn over the part of it that was moving —
// by the second pass the band was a web of settled wiring the live edges had to
// fight through. `ctx` carries what phase we are in and what the planner decided.
//
// The case-to-gate edge does two jobs at different times, so it is coloured per
// frame: the planner's verdict during the check, the retriever's dispatch during
// the pass. An edge with nothing to say right now is `undrawn`, not dim.
function edgeState(edge, t, ctx) {
  // The group's arc back into the case: the picture of the case being updated,
  // so it is up for the merge step and only for the pass reporting back.
  if (edge.hasClass("grp-out")) {
    if (!ctx.returning) return "undrawn";
    const gid = edge.data("group");
    if (stateAt(`grp:${gid}`, t) !== "done") return "undrawn";
    if (!endedSince(`grp:${gid}`, t, passStartBefore(t))) return "undrawn";
    return stateAt("phase:merge_and_update", t) === "active" ? "flowing" : "st-done";
  }
  if (edge.hasClass("gate-in")) {
    const gid = edge.data("group");
    const ran = stateAt(`grp:${gid}`, t);
    // During a pass: the retriever handing this group its work.
    if (ctx.dispatching && ctx.currentGroups.has(gid)) {
      if (ran === "active") return "flowing wire-dispatch";
      return ran === "done" ? "st-done wire-dispatch" : "undrawn";
    }
    // During a check: the planner reaching out to every group still in play and
    // saying what it decided. The line goes out *first*, dashed and colourless
    // while the answer is still pending, and lands as green or pink when the
    // check settles — so the verdict arrives along the wire rather than being
    // written on the gate before anything reached it.
    if (ctx.deciding && ran !== "done") {
      const verdict = ctx.verdicts.get(gid) || "pending";
      return verdict === "pending" ? "flowing wire-pending" : `st-done wire-${verdict}`;
    }
    return "undrawn";
  }
  // The backbone — the one edge that is always drawn. It carries the corpus
  // description into the case, which is the thing the first planning check
  // reads, so that check is when it flows.
  if (edge.hasClass("spine")) {
    const planning = ["check_state", "plan_extraction"]
      .some((node) => stateAt(`phase:${node}`, t) === "active");
    if (planning) return "flowing";
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
    nodeIds = present.filter(
      (id) => !(SUBSUMED_BY[id] || []).some((by) => present.includes(by))
    );
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

/* --- Panel 2: cards that open on demand ----------------------------------
 *
 * A scan step is seven characterized notes and an extraction pass is a dozen
 * variables across several groups, all expanded — the panel opened on the middle
 * of somebody's summary and the shape of the step was three scrolls away. The
 * containers collapse instead: the step's shape first, one card's contents when
 * asked for.
 *
 * Which cards are open is kept out here rather than in the DOM, because Panel 2
 * re-renders from scratch on every cursor move and a card opened to be talked
 * about should survive the presenter scrubbing the map underneath it.
 */
const openCards = new Set();

function cardSection(key, classes, head, body) {
  return `<details class="node-detail ${classes}" data-card="${esc(key)}"${
    openCards.has(key) ? " open" : ""
  }>${head}<div class="card-body">${body}</div></details>`;
}

// `toggle` does not bubble, so this listens in the capture phase.
function watchCards(root) {
  root.addEventListener(
    "toggle",
    (evt) => {
      const el = evt.target;
      if (!el || el.tagName !== "DETAILS" || !el.dataset.card) return;
      if (el.open) openCards.add(el.dataset.card);
      else openCards.delete(el.dataset.card);
    },
    true
  );
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

  const head = `<summary class="node-head agent-${group.agent || "orchestrator"}">
      <span class="node-head-title">${esc(group.label || group.key)}</span>
      <span class="node-head-id">${vars.length ? `${settled}/${vars.length}` : ""}</span>
      <span class="status-pill status-${esc(group.status)}">${esc(group.status)}</span>
    </summary>`;

  return cardSection(
    group.key,
    "group",
    head,
    `${retrieval}${calls}${cards}${
      group.error ? `<pre class="code">${esc(fmt(group.error))}</pre>` : ""
    }`
  );
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

  const head = `<summary class="node-head agent-${inst.agent || "scanner"}">
      <span class="node-head-title">${esc(inst.label || inst.key)}</span>
      <span class="node-head-id"></span>
      <span class="status-pill status-${status}">${status}</span>
    </summary>`;

  return cardSection(
    inst.key,
    `instance status-${status}`,
    head,
    `${viewNote(r, inst.input, calls)}${other}${collapsible("Result", inst.result, "result")}${
      inst.error ? `<pre class="code">${esc(fmt(inst.error))}</pre>` : ""
    }`
  );
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
//
// One row per verdict rather than one flat run of chips. Flat, every group wore
// a green ✓ unless the *retriever* had already discarded it, so a plan that
// ruled three groups out read as ten groups all passing — and disagreed with the
// map sitting directly above it. The verdict comes from the same annotation the
// gate discs read, and each row is the colour that verdict has on the map.
const PLAN_ROWS = [
  { verdict: "ungated", glyph: "↓", label: "Ungated", hint: "no gate — always extracted" },
  { verdict: "open", glyph: "✓", label: "Passed", hint: "cleared the corpus gate" },
  { verdict: "shut", glyph: "✗", label: "Blocked", hint: "ruled out for this corpus" },
];

function viewGate(prog) {
  const groups = prog.groups || [];
  if (!groups.length) return "";

  const byVerdict = new Map();
  for (const g of groups) {
    const verdict = annotationVerdict(g.annotation || "");
    if (!byVerdict.has(verdict)) byVerdict.set(verdict, []);
    byVerdict.get(verdict).push(g);
  }

  const rows = PLAN_ROWS.filter((row) => byVerdict.has(row.verdict))
    .map((row) => {
      const members = byVerdict.get(row.verdict);
      const chips = members
        .map(
          (g) =>
            `<span class="gate-chip ${row.verdict}">${row.glyph} ${esc(
              g.name || g.group_id
            )}</span>`
        )
        .join("");
      return `<div class="plan-row ${row.verdict}">
        <div class="plan-row-head">
          <b>${row.label}</b><span class="plan-count">${members.length}</span>
          <span class="muted">${esc(row.hint)}</span>
        </div>
        <div class="chips">${chips}</div>
      </div>`;
    })
    .join("");

  return `<div class="headline-fact">
    <b>Extraction plan</b>
    <div class="muted" style="font-size:.74rem">${groups.length} group(s) considered.</div>
    ${rows}
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
        <summary>${NOTE_ICON} In note #${esc(nid)} <span class="muted">${esc(note.note_type || "")}</span></summary>
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

// How tall the variables panel may grow before the map hits its floor, and the
// clamp both the splitter drag and the first open share so neither can push the
// map below MAP_MIN.
//
// The grid's own padding and row gap are not available to either row, so they
// have to come off first — `clientHeight` includes the padding, and subtracting
// MAP_MIN from it alone left the map about 40px short of its floor.
function maxVarsHeight(grid) {
  const cs = getComputedStyle(grid);
  const gap = parseFloat(cs.rowGap) || 0;
  const padding = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
  return Math.max(VARS_MIN, grid.clientHeight - padding - gap - MAP_MIN);
}

function setVarsHeight(grid, px) {
  const clamped = Math.max(VARS_MIN, Math.min(px, maxVarsHeight(grid)));
  grid.style.setProperty("--vars-h", `${Math.round(clamped)}px`);
  return clamped;
}

/* The variables panel is put away by default and pulled up from the bottom edge
 * when it is wanted. The map is the thing being presented; the variable table is
 * a reference the presenter opens to answer a question and closes again, and
 * left open it was taking a third of the map's height for the whole talk.
 *
 * Collapsed is a class on the grid rather than a height, so the row falls back
 * to `auto` and the inline `--vars-h` the splitter wrote is simply not consulted
 * until the panel opens again — reopening lands on the size it was dragged to.
 */
function wireVarsPane() {
  const toggle = document.getElementById("vars-toggle");
  const grid = document.querySelector(".grid");
  if (!toggle || !grid) return;

  const apply = (collapsed) => {
    // Opened before it has ever been dragged, it takes everything the map can
    // spare. Someone reaching for the variable table wants to read the table,
    // and a third of a screen shows a handful of rows out of forty-four; the
    // map is one keystroke away again. Once dragged, that size wins instead —
    // and double-clicking the splitter to forget it comes back here.
    if (!collapsed && !localStorage.getItem(VARS_H_KEY)) {
      setVarsHeight(grid, maxVarsHeight(grid));
    }
    grid.classList.toggle("vars-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.title = collapsed ? "Show variables (V)" : "Hide variables (V)";
    localStorage.setItem(VARS_OPEN_KEY, collapsed ? "0" : "1");
    // Explicitly, rather than leaving it to the map's ResizeObserver: this is
    // the largest shape change the map ever sees, and we know for certain it
    // just happened. The observer stays as the catch-all for the splitter drag
    // and the window, but it is not reliably delivered for a container that
    // resizes because a grid track changed.
    refitMap();
  };
  const toggleOpen = () => apply(!grid.classList.contains("vars-collapsed"));

  // Closed unless this presenter has opened it before.
  apply(localStorage.getItem(VARS_OPEN_KEY) !== "1");

  toggle.addEventListener("click", toggleOpen);
  toggle.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleOpen();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "v" && e.key !== "V") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    toggleOpen();
  });
}

// Drag the boundary between the workflow map and the variables panel. The size
// is written as a pixel `--vars-h` on the grid; the map's ResizeObserver re-fits
// Cytoscape as it changes, so the flowchart redraws live during the drag.
function wireSplitter() {
  const handle = document.getElementById("row-split");
  const grid = document.querySelector(".grid");
  if (!handle || !grid) return;

  const setHeight = (px) => setVarsHeight(grid, px);

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
