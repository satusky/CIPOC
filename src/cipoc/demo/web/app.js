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
function buildMap(graph) {
  if (graph.coarse_map && Object.keys(graph.coarse_map).length) {
    COARSE = graph.coarse_map;
    rebuildCoarseMembers();
  }
  const agentColors = graph.agent_colors || {};
  applyAgentColors(agentColors);
  renderLegend(agentColors);

  const nodes = graph.elements.nodes.map((n) => ({
    data: { ...n.data, baseLabel: n.data.label },
  }));
  const edges = graph.elements.edges.map((e) => ({ data: { ...e.data } }));

  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: { nodes, edges },
    // The chart's own style already colors nodes/edges by agent and kind; we only
    // layer run-state (visited / current / active / traversed) on top of it.
    style: [...(graph.style || []), ...stateStyles()],
    layout: graph.layout || { name: "breadthfirst", directed: true },
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
    { selector: "edge.dim", style: { opacity: 0.18 } },
    {
      selector: "edge.traversed",
      style: {
        opacity: 1,
        width: 4,
        "line-color": "#5b4bc0",
        "target-arrow-color": "#5b4bc0",
        "z-index": 15,
      },
    },
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

function renderDetail(view) {
  const snap = view.snapshot;
  const step = view.step;

  // A collapsed fan-out step (e.g. "Characterize notes") shows one card per
  // instance instead of one merged card per map node.
  if (!focusBlock && step && step.fanout) {
    renderFanoutDetail(step, snap);
    return;
  }

  let title, subtitle, agent, nodeIds;
  if (focusBlock) {
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
    nodeIds = nodeIds.filter((id) => snap.details[id]);
  } else {
    title = "Run start";
    subtitle = "";
    agent = null;
    nodeIds = [];
  }

  els.detailNode.textContent = step && step.map_node_id ? step.map_node_id : "";

  const parts = [];
  parts.push(`<div class="detail-headline">
      <h3 class="detail-title">${esc(title)}</h3>
      <div class="detail-meta">
        ${subtitle ? `<span class="muted">${esc(subtitle)}</span>` : ""}
        ${agent ? `<span class="chip agent-${agent}">${esc(agent)}</span>` : ""}
        ${focusBlock ? `<span class="chip">click background to unpin</span>` : ""}
      </div>
    </div>`);

  if (nodeIds.length === 0) {
    parts.push(`<p class="empty">No model activity captured for this step.</p>`);
  } else {
    for (const id of nodeIds) parts.push(renderNodeDetail(id, snap.details[id], snap));
  }
  els.detail.innerHTML = parts.join("");
}

function renderNodeDetail(id, detail, snap) {
  const status = detail.status || "idle";
  const count =
    detail.count > 1 ? `<span class="summary-tag">×${detail.count} instances</span>` : "";
  const headline = componentHeadline(id, detail, snap);
  const calls = (detail.llm_calls || []).map(renderLLMCall).join("");

  return `<section class="node-detail">
    <div class="detail-meta" style="margin:.2rem 0 .5rem">
      <span class="chip agent-${detail.agent || "orchestrator"}">${esc(id)}</span>
      <span class="status-pill status-${status}">${status}</span>
      ${count}
    </div>
    ${headline}
    ${calls || ""}
    ${collapsible("Task input", detail.input, "input")}
    ${collapsible("Task result", detail.result, "result")}
    ${
      detail.error
        ? `<pre class="code">${esc(fmt(detail.error))}</pre>`
        : ""
    }
  </section>`;
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
  const calls = (inst.llm_calls || []).map(renderLLMCall).join("");

  return `<section class="node-detail instance status-${status}">
    <div class="detail-meta" style="margin:.2rem 0 .5rem">
      <span class="chip agent-${inst.agent || "scanner"}">${esc(inst.label || inst.key)}</span>
      <span class="status-pill status-${status}">${status}</span>
    </div>
    ${viewNote(r, inst.input)}
    ${calls || ""}
    ${collapsible("Result", inst.result, "result")}
    ${inst.error ? `<pre class="code">${esc(fmt(inst.error))}</pre>` : ""}
  </section>`;
}

// One note's characterization as three fixed slots — summary, detected concepts,
// and cancer mentions. Rendering all three always makes the card a stable
// skeleton: in live mode each slot shows a "pending…" placeholder until its part
// of the result streams in (keyed on the field's presence, since the scanner
// sub-agent's values arrive summary → concepts → mentions), then fills in place.
// (componentHeadline only dispatches one view; a note has all three, so compose
// them here.)
function viewNote(r, input) {
  const summary =
    "summary" in r ? viewSummary(r, { note: input }) : pendingSlot("Note summary");
  const concepts =
    "concepts" in r ? viewConcepts(r.concepts || {}) : pendingSlot("Concepts detected");
  let mentions;
  if ("cancer_mentions" in r) {
    const m = r.cancer_mentions || [];
    mentions = m.length
      ? viewCancerMentions(m, r.cancer_status)
      : emptySlot("Cancer mentions", "none found");
  } else {
    mentions = pendingSlot("Cancer mentions");
  }
  return summary + concepts + mentions;
}

function pendingSlot(label) {
  return `<div class="headline-fact pending"><b>${esc(label)}</b>
    <span class="pending-dot">pending…</span></div>`;
}

function emptySlot(label, note) {
  return `<div class="headline-fact"><b>${esc(label)}</b>
    <span class="muted" style="font-size:.78rem"> ${esc(note)}</span></div>`;
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

  if (r.note_corpus_descriptors) return viewCorpus(r);
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
function viewSummary(r, inp) {
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
  </div>`;
}

function viewConcepts(concepts) {
  const chips = Object.entries(concepts)
    .map(([name, c]) => {
      const present = c && c.presence;
      const conf = c && c.confidence ? ` <span class="conf">${esc(c.confidence)}</span>` : "";
      return `<span class="concept-chip ${present ? "present" : "absent"}">${present ? "✓" : "○"} ${esc(name)}${present ? conf : ""}</span>`;
    })
    .join("");
  return `<div class="headline-fact"><b>Concepts detected</b><div class="chips">${chips}</div></div>`;
}

function viewCancerMentions(mentions, statuses) {
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
  return `<div class="headline-fact"><b>Cancer mentions${st}</b>${cards}</div>`;
}

// --- characterize: corpus-level descriptors -----------------------------
function viewCorpus(r) {
  const d = r.note_corpus_descriptors || {};
  const facts = r.case_facts || {};
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
  const factRows = Object.entries(facts)
    .filter(([, v]) => v != null && v !== "")
    .map(([k, v]) => stat(k.replace(/_/g, " "), fmt(v)));
  return `<div class="headline-fact">
    <b>Corpus characterization</b>
    <div class="stat-grid">${rows.join("")}</div>
    ${factRows.length ? `<div class="stat-sub muted">Case facts</div><div class="stat-grid">${factRows.join("")}</div>` : ""}
  </div>`;
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
function extractionRow(v) {
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
      <b>${esc(String(v.item_id))}</b>
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
  const rows = vars.map(extractionRow).join("");
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

function renderLLMCall(call) {
  const usage = call.usage || {};
  const tok =
    usage.total_tokens != null
      ? `${usage.total_tokens} tok (${usage.input_tokens ?? "?"} in / ${usage.output_tokens ?? "?"} out)`
      : "";
  const messages = (call.prompt_messages || [])
    .map(
      (m) => `<div class="msg">
        <div class="role">${esc(m.role || "message")}</div>
        <pre class="code plain">${esc(fmt(m.content))}</pre>
      </div>`
    )
    .join("");
  const reasoning = call.reasoning
    ? `<div class="msg"><div class="role">reasoning</div><pre class="code plain">${esc(call.reasoning)}</pre></div>`
    : "";
  const response = `<div class="msg"><div class="role">response</div><pre class="code">${esc(fmt(call.response))}</pre></div>`;
  const err = call.error ? `<div class="msg"><div class="role">error</div><pre class="code">${esc(call.error)}</pre></div>` : "";

  return `<details class="block llm-call">
    <summary><span class="llm-head" style="all:unset;display:flex;flex:1;gap:.5rem;align-items:center">
      🧠 Model call <span class="muted">${esc(call.model || "")}</span>
      <span class="tok">${tok}</span></span></summary>
    <div class="block-body">${messages}${reasoning}${response}${err}</div>
  </details>`;
}

function collapsible(label, value, tag) {
  if (value === null || value === undefined) return "";
  const empty = typeof value === "object" && Object.keys(value).length === 0;
  return `<details class="block">
    <summary>${esc(label)}${empty ? `<span class="summary-tag">empty</span>` : ""}</summary>
    <div class="block-body"><pre class="code plain">${esc(fmt(value))}</pre></div>
  </details>`;
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
