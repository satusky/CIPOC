"use strict";
/* CIPOC workbench — shell, data indexing, routing and shared helpers.
 *
 * Input is one JSON file: the orchestrator's final output state. Nothing here
 * re-implements orchestration logic; every status, reason and value is read
 * from the state as recorded. The one thing the UI derives is *presentation*
 * grouping (waves, per-group roll-ups), which is layout, not a decision.
 *
 * All model- and note-produced text reaches the DOM through h(), which sets
 * textContent — never innerHTML — so nothing in the state can inject markup.
 */

const STATE_URL = "case_state.json";

/* ------------------------------------------------------------- DOM helpers */

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") throw new Error("h(): raw HTML is not allowed");
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const $ = (sel) => document.querySelector(sel);
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };

/* --------------------------------------------------------------- run state */

const App = {
  raw: null,
  view: "variables",
  mode: "control",
  grouped: true,
  selection: null,          // {kind: 'note'|'group'|'variable', id}
  noteFilter: "",
  varFilter: "",
  sort: { key: "item_id", dir: 1 },

  notes: new Map(),         // note_id (string) -> ProcessedClinicalNote
  groups: [],               // TargetGroup[], in configured order
  groupById: new Map(),
  results: new Map(),       // item_id (number) -> CaseVariableResult
  variables: [],            // flattened {item_id, name, group_id, result}
  groupOfItem: new Map(),
  exchanges: {},
  noteSelections: {},       // note_selection, keyed group:<id>
  attempts: {},
};

const STATUS = {
  extracted:       { label: "Extracted" },
  structured_data: { label: "Structured" },
  not_found:       { label: "Not found" },
  not_applicable:  { label: "Not applicable" },
  blocked:         { label: "Blocked" },
  error:           { label: "Error" },
  pending:         { label: "Pending" },
};
const statusLabel = (s) => (STATUS[s] || { label: s }).label;

/* ------------------------------------------------------------------ index */

function indexState(raw) {
  App.raw = raw;

  App.notes = new Map(Object.entries(raw.note_corpus || {}));

  App.groups = raw.target_variables || [];
  App.groupById = new Map(App.groups.map((g) => [g.group_id, g]));

  App.results = new Map(
    Object.entries(raw.variable_results || {}).map(([k, v]) => [Number(k), v])
  );

  App.variables = [];
  App.groupOfItem = new Map();
  for (const group of App.groups) {
    for (const variable of group.variables || []) {
      App.groupOfItem.set(variable.item_id, group.group_id);
      App.variables.push({
        item_id: variable.item_id,
        name: variable.name,
        group_id: group.group_id,
        group_name: group.name,
        result: App.results.get(variable.item_id) || { item_id: variable.item_id, status: "pending" },
      });
    }
  }

  App.exchanges = raw.llm_exchanges || {};
  App.noteSelections = raw.note_selection || {};
  App.attempts = raw.variable_attempts || {};
}

/* Lookups over the recorded side-channels. All tolerate absence: a state dump
 * that predates prompt capture renders every view, minus those panels. */
const exchangesFor = (key) => App.exchanges[key] || [];
const noteExchanges = (noteId) => exchangesFor("note:" + noteId);
const groupExchanges = (groupId) => exchangesFor("group:" + groupId);
const variableKey = (itemId) => "group:" + App.groupOfItem.get(itemId) + "/variable:" + itemId;
const variableExchanges = (itemId) => exchangesFor(variableKey(itemId));
const variableAttempts = (itemId) => App.attempts[variableKey(itemId)] || [];
const noteSelection = (groupId) => App.noteSelections["group:" + groupId] || null;

const hasCapture = () => Object.keys(App.exchanges).length > 0;

/* Which groups considered, rejected or selected a note. */
function groupsTouchingNote(noteId) {
  const id = String(noteId);
  const out = [];
  for (const group of App.groups) {
    const sel = noteSelection(group.group_id);
    if (!sel) continue;
    const hit = (list) => (list || []).some((n) => String(n) === id);
    let role = null;
    let reason = null;
    if (hit(sel.selected_note_ids)) role = "selected";
    else if (Object.prototype.hasOwnProperty.call(sel.filtered_out || {}, id)) {
      role = "filtered out";
      reason = sel.filtered_out[id];
    } else if (hit(sel.candidate_note_ids)) {
      role = "not selected";
      reason = "Survived the note filter but the retriever did not judge it relevant.";
    }
    if (role) out.push({ group, role, reason });
  }
  return out;
}

/* Variables whose evidence cites a note. */
function variablesCitingNote(noteId) {
  const id = String(noteId);
  return App.variables.filter((v) =>
    (((v.result.extraction || {}).spans) || []).some((s) => String(s.note_id) === id)
  );
}

/* ------------------------------------------------------ group roll-up state
 * A roll-up of recorded per-variable statuses. No gate is re-evaluated here;
 * the exclusion reason is read off the variables the orchestrator stamped.
 */
function groupState(group) {
  const items = (group.variables || []).map(
    (v) => App.results.get(v.item_id) || { status: "pending" }
  );
  const counts = {};
  for (const r of items) counts[r.status] = (counts[r.status] || 0) + 1;

  const total = items.length;
  const na = counts.not_applicable || 0;
  const unresolved = (counts.error || 0) + (counts.blocked || 0);
  const coded = (counts.extracted || 0) + (counts.structured_data || 0);

  let kind = "ran";
  let label = "complete";
  if (total > 0 && na === total) {
    kind = "excluded";
    label = "excluded";
  } else if (counts.pending) {
    label = "pending";
  } else if (unresolved) {
    kind = "problem";
    label = unresolved + " unresolved";
  } else if (coded === 0) {
    label = "nothing coded";
  }

  let reason = null;
  if (kind === "excluded") {
    const first = items.find((r) => r.reason);
    reason = first ? first.reason : null;
  }

  return { counts, total, coded, unresolved, kind, label, reason, items };
}

/* Waves: `initial` runs first; `dependent` groups follow. `depends_on` is
 * honoured when the state carries it, so a config that adds it renders a third
 * wave without a code change here. */
function waves() {
  const buckets = new Map();
  for (const group of App.groups) {
    const dependent = group.stage === "dependent";
    const gated = dependent && (group.depends_on || []).length > 0;
    const index = !dependent ? 0 : gated ? 2 : 1;
    if (!buckets.has(index)) buckets.set(index, []);
    buckets.get(index).push(group);
  }
  const labels = ["initial", "dependent", "dependent · after prerequisites"];
  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([index, groups], position) => ({
      title: "wave " + (position + 1) + " · " + labels[index],
      groups,
    }));
}

/* ---------------------------------------------------------------- tooltip */

const Tooltip = {
  node: null,
  show(target, build) {
    if (!this.node) this.node = $("#tooltip");
    clear(this.node).append(build());
    this.node.hidden = false;
    const box = target.getBoundingClientRect();
    const own = this.node.getBoundingClientRect();
    let left = box.left;
    if (left + own.width > window.innerWidth - 12) left = window.innerWidth - own.width - 12;
    let top = box.bottom + 8;
    if (top + own.height > window.innerHeight - 12) top = box.top - own.height - 8;
    this.node.style.left = Math.max(8, left) + "px";
    this.node.style.top = Math.max(8, top) + "px";
  },
  hide() { if (this.node) this.node.hidden = true; },
};

function hoverable(node, build) {
  node.addEventListener("mouseenter", () => Tooltip.show(node, build));
  node.addEventListener("focus", () => Tooltip.show(node, build));
  node.addEventListener("mouseleave", () => Tooltip.hide());
  node.addEventListener("blur", () => Tooltip.hide());
  return node;
}

/* --------------------------------------------------------------- routing */

function select(kind, id) {
  App.selection = { kind, id: String(id) };
  renderDetail();
  markSelection();
}

function clearSelection() {
  App.selection = null;
  $("#detail").hidden = true;
  Tooltip.hide();
  markSelection();
}

const isSelected = (kind, id) =>
  !!App.selection && App.selection.kind === kind && App.selection.id === String(id);

function markSelection() {
  for (const node of document.querySelectorAll("[data-entity]")) {
    const index = node.dataset.entity.indexOf(":");
    const kind = node.dataset.entity.slice(0, index);
    const id = node.dataset.entity.slice(index + 1);
    if (isSelected(kind, id)) node.setAttribute("aria-current", "true");
    else node.removeAttribute("aria-current");
  }
}

function setView(view) {
  App.view = view;
  for (const tab of document.querySelectorAll(".view-tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  }
  $("#view-notes").hidden = view !== "notes";
  $("#view-variables").hidden = view !== "variables";
  render();
}

function setMode(mode) {
  App.mode = mode;
  for (const tab of document.querySelectorAll(".mode-tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === mode));
  }
  $("#control").hidden = mode !== "control";
  $("#table-wrap").hidden = mode !== "table";
  $("#group-toggle").parentElement.style.display = mode === "table" ? "" : "none";
  render();
}

function render() {
  if (App.view === "notes") renderNotes();
  else if (App.mode === "control") renderControlRoom();
  else renderTable();
  markSelection();
}

/* --------------------------------------------------------------- chrome */

function renderChrome() {
  const counts = {};
  for (const v of App.variables) counts[v.result.status] = (counts[v.result.status] || 0) + 1;
  const flags = ((App.raw.report || {}).flags || []).length;

  clear($("#case-summary")).append(
    h("span", {}, h("b", { text: String(App.notes.size) }), " notes"),
    h("span", {}, h("b", { text: String(App.variables.length) }), " variables"),
    h("span", {}, h("b", { text: String(counts.extracted || 0) }), " extracted"),
    h("span", {}, h("b", { text: String((counts.error || 0) + (counts.blocked || 0)) }), " unresolved"),
    h("span", { class: flags ? "chip warn" : "chip" },
      flags ? flags + " review flag" + (flags === 1 ? "" : "s") : "no review flags")
  );

  const facts = App.raw.case_facts || {};
  const order = ["primary_site", "gross_primary_site", "histology", "behavior", "sex", "date_of_diagnosis"];
  const rail = clear($("#facts"));
  rail.append(h("span", { class: "faint", style: "font-size:11px;text-transform:uppercase;letter-spacing:.08em" },
    "Case facts"));
  for (const key of order) {
    const value = facts[key];
    rail.append(h("span", { class: "fact" + (value ? "" : " unset") },
      h("span", { class: "k", text: key.replace(/_/g, " ") }),
      h("span", { class: "v", text: value == null || value === "" ? "unknown" : String(value) })
    ));
  }
  if (App.raw.fatal_blocker) {
    rail.append(h("span", { class: "chip bad", text: "fatal: " + App.raw.fatal_blocker }));
  }
  if (!hasCapture()) {
    rail.append(h("span", {
      class: "chip",
      title: "This state dump carries no llm_exchanges channel, so prompt and response panels are unavailable.",
      text: "no prompt capture",
    }));
  }
}

/* ------------------------------------------------------------------ boot */

function wire() {
  for (const tab of document.querySelectorAll(".view-tab")) {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  }
  for (const tab of document.querySelectorAll(".mode-tab")) {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  }
  $("#detail-close").addEventListener("click", clearSelection);
  $("#note-filter").addEventListener("input", (e) => {
    App.noteFilter = e.target.value.trim().toLowerCase();
    renderNotes();
  });
  $("#var-filter").addEventListener("input", (e) => {
    App.varFilter = e.target.value.trim().toLowerCase();
    render();
  });
  $("#group-toggle").addEventListener("change", (e) => {
    App.grouped = e.target.checked;
    render();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") clearSelection();
  });
  window.addEventListener("scroll", () => Tooltip.hide(), true);
}

function bootError(err) {
  const boot = $("#boot");
  boot.className = "boot error";
  clear(boot).append(
    h("div", {},
      h("p", { text: "Could not load " + STATE_URL + " — " + err.message }),
      h("p", { class: "muted" },
        "Browsers block file:// fetches. Serve this directory, e.g. ",
        h("code", { text: "python3 -m http.server -d src/cipoc/workbench/web 8000" }),
        " then open ",
        h("code", { text: "http://127.0.0.1:8000" }),
        "."))
  );
}

async function boot() {
  wire();
  let raw;
  try {
    const response = await fetch(STATE_URL, { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    raw = await response.json();
  } catch (err) {
    bootError(err);
    return;
  }
  indexState(raw);
  renderChrome();
  setMode(App.mode);
  setView(App.view);
  $("#boot").hidden = true;
}

document.addEventListener("DOMContentLoaded", boot);
