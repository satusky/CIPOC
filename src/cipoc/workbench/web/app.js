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
    /* Not Object.assign: assigning null to a dataset property stringifies it,
       so `{annotated: null}` yields data-annotated="null" — present, and matched
       by every [data-annotated] selector. Skip absent keys the same way the
       top-level props above do. */
    else if (key === "dataset") {
      for (const [name, item] of Object.entries(value)) {
        if (item === null || item === undefined || item === false) continue;
        node.dataset[name] = item;
      }
    }
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

/* The lens the control room opens on, and the fallback whenever a stored or
   requested one is unavailable. Confidence is the reading that varies most
   between runs and the one this view was built to scan for; status is a click
   away, and the fill rule marks every valueless variable under any lens. */
const DEFAULT_LENS = "confidence";

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

  truth: new Map(),         // item_id (number) -> expected value (string)
  truthSource: null,        // where the reference file came from, for the chrome

  lens: DEFAULT_LENS,       // which metric the control room colours by
  classFilter: new Set(),   // lens classes the legend has narrowed to; empty = all

  feedback: { variable: {}, group: {}, note: {} },
  feedbackDraft: new Map(),  // kind:id -> unsaved edits, kept across re-renders
  feedbackWritable: false,   // false under a plain static server

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

const CONFIDENCE_LEVELS = ["max", "high", "medium", "low"];

const hasValue = (result) => result.value != null && result.value !== "";

/* ----------------------------------------------------------------- lenses
 *
 * The control room shows ONE metric at a time. The active lens colours both
 * channels of every bubble — the 8px dot and the 1px outline — from its own
 * ramp, so there is a single vocabulary on screen and a single legend.
 *
 * This replaces an earlier scheme where the dot meant confidence for a value
 * that existed and status for one that did not, while a trailing glyph in a
 * fifth grid track meant the ground-truth verdict. That kept confidence and
 * correctness legible at once — a max-confidence wrong answer being the thing
 * worth finding — but it cost two extra legend rows and made every bubble a
 * two-vocabulary decode. The cross-read now costs one click, and both values
 * are still side by side in the tooltip, the detail pane and the table.
 *
 * Each entry answers three questions and nothing else: which classes exist and
 * in what order (`classes`), what a class is called (`label`), and which class
 * a variable falls in (`reading`). Adding a lens is a table entry, not a
 * branch anywhere below.
 *
 * `classes` is a thunk because the accuracy lens reads VERDICTS out of
 * truth.js, which is a later <script> — the binding does not exist yet when
 * this object is built, only by the time anything renders.
 */
const LENSES = {
  confidence: {
    label: "Confidence",
    prefix: "c-",
    classes: () => CONFIDENCE_LEVELS.concat("unrated"),
    labelFor: (cls) => cls,
    /* `structured_data` carries a value but never an extraction, so it has no
       rating to show; it lands in `unrated` beside the valueless results and is
       told apart from them by the fill rule below. */
    reading: (entry) => {
      const level = (entry.result.extraction || {}).presence_confidence;
      return hasValue(entry.result) && CONFIDENCE_LEVELS.includes(level) ? level : "unrated";
    },
    available: () => true,
  },
  status: {
    label: "Status",
    prefix: "s-",
    classes: () => Object.keys(STATUS),
    labelFor: statusLabel,
    reading: (entry) => entry.result.status,
    available: () => true,
  },
  accuracy: {
    label: "Accuracy",
    prefix: "m-",
    classes: () => VERDICTS,
    labelFor: (cls) => VERDICT_LABEL[cls],
    reading: (entry) => verdictFor(entry).verdict,
    available: () => hasTruth(),
    /* The one lens whose indicator is a glyph rather than a dot, and it has to
       be. Its two most consequential classes are `match` and `mismatch` —
       correct and wrong — which are green and red, ΔE 2.2 under deuteranopia:
       indistinguishable. Both carry a value, so the hollow rule cannot separate
       them either. Shape is the channel that survives, and these verdicts
       already had a designed glyph vocabulary. It rides in the indicator column
       the dot would have used, so this costs no width and shifts nothing. */
    mark: (cls) => VERDICT_MARK[cls],
  },
};

const activeLens = () => LENSES[App.lens] || LENSES[DEFAULT_LENS];

/* {key, cls, hollow} for one App.variables entry under a lens.
 *
 * `hollow` stays exactly what it has always meant — this variable holds no
 * value — and is deliberately NOT redefined as "the lens has no reading here".
 * It is already correct under all three lenses (no value to rate, no value to
 * compare), and it is the secondary encoding two of the three ramps need: it
 * is what separates `missed` from `mismatch` and `not_found` from
 * `structured_data`, pairs that colour alone does not resolve under CVD. */
function indicatorFor(entry, lensId) {
  const lens = LENSES[lensId] || LENSES[DEFAULT_LENS];
  const key = lens.reading(entry);
  return {
    key,
    cls: lens.prefix + key,
    hollow: !hasValue(entry.result),
    mark: lens.mark ? lens.mark(key) : null,
  };
}

/* The status dot for the table and the detail pane. Those two show every axis
   at once in labelled columns, so their dot is fixed on status and does NOT
   follow the lens — the lens is the control room's way of choosing what to
   show, and a table has no such shortage of room. */
const statusDot = (result) =>
  "s-" + result.status + (hasValue(result) ? "" : " hollow");

/* Counts per class for the active lens, in the lens's own order, dropping the
   classes this run never produced — the legend then shows what actually
   happened rather than the vocabulary in the abstract. */
function lensTally(lensId) {
  const lens = LENSES[lensId] || LENSES[DEFAULT_LENS];
  const counts = new Map();
  for (const entry of App.variables) {
    const { key, hollow } = indicatorFor(entry, lensId);
    const row = counts.get(key) || { key, count: 0, valued: 0 };
    row.count += 1;
    if (!hollow) row.valued += 1;
    counts.set(key, row);
  }
  return lens.classes()
    .filter((cls) => counts.has(cls))
    .map((cls) => ({
      ...counts.get(cls),
      label: lens.labelFor(cls),
      cls: lens.prefix + cls,
      mark: lens.mark ? lens.mark(cls) : null,
    }));
}

/* The legend narrows the grid to one or more classes; it ANDs with the text
   filter rather than replacing it. */
function passesLens(entry) {
  if (!App.classFilter.size) return true;
  return App.classFilter.has(indicatorFor(entry, App.lens).key);
}

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

/* ----------------------------------------------------------------- theme */

const THEME_KEY = "cipoc-theme";
const storedTheme = () => { try { return localStorage.getItem(THEME_KEY); } catch (err) { return null; } };

const Theme = {
  current: () => (document.documentElement.dataset.theme === "light" ? "light" : "dark"),

  apply(theme) {
    document.documentElement.dataset.theme = theme;
    const next = theme === "light" ? "dark" : "light";
    const button = $("#theme-toggle");
    button.textContent = theme === "light" ? "\u263D" : "\u2600";
    button.title = "Switch to " + next + " theme";
    button.setAttribute("aria-label", button.title);
  },

  toggle() {
    const next = Theme.current() === "light" ? "dark" : "light";
    try { localStorage.setItem(THEME_KEY, next); } catch (err) { /* blocked */ }
    Theme.apply(next);
  },

  /* The head script has already painted the resolved theme; this only labels
   * the control and keeps following the OS until the user states a
   * preference of their own. */
  init() {
    Theme.apply(Theme.current());
    $("#theme-toggle").addEventListener("click", Theme.toggle);
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
      if (!storedTheme()) Theme.apply(e.matches ? "light" : "dark");
    });
  },
};

/* --------------------------------------------------------------- routing */

/* Clicking the entity that is already open closes the panel; clicking a
 * different one keeps it open and swaps the contents. Cross-links inside the
 * panel navigate with show() so following one never closes it. */
function select(kind, id) {
  if (isSelected(kind, id)) clearSelection();
  else show(kind, id);
}

function show(kind, id) {
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

/* The lens is remembered the way the theme is. A stored `accuracy` is only
   honoured when a reference file is actually loaded — restoring it without one
   would select a tab that is not on screen and paint every bubble `untested`;
   it falls back to DEFAULT_LENS instead. */
/* Bumped when the default changed: the old key had been written on every load
   by the restore below, so a stored `status` recorded "this page has been
   opened", not "this reader chose status", and would have pinned every
   existing browser to the old default forever. */
const LENS_KEY = "cipoc-lens-v2";
const storedLens = () => { try { return localStorage.getItem(LENS_KEY); } catch (err) { return null; } };

/* `remember` is false for the restore at startup — only a choice gets stored,
   so the remembered lens stays a preference rather than a snapshot of whatever
   the default happened to be. */
function setLens(lens, remember = true) {
  if (!LENSES[lens] || !LENSES[lens].available()) lens = DEFAULT_LENS;
  App.lens = lens;
  if (remember) { try { localStorage.setItem(LENS_KEY, lens); } catch (err) { /* blocked */ } }
  App.classFilter.clear();
  for (const tab of document.querySelectorAll(".lens-tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.lens === lens));
    tab.hidden = !LENSES[tab.dataset.lens].available();
  }
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
  const flags = ((App.raw.report || {}).flags || []).length;
  const acc = hasTruth() ? caseAccuracy() : null;

  /* The accuracy lens exists only when there is something to compare against;
     with no reference file the toolbar is unchanged. */
  for (const tab of document.querySelectorAll(".lens-tab")) {
    tab.hidden = !LENSES[tab.dataset.lens].available();
  }

  /* The per-status counts that used to sit here (`24 extracted`, `2
     unresolved`) are now the status legend, which carries every status rather
     than the two this line had room for. */
  clear($("#case-summary")).append(
    h("span", {}, h("b", { text: String(App.notes.size) }), " notes"),
    h("span", {}, h("b", { text: String(App.variables.length) }), " variables"),
    h("span", { class: flags ? "chip warn" : "chip" },
      flags ? flags + " review flag" + (flags === 1 ? "" : "s") : "no review flags")
  );
  /* Appended separately rather than as a conditional argument above:
     Element.append() renders a null argument as the literal text "null",
     unlike h(), which skips falsy children. */
  const annotations = annotationCount();
  if (annotations) {
    $("#case-summary").append(h("span", { class: "chip on",
      text: annotations + " annotation" + (annotations === 1 ? "" : "s") }));
  }
  if (acc && acc.tested) {
    $("#case-summary").append(hoverable(h("span", {
      class: "chip " + (acc.correct === acc.tested ? "good" : "warn"),
      text: acc.correct + "/" + acc.tested + " correct",
    }), accuracyTooltip));
  }

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
  Theme.init();
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
  for (const tab of document.querySelectorAll(".lens-tab")) {
    tab.addEventListener("click", () => setLens(tab.dataset.lens));
  }
  $("#truth-file").addEventListener("change", onTruthFile);
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
  /* A reference file is optional and independent of the run: absence is the
     normal case and leaves every view exactly as it was. loadTruth() is in
     truth.js and never throws — it returns false when nothing is served. */
  await loadTruth();
  await loadFeedback();
  renderChrome();
  setLens(storedLens() || App.lens, false);
  setMode(App.mode);
  setView(App.view);
  $("#boot").hidden = true;
}

document.addEventListener("DOMContentLoaded", boot);
