"use strict";
/* CIPOC workbench — shell, data indexing, routing and shared helpers.
 *
 * Input is one JSON file: a canonical OrchestratorRunResult. Nothing here
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
   requested one is unavailable. There is only ever one other (accuracy, and
   only with a reference file loaded), so most runs never show a toggle. */
const DEFAULT_LENS = "confidence";

/* --------------------------------------------------------------- run state */

const App = {
  schemaVersion: null,
  run: {},
  case: {},
  inputs: {},
  corpus: {},
  observability: {},
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
  noteDigests: new Map(),
  descriptors: {},
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
 * A bubble's dot and outline answer whichever question the result actually
 * raises: for a coded value, how far to trust it (confidence); for an empty
 * one, why it is empty (status), drawn hollow so the two never read alike.
 * That is one reading, not two — `extracted` and `not found` are not rival
 * answers to the same question, and no variable is ever eligible for both
 * halves. Splitting them across two tabs made the reader toggle to learn
 * something the single dot had already told them.
 *
 * The lens picks between *metrics that genuinely compete for the same pixel*:
 * this one and, when a reference file is loaded, the ground-truth verdict.
 * With no reference file there is nothing to choose and the toggle is hidden.
 *
 * Each entry answers three questions and nothing else: which classes exist and
 * in what order (`classes`), what a class is called (`labelFor`), and which
 * class a variable falls in (`reading`). `classFor` maps a class to its CSS
 * class — per class, not per lens, because this lens draws from two ramps.
 *
 * `classes` is a thunk because the accuracy lens reads VERDICTS out of
 * truth.js, which is a later <script> — the binding does not exist yet when
 * this object is built, only by the time anything renders.
 */
const LENSES = {
  confidence: {
    label: "Confidence",
    /* Confidence levels first, best to worst, then the reasons a variable has
       no value to rate. Zero-count classes are dropped by lensTally, so the
       legend only ever spells out the halves this run actually produced. */
    classes: () => CONFIDENCE_LEVELS.concat(Object.keys(STATUS)),
    classFor: (cls) => (CONFIDENCE_LEVELS.includes(cls) ? "c-" : "s-") + cls,
    /* Lower-cased: the legend sets the confidence terms beside the status ones
       in one row, and STATUS.label is title-cased for the table's own column. */
    labelFor: (cls) => (CONFIDENCE_LEVELS.includes(cls) ? cls : statusLabel(cls).toLowerCase()),
    /* `structured_data` carries a value but never an extraction, so it has no
       rating to show and falls through to its status colour — filled, because
       it does hold a value. */
    reading: (entry) => {
      const level = (entry.result.extraction || {}).presence_confidence;
      return hasValue(entry.result) && CONFIDENCE_LEVELS.includes(level)
        ? level : entry.result.status;
    },
    /* What a screen reader says for a bubble. Separate from labelFor because
       the legend wants the bare term and speech wants the sentence. */
    aria: (cls) => (CONFIDENCE_LEVELS.includes(cls) ? cls + " confidence" : statusLabel(cls)),
    available: () => true,
  },
  accuracy: {
    label: "Accuracy",
    classes: () => VERDICTS,
    classFor: (cls) => "m-" + cls,
    labelFor: (cls) => VERDICT_LABEL[cls],
    reading: (entry) => verdictFor(entry).verdict,
    aria: (cls) => "verdict " + VERDICT_LABEL[cls],
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

/* True only when there is a real choice to offer. */
const lensChoices = () => Object.keys(LENSES).filter((id) => LENSES[id].available());

const activeLens = () => LENSES[App.lens] || LENSES[DEFAULT_LENS];

/* {key, cls, hollow} for one App.variables entry under a lens.
 *
 * `hollow` means exactly one thing under both lenses — this variable holds no
 * value. It is what tells the confidence half of a reading from the status
 * half at a glance, and it is the secondary encoding the ramps need where
 * colour alone fails under CVD: `missed` vs `mismatch`, `not_found` vs
 * `structured_data`. */
function indicatorFor(entry, lensId) {
  const lens = LENSES[lensId] || LENSES[DEFAULT_LENS];
  const key = lens.reading(entry);
  return {
    key,
    cls: lens.classFor(key),
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
      cls: lens.classFor(cls),
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

function normalizeRunResult(result) {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("Expected an OrchestratorRunResult JSON object.");
  }
  if (result.schema_version !== "1.0") {
    const version = result.schema_version == null ? "missing" : String(result.schema_version);
    throw new Error("Unsupported schema_version " + version + "; expected 1.0.");
  }

  for (const domain of ["run", "case", "inputs", "corpus", "observability"]) {
    if (!result[domain] || typeof result[domain] !== "object" || Array.isArray(result[domain])) {
      throw new Error("OrchestratorRunResult is missing the " + domain + " object.");
    }
  }
  return result;
}

function indexState(raw) {
  const result = normalizeRunResult(raw);
  App.schemaVersion = result.schema_version;
  App.run = result.run;
  App.case = result.case;
  App.inputs = result.inputs;
  App.corpus = result.corpus;
  App.observability = result.observability;

  App.notes = new Map(Object.entries(App.corpus.note_corpus || {}));
  App.noteDigests = new Map(Object.entries(App.corpus.note_digests || {}));
  App.descriptors = App.corpus.note_corpus_descriptors || {};

  App.groups = App.inputs.target_variables || [];
  App.groupById = new Map(App.groups.map((g) => [g.group_id, g]));

  App.results = new Map(
    Object.entries(App.case.variable_results || {}).map(([k, v]) => [Number(k), v])
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

  App.exchanges = App.observability.llm_exchanges || {};
  App.noteSelections = App.case.note_selection || {};
  App.attempts = App.observability.variable_attempts || {};
}

/* Lookups over the recorded observability channels. */
const exchangesFor = (key) => App.exchanges[key] || [];
const noteExchanges = (noteId) => exchangesFor("note:" + noteId);
const groupExchanges = (groupId) => exchangesFor("group:" + groupId);
const variableKey = (itemId) => "group:" + App.groupOfItem.get(itemId) + "/variable:" + itemId;
const variableExchanges = (itemId) => exchangesFor(variableKey(itemId));
const variableAttempts = (itemId) => App.attempts[variableKey(itemId)] || [];
const noteSelection = (groupId) => App.noteSelections["group:" + groupId] || null;

const hasCapture = () => App.observability.llm_content_captured === true;

const NOTE_SELECTION_REJECTION_MESSAGES = {
  note_type_mismatch: "Note type did not match the configured note filter.",
  cancer_status_mismatch: "Cancer status did not match the configured note filter.",
  missing_or_invalid_date: "Note date was missing or invalid.",
  outside_date_window: "Note date was outside the configured date window.",
};

const NOTE_SELECTION_UNEVALUATED_MESSAGES = {
  keyword_filter_disabled: "Keyword filtering was configured but was not evaluated.",
  temporal_anchor_unavailable: "The date-window check was not evaluated because no temporal anchor was available.",
};

function presentationMessage(messages, code) {
  return messages[code] || String(code).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()) + ".";
}

function rejectionMessage(codes) {
  return (codes || []).map((code) =>
    presentationMessage(NOTE_SELECTION_REJECTION_MESSAGES, code)).join(" ");
}

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
    else if (hit(sel.discarded_note_ids)) {
      role = "discarded proposal";
      reason = "The retriever proposed this ID even though it was not offered.";
    }
    else if (Object.prototype.hasOwnProperty.call(sel.rejected_note_ids || {}, id)) {
      role = "filtered out";
      reason = rejectionMessage(sel.rejected_note_ids[id]);
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
  syncLensTabs();
  render();
}

/* The toggle is chrome for a choice; with one lens available there is no
   choice, so the whole control goes away rather than sitting there as a
   single pressed button. */
function syncLensTabs() {
  for (const tab of document.querySelectorAll(".lens-tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.lens === App.lens));
    tab.hidden = !LENSES[tab.dataset.lens].available();
  }
  $(".lenses").hidden = lensChoices().length < 2;

  /* Tells styles.css whether to reserve the bubble's leading glyph column.
     Keyed on lens AVAILABILITY, not on App.lens: with a reference loaded the
     column is reserved under both lenses, so toggling between them still shifts
     no label sideways — the invariant the fixed track was there to protect.
     With no reference the accuracy lens cannot be chosen at all, and the column
     would be 20px of dead indent on every row.

     `delete` rather than assigning null, which would write the string "null"
     and leave the attribute present — and [data-marks] matches on presence. */
  if (lensChoices().some((id) => LENSES[id].mark)) {
    document.documentElement.dataset.marks = "1";
  } else {
    delete document.documentElement.dataset.marks;
  }
}

function render() {
  if (App.view === "notes") renderNotes();
  else if (App.mode === "control") renderControlRoom();
  else renderTable();
  markSelection();
}

/* --------------------------------------------------------------- chrome */

function renderChrome() {
  const flags = ((App.case.report || {}).flags || []).length;
  const acc = hasTruth() ? caseAccuracy() : null;

  /* The accuracy lens exists only when there is something to compare against;
     loading a reference file is what makes the toggle appear at all. */
  syncLensTabs();

  /* The per-status counts that used to sit here (`24 extracted`, `2
     unresolved`) are now the status legend, which carries every status rather
     than the two this line had room for. */
  clear($("#case-summary")).append(
    h("span", {}, h("b", { text: String(App.notes.size) }), " notes"),
    h("span", {}, h("b", { text: String(App.variables.length) }), " variables"),
    h("span", { class: flags ? "chip warn" : "chip" },
      flags ? flags + " review flag" + (flags === 1 ? "" : "s") : "no review flags")
  );
  if (App.run.run_id) {
    const duration = Number.isFinite(App.run.duration_seconds)
      ? " · " + App.run.duration_seconds.toFixed(1) + "s" : "";
    $("#case-summary").append(h("span", {
      title: [App.run.started_at, App.run.finished_at].filter(Boolean).join(" to "),
      text: "run " + String(App.run.run_id).slice(0, 8) + duration,
    }));
  }
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

  const facts = App.case.case_facts || {};
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
  if (App.case.fatal_blocker) {
    rail.append(h("span", { class: "chip bad", text: "fatal: " + App.case.fatal_blocker }));
  }
  if (!hasCapture()) {
    rail.append(h("span", {
      class: "chip",
      title: "LLM content capture was disabled, so prompt and response bodies are unavailable.",
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
        "Browsers block file:// fetches. Start the installed server with ",
        h("code", { text: "cipoc-workbench serve" }),
        " then open ",
        h("code", { text: "http://127.0.0.1:8000" }),
        "."))
  );
}

async function boot() {
  wire();
  try {
    const response = await fetch(STATE_URL, { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    indexState(await response.json());
  } catch (err) {
    bootError(err);
    return;
  }
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
