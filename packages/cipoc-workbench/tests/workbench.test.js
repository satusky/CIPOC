"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { loadWorkbench, canonicalResult, descendants } = require("./helpers");

for (const schemaVersion of ["1.0", "1.1", "1.2"]) {
test(`recognizes canonical ${schemaVersion} and indexes every result domain without rewriting it`, () => {
  const { context } = loadWorkbench();
  const result = canonicalResult(schemaVersion);
  const original = JSON.stringify(result);
  context.result = result;
  vm.runInContext("indexState(result)", context);

  assert.equal(vm.runInContext("App.schemaVersion", context), schemaVersion);
  assert.equal(vm.runInContext("App.run.run_id", context), result.run.run_id);
  assert.equal(vm.runInContext("App.case.variable_results['400'].value", context), "C504");
  assert.equal(vm.runInContext("App.groups[0].group_id", context), "diagnosis");
  assert.equal(vm.runInContext("App.notes.get('note-2').note_id", context), "note-2");
  assert.equal(vm.runInContext("App.noteDigests.get('note-2').summary", context), "Digest");
  assert.equal(vm.runInContext("App.descriptors.note_count", context), 3);
  assert.equal(vm.runInContext("variableAttempts(400).length", context), 1);
  assert.equal(vm.runInContext("noteExchanges('note-2').length", context), 1);
  assert.equal(vm.runInContext("groupExchanges('diagnosis').length", context), 2);
  assert.equal(vm.runInContext("variableExchanges(400).length", context), 1);
  assert.equal(JSON.stringify(result), original);
  assert.equal(vm.runInContext("normalizeRunResult(result)", context), result);

  for (const version of [undefined, 1.1, "0.9", "2.0"]) {
    context.bad = { ...result, schema_version: version };
    assert.throws(() => vm.runInContext("indexState(bad)", context), /schema_version/);
  }
  context.bad = { ...result, case: undefined };
  assert.throws(() => vm.runInContext("indexState(bad)", context), /missing the case object/);
});

test(`all views render canonical ${schemaVersion} data and typed note-selection provenance`, () => {
  const { context, document } = loadWorkbench();
  context.result = canonicalResult(schemaVersion);
  vm.runInContext("indexState(result)", context);

  vm.runInContext("renderNotes()", context);
  assert.match(document.querySelector("#notes-list").textContent, /note-10/);

  vm.runInContext("renderChrome()", context);
  assert.match(document.querySelector("#case-summary").textContent, /run 123e4567 · 2\.0s/);
  assert.ok(document.querySelector("#facts").textContent.includes(
    "telemetry " + (schemaVersion !== "1.0" ? "complete" : "unknown")));

  vm.runInContext("renderControlRoom()", context);
  assert.match(document.querySelector("#control").textContent, /Primary Site/);

  vm.runInContext("renderTable()", context);
  assert.match(document.querySelector("#table-wrap").textContent, /C504/);

  context.noteView = vm.runInContext("noteDetail('note-2')", context);
  assert.match(context.noteView.textContent, /Scanner calls \(1\)/);
  assert.match(context.noteView.textContent, /Prompt and response bodies were not captured/);
  assert.match(context.noteView.textContent, /hormonal therapyunknown \(not recorded\)/);
  assert.match(context.noteView.textContent, /immunotherapyunknown \(not recorded\)/);

  context.rejectedNoteView = vm.runInContext("noteDetail('note-A')", context);
  assert.match(context.rejectedNoteView.textContent, /Note type did not match the configured note filter/);

  context.groupView = vm.runInContext("groupDetail('diagnosis')", context);
  assert.match(context.groupView.textContent, /Requested items: 400/);
  assert.match(context.groupView.textContent, /surgery: present/);
  assert.match(context.groupView.textContent, /hormonal_therapy: unknown \(not recorded\)/);
  assert.match(context.groupView.textContent, /immunotherapy: unknown \(not recorded\)/);
  assert.doesNotMatch(context.groupView.textContent, /chemotherapy: absent/);
  assert.match(context.groupView.textContent, /Note type did not match the configured note filter/);
  assert.match(context.groupView.textContent, /Cancer status did not match the configured note filter/);
  assert.match(context.groupView.textContent, /Keyword filtering was configured but was not evaluated/);
  assert.match(context.groupView.textContent, /no temporal anchor was available/);
  assert.match(context.groupView.textContent, /invented-Xdiscarded proposal/);

  context.variableView = vm.runInContext("variableDetail(400)", context);
  assert.match(context.variableView.textContent, /Attempts \(1\)/);
  assert.match(context.variableView.textContent, /Variable-level calls \(1\)/);
  assert.match(context.variableView.textContent, /Group extraction call/);
});
}

test("string and integer primary citations navigate without losing leading zeros or scalar types", () => {
  for (const schemaVersion of ["1.0", "1.1"]) {
    for (const noteId of [0, 1, "001", "note-A"]) {
      const { context, document } = loadWorkbench();
      const result = canonicalResult(schemaVersion);
      result.corpus.note_corpus = Object.fromEntries([0, 1, "001", "note-A"].map((id) => [String(id), {
        ...result.corpus.note_corpus["note-2"], note_id: id, content: "Evidence in " + id,
      }]));
      const extraction = result.case.variable_results[400].extraction;
      extraction.most_important_note = noteId;
      extraction.spans = [{ note_id: String(noteId), text: "Evidence in " + noteId }];
      result.observability.variable_attempts["group:diagnosis/variable:400"][0].candidate = { ...extraction };
      context.result = JSON.parse(JSON.stringify(result));
      vm.runInContext("indexState(result)", context);
      assert.equal(vm.runInContext("App.results.get(400).extraction.most_important_note", context), noteId);

      const view = vm.runInContext("variableDetail(400)", context);
      assert.ok(view.textContent.includes("Cited note " + noteId));
      const citation = descendants(view).find((node) => node.tag === "button" && node.textContent === "#" + noteId);
      assert.ok(citation);
      citation.click();
      assert.equal(vm.runInContext("App.selection.id", context), String(noteId));
      assert.ok(document.querySelector("#detail-body").textContent.includes("Evidence in " + noteId));
      assert.equal(vm.runInContext("App.notes.get(App.selection.id).note_id", context), noteId);

      // A primary citation alone must also appear in the note's reverse links.
      context.result.case.variable_results[400].extraction.spans = [];
      assert.equal(vm.runInContext("variablesCitingNote(App.selection.id).length", context), 1);
      const attempt = vm.runInContext("attemptCard(variableAttempts(400)[0])", context);
      assert.ok(descendants(attempt).some((node) => node.tag === "button" && node.textContent === "#" + noteId));
    }
  }
});

test("null primary citations do not resolve to a note named null; selection keeps 001 distinct from 1", () => {
  const { context } = loadWorkbench();
  const result = canonicalResult("1.1");
  result.case.variable_results[400].extraction.most_important_note = null;
  result.corpus.note_corpus.null = { ...result.corpus.note_corpus["note-2"], note_id: "null" };
  result.case.note_selection["group:diagnosis"].candidate_note_ids = [1, "001"];
  result.case.note_selection["group:diagnosis"].selected_note_ids = ["001"];
  context.result = result;
  vm.runInContext("indexState(result)", context);
  assert.doesNotMatch(vm.runInContext("variableDetail(400).textContent", context), /Cited note null/);
  assert.equal(vm.runInContext("variablesCitingNote('null').length", context), 0);
  assert.equal(vm.runInContext("groupsTouchingNote('001')[0].role", context), "selected");
  assert.equal(vm.runInContext("groupsTouchingNote(1)[0].role", context), "not selected");
});

test("partial telemetry retains diagnostic invocations without fabricating semantic attempts", () => {
  for (const capture of [true, false]) {
    const { context, document } = loadWorkbench();
    const result = canonicalResult("1.1");
    const telemetry = result.observability;
    telemetry.collection_status = "partial";
    telemetry.collection_issues = [{ code: "unbound_task", message: "No exact task binding for the repair." }];
    telemetry.llm_content_captured = capture;
    for (const exchange of Object.values(telemetry.llm_exchanges).flat()) {
      exchange.prompt_messages = capture ? [] : null;
    }
    telemetry.unattributed_exchanges = [{
      invocation_id: "call-001", namespace: ["extract_branch:abc", "repair_variable:def"],
      agent: "extractor", node: "repair_variable", model: "resolved-model",
      usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12,
        input_token_details: { cache_read: 4 } },
      error: "provider diagnostic", retry_ordinal: 2,
      prompt_messages: capture ? [{ role: "human", content: "Repair", truncated: true, original_char_count: 100 }] : null,
      response: capture ? { value: "C504" } : null,
      task_id: "extra-task-metadata",
    }];
    const attempts = JSON.stringify(telemetry.variable_attempts);
    const exchanges = JSON.stringify(telemetry.llm_exchanges);
    const summary = JSON.stringify(telemetry.llm_usage_summary);
    context.result = result;
    vm.runInContext("indexState(result); renderChrome()", context);
    const button = descendants(document.querySelector("#facts")).find((node) => node.tag === "button");
    assert.match(button.textContent, /telemetry partial \(1 issue\)/);
    button.click();
    const text = document.querySelector("#view-run").textContent;
    assert.equal(document.querySelector("#view-run").hidden, false);
    assert.equal(document.querySelector("#detail").hidden, true);
    assert.match(text, /unbound_task/);
    assert.match(text, /No exact task binding for the repair/);
    assert.match(text, /Unattributed invocations \(1\)/);
    assert.match(text, /call-001/);
    assert.match(text, /extract_branch:abc/);
    assert.match(text, /resolved-model/);
    assert.match(text, /transport retry 2/);
    assert.match(text, /provider diagnostic/);
    assert.match(text, /input tokens10output tokens2total tokens12/);
    assert.match(text, /cache_read/);
    assert.match(text, /extra-task-metadata/);
    assert.doesNotMatch(text, /[Aa]ttempt [12]/);
    if (capture) {
      assert.match(text, /truncated from 100 characters/);
      assert.match(text, /Response/);
      assert.match(text, /C504/);
    } else {
      assert.match(text, /Prompt and response bodies were not captured/);
    }
    assert.match(vm.runInContext("variableDetail(400).textContent", context), /Telemetry collection: partial/);
    assert.doesNotMatch(vm.runInContext("variableDetail(400).textContent", context), /call-001/);
    assert.equal(JSON.stringify(telemetry.variable_attempts), attempts);
    assert.equal(JSON.stringify(telemetry.llm_exchanges), exchanges);
    assert.equal(JSON.stringify(telemetry.llm_usage_summary), summary);
  }
});

test("unavailable telemetry and null usage are explicit, never zero totals or inferred attempts", () => {
  const { context, document } = loadWorkbench();
  context.result = canonicalResult("1.1");
  Object.assign(context.result.observability, {
    collection_status: "unavailable",
    collection_issues: [{ code: "snapshot_failed", message: "Telemetry finalization failed." }],
    llm_usage_summary: null, llm_exchanges: {}, variable_attempts: {},
  });
  vm.runInContext("indexState(result); renderChrome(); setView('run')", context);
  const text = document.querySelector("#view-run").textContent;
  assert.match(document.querySelector("#facts").textContent, /telemetry unavailable/);
  assert.match(text, /snapshot_failed/);
  assert.match(text, /Usage summary unavailable. Missing usage is not zero usage/);
  assert.doesNotMatch(text, /total tokens0/);
  assert.equal(vm.runInContext("App.observability.llm_usage_summary", context), null);
  const variable = vm.runInContext("variableDetail(400).textContent", context);
  assert.match(variable, /No attempt records in this run artifact/);
  assert.match(variable, /Missing records do not establish that no calls occurred/);
  assert.doesNotMatch(variable, /Attempt 1/);
  assert.match(vm.runInContext("usageDetails(null).textContent", context), /Usage unavailable/);
  assert.match(vm.runInContext("usageDetails({}).textContent", context), /total tokensnot recorded/);
});

test("legacy collection status stays unknown while reported zero usage remains a real zero", () => {
  const { context } = loadWorkbench();
  context.result = canonicalResult();
  vm.runInContext("indexState(result)", context);
  assert.match(vm.runInContext("telemetryDetail().textContent", context), /unknown \(not recorded in schema 1.0\)/);
  assert.match(vm.runInContext("telemetryDetail().textContent", context), /total tokens48/);
  const zero = vm.runInContext("usageDetails({input_tokens: 0, output_tokens: 0, total_tokens: 0}).textContent", context);
  assert.match(zero, /input tokens0output tokens0total tokens0/);
  assert.doesNotMatch(zero, /unavailable/);

  context.exchange = { agent: "extractor", node: "repair_variable", attempt: 2, retry_ordinal: 1, usage: null };
  const call = vm.runInContext("exchangeCard(exchange, 0).textContent", context);
  assert.match(call, /attempt 2/);
  assert.match(call, /transport retry 1/);
  assert.match(call, /Usage unavailable/);
});

test("hormonal therapy and immunotherapy are independent treatment gate evidence", () => {
  for (const name of ["hormonal_therapy", "immunotherapy"]) {
    const { context } = loadWorkbench();
    const result = canonicalResult("1.1");
    const concepts = Object.fromEntries(["surgery", "chemotherapy", "radiation", "hormonal_therapy", "immunotherapy"]
      .map((key) => [key, { presence: key === name, evidence: [] }]));
    result.corpus.note_corpus_descriptors.concepts = concepts;
    result.corpus.note_corpus["note-2"].concepts = concepts;
    context.result = result;
    vm.runInContext("indexState(result)", context);
    const text = vm.runInContext("groupDetail('diagnosis').textContent", context);
    assert.ok(text.includes(name + ": present"));
    assert.match(text, /chemotherapy: absent/);
    const note = vm.runInContext("noteDetail('note-2').textContent", context);
    assert.ok(note.includes(name.replace(/_/g, " ") + "present"));
  }
});

test("applicability preserves breast OR (cutaneous site AND melanoma) and historical leaves", () => {
  const { context } = loadWorkbench();
  const result = canonicalResult("1.1");
  result.inputs.target_variables[0].applies_to = {
    any_of: [
      { gross_primary_sites: ["breast"], histology_families: [] },
      { all_of: [{ primary_sites: ["C440-C449", "C632"] }, { histology_families: ["melanoma"] }] },
    ],
  };
  result.case.case_facts = { primary_site: "C441", histology: "8720" };
  context.result = result;
  vm.runInContext("indexState(result)", context);
  const expected = "(gross primary sites: breast OR (primary sites: C440-C449, C632 AND histology families: melanoma))";
  assert.equal(vm.runInContext("applicabilityLabel(App.groups[0].applies_to)", context), expected);
  assert.ok(vm.runInContext("groupDetail('diagnosis').textContent", context).includes(expected));
  assert.ok(vm.runInContext("groupTooltip(App.groups[0], groupState(App.groups[0])).textContent", context).includes(expected));
  assert.match(vm.runInContext("groupDetail('diagnosis').textContent", context), /primary_site=C441/);
  assert.match(vm.runInContext("groupDetail('diagnosis').textContent", context), /histology=8720/);
  assert.equal(vm.runInContext("applicabilityLabel({gross_primary_sites: ['breast'], histology_families: ['melanoma']})", context),
    "gross primary sites: breast / histology families: melanoma");
  assert.equal(vm.runInContext("applicabilityLabel({new_restriction: ['keep-visible']})", context), "new restriction: keep-visible");
});

test("alphanumeric note IDs remain strings and use lexical tie-breaking", () => {
  const { context } = loadWorkbench();
  context.result = canonicalResult();
  vm.runInContext("indexState(result)", context);

  assert.deepEqual(
    Array.from(vm.runInContext("sortedNotes().map((note) => note.note_id)", context)),
    ["note-10", "note-2", "note-A"],
  );
  assert.equal(vm.runInContext("App.notes.has('note-2')", context), true);
  assert.equal(vm.runInContext("App.notes.has('NaN')", context), false);
});

test("the packaged example is a canonical artifact", () => {
  const { context, document } = loadWorkbench();
  const examplePath = path.join(__dirname, "..", "src", "cipoc_workbench", "example", "case_state.json");
  context.result = JSON.parse(fs.readFileSync(examplePath, "utf8"));

  assert.doesNotThrow(() => vm.runInContext("indexState(result)", context));
  assert.equal(vm.runInContext("App.schemaVersion", context), "1.0");
  assert.ok(vm.runInContext("App.notes.size", context) > 0);
  assert.ok(vm.runInContext("App.variables.length", context) > 0);

  assert.doesNotThrow(() => vm.runInContext(
    "renderChrome(); renderNotes(); renderControlRoom(); renderTable()", context));
  assert.ok(document.querySelector("#notes-list").textContent.length > 0);
  assert.ok(document.querySelector("#control").textContent.length > 0);
  assert.ok(document.querySelector("#table-wrap").textContent.length > 0);

  assert.doesNotThrow(() => vm.runInContext("noteDetail(App.notes.keys().next().value)", context));
  assert.doesNotThrow(() => vm.runInContext("groupDetail(App.groups[0].group_id)", context));
  assert.doesNotThrow(() => vm.runInContext("variableDetail(App.variables[0].item_id)", context));
});

const settle = () => new Promise((resolve) => setImmediate(resolve));
const runB = () => {
  const result = canonicalResult("1.2");
  result.run.run_id = "223e4567-e89b-42d3-a456-426614174000";
  result.case.variable_results[400].value = "C509";
  return result;
};
const feedbackDocument = (runId, note = "", writable = true) => ({
  run_id: runId, writable, read_only_reason: writable ? null : "Use --feedback-dir for this run.",
  annotations: { variable: note ? { 400: { flags: [], expected: null, note } } : {}, group: {}, note: {} },
});
async function openRun(harness, result = canonicalResult(), name = "run.json") {
  harness.context.file = { name, content: JSON.stringify(result) };
  return harness.evaluate("loadRunFile(file)");
}

function reviewSnapshot(h) {
  const nodes = [".views", "#case-summary", "#facts", "#run-source", "#view-notes", "#view-variables", "#view-run", "#detail", "#tooltip"]
    .flatMap((selector) => descendants(h.document.querySelector(selector)));
  const state = h.evaluate("({...App})");
  delete state.loadGeneration;
  return {
    state,
    drafts: h.evaluate("JSON.stringify([...App.feedbackDraft])"),
    truth: h.evaluate("JSON.stringify([...App.truth])"),
    marks: h.document.documentElement.dataset.marks,
    dom: nodes.map((node) => ({ node, parent: node.parentElement, children: [...node.childNodes],
      attributes: { ...node.attributes }, value: node.value, checked: node.checked,
      text: node.tag === "#text" ? node.textContent : null, display: node.style.display,
      scrollTop: node.scrollTop, scrollLeft: node.scrollLeft,
    })),
  };
}

test("renderer-consumed malformed leaves cannot replace a run, its drafts, truth, or DOM", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("indexTruth({400:'C509'},'keep'); setLens('accuracy'); setView('notes'); show('variable',400); setDraft('variable',400,{note:'keep draft'})");
  const before = reviewSnapshot(h);
  for (const change of [
    (r) => { r.corpus.note_corpus['note-2'].note_id = { toString: null }; },
    (r) => { r.corpus.note_corpus['note-2'].flags = [null]; },
    (r) => { r.corpus.note_corpus['note-2'].flags = [17]; },
    (r) => { r.corpus.note_corpus['note-2'].cancer_status = [{}]; },
    (r) => { r.inputs.target_variables[0].name = { toString: null }; },
    (r) => { r.inputs.target_variables[0].variables[0].name = null; },
    (r) => { r.case.variable_results[400].status = []; },
    (r) => { r.case.variable_results[400].value = 12; },
    (r) => { r.case.variable_results[400].extraction.spans[0].text = {}; },
    (r) => { r.case.variable_results[400].extraction.most_important_note = {}; },
    (r) => { r.case.note_selection['group:diagnosis'].selected_note_ids = [{}]; },
    (r) => { r.observability.llm_exchanges['note:note-2'][0].model = {}; },
  ]) {
    const bad = runB(); change(bad);
    assert.equal(await openRun(h, bad), false);
    assert.equal(h.document.querySelector('#run-confirm').open, false);
    assert.deepEqual(reviewSnapshot(h), before);
  }
  // Historical labels and codes remain recorded values, not recoded enums.
  const historical = runB();
  historical.case.variable_results[400].status = 'historical_custom_status';
  historical.case.variable_results[400].value = 'unfamiliar-code';
  historical.corpus.note_corpus['note-2'].flags = ['historical-keyword'];
  const loading = openRun(h, historical); await settle();
  h.document.querySelector('#run-confirm-discard').click();
  assert.equal(await loading, true);
  h.evaluate("App.noteFilter='nothing-matches'; renderNotes(); App.varFilter='nothing-matches'; setView('variables'); setMode('table'); setView('run')");
  assert.equal(h.evaluate("App.results.get(400).value"), 'unfamiliar-code');
});

test("post-validation rendering failures restore all prior nodes, form edits, state and inactive views", async () => {
  for (const renderer of ['renderNotes', 'renderControlRoom', 'renderTable', 'renderRun']) {
    const h = loadWorkbench({ manualFetch: true });
    await openRun(h);
    h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
    h.evaluate("wire(); indexTruth({400:'C509'},'keep'); setLens('accuracy'); setView('notes'); show('variable',400)");
    const textarea = h.document.querySelector('.fb-text');
    textarea.value = 'draft survives'; textarea.dispatchEvent({ type: 'input' });
    h.document.querySelector('#truth-file').value = 'keep-reference.json';
    h.document.querySelector('#detail-body').scrollTop = 57;
    h.evaluate("$('#tooltip').append(h('p',{text:'keep tooltip'})); $('#tooltip').hidden=false");
    const before = reviewSnapshot(h);
    h.context.originalRenderer = h.evaluate(renderer);
    h.evaluate(renderer + ` = function() { originalRenderer(); throw new Error('injected rendering failure'); }`);
    const loading = openRun(h, runB()); await settle();
    h.document.querySelector('#run-confirm-discard').click();
    assert.equal(await loading, false);
    assert.deepEqual(reviewSnapshot(h), before);
    assert.equal(h.document.querySelector('#truth-file').value, 'keep-reference.json');
    assert.equal(h.requests.length, 1, 'failed activation must not start feedback loading');
    assert.match(h.document.querySelector('#run-load-status').textContent, /injected rendering failure/);
    h.evaluate(renderer + ' = originalRenderer');
    // The restored elements, not clones lacking handlers, remain interactive.
    textarea.value = 'still editable'; textarea.dispatchEvent({ type: 'input' });
    assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), 'still editable');
    h.document.querySelector('[data-view="variables"]').click();
    h.document.querySelector('[data-mode="table"]').click();
    assert.match(h.document.querySelector('#table-wrap').textContent, /C504/);
  }
});

test("harness reads real scripts and containers, missing selectors stay null, and tab clicks route", () => {
  const h = loadWorkbench();
  const { document, evaluate, context, scripts } = h;
  assert.equal(document.querySelector("#does-not-exist"), null);
  assert.equal(document.querySelector(".does-not-exist"), null);
  assert.ok(scripts.indexOf("run.js") > scripts.indexOf("detail.js"));
  assert.equal(document.querySelectorAll("script[src]").length, scripts.length);
  assert.equal(document.querySelector("#group-toggle").parentElement.tag, "label");
  assert.equal(document.querySelector("#boot").style.position, "static");
  context.result = canonicalResult();
  evaluate("indexState(result); wire(); renderChrome(); setMode(App.mode); setView(App.view)");
  for (const view of ["notes", "run", "variables"]) {
    document.querySelector('[data-view="' + view + '"]').click();
    assert.equal(evaluate("App.view"), view);
    for (const other of ["notes", "variables", "run"]) {
      assert.equal(document.querySelector("#view-" + other).hidden, other !== view);
      assert.equal(document.querySelector('[data-view="' + other + '"]').getAttribute("aria-selected"), String(other === view));
    }
  }
  for (const mode of ["table", "control"]) {
    document.querySelector('[data-mode="' + mode + '"]').click();
    assert.equal(evaluate("App.mode"), mode);
    assert.equal(document.querySelector("#control").hidden, mode !== "control");
    assert.equal(document.querySelector("#table-wrap").hidden, mode !== "table");
    assert.equal(document.querySelector('[data-mode="' + mode + '"]').getAttribute("aria-selected"), "true");
    assert.equal(document.querySelector("#group-toggle").parentElement.style.display, mode === "table" ? "" : "none");
  }
  document.querySelector("#group-toggle").click();
  assert.equal(evaluate("App.grouped"), false);
  const entity = document.querySelector('[data-entity="variable:400"]');
  assert.ok(entity);
  entity.click();
  assert.equal(document.querySelector("#detail").hidden, false);
  assert.equal(entity.getAttribute("aria-current"), "true");
  assert.equal(entity.dataset.entity, "variable:400");
  document.querySelector(".telemetry-link").click();
  assert.equal(evaluate("App.selection"), null);
  assert.equal(document.querySelector("#view-run").hidden, false);
  assert.equal(document.querySelector("#detail").hidden, true);
});

test("local activation clears case-local state but preserves view, layout, grouping and theme", async () => {
  const h = loadWorkbench({ manualFetch: true });
  assert.equal(await openRun(h), true);
  h.evaluate(`
    Theme.apply('light'); setView('notes'); setMode('table'); App.grouped = false;
    indexTruth({400:'wrong'}, 'previous-reference.json'); setLens('accuracy');
    App.classFilter.add('mismatch'); App.noteFilter = 'old'; App.varFilter = 'old';
    $('#note-filter').value = 'old'; $('#var-filter').value = 'old';
    App.feedback.variable['400'] = {flags:[],note:'old review'};
    show('variable',400); $('#tooltip').append(h('p',{text:'old tooltip'})); $('#tooltip').hidden = false;
  `);
  assert.equal(await openRun(h, runB(), "<img onerror=bad>.json"), true);
  assert.equal(h.evaluate("App.run.run_id"), runB().run.run_id);
  assert.equal(h.evaluate("App.truth.size"), 0);
  assert.equal(h.evaluate("App.truthSource"), null);
  assert.equal(h.evaluate("App.selection"), null);
  assert.equal(h.evaluate("App.lens"), "confidence");
  assert.equal(h.evaluate("App.classFilter.size"), 0);
  assert.equal(h.evaluate("annotationCount()"), 0);
  assert.equal(h.evaluate("App.feedbackWritable"), false);
  assert.equal(h.evaluate("App.feedbackStatus"), "loading");
  assert.equal(h.evaluate("App.view + '/' + App.mode + '/' + App.grouped + '/' + Theme.current()"), "notes/table/false/light");
  for (const id of ["note-filter", "var-filter"]) assert.equal(h.document.querySelector("#" + id).value, "");
  assert.equal(h.document.querySelector("#group-toggle").checked, false);
  assert.equal(h.document.querySelector("#tooltip").textContent, "");
  assert.equal(h.document.querySelector("#tooltip").hidden, true);
  assert.equal(h.document.querySelector("#detail-body").textContent, "");
  assert.equal(h.evaluate("App.runSource"), "<img onerror=bad>.json");
  assert.match(h.document.querySelector("#run-source").textContent, /<img onerror=bad>\.json/);
  assert.equal(h.document.querySelector("#run-source img"), null);
  assert.ok(h.requests.every((r) => r.url.startsWith("/api/runs/") && !r.options.body));
  assert.ok([...h.storage.keys()].every((key) => key.startsWith("cipoc-lens")));
});

test("bad JSON, versions, UUIDs, failure envelopes and malformed nested containers never replace state or drafts", async () => {
  const h = loadWorkbench();
  await openRun(h);
  h.evaluate("App.feedbackDraft.set('variable:400',{flags:[],note:'keep me'}); indexTruth({400:'C504'},'keep'); show('variable',400)");
  const previous = h.evaluate("({run:App.run, notes:App.notes, results:App.results, truth:App.truth, draft:App.feedbackDraft, activation:App.activation})");
  const changes = [
    (r) => { r.schema_version = "2.0"; },
    (r) => { r.run.run_id = "../other"; },
    (r) => { r.run.run_id = r.run.run_id.toUpperCase(); },
    (r) => { r.run.run_id += "\n"; },
    (r) => { r.run.status = "failed"; },
    (r) => { delete r.case; },
    (r) => { r.inputs.target_variables = {}; },
    (r) => { r.inputs.target_variables[0].variables = [null]; },
    (r) => { r.case.variable_results[400] = null; },
    (r) => { r.case.variable_results[400].extraction.spans = {}; },
    (r) => { r.corpus.note_corpus["note-2"] = null; },
    (r) => { r.corpus.note_corpus["note-2"].concepts.cancer = null; },
    (r) => { r.case.note_selection["group:diagnosis"].selected_note_ids = {}; },
    (r) => { r.observability.llm_exchanges["note:note-2"] = {}; },
    (r) => { r.observability.variable_attempts["group:diagnosis/variable:400"] = [null]; },
    (r) => { r.observability.unattributed_exchanges = {}; },
  ];
  for (const change of changes) {
    const bad = runB(); change(bad);
    assert.equal(await openRun(h, bad), false);
    assert.equal(h.document.querySelector("#run-confirm").open, false);
    for (const key of Object.keys(previous)) assert.equal(h.evaluate("App." + (key === "draft" ? "feedbackDraft" : key)), previous[key]);
    assert.match(h.document.querySelector("#run-load-status").textContent, /Could not load/);
  }
  h.context.file = { name: "broken.json", content: "{" };
  assert.equal(await h.evaluate("loadRunFile(file)"), false);
  assert.equal(await h.evaluate("loadRunFile(null)"), false);
  assert.equal(h.evaluate("App.run"), previous.run);
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), "keep me");
  assert.equal(h.evaluate("App.selection.id"), "400");
});

test("validated run replacement offers Cancel or Discard and Load without discarding drafts early", async () => {
  const h = loadWorkbench();
  await openRun(h);
  h.evaluate("App.feedbackDraft.set('variable:400',{flags:[],note:'draft'})");
  let opening = openRun(h, runB());
  await settle();
  assert.equal(h.document.querySelector("#run-confirm").open, true);
  assert.equal(h.evaluate("App.run.run_id"), canonicalResult().run.run_id);
  h.document.querySelector("#run-confirm-cancel").click();
  assert.equal(await opening, false);
  assert.equal(h.evaluate("App.feedbackDraft.size"), 1);
  opening = openRun(h, runB());
  await settle();
  h.document.querySelector("#run-confirm-discard").click();
  assert.equal(await opening, true);
  assert.equal(h.evaluate("App.feedbackDraft.size"), 0);
  assert.equal(h.evaluate("App.run.run_id"), runB().run.run_id);
});

test("latest selected file wins out-of-order reads, and cancelled or failed newer selections do not resurrect older reads", async () => {
  for (const end of ["success", "cancel", "invalid", "read error", "abort"]) {
    const h = loadWorkbench({ manualFiles: true });
    const first = openRun(h);
    h.reads[0].resolve(); await first;
    const old = openRun(h, runB(), "same.json");
    const replacement = canonicalResult("1.1");
    replacement.run.run_id = "323e4567-e89b-42d3-a456-426614174000";
    let latest;
    if (end === "cancel") latest = h.evaluate("loadRunFile(null)");
    else {
      latest = openRun(h, replacement, "same.json");
      if (end === "invalid") h.reads[2].resolve("invalid json");
      else if (end === "read error") h.reads[2].reject();
      else if (end === "abort") h.reads[2].abort();
      else h.reads[2].resolve();
    }
    assert.equal(await latest, end === "success");
    h.reads[1].resolve();
    assert.equal(await old, false);
    assert.equal(h.evaluate("App.run.run_id"), end === "success" ? replacement.run.run_id : canonicalResult().run.run_id);
  }
});

test("file input listener resets selection and reopening the same filename reactivates the artifact", async () => {
  const h = loadWorkbench();
  h.evaluate("wire()");
  const input = h.document.querySelector("#run-file");
  for (const result of [canonicalResult(), runB(), canonicalResult()]) {
    input.files = [{ name: "same.json", content: JSON.stringify(result) }];
    input.value = "same.json";
    input.dispatchEvent({ type: "change" });
    assert.equal(input.value, "");
    await Promise.all(input.eventResults);
    assert.equal(h.evaluate("App.run.run_id"), result.run.run_id);
  }
  assert.equal(h.evaluate("App.activation"), 3);
  assert.equal(h.evaluate("App.runSource"), "same.json");
});

test("native run-input cancellation invalidates pending run reads without disturbing the current review", async () => {
  const h = loadWorkbench({ manualFiles: true, manualFetch: true });
  h.evaluate('wire()');
  const initial = openRun(h); h.reads[0].resolve(); await initial;
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("indexTruth({400:'keep'},'keep'); show('variable',400); setDraft('variable',400,{note:'keep draft'})");
  const before = reviewSnapshot(h);
  const pending = openRun(h, runB());
  const input = h.document.querySelector('#run-file');
  input.dispatchEvent({ type: 'cancel' });
  await Promise.all(input.eventResults);
  h.reads[1].resolve();
  assert.equal(await pending, false);
  assert.deepEqual(reviewSnapshot(h), before);
  assert.equal(h.document.querySelector('#run-confirm').open, false);
  assert.match(h.document.querySelector('#run-load-status').textContent, /No file selected/);
});

test("empty startup does not parse a body or present phantom case metrics, feedback, or reference data", async () => {
  const requests = [];
  let parsedEmptyBody = false;
  const h = loadWorkbench({ fetch: async (url) => {
    requests.push(url);
    return { ok: url === "case_state.json", status: url === "case_state.json" ? 204 : 404,
      json: async () => { parsedEmptyBody = true; throw new Error("Empty response has no JSON body"); } };
  } });
  await h.evaluate("boot()");
  assert.equal(parsedEmptyBody, false);
  assert.deepEqual(requests.sort(), ["case_state.json", "endpoint_catalog.json"]);
  assert.equal(h.evaluate("App.run.run_id"), undefined);
  assert.equal(h.evaluate("App.activation"), 0);
  assert.equal(h.document.querySelector("#boot").hidden, false);
  assert.equal(h.document.querySelector("#boot").classList.contains("error"), false);
  assert.match(h.document.querySelector("#boot").textContent, /No run loaded/);
  assert.match(h.document.querySelector("#run-load-status").textContent, /No startup run configured/);
  assert.equal(h.document.querySelector("#run-file").disabled, false);
  assert.equal(h.document.querySelector("#theme-toggle").disabled, false);
  for (const view of ["notes", "variables", "run"]) {
    const tab = h.document.querySelector('[data-view="' + view + '"]');
    assert.equal(tab.disabled, true);
    assert.equal(tab.getAttribute("aria-selected"), "false");
    tab.click();
    assert.equal(h.document.querySelector("#view-" + view).hidden, true);
  }
  h.evaluate("setView('run'); render(); renderChrome(); show('telemetry','run')");
  assert.equal(h.document.querySelector("#view-run").textContent, "");
  assert.equal(h.document.querySelector("#case-summary").textContent, "");
  assert.equal(h.document.querySelector("#facts").textContent, "");
  assert.equal(h.document.querySelector("#facts").hidden, true);
  assert.equal(h.document.querySelector("#detail").hidden, true);
  assert.equal(await h.evaluate("loadFeedback()"), false);
  assert.equal(h.evaluate("App.feedbackReason"), "No run loaded.");
  assert.equal(requests.length, 2);
  const theme = h.evaluate("Theme.current()");
  h.document.querySelector("#theme-toggle").click();
  assert.notEqual(h.evaluate("Theme.current()"), theme);
});

test("first local load activates an empty workbench and uses only run-scoped feedback", async () => {
  const h = loadWorkbench({ manualFetch: true });
  const boot = h.evaluate("boot()");
  h.requests.find((r) => r.url === "case_state.json").resolve(undefined, 204);
  await boot;
  assert.equal(await openRun(h), true);
  assert.equal(h.document.querySelector("#boot").hidden, true);
  assert.equal(h.document.querySelector("#facts").hidden, false);
  for (const tab of h.document.querySelectorAll(".view-tab")) assert.equal(tab.disabled, false);
  h.document.querySelector('[data-view="run"]').click();
  assert.equal(h.document.querySelector("#view-run").hidden, false);
  assert.equal(h.requests.filter((r) => r.url.includes("ground-truth")).length, 0);
  const feedback = h.requests.find((r) => r.url.includes("/feedback"));
  assert.equal(feedback.url, "/api/runs/" + canonicalResult().run.run_id + "/feedback");
  feedback.resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  assert.equal(h.evaluate("App.feedbackWritable"), true);
});

test("cancelled, malformed and render-failing first files preserve the empty screen and disabled controls", async () => {
  const h = loadWorkbench({ manualFetch: true });
  const boot = h.evaluate("boot()");
  h.requests.find((r) => r.url === "case_state.json").resolve(undefined, 204);
  await boot;
  h.document.querySelector("#run-file").dispatchEvent({ type: "cancel" });
  assert.match(h.document.querySelector("#run-load-status").textContent, /No file selected/);
  assert.doesNotMatch(h.document.querySelector("#run-load-status").textContent, /Current run kept/);
  const before = reviewSnapshot(h);
  h.context.file = { name: "broken.json", content: "{" };
  assert.equal(await h.evaluate("loadRunFile(file)"), false);
  assert.deepEqual(reviewSnapshot(h), before);
  h.evaluate("const originalRunRenderer = renderRun; renderRun = function() { originalRunRenderer(); throw new Error('injected render failure'); }");
  assert.equal(await openRun(h), false);
  assert.deepEqual(reviewSnapshot(h), before);
  assert.equal(h.document.querySelector("#boot").hidden, false);
  assert.equal(h.document.querySelector("#boot").classList.contains("error"), false);
  for (const tab of h.document.querySelectorAll(".view-tab")) assert.equal(tab.disabled, true);
  assert.equal(h.requests.filter((r) => r.url.includes("/api/")).length, 0);
  h.evaluate("renderRun = originalRunRenderer");
  assert.equal(await openRun(h), true);
});

test("only explicit no-content startup is empty; HTTP, JSON and artifact failures remain errors", async () => {
  for (const scenario of ["404", "500", "invalid JSON", "invalid artifact"]) {
    const h = loadWorkbench({ fetch: async (url) => ({
      ok: url === "case_state.json" && !["404", "500"].includes(scenario),
      status: url !== "case_state.json" ? 404 : scenario === "500" ? 500 : scenario === "404" ? 404 : 200,
      json: async () => { if (scenario === "invalid JSON") throw new SyntaxError("Invalid JSON"); return {}; },
    }) });
    await h.evaluate("boot()");
    assert.equal(h.document.querySelector("#boot").classList.contains("error"), true, scenario);
    assert.match(h.document.querySelector("#run-load-status").textContent, /Startup load failed/);
    assert.equal(h.document.querySelector("#run-file").disabled, false);
    assert.equal(h.document.querySelector("#facts").hidden, true);
    assert.equal(h.evaluate("App.run.run_id"), undefined);
  }
});

test("late empty startup response cannot reset a pending or completed local selection", async () => {
  for (const completed of [false, true]) {
    const h = loadWorkbench({ manualFetch: true, manualFiles: true });
    const boot = h.evaluate("boot()");
    const local = openRun(h, runB());
    if (completed) { h.reads[0].resolve(); assert.equal(await local, true); }
    const status = h.document.querySelector("#run-load-status").textContent;
    h.requests.find((r) => r.url === "case_state.json").resolve(undefined, 204);
    await boot;
    assert.equal(h.document.querySelector("#run-load-status").textContent, status);
    if (!completed) { h.reads[0].resolve(); assert.equal(await local, true); }
    assert.equal(h.evaluate("App.run.run_id"), runB().run.run_id);
    assert.equal(h.document.querySelector("#boot").hidden, true);
    assert.equal(h.requests.filter((r) => r.url.includes("ground-truth")).length, 0);
  }
});

test("refresh returns to empty without state or reloads the explicit startup artifact", async () => {
  for (const configured of [false, true]) {
    const first = loadWorkbench({ manualFetch: true });
    const boot = first.evaluate("boot()");
    first.requests.find((r) => r.url === "case_state.json").resolve(configured ? canonicalResult() : undefined, configured ? 200 : 204);
    await boot;
    assert.equal(await openRun(first, runB()), true);
    first.document.querySelector("#theme-toggle").click();
    const refreshed = loadWorkbench({ manualFetch: true, storage: first.storage });
    const reloading = refreshed.evaluate("boot()");
    refreshed.requests.find((r) => r.url === "case_state.json").resolve(configured ? canonicalResult() : undefined, configured ? 200 : 204);
    await reloading;
    assert.equal(refreshed.evaluate("App.run.run_id"), configured ? canonicalResult().run.run_id : undefined);
    assert.equal(refreshed.document.querySelector("#boot").hidden, configured);
    assert.equal(refreshed.evaluate("Theme.current()"), "light");
    assert.ok([...first.storage.keys()].every((key) => key.startsWith("cipoc-theme") || key.startsWith("cipoc-lens")));
  }
});

test("late startup fetch cannot replace a local selection, including a pending local read", async () => {
  const h = loadWorkbench({ manualFetch: true, manualFiles: true });
  const boot = h.evaluate("boot()");
  const local = openRun(h, runB());
  h.requests.find((r) => r.url === "case_state.json").resolve(canonicalResult());
  await boot;
  assert.equal(h.evaluate("App.activation"), 0);
  h.reads[0].resolve();
  assert.equal(await local, true);
  assert.equal(h.evaluate("App.run.run_id"), runB().run.run_id);
  assert.equal(h.requests.filter((r) => r.url.includes("ground-truth")).length, 0);
});

test("startup error leaves Load Run accessible, and catalog or feedback failure never prevents local review", async () => {
  const h = loadWorkbench({ manualFetch: true });
  const boot = h.evaluate("boot()");
  h.requests.find((r) => r.url === "case_state.json").reject(new Error("offline"));
  await boot;
  assert.match(h.document.querySelector("#boot").textContent, /Use Load Run/);
  assert.equal(h.document.querySelector("#boot").contains(h.document.querySelector("#run-file")), false);
  assert.equal(await openRun(h), true);
  assert.equal(h.document.querySelector("#boot").hidden, true);
  const feedback = h.requests.find((r) => r.url.includes("/feedback"));
  feedback.reject(new Error("API not installed"));
  await settle();
  assert.equal(h.evaluate("App.feedbackWritable"), false);
  h.evaluate("show('variable',400)");
  assert.match(h.document.querySelector("#detail-body").textContent, /Feedback unavailable: API not installed/);
  assert.match(h.document.querySelector("#control").textContent, /Primary Site/);
  assert.equal(h.document.querySelector(".fb-save").disabled, true);
});

test("startup ground truth is non-blocking and cannot follow a local switch or supersede a local reference", async () => {
  const h = loadWorkbench({ manualFetch: true, manualFiles: true });
  const boot = h.evaluate("boot()");
  h.requests.find((r) => r.url === "case_state.json").resolve(canonicalResult());
  await boot;
  assert.equal(h.document.querySelector("#boot").hidden, true);
  const truth = h.requests.find((r) => r.url === "/api/runs/" + canonicalResult().run.run_id + "/ground-truth");
  h.context.event = { target: { files: [{ name: "local-truth.json", content: '{"400":"local"}' }] } };
  const localTruth = h.evaluate("onTruthFile(event)");
  h.reads[0].resolve(); await localTruth;
  truth.resolve({ run_id: canonicalResult().run.run_id, values: { 400: "stale server reference" } });
  await settle();
  assert.equal(h.evaluate("truthFor(400)"), "local");
  const referenceRead = h.evaluate("onTruthFile(event)");
  const localRun = openRun(h, runB());
  h.reads[2].resolve(); await localRun;
  h.reads[1].resolve(); await referenceRead;
  assert.equal(h.evaluate("App.truth.size"), 0);
  assert.equal(h.evaluate("App.truthSource"), null);
  assert.equal(h.requests.filter((r) => r.url.includes("ground-truth")).length, 1);
});

test("reference picker also uses latest-read wins and reports only current read errors", async () => {
  const h = loadWorkbench({ manualFiles: true });
  const opening = openRun(h); h.reads[0].resolve(); await opening;
  h.context.event = { target: { files: [{ name: "truth.json", content: '{"400":"first"}' }] } };
  const first = h.evaluate("onTruthFile(event)");
  h.context.event.target.files[0].content = '{"400":"second"}';
  const second = h.evaluate("onTruthFile(event)");
  h.reads[2].resolve(); await second;
  h.reads[1].reject(); await first;
  assert.equal(h.evaluate("truthFor(400)"), "second");
  assert.equal(h.alerts.length, 0);
  const failed = h.evaluate("onTruthFile(event)");
  h.reads[3].reject(); await failed;
  assert.match(h.alerts[0], /Could not load ground truth/);
});

test("current activation reference reads survive invalid, cancelled, and save-blocked run selections", async () => {
  for (const source of ['server', 'local']) for (const rejection of ['invalid', 'native cancel', 'draft cancel', 'pending save']) {
    const h = loadWorkbench({ manualFiles: true, manualFetch: true });
    h.evaluate('wire()');
    const initial = openRun(h); h.reads[0].resolve(); await initial;
    const runId = canonicalResult().run.run_id;
    h.requests[0].resolve(feedbackDocument(runId)); await settle();
    let saving, put;
    if (rejection === 'pending save') {
      saving = h.evaluate("saveAnnotation('variable',400,{flags:[],note:'saving'},h('span'))");
      put = h.requests.at(-1);
    }
    if (rejection === 'draft cancel') h.evaluate("setDraft('variable',400,{note:'keep draft'})");
    h.context.truthEvent = { target: { files: [{ name: 'reference.json', content: '{"400":"C509"}' }] } };
    const reference = h.evaluate(source === 'server' ? 'loadTruth()' : 'onTruthFile(truthEvent)');
    const read = source === 'server' ? h.requests.at(-1) : h.reads.at(-1);
    if (rejection === 'native cancel') {
      h.document.querySelector('#run-file').dispatchEvent({ type: 'cancel' });
    } else {
      const result = runB();
      if (rejection === 'invalid') result.schema_version = '99.0';
      const selection = openRun(h, result);
      if (rejection !== 'pending save') h.reads.at(-1).resolve();
      if (rejection === 'draft cancel') {
        await settle();
        h.document.querySelector('#run-confirm-cancel').click();
      }
      assert.equal(await selection, false);
    }
    if (source === 'server') read.resolve({ run_id: runId, values: { 400: 'C509' } });
    else read.resolve();
    await reference;
    assert.equal(h.evaluate('truthFor(400)'), 'C509', source + ': ' + rejection);
    assert.equal(h.evaluate('App.truthSource'), source === 'server' ? 'server' : 'reference.json');
    assert.equal(h.evaluate('App.run.run_id'), runId);
    if (saving) {
      put.resolve({ run_id: runId, kind: 'variable', id: '400', annotation: { flags: [], note: 'saved' } });
      assert.equal(await saving, true);
    }
  }
});

test("automatic truth requires an explicit matching run binding, with no unbound fallback", async () => {
  const a = canonicalResult(), b = runB();
  for (const response of [
    { body: { run_id: a.run.run_id, values: { 400: 'A reference' } }, status: 200 },
    { body: { detail: 'The reference belongs to the startup run only.' }, status: 409 },
    { body: {}, status: 404 },
    { body: { 400: 'unbound values' }, status: 200 },
    { body: { run_id: b.run.run_id, values: [] }, status: 200 },
  ]) {
    const h = loadWorkbench({ manualFetch: true });
    // Simulate case_state.json being replaced with B after the server bound truth to A.
    const boot = h.evaluate('boot()');
    h.requests.find((r) => r.url === 'case_state.json').resolve(b);
    await boot;
    const url = '/api/runs/' + b.run.run_id + '/ground-truth';
    const truth = h.requests.find((r) => r.url === url);
    assert.ok(truth);
    truth.resolve(response.body, response.status); await settle();
    assert.equal(h.evaluate('App.truth.size'), 0);
    assert.equal(h.evaluate('App.truthSource'), null);
    assert.deepEqual(h.requests.filter((r) => /ground[_-]truth/.test(r.url)).map((r) => r.url), [url]);
    assert.equal(h.evaluate('App.run.run_id'), b.run.run_id);
  }
  const h = loadWorkbench({ manualFetch: true });
  const boot = h.evaluate('boot()');
  h.requests.find((r) => r.url === 'case_state.json').resolve(a); await boot;
  const pending = h.requests.find((r) => r.url.endsWith('/ground-truth'));
  await openRun(h, b);
  pending.resolve({ run_id: a.run.run_id, values: { 400: 'A reference' } }); await settle();
  assert.equal(h.evaluate('App.truth.size'), 0);
});

test("legacy sparse annotation documents stay readable without mutating server response metadata", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  // The persisted compatibility fixture from test_server.py, with the server's
  // explicit run binding, write capability and empty group bucket added on GET.
  const document = {
    schema_version: 'legacy', state_file: 'original.json', reviewer: { id: 'reviewer-1' }, updated_at: 'yesterday',
    run_id: canonicalResult().run.run_id, writable: true, read_only_reason: null,
    annotations: { note: { '001': { note: 'Keep me', custom: 'retained' } }, variable: { '400': { flags: ['wrong'] } }, group: {} },
  };
  const before = JSON.stringify(document);
  h.requests[0].resolve(document); await settle();
  assert.equal(h.evaluate('App.feedbackWritable'), true);
  assert.equal(h.evaluate("annotationFor('note','001').custom"), 'retained');
  assert.equal(h.evaluate("currentAnnotation('note','001').flags.length"), 0);
  const note = h.evaluate("feedbackSection('note','001')");
  assert.equal(note.querySelector('.fb-text').value, 'Keep me');
  assert.equal(note.querySelector('.fb-save').disabled, false);
  assert.equal(h.evaluate("currentAnnotation('variable',400).note"), '');
  assert.equal(JSON.stringify(document), before);
  assert.equal(h.requests.length, 1, 'reading legacy feedback must not trigger a migration write');
});

test("feedback saves keep arbitrary entity IDs in the query, not URL path segments", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  const runId = canonicalResult().run.run_id;
  h.requests[0].resolve(feedbackDocument(runId)); await settle();
  for (const id of ['.', '..', 'a/b', 'a\\b', 'a?b#c', '%2f', '001', 'a & b', '../group/other']) {
    h.context.entityId = id;
    const saving = h.evaluate("saveAnnotation('note',entityId,{flags:[],note:'keep exact ID'},h('span'))");
    const put = h.requests.at(-1);
    assert.equal(put.url, 'api/runs/' + runId + '/feedback/note?entity_id=' + encodeURIComponent(id));
    const parsed = new URL(put.url, 'http://127.0.0.1:8000/');
    assert.equal(parsed.pathname, '/api/runs/' + runId + '/feedback/note');
    assert.equal(parsed.searchParams.get('entity_id'), id);
    put.resolve({ run_id: runId, kind: 'note', id, annotation: { flags: [], note: 'keep exact ID' } });
    assert.equal(await saving, true);
    assert.equal(h.evaluate("annotationFor('note',entityId).note"), 'keep exact ID');
  }
  const saving = h.evaluate("saveAnnotation('note','a/b',{flags:[],note:'must not change identity'},h('span'))");
  h.requests.at(-1).resolve({ run_id: runId, kind: 'note', id: 'a', annotation: null });
  assert.equal(await saving, false);
  assert.equal(h.evaluate("App.feedbackDraft.get('note:a/b').note"), 'must not change identity');
});

test("feedback A/B/A loads use canonical identity and activation, never overlapping entity IDs", async () => {
  const h = loadWorkbench({ manualFetch: true });
  const a = canonicalResult(), b = runB();
  await openRun(h, a);
  const firstA = h.requests.at(-1);
  await openRun(h, b);
  const requestB = h.requests.at(-1);
  requestB.resolve(feedbackDocument(b.run.run_id, "B review")); await settle();
  assert.equal(h.evaluate("annotationFor('variable',400).note"), "B review");
  await openRun(h, a);
  const secondA = h.requests.at(-1);
  assert.equal(h.evaluate("annotationFor('variable',400)"), null);
  firstA.resolve(feedbackDocument(a.run.run_id, "stale A response")); await settle();
  assert.equal(h.evaluate("annotationFor('variable',400)"), null);
  secondA.resolve(feedbackDocument(a.run.run_id, "saved A review")); await settle();
  assert.equal(h.evaluate("annotationFor('variable',400).note"), "saved A review");
  assert.equal(h.evaluate("App.feedbackWritable"), true);
  assert.deepEqual(h.requests.map((r) => r.url), [a, b, a].map((r) => "/api/runs/" + r.run.run_id + "/feedback"));
});

test("read-only capability, identity mismatch and corrupt feedback are explicit and cannot write", async () => {
  const a = canonicalResult();
  const scenarios = [
    { ...feedbackDocument(a.run.run_id, "saved", false) },
    { ...feedbackDocument(runB().run.run_id) },
    { ...feedbackDocument(a.run.run_id), writable: undefined },
    { ...feedbackDocument(a.run.run_id), annotations: { variable: {}, group: [], note: {} } },
    { ...feedbackDocument(a.run.run_id), annotations: { variable: { 400: null }, group: {}, note: {} } },
  ];
  for (const document of scenarios) {
    const h = loadWorkbench({ manualFetch: true });
    await openRun(h, a);
    h.requests[0].resolve(document); await settle();
    assert.equal(h.evaluate("App.feedbackWritable"), false);
    h.evaluate("show('variable',400)");
    assert.equal(h.document.querySelector(".fb-save").disabled, true);
    assert.equal(h.document.querySelector(".fb-text").disabled, true);
    assert.match(h.document.querySelector("#detail-body").textContent, /Use --feedback-dir|Feedback unavailable/);
    assert.equal(await h.evaluate("saveAnnotation('variable',400,{flags:[],note:'no write'},h('span'))"), false);
    assert.equal(h.requests.length, 1);
  }
});

test("pending saves snapshot prefilled values, block switching and remain disabled across navigation; failures retain drafts", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("indexTruth({400:'C509'},'reference'); show('variable',400)");
  let text = h.document.querySelector(".fb-text");
  text.value = "submitted draft"; text.dispatchEvent({ type: "input" });
  const save = h.document.querySelector(".fb-save");
  save.click();
  const saving = save.eventResults[0];
  const put = h.requests.at(-1);
  assert.equal(put.options.method, "PUT");
  assert.deepEqual(JSON.parse(put.options.body), { flags: [], expected: "C509", note: "submitted draft" });
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').expected"), "C509");
  assert.equal(h.document.querySelector(".fb-text").disabled, true);
  h.evaluate("show('note','note-2'); show('variable',400)");
  for (const control of h.document.querySelectorAll(".flag-grid input, .fb-text, .fb-input, .fb-save")) assert.equal(control.disabled, true);
  const count = h.requests.length;
  h.document.querySelector(".fb-save").click();
  assert.equal(h.requests.length, count);
  assert.equal(await openRun(h, runB()), false);
  assert.match(h.document.querySelector("#run-load-status").textContent, /pending feedback save/);
  put.reject(new Error("disk full"));
  assert.equal(await saving, false);
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), "submitted draft");
  assert.equal(h.evaluate("App.feedbackPending.size"), 0);
  h.evaluate("show('note','note-2'); show('variable',400)");
  assert.equal(h.document.querySelector(".fb-text").value, "submitted draft");
  assert.equal(h.document.querySelector(".fb-save").disabled, false);
  assert.match(h.document.querySelector("#detail-body").textContent, /Not saved: disk full/);
  const retry = h.document.querySelector(".fb-save"); retry.click();
  const annotation = JSON.parse(h.requests.at(-1).options.body);
  h.requests.at(-1).resolve({ run_id: canonicalResult().run.run_id, kind: "variable", id: "400", annotation });
  assert.equal(await retry.eventResults[0], true);
  assert.equal(h.evaluate("App.feedbackDraft.size"), 0);
  assert.equal(h.evaluate("annotationFor('variable',400).note"), "submitted draft");
});

test("save completions do not resurrect a closed dossier or erase a newer draft", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("show('variable',400)");
  const saving = h.evaluate("saveAnnotation('variable',400,{flags:[],note:'submitted'},h('span'))");
  h.evaluate("App.feedbackDraft.set('variable:400',{flags:[],note:'newer edit'}); clearSelection()");
  h.requests.at(-1).resolve({ run_id: canonicalResult().run.run_id, kind: "variable", id: "400", annotation: { flags: [], note: "saved" } });
  assert.equal(await saving, true);
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), "newer edit");
  assert.equal(h.evaluate("App.selection"), null);
  assert.equal(h.document.querySelector("#detail").hidden, true);
});

test("intentionally empty submitted expected values stay empty after a failed save and navigation", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("indexTruth({400:'C509'},'reference'); show('variable',400)");
  const text = h.document.querySelector(".fb-text");
  text.value = "keep prefill"; text.dispatchEvent({ type: "input" });
  h.evaluate("show('note','note-2'); show('variable',400)");
  assert.equal(h.document.querySelector(".fb-input").value, "C509");
  const expected = h.document.querySelector(".fb-input");
  expected.value = ""; expected.dispatchEvent({ type: "input" });
  const save = h.document.querySelector(".fb-save"); save.click();
  assert.equal(JSON.parse(h.requests.at(-1).options.body).expected, null);
  h.evaluate("show('note','note-2'); show('variable',400)");
  assert.equal(h.document.querySelector(".fb-input").value, "");
  h.requests.at(-1).reject(new Error("offline")); await save.eventResults[0];
  h.evaluate("show('note','note-2'); show('variable',400)");
  assert.equal(h.document.querySelector(".fb-input").value, "");
});

test("save response identity errors preserve drafts and old detached forms cannot edit a new activation", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  h.evaluate("show('variable',400)");
  const oldText = h.document.querySelector(".fb-text");
  const oldSave = h.document.querySelector(".fb-save");
  const saving = h.evaluate("saveAnnotation('variable',400,{flags:[],note:'keep draft'},h('span'))");
  h.requests.at(-1).resolve({ run_id: runB().run.run_id, kind: "variable", id: "400", annotation: null });
  assert.equal(await saving, false);
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), "keep draft");
  assert.match(h.document.querySelector("#detail-body").textContent, /response identity/);
  const opening = openRun(h, runB()); await settle();
  h.document.querySelector("#run-confirm-discard").click(); await opening;
  h.requests.at(-1).resolve(feedbackDocument(runB().run.run_id)); await settle();
  oldText.value = "stale edit"; oldText.dispatchEvent({ type: "input" }); oldSave.click();
  assert.equal(h.evaluate("App.feedbackDraft.size"), 0);
  assert.equal(h.requests.filter((r) => r.options.method === "PUT").length, 1);
});

test("defensive save completion guard includes activation even when the run UUID is unchanged", async () => {
  const h = loadWorkbench({ manualFetch: true });
  await openRun(h);
  h.requests[0].resolve(feedbackDocument(canonicalResult().run.run_id)); await settle();
  const saving = h.evaluate("saveAnnotation('variable',400,{flags:[],note:'old'},h('span'))");
  const oldPut = h.requests.at(-1);
  // The UI blocks this switch; simulate an external activation to exercise the callback guard.
  h.evaluate("App.feedbackPending.clear(); App.feedbackDraft.clear()");
  await openRun(h);
  h.requests.at(-1).resolve(feedbackDocument(canonicalResult().run.run_id, "current saved")); await settle();
  h.evaluate("App.feedbackDraft.set('variable:400',{flags:[],note:'current draft'})");
  oldPut.resolve({ run_id: canonicalResult().run.run_id, kind: "variable", id: "400", annotation: { flags: [], note: "stale save" } });
  assert.equal(await saving, false);
  assert.equal(h.evaluate("annotationFor('variable',400).note"), "current saved");
  assert.equal(h.evaluate("App.feedbackDraft.get('variable:400').note"), "current draft");
});
