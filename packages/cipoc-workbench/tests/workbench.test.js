"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class FakeNode {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.className = "";
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this._text = "";
    this.listeners = {};
  }

  append(...children) {
    for (const child of children.flat(Infinity)) {
      this.children.push(child instanceof FakeNode ? child : new FakeText(String(child)));
    }
  }

  appendChild(child) { this.append(child); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  click() { this.listeners.click?.({ stopPropagation() {} }); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  get firstChild() { return this.children[0] || null; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
}

class FakeText extends FakeNode {
  constructor(text) { super("#text"); this._text = text; }
}

function descendants(node) {
  return [node, ...node.children.flatMap(descendants)];
}

function canonicalResult(schemaVersion = "1.0") {
  const note = (noteId, content) => ({
    note_id: noteId,
    date: "2026-09-04",
    note_type: "Pathology",
    content,
    summary: "Breast cancer pathology.",
    concepts: {
      cancer: { presence: true, confidence: "high", evidence: [{ note_id: noteId, text: content }] },
    },
    cancer_status: ["current"],
    cancer_mentions: [],
    flags: ["breast"],
  });

  const exchange = (entityKey, agent, node) => ({
    entity_key: entityKey,
    agent,
    node,
    attempt: 1,
    retry_ordinal: null,
    model: "test-model",
    prompt_messages: null,
    response: null,
    usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
    error: null,
  });

  return {
    schema_version: schemaVersion,
    run: {
      run_id: "123e4567-e89b-42d3-a456-426614174000",
      started_at: "2026-09-04T12:00:00Z",
      finished_at: "2026-09-04T12:00:02Z",
      duration_seconds: 2,
      status: "completed",
      config_fingerprint: {},
      contains_phi: true,
    },
    case: {
      case_facts: { gross_primary_site: "breast" },
      variable_results: {
        400: {
          item_id: 400,
          status: "extracted",
          value: "C504",
          extraction: {
            item_id: 400,
            value: "C504",
            explanation: "Upper-outer quadrant.",
            most_important_note: "note-2",
            spans: [{ note_id: "note-2", text: "Upper-outer quadrant" }],
            presence_confidence: "high",
            validation_errors: [],
            is_valid: true,
            extraction_attempts: 1,
          },
          reason: null,
          blocking_item_ids: [],
        },
      },
      note_selection: {
        "group:diagnosis": {
          group_id: "diagnosis",
          requested_item_ids: [400],
          candidate_note_ids: ["note-10", "note-2"],
          rejected_note_ids: { "note-A": ["note_type_mismatch", "cancer_status_mismatch"] },
          selected_note_ids: ["note-2"],
          discarded_note_ids: ["invented-X"],
          unevaluated_checks: ["keyword_filter_disabled", "temporal_anchor_unavailable"],
        },
      },
      fatal_blocker: null,
      report: { flags: [] },
    },
    inputs: {
      target_variables: [{
        group_id: "diagnosis",
        name: "Diagnosis",
        extract_as_group: true,
        stage: "initial",
        gate: ["treatment_present"],
        applies_to: null,
        note_filter: { note_types: ["Pathology"], keywords: ["breast"], cancer_status: [], within_days: 30 },
        variables: [{ item_id: 400, name: "Primary Site" }],
      }],
      structured_data: {},
    },
    corpus: {
      note_corpus: {
        "note-2": note("note-2", "Upper-outer quadrant"),
        "note-A": note("note-A", "Other note"),
        "note-10": note("note-10", "Earlier lexical identifier"),
      },
      note_digests: {
        "note-2": { note_id: "note-2", note_type: "Pathology", summary: "Digest" },
      },
      note_corpus_descriptors: {
        note_count: 3,
        concepts: { surgery: { presence: true } },
      },
    },
    observability: {
      ...(schemaVersion === "1.1" ? {
        collection_status: "complete", collection_issues: [], unattributed_exchanges: [],
      } : {}),
      llm_content_captured: false,
      max_content_chars: null,
      content_truncated: false,
      variable_attempts: {
        "group:diagnosis/variable:400": [{
          attempt: 1,
          mode: "group",
          candidate: { value: "C504", presence_confidence: "high", explanation: "Candidate" },
          validation_errors: [],
          is_valid: true,
        }],
      },
      llm_exchanges: {
        "note:note-2": [exchange("note:note-2", "note_scanner", "detect_concepts")],
        "group:diagnosis": [
          exchange("group:diagnosis", "note_retriever", "identify_relevant_notes"),
          exchange("group:diagnosis", "extractor", "extract_group_values"),
        ],
        "group:diagnosis/variable:400": [exchange("group:diagnosis/variable:400", "extractor", "extract_variable")],
      },
      llm_usage_summary: {
        input_tokens: 40, output_tokens: 8, total_tokens: 48,
        logical_calls: 4, model_invocations: 4, successful_invocations: 4,
        failed_invocations: 0, retry_invocations: 0,
        usage_reported_invocations: 4, missing_usage_invocations: 0,
        by_agent: {}, by_node: {}, by_model: {},
      },
    },
  };
}

function loadWorkbench() {
  const nodes = new Map();
  const document = {
    documentElement: new FakeNode("html"),
    createElement: (tag) => new FakeNode(tag),
    createTextNode: (text) => new FakeText(String(text)),
    addEventListener() {},
    querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, new FakeNode(selector));
      return nodes.get(selector);
    },
    querySelectorAll() { return []; },
  };
  const context = vm.createContext({
    console,
    document,
    Node: FakeNode,
    localStorage: { getItem: () => null, setItem() {} },
    window: {
      innerWidth: 1200,
      innerHeight: 800,
      addEventListener() {},
      matchMedia: () => ({ matches: false, addEventListener() {} }),
    },
  });
  const web = path.join(__dirname, "..", "src", "cipoc_workbench", "web");
  for (const file of ["app.js", "truth.js", "feedback.js", "notes.js", "control.js", "table.js", "detail.js"]) {
    vm.runInContext(fs.readFileSync(path.join(web, file), "utf8"), context, { filename: file });
  }
  return { context, document };
}

for (const schemaVersion of ["1.0", "1.1"]) {
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

  for (const version of [undefined, 1.1, "0.9", "1.2", "2.0"]) {
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
    "telemetry " + (schemaVersion === "1.1" ? "complete" : "unknown")));

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
    const text = document.querySelector("#detail-body").textContent;
    assert.equal(document.querySelector("#detail-title").textContent, "Run telemetry");
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
  vm.runInContext("indexState(result); renderChrome(); show('telemetry', 'run')", context);
  const text = document.querySelector("#detail-body").textContent;
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
