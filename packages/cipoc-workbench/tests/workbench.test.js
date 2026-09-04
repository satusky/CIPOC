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
  }

  append(...children) {
    for (const child of children.flat(Infinity)) {
      this.children.push(child instanceof FakeNode ? child : new FakeText(String(child)));
    }
  }

  appendChild(child) { this.append(child); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  addEventListener() {}
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  get firstChild() { return this.children[0] || null; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
}

class FakeText extends FakeNode {
  constructor(text) { super("#text"); this._text = text; }
}

function canonicalResult() {
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
    schema_version: "1.0",
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
      llm_usage_summary: {},
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

test("recognizes only the canonical 1.0 schema and indexes every result domain", () => {
  const { context } = loadWorkbench();
  const result = canonicalResult();
  context.result = result;
  vm.runInContext("indexState(result)", context);

  assert.equal(vm.runInContext("App.schemaVersion", context), "1.0");
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

  for (const version of [undefined, "0.9", "2.0"]) {
    context.bad = { ...result, schema_version: version };
    assert.throws(() => vm.runInContext("indexState(bad)", context), /schema_version/);
  }
});

test("all current views render canonical nested data and typed note-selection provenance", () => {
  const { context, document } = loadWorkbench();
  context.result = canonicalResult();
  vm.runInContext("indexState(result)", context);

  vm.runInContext("renderNotes()", context);
  assert.match(document.querySelector("#notes-list").textContent, /note-10/);

  vm.runInContext("renderChrome()", context);
  assert.match(document.querySelector("#case-summary").textContent, /run 123e4567 · 2\.0s/);

  vm.runInContext("renderControlRoom()", context);
  assert.match(document.querySelector("#control").textContent, /Primary Site/);

  vm.runInContext("renderTable()", context);
  assert.match(document.querySelector("#table-wrap").textContent, /C504/);

  context.noteView = vm.runInContext("noteDetail('note-2')", context);
  assert.match(context.noteView.textContent, /Scanner calls \(1\)/);
  assert.match(context.noteView.textContent, /Prompt and response bodies were not captured/);

  context.rejectedNoteView = vm.runInContext("noteDetail('note-A')", context);
  assert.match(context.rejectedNoteView.textContent, /Note type did not match the configured note filter/);

  context.groupView = vm.runInContext("groupDetail('diagnosis')", context);
  assert.match(context.groupView.textContent, /Requested items: 400/);
  assert.match(context.groupView.textContent, /surgery: present/);
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
