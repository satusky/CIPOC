"use strict";

/* Results for items no group requested: CS Site-Specific Factors extracted in
 * place of requested items (case.ssf_equivalents, 1.3), and any other result
 * outside the configured groups. The artifact is synthetic, hand-built on the
 * shared canonical fixture. */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { loadWorkbench, canonicalResult } = require("./helpers");

const extracted = (itemId, value) => ({
  item_id: itemId, status: "extracted", value, reason: null, blocking_item_ids: [],
  extraction: {
    item_id: itemId, value, explanation: "From the pathology note.", most_important_note: "note-2",
    spans: [{ note_id: "note-2", text: "Upper-outer quadrant" }], presence_confidence: "high",
    validation_errors: [], is_valid: true, extraction_attempts: 1,
  },
});
const closed = (itemId, status, reason) => ({ item_id: itemId, status, value: null, reason, blocking_item_ids: [] });

/* A 2012 case: FIGO Stage held in CS SSF2 (2890), both grade items in one
 * shared CS SSF7 (2861) beside Grade (440), a Gleason item BLOCKED by a schema
 * disagreement, and 2879 with a result but no record. */
function ssfResult() {
  const result = canonicalResult("1.3");
  result.case.case_facts = { primary_site: "C569", date_of_diagnosis: "20120315" };
  result.inputs.target_variables.push(
    { group_id: "site_specific_codes", name: "Site Specific Codes", stage: "dependent", extract_as_group: true,
      variables: [{ item_id: 3843, name: "Grade Clinical" }, { item_id: 3844, name: "Grade Pathological" },
        { item_id: 440, name: "Grade" }], replaces: { 440: [3843, 3844] } },
    { group_id: "gynecologic_only", name: "Gynecologic Only", stage: "dependent", extract_as_group: true,
      applies_to: { gross_primary_sites: ["ovary"] }, variables: [{ item_id: 3836, name: "FIGO Stage" }] },
    { group_id: "prostate_only", name: "Prostate Only", stage: "dependent", extract_as_group: true,
      variables: [{ item_id: 3838, name: "Gleason Patterns Clinical" }] },
  );
  Object.assign(result.case.variable_results, {
    3836: closed(3836, "not_applicable",
      "Not collected for 2012 diagnoses: FIGO Stage (3836) applies from 2018; CS SSF2 (2890, FIGO Stage) holds it for this year."),
    2890: extracted(2890, "300"),
    3843: closed(3843, "not_applicable",
      "Not collected for 2012 diagnoses: Grade Clinical (3843) applies from 2018; Grade (440) and CS SSF7 (2861, Nottingham) hold it for this year."),
    3844: closed(3844, "not_applicable",
      "Not collected for 2012 diagnoses: Grade Pathological (3844) applies from 2018; Grade (440) and CS SSF7 (2861, Nottingham) hold it for this year."),
    440: extracted(440, "2"),
    2861: extracted(2861, "020"),
    3838: closed(3838, "blocked",
      "Not collected for 2012 diagnoses: Gleason Patterns Clinical (3838) applies from 2018; the candidate schemas disagree on the CS Site-Specific Factor that holds it: cs/prostate holds it in CS SSF7 (2861, Gleason); cs/other holds no SSF for it."),
    2879: extracted(2879, "010"),
  });
  result.case.ssf_equivalents = {
    2890: { item_id: 2890, label: "CS SSF2", name: "FIGO Stage", replaces: [3836],
      schemas: ["cs/ovary"], group_id: "gynecologic_only" },
    2861: { item_id: 2861, label: "CS SSF7", name: "Nottingham or Bloom-Richardson (BR) Score/Grade",
      replaces: [3843, 3844], schemas: ["cs/breast"], group_id: "site_specific_codes" },
  };
  Object.assign(result.observability.variable_attempts, {
    "group:gynecologic_only/variable:2890": [{ attempt: 1, mode: "group",
      candidate: { value: "300", presence_confidence: "high" }, validation_errors: [], is_valid: true }],
    "group:elsewhere/variable:2879": [{ attempt: 1, mode: "individual",
      candidate: { value: "010", presence_confidence: "high" }, validation_errors: [], is_valid: true }],
  });
  return result;
}

function indexed(result = ssfResult()) {
  const harness = loadWorkbench();
  harness.context.result = result;
  harness.evaluate("indexState(result)");
  return harness;
}

const entryOf = (h, itemId) => h.evaluate(`(() => {
  const v = App.variables.find((v) => v.item_id === ${itemId});
  return v && { name: v.name, group_id: v.group_id, group_name: v.group_name, requested: v.requested };
})()`);

test("unrequested results are indexed under their own item numbers without rewriting the artifact", () => {
  const result = ssfResult();
  const original = JSON.stringify(result);
  const h = indexed(result);

  assert.deepEqual({ ...entryOf(h, 2890) },
    { name: "CS SSF2 · FIGO Stage", group_id: "gynecologic_only", group_name: "Gynecologic Only", requested: false });
  assert.deepEqual({ ...entryOf(h, 2861) }, { name: "CS SSF7 · Nottingham or Bloom-Richardson (BR) Score/Grade",
    group_id: "site_specific_codes", group_name: "Site Specific Codes", requested: false });
  // No record: the "Other results" group, appended after the configured ones.
  assert.deepEqual({ ...entryOf(h, 2879) },
    { name: "Item 2879", group_id: "not-requested", group_name: "Other results", requested: false });
  assert.equal(entryOf(h, 3836).requested, true);
  assert.equal(h.evaluate("App.variables.length"), 9);
  assert.deepEqual([...h.evaluate("App.groups.map((g) => g.group_id)")],
    ["diagnosis", "site_specific_codes", "gynecologic_only", "prostate_only", "not-requested"]);
  assert.equal(h.evaluate("App.inputs.target_variables.length"), 4);
  assert.deepEqual([...h.evaluate("App.replacedBy.get(3844).map((r) => r.item_id)")], [2861]);

  // Telemetry: the SSF under its group's key; an unplaced result by its item ID.
  assert.equal(h.evaluate("variableAttempts(2890).length"), 1);
  assert.equal(h.evaluate("variableAttempts(2879).length"), 1);
  assert.equal(JSON.stringify(result), original);
});

test("the control room shows each SSF in its group, dashed, and other results after the waves", () => {
  const h = indexed();
  h.evaluate("renderControlRoom()");
  const control = h.document.querySelector("#control");
  const bubble = (itemId) => control.querySelector(`[data-entity="variable:${itemId}"]`);

  for (const itemId of [2890, 2861, 2879]) assert.equal(bubble(itemId).getAttribute("data-unrequested"), "1");
  for (const itemId of [3836, 440, 400]) assert.equal(bubble(itemId).getAttribute("data-unrequested"), null);
  assert.match(bubble(2890).textContent, /^2890:CS SSF2 · FIGO Stage300$/);
  assert.match(bubble(2890).getAttribute("aria-label"), /, not requested$/);

  const card = (groupId) => control.querySelector(`[data-entity="group:${groupId}"]`);
  // 3836 alone is not applicable, but its SSF was extracted: the group ran.
  assert.ok(card("gynecologic_only").contains(bubble(2890)));
  assert.ok(!card("gynecologic_only").classList.contains("excluded"));
  assert.ok(card("site_specific_codes").contains(bubble(2861)));
  assert.ok(card("not-requested").contains(bubble(2879)));
  assert.deepEqual([...h.evaluate("waves().map((w) => w.title)")],
    ["wave 1 · initial", "wave 2 · dependent", "not requested"]);

  const tooltip = (itemId) => h.evaluate(
    `bubbleTooltip(App.variables.find((v) => v.item_id === ${itemId})).textContent`);
  assert.match(tooltip(2890), /requestedno, in place of 3836 FIGO Stage/);
  assert.match(tooltip(2861), /in place of 3843 Grade Clinical and 3844 Grade Pathological/);
  assert.match(tooltip(3836), /replaced by2890 CS SSF2 · FIGO Stage/);
  assert.match(tooltip(2879), /requestedno$/);
});

test("the table marks unrequested rows and replaced items, and filtering finds them", () => {
  const h = indexed();
  h.evaluate("renderTable()");
  const table = h.document.querySelector("#table-wrap");
  const row = (itemId) => table.querySelector(`[data-entity="variable:${itemId}"]`);
  assert.match(row(2890).textContent, /CS SSF2 · FIGO Stagenot requested/);
  assert.match(row(3836).textContent, /FIGO Stagereplaced by 2890/);
  assert.match(row(3843).textContent, /Grade Clinicalreplaced by 2861/);
  assert.doesNotMatch(row(440).textContent, /not requested|replaced by/);
  assert.match(table.textContent, /Other results · 1 of 1 · complete/);
  assert.match(table.textContent, /Gynecologic Only · 2 of 2 · complete/);

  h.evaluate("App.varFilter = 'not requested'; renderTable()");
  assert.deepEqual(table.querySelectorAll("[data-entity]").map((r) => r.getAttribute("data-entity")).sort(),
    ["variable:2861", "variable:2879", "variable:2890"]);
  h.evaluate("App.varFilter = 'ssf2'; renderTable()");
  assert.deepEqual(table.querySelectorAll("[data-entity]").map((r) => r.getAttribute("data-entity")), ["variable:2890"]);
});

test("dossiers link each SSF and the items it replaced both ways", () => {
  const h = indexed();
  const view = (itemId) => h.evaluate(`variableDetail(${itemId}).textContent`);

  assert.match(view(2890),
    /In place ofrequested3836 FIGO StageSSFCS SSF2 \(2890\)nameFIGO Stageschemascs\/ovaryNot requested: the diagnosis year predates/);
  assert.match(view(2861), /requested3843 Grade Clinical3844 Grade Pathological/);
  assert.match(view(3836), /Replaced byCS SSF22890 FIGO Stage · cs\/ovary/);
  assert.match(view(3844), /Replaced byCS SSF72861 Nottingham/);
  assert.match(view(3838), /reasonNot collected for 2012 diagnoses: .*the candidate schemas disagree/);
  assert.doesNotMatch(view(3838), /Replaced by/);
  assert.match(view(2879), /Not requestedNot requested; no record says why this result is in the run\./);
  assert.match(view(2890), /Attempts \(1\)/);
  assert.doesNotMatch(view(440), /In place of|Replaced by|Not requested/);

  h.evaluate("show('variable', 2890)");
  assert.equal(h.document.querySelector("#detail-title").textContent, "2890 · CS SSF2 · FIGO Stage");

  // The SSF is one of its group's variables; the other-results group has no gating or feedback.
  assert.match(h.evaluate("groupDetail('gynecologic_only').textContent"),
    /Variables \(2\).*2890 CS SSF2 · FIGO StageExtracted300not requested/);
  const other = h.evaluate("groupDetail('not-requested').textContent");
  assert.match(other, /^AboutResults for items no group requested.*Variables \(1\)2879 Item 2879Extracted010not requested$/);
  assert.doesNotMatch(other, /Gating|Feedback/);
});

test("unrequested results count in the totals, the legend and group roll-ups", () => {
  const h = indexed();
  h.evaluate("renderChrome()");
  const summary = h.document.querySelector("#case-summary").textContent;
  assert.match(summary, /9 variables/);
  assert.match(summary, /3 not requested/);

  const tally = Object.fromEntries(h.evaluate("lensTally('confidence')").map((row) => [row.key, row.count]));
  assert.deepEqual({ ...tally }, { high: 5, not_applicable: 3, blocked: 1 });
  assert.equal(h.evaluate("groupState(App.groupById.get('gynecologic_only')).total"), 2);
  assert.equal(h.evaluate("groupState(App.groupById.get('gynecologic_only')).coded"), 1);
  assert.equal(h.evaluate("groupState(App.groupById.get('prostate_only')).label"), "1 unresolved");

  // A reference value for an SSF is scored like any other.
  h.evaluate("indexTruth({2890: '300', 2879: '011'}, 'test')");
  assert.equal(h.evaluate("verdictFor(App.variables.find((v) => v.item_id === 2890)).verdict"), "match");
  assert.equal(h.evaluate("caseAccuracy().tested"), 2);
});

test("feedback attaches to an SSF result by its own item number", async () => {
  const h = loadWorkbench({ manualFetch: true });
  h.context.file = { name: "ssf.json", content: JSON.stringify(ssfResult()) };
  await h.evaluate("loadRunFile(file)");
  const runId = canonicalResult().run.run_id;
  h.requests[0].resolve({ run_id: runId, writable: true, read_only_reason: null,
    annotations: { variable: {}, group: {}, note: {} } });
  await new Promise((resolve) => setImmediate(resolve));

  const saving = h.evaluate("saveAnnotation('variable', 2890, {flags: [], note: 'FIGO from the op note'}, h('span'))");
  const put = h.requests.at(-1);
  assert.equal(put.url, "api/runs/" + runId + "/feedback/variable?entity_id=2890");
  put.resolve({ run_id: runId, kind: "variable", id: "2890", annotation: { flags: [], note: "FIGO from the op note" } });
  assert.equal(await saving, true);
  h.evaluate("renderControlRoom()");
  assert.equal(h.document.querySelector('#control [data-entity="variable:2890"]').getAttribute("data-annotated"), "1");
  assert.equal(h.evaluate("annotationFor('variable', 2890).note"), "FIGO from the op note");
  assert.equal(h.evaluate("currentAnnotation('variable', '2890').note"), "FIGO from the op note");
});

test("malformed SSF records are rejected when the artifact loads", () => {
  const h = loadWorkbench();
  const cases = [
    [(r) => { r.case.ssf_equivalents = []; }, /case.ssf_equivalents must be an object/],
    [(r) => { r.case.ssf_equivalents = { 2890: { ...r.case.ssf_equivalents[2890], item_id: 2891 } }; },
      /integer item_id matching its key/],
    [(r) => { r.case.ssf_equivalents[2890].replaces = "3836"; }, /ssf_equivalents.replaces must be an array/],
    [(r) => { r.case.ssf_equivalents[2890].replaces = ["3836"]; }, /must list integer item IDs/],
    [(r) => { r.case.ssf_equivalents[2890].label = 2; }, /ssf_equivalents.label must be text/],
    [(r) => { r.case.ssf_equivalents[2890].group_id = 7; }, /ssf_equivalents.group_id must be text/],
  ];
  for (const [mutate, message] of cases) {
    h.context.bad = ssfResult();
    mutate(h.context.bad);
    assert.throws(() => h.evaluate("indexState(bad)"), message);
  }
  // A record whose group isn't configured puts its SSF in "Other results".
  const result = ssfResult();
  result.case.ssf_equivalents[2890].group_id = null;
  const indexedNull = indexed(result);
  assert.equal(indexedNull.evaluate("App.variables.find((v) => v.item_id === 2890).group_id"), "not-requested");
  assert.equal(indexedNull.evaluate("App.variables.find((v) => v.item_id === 2890).name"), "CS SSF2 · FIGO Stage");
});

test("artifacts with only requested results index and render as before", () => {
  const example = JSON.parse(fs.readFileSync(
    path.join(__dirname, "..", "src", "cipoc_workbench", "example", "case_state.json"), "utf8"));
  const artifacts = [["bundled 1.0 example", example],
    ...["1.0", "1.1", "1.2", "1.3"].map((version) => [version, canonicalResult(version)])];
  for (const [label, result] of artifacts) {
    const h = indexed(result);
    assert.ok(h.evaluate("App.groups === App.inputs.target_variables"), label);
    assert.equal(h.evaluate("App.variables.filter(isUnrequested).length"), 0, label);
    assert.equal(h.evaluate("App.replacedBy.size + App.unrequestedKeys.size"), 0, label);
    h.evaluate("renderChrome(); renderControlRoom(); renderTable()");
    for (const selector of ["#case-summary", "#control", "#table-wrap"]) {
      assert.doesNotMatch(h.document.querySelector(selector).textContent, /not requested|Other results|replaced by/, label);
    }
    assert.equal(h.document.querySelector("#control").querySelectorAll("[data-unrequested]").length, 0, label);
    assert.ok(!h.evaluate("waves().some((w) => w.title === 'not requested')"), label);
  }
});
