"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { loadWorkbench, canonicalResult } = require("./helpers");

const catalog = () => ({ currency: "USD", source: "Test contract", version: "test-v1", models: {
  "test-model": { input_per_million: 2, output_per_million: 10, cached_input_per_million: 1 },
} });
const invocation = (overrides = {}) => ({ model: "test-model", usage: {
  input_tokens: 100, output_tokens: 20, total_tokens: 120,
}, ...overrides });

function metricsHarness() {
  const harness = loadWorkbench();
  harness.context.rawCatalog = catalog();
  harness.evaluate("Pricing.catalog = validateCatalog(rawCatalog); Pricing.source = 'test.json'");
  harness.context.result = canonicalResult("1.2");
  harness.evaluate("indexState(result)");
  return harness;
}

test("pricing validates contract rates and exact model keys without guessing aliases", () => {
  const { context, evaluate } = metricsHarness();
  for (const invalid of [null, [], {}, { ...catalog(), currency: "usd" }, { ...catalog(), version: "" },
    { ...catalog(), models: { x: { input_per_million: -1 } } },
    { ...catalog(), models: { x: { input_per_million: "2" } } },
    { ...catalog(), models: { x: { input_per_million: Infinity } } }]) {
    context.invalid = invalid;
    assert.throws(() => evaluate("validateCatalog(invalid)"));
  }
  context.call = invocation();
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), 0.0004);
  context.call.model = "TEST-MODEL";
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /exact-model/);
  context.call.model = "toString";
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /exact-model/);
  context.rawCatalog = catalog();
  context.rawCatalog.models.null = context.rawCatalog.models["test-model"];
  context.rawCatalog.models.undefined = context.rawCatalog.models["test-model"];
  evaluate("Pricing.catalog = validateCatalog(rawCatalog)");
  context.call.model = null;
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /Recorded model unavailable/);
  delete context.call.model;
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /Recorded model unavailable/);
  context.rawCatalog = { ...catalog(), models: {} };
  assert.doesNotThrow(() => evaluate("validateCatalog(rawCatalog)"));
});

test("unverified zeros are withheld, explicit incomplete provenance overrides positive legacy counts", () => {
  const { context, evaluate } = metricsHarness();
  context.call = invocation({ usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } });
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /unverified/);
  context.call.usage_reported_fields = ["input_tokens", "output_tokens", "total_tokens"];
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), 0);
  context.call = invocation({ usage_reported_fields: [] });
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), null);
  context.call.usage_reported_fields = ["input_tokens", "total_tokens"];
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), null);
  context.call.usage_reported_fields = null;
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), 0.0004);
  context.call.usage = null;
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /usage unavailable/);
});

test("cost never coerces malformed counts, invents cached rates or adds token breakdowns", () => {
  const { context, evaluate } = metricsHarness();
  context.call = invocation({ usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120,
    input_token_details: { cache_read: 40 }, output_token_details: { reasoning: 10 } } });
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), 0.00036);
  evaluate("Pricing.catalog.models['test-model'].cached_input_per_million = null");
  assert.match(evaluate("invocationCost(call, Pricing.catalog).reason"), /Cached-input rate/);
  for (const usage of [
    { input_tokens: 100, output_tokens: 20, total_tokens: 100 },
    { input_tokens: "100", output_tokens: 20, total_tokens: 120 },
    { input_tokens: true, output_tokens: 20, total_tokens: 21 },
    { input_tokens: -1, output_tokens: 20, total_tokens: 19 },
    { input_tokens: Number.MAX_SAFE_INTEGER + 1, output_tokens: 20, total_tokens: Number.MAX_SAFE_INTEGER + 21 },
    { input_tokens: 100, output_tokens: 20, total_tokens: 120, input_token_details: { cache_read: 101 } },
    { input_tokens: 100, output_tokens: 20, total_tokens: 120, output_token_details: { reasoning: 21 } },
    { input_tokens: 100, output_tokens: 20, total_tokens: 120, input_token_details: { audio: 5 } },
    { input_tokens: 100, output_tokens: 20, total_tokens: 120, output_token_details: { provider_specific: 5 } },
  ]) {
    context.call = invocation({ usage });
    assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), null);
  }
  context.call = invocation();
  evaluate("Pricing.catalog.models['test-model'].input_per_million = 0; Pricing.catalog.models['test-model'].output_per_million = 0");
  assert.equal(evaluate("invocationCost(call, Pricing.catalog).amount"), 0);
  assert.equal(evaluate("costText(0)"), "USD 0.00");
  assert.equal(evaluate("costText(0.00000001)"), "USD <0.000001");
});

test("cost aggregates include failed, retried and diagnostic calls exactly once without altering summaries", () => {
  const { context, evaluate } = metricsHarness();
  const telemetry = context.result.observability;
  telemetry.collection_status = "partial";
  telemetry.collection_issues = [{ code: "unbound_task", message: "No task binding" }];
  telemetry.unattributed_exchanges = [invocation({ invocation_id: "diagnostic", error: "429", retry_ordinal: 1 })];
  telemetry.llm_exchanges["note:note-2"][0].usage = null;
  const before = JSON.stringify(context.result);
  evaluate("renderChrome(); setView('run')");
  const stats = evaluate("costSummary(retainedInvocations(App.observability), Pricing.catalog)");
  assert.equal(stats.costed, 4);
  assert.equal(stats.missing, 1);
  assert.ok(Math.abs(stats.amount - 0.00052) < 1e-12);
  assert.equal(JSON.stringify(context.result), before);
  const text = evaluate("document.querySelector('#view-run').textContent");
  assert.match(text, /Totals and estimates may be incomplete/);
  assert.match(text, /does not reconcile/);
  assert.match(text, /No task binding/);
  assert.match(text, /diagnostic/);
});

test("invalid invocation identity withholds cost and timing sums but retains individual cards", () => {
  const { context, evaluate } = metricsHarness();
  const call = context.result.observability.llm_exchanges["note:note-2"][0];
  call.invocation_id = "duplicate";
  call.service_seconds = 2;
  context.result.observability.unattributed_exchanges = [{ ...call }];
  evaluate("setView('run')");
  assert.match(evaluate("document.querySelector('#view-run').textContent"), /duplicate invocation identity/);
  assert.equal(evaluate("costSummary(retainedInvocations(App.observability), Pricing.catalog, aggregateIssue(App.observability, retainedInvocations(App.observability))).amount"), null);
  assert.equal(evaluate("timingSummary(retainedInvocations(App.observability), 'unsafe').service_seconds.sum"), null);
  context.result.observability.unattributed_exchanges = [];
  context.result.observability.collection_issues = [{ code: "invalid_invocation_sequence", message: "Retry gap" }];
  assert.match(evaluate("aggregateIssue(App.observability, retainedInvocations(App.observability))"), /sequence/);
});

test("header flags retained-summary mismatch even when collection and provider usage are complete", () => {
  const { context, document, evaluate } = metricsHarness();
  context.result.observability.llm_usage_summary.model_invocations = 5;
  evaluate("renderChrome()");
  const estimate = [...document.querySelectorAll("#case-summary .chip")].find((node) => node.textContent.startsWith("known estimate"));
  assert.ok(estimate.classList.contains("warn"));
  assert.match(estimate.getAttribute("title"), /does not reconcile/);
});

test("each breakdown row explains its own unpriceable invocations", () => {
  const { context, document, evaluate } = metricsHarness();
  context.result.observability.llm_usage_summary.by_model = {
    "test-model": {}, "missing-rate": {},
  };
  const calls = context.result.observability.llm_exchanges["group:diagnosis"];
  calls[0].model = "missing-rate";
  calls[1].usage_reported_fields = [];
  evaluate("setView('run')");
  const rows = document.querySelectorAll(".metrics-table tbody tr");
  assert.match(rows.find((row) => row.children[0].textContent === "missing-rate").textContent, /No exact-model pricing/);
  assert.match(rows.find((row) => row.children[0].textContent === "test-model").textContent, /Required counts not verified/);
  assert.match(rows[0].textContent, /1 unpriceable/);
});

test("timing separates monotonic sums, queue waits, interval unions and missing measurements", () => {
  const { context, evaluate } = metricsHarness();
  context.records = [
    { started_at: "2026-09-04T12:00:02Z", finished_at: "2026-09-04T12:00:05Z", service_seconds: 3, queue_seconds: 0 },
    { started_at: "2026-09-04T12:00:00Z", finished_at: "2026-09-04T12:00:03Z", service_seconds: 3, queue_seconds: 2 },
    { started_at: "2026-09-04T12:00:08Z", finished_at: "2026-09-04T12:00:07Z", service_seconds: 1, queue_seconds: null },
    { started_at: "2026-09-04T12:00:10Z", finished_at: "2026-09-04T12:01:10Z", service_seconds: 2 },
    { service_seconds: 0 }, {},
  ];
  const timing = evaluate("timingSummary(records)");
  assert.equal(timing.service_seconds.sum, 9);
  assert.equal(timing.service_seconds.count, 5);
  assert.equal(timing.service_seconds.mean, 1.8);
  assert.equal(timing.service_seconds.max, 3);
  assert.equal(timing.queue_seconds.count, 2);
  assert.equal(timing.queue_seconds.sum, 2);
  assert.equal(timing.interval_seconds, 5);
  assert.equal(timing.interval_covered, 2);
  assert.equal(timing.interval_invalid, 2);
  assert.equal(evaluate("timingSummary([{}]).service_seconds.sum"), null);
  assert.equal(evaluate("timingSummary([{service_seconds:0,queue_seconds:0}]).queue_seconds.sum"), 0);
  assert.equal(evaluate("timingSummary([{service_seconds:Infinity,queue_seconds:-1}]).service_seconds.sum"), null);
});

test("dashboard renders recorded buckets, retry-only models, valid not-found attempts and capture limits", () => {
  const { context, document, evaluate } = metricsHarness();
  const telemetry = context.result.observability;
  const bucket = { input_tokens: 11, output_tokens: 7, total_tokens: 18,
    logical_calls: 0, model_invocations: 1, successful_invocations: 1, failed_invocations: 0,
    retry_invocations: 1, usage_reported_invocations: 1, missing_usage_invocations: 0 };
  telemetry.llm_usage_summary.by_model = { "resolved-model": bucket };
  telemetry.llm_usage_summary.by_agent = { extractor: { ...bucket } };
  telemetry.llm_usage_summary.by_node = { repair: { ...bucket } };
  telemetry.max_content_chars = 50;
  telemetry.content_truncated = true;
  telemetry.variable_attempts["group:diagnosis/variable:400"].push({ attempt: 2, mode: "repair", candidate: null, is_valid: true });
  const before = JSON.stringify(telemetry);
  evaluate("setView('run')");
  assert.equal(document.querySelectorAll(".metrics-table").length, 3);
  assert.equal(document.querySelectorAll(".metrics-table tbody tr").length, 3);
  const modelRow = document.querySelectorAll(".metrics-table tbody tr")[2];
  assert.equal(modelRow.children[0].textContent, "resolved-model");
  assert.equal(modelRow.children[1].textContent, "0");
  assert.equal(modelRow.children[8].textContent, "18");
  assert.match(document.querySelector("#view-run").textContent, /prompt character limit50/);
  assert.match(document.querySelector("#view-run").textContent, /valid verdicts2/);
  assert.match(document.querySelector("#view-run").textContent, /repair mode1/);
  assert.match(document.querySelector("#view-run").textContent, /valid verdict can be not-found/);
  assert.equal(JSON.stringify(telemetry), before);
});

test("legacy, unavailable and empty summaries stay unavailable instead of being reconstructed", () => {
  const { context, document, evaluate } = metricsHarness();
  for (const summary of [undefined, null, {}]) {
    context.result.observability.llm_usage_summary = summary;
    context.result.observability.collection_status = null;
    evaluate("renderChrome(); setView('run')");
    assert.match(document.querySelector("#case-summary").textContent, /tokens unavailable/);
    assert.match(document.querySelector("#view-run").textContent, /not recorded/);
    assert.match(document.querySelector("#view-run").textContent, /unknown/);
    assert.equal(context.result.observability.llm_usage_summary, summary);
  }
});

test("catalog loads are optional, latest-wins, and a bad replacement keeps prior catalog visible", async () => {
  const harness = loadWorkbench({ manualFetch: true, manualFiles: true });
  const { context, evaluate, requests, reads } = harness;
  context.result = canonicalResult("1.2");
  evaluate("indexState(result)");
  const startup = evaluate("loadEndpointCatalog()");
  const chosen = { ...catalog(), source: "Chosen", version: "v2" };
  context.file = { name: "local-pricing.json", content: JSON.stringify(chosen) };
  const local = evaluate("loadEndpointCatalog(file)");
  reads[0].resolve();
  await local;
  requests[0].resolve(catalog());
  await startup;
  assert.equal(evaluate("Pricing.catalog.source"), "Chosen");
  context.file = { name: "bad.json", content: "{" };
  const bad = evaluate("loadEndpointCatalog(file)");
  reads[1].resolve();
  await bad;
  assert.equal(evaluate("Pricing.catalog.source"), "Chosen");
  assert.ok(evaluate("Pricing.error"));
  evaluate("setView('run')");
  assert.match(harness.document.querySelector("#view-run").textContent, /Previous catalog remains active/);
  assert.equal(evaluate("App.run.run_id"), context.result.run.run_id);
});

test("pricing picker uses local file reading and resets its native selection", async () => {
  const harness = loadWorkbench({ manualFiles: true });
  const { context, document, reads, evaluate } = harness;
  context.result = canonicalResult();
  evaluate("indexState(result); setView('run')");
  const input = document.querySelector('input[aria-label="Load pricing catalog"]');
  input.value = "same.json";
  input.files = [{ name: "same.json", content: JSON.stringify(catalog()) }];
  input.dispatchEvent({ type: "change" });
  assert.equal(input.value, "");
  reads[0].resolve();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(evaluate("Pricing.catalog.version"), "test-v1");
});
