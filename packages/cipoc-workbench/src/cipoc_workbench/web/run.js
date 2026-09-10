"use strict";
/* View-time calculations only. Recorded usage buckets and clinical results
 * remain untouched; retained invocations are not a substitute for a summary. */

const Pricing = { catalog: null, source: null, error: null, generation: 0 };
const tokenCount = (value) => Number.isSafeInteger(value) && value >= 0;
const durationValue = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const metricCount = (value) => tokenCount(value) ? value.toLocaleString("en-US") : "not recorded";
const secondsText = (value) => durationValue(value) ? value.toFixed(3) + " s" : "unavailable";

function validateCatalog(raw) {
  if (!isRecord(raw) || !/^[A-Z]{3}$/.test(raw.currency || "") ||
      typeof raw.source !== "string" || !raw.source.trim() ||
      typeof raw.version !== "string" || !raw.version.trim() || !isRecord(raw.models)) {
    throw new Error("Expected currency (three uppercase letters), source, version and models.");
  }
  const models = Object.create(null);
  for (const [model, rates] of Object.entries(raw.models)) {
    if (!model.trim() || !isRecord(rates)) throw new Error("Each model needs a name and rate object.");
    models[model] = {};
    for (const key of ["input_per_million", "output_per_million", "cached_input_per_million"]) {
      const value = rates[key] ?? null;
      if (value !== null && !durationValue(value)) throw new Error(model + ": " + key + " must be a nonnegative finite number or null.");
      models[model][key] = value;
    }
  }
  return { currency: raw.currency, source: raw.source, version: raw.version, models };
}

function invocationCost(exchange, catalog) {
  const unavailable = (reason) => ({ amount: null, reason });
  const usage = exchange.usage;
  if (usage == null) return unavailable("Provider usage unavailable");
  if (!catalog) return unavailable("Pricing catalog unavailable");
  if (typeof exchange.model !== "string" || !exchange.model.trim()) return unavailable("Recorded model unavailable");
  const rates = Object.hasOwn(catalog.models, exchange.model) ? catalog.models[exchange.model] : null;
  if (!rates) return unavailable("No exact-model pricing");
  if (rates.input_per_million == null || rates.output_per_million == null) return unavailable("Input/output rates unavailable");
  const input = usage.input_tokens, output = usage.output_tokens, total = usage.total_tokens;
  if (![input, output, total].every(tokenCount) || !tokenCount(input + output) || total !== input + output) {
    return unavailable("Invalid or inconsistent token counts");
  }
  const reported = exchange.usage_reported_fields;
  if (reported != null) {
    if (!Array.isArray(reported) || !reported.includes("input_tokens") || !reported.includes("output_tokens")) {
      return unavailable("Required counts not verified as provider-reported");
    }
  } else if (input === 0 || output === 0) {
    return unavailable("Zero count has unverified provenance");
  }
  const inputDetails = usage.input_token_details ?? {}, outputDetails = usage.output_token_details ?? {};
  for (const [details, allowed] of [[inputDetails, ["cache_read"]], [outputDetails, ["reasoning"]]]) {
    if (!isRecord(details) || Object.values(details).some((count) => !tokenCount(count))) {
      return unavailable("Invalid token details");
    }
    if (Object.entries(details).some(([key, count]) => count > 0 && !allowed.includes(key))) {
      return unavailable("Unsupported billing-specific token details");
    }
  }
  const cached = inputDetails.cache_read ?? 0;
  if (cached > input || (outputDetails.reasoning ?? 0) > output) return unavailable("Token detail exceeds its total");
  if (cached > 0 && rates.cached_input_per_million == null) return unavailable("Cached-input rate unavailable");
  const terms = [(input - cached) * rates.input_per_million,
    cached > 0 ? cached * rates.cached_input_per_million : 0, output * rates.output_per_million];
  const amount = terms.reduce((a, b) => a + b, 0) / 1_000_000;
  if (!Number.isFinite(amount) || (amount === 0 && terms.some((term) => term > 0))) return unavailable("Estimate exceeds numeric precision");
  return { amount, reason: null };
}

function costText(amount, catalog = Pricing.catalog) {
  if (amount == null || !catalog) return "unavailable";
  const value = amount === 0 ? "0.00" : amount < 0.000001 ? "<0.000001" :
    amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 6 });
  return catalog.currency + " " + value;
}

function retainedInvocations(telemetry) {
  return Object.values(telemetry.llm_exchanges || {}).flat().concat(telemetry.unattributed_exchanges || []);
}

function aggregateIssue(telemetry, records) {
  if ((telemetry.collection_issues || []).some((issue) => /invalid_invocation_sequence|duplicate.*invocation/.test(issue.code))) {
    return "Derived aggregates withheld: invocation sequence is unreliable.";
  }
  const ids = new Set();
  for (const record of records) {
    if (record.invocation_id == null) continue;
    if (typeof record.invocation_id !== "string" || !record.invocation_id) return "Derived aggregates withheld: invalid invocation identity.";
    if (ids.has(record.invocation_id)) return "Derived aggregates withheld: duplicate invocation identity.";
    ids.add(record.invocation_id);
  }
  return null;
}

function costSummary(records, catalog, issue = null) {
  let amount = 0, costed = 0;
  const reasons = new Map();
  for (const record of records) {
    const cost = invocationCost(record, catalog);
    if (cost.amount === null) reasons.set(cost.reason, (reasons.get(cost.reason) || 0) + 1);
    else { amount += cost.amount; costed++; }
  }
  return { amount: !issue && costed && Number.isFinite(amount) ? amount : null,
    costed, missing: records.length - costed, reasons, issue };
}

function timingSummary(records, issue = null) {
  const result = { interval_seconds: null, interval_covered: 0, interval_invalid: 0 };
  for (const field of ["service_seconds", "queue_seconds"]) {
    let sum = 0, count = 0, max = 0;
    for (const record of records) if (durationValue(record[field])) {
      sum += record[field]; count++; max = Math.max(max, record[field]);
    }
    const available = !issue && count > 0 && Number.isFinite(sum);
    result[field] = { count, sum: available ? sum : null,
      mean: available ? sum / count : null, max: available ? max : null };
  }
  const intervals = [];
  const timestamp = (value) => typeof value === "string" && /T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? Date.parse(value) : NaN;
  for (const record of records) {
    if (record.started_at == null || record.finished_at == null) continue;
    const start = timestamp(record.started_at), finish = timestamp(record.finished_at);
    const elapsed = (finish - start) / 1000;
    // Wall-clock jumps do not invalidate the separately recorded monotonic time.
    if (!Number.isFinite(elapsed) || elapsed < 0 ||
        (durationValue(record.service_seconds) && Math.abs(elapsed - record.service_seconds) > 0.05)) {
      result.interval_invalid++; continue;
    }
    intervals.push([start, finish]);
  }
  result.interval_covered = intervals.length;
  if (!issue && intervals.length) result.interval_seconds = mergeRanges(intervals)
    .reduce((sum, [start, finish]) => sum + (finish - start) / 1000, 0);
  return result;
}

function metricCard(label, value, detail, warning = false) {
  return h("div", { class: "metric-card" + (warning ? " metric-warning" : "") },
    h("span", { class: "metric-label", text: label }), h("strong", { text: value }),
    h("span", { class: "faint", text: detail }));
}

function runSummaryChips() {
  const summary = App.observability.llm_usage_summary;
  const records = retainedInvocations(App.observability);
  const cost = costSummary(records, Pricing.catalog, aggregateIssue(App.observability, records));
  const complete = collectionStatus() === "complete" && summary?.missing_usage_invocations === 0;
  const countMatches = tokenCount(summary?.model_invocations) && summary.model_invocations === records.length;
  return [h("span", { class: "chip" + (complete ? "" : " warn"),
    text: summary && tokenCount(summary.total_tokens) ? metricCount(summary.total_tokens) + " recorded tokens" : "tokens unavailable" }),
    h("span", { class: "chip" + (cost.missing || !complete || !countMatches || cost.amount == null ? " warn" : ""),
      title: countMatches ? "Known estimate over retained invocations, not a provider invoice." :
        "Retained invocation count does not reconcile with an available recorded summary; cost coverage is incomplete.",
      text: "known estimate " + costText(cost.amount) + " (" + cost.costed + "/" + records.length + " retained calls)" })];
}

function exchangeMetrics(exchange) {
  const cost = invocationCost(exchange, Pricing.catalog);
  return kv([
    ["observed model-call duration", secondsText(exchange.service_seconds)],
    ["local queue wait", secondsText(exchange.queue_seconds)],
    ["known estimated cost", cost.amount == null ? "unavailable: " + cost.reason : costText(cost.amount)],
    ["verified provider scalars", exchange.usage_reported_fields == null ? "not recorded / unavailable" :
      Array.isArray(exchange.usage_reported_fields) ? exchange.usage_reported_fields.join(", ") || "none verified" : "invalid provenance"],
  ]);
}

function usageBreakdown(dimension, label, telemetry, records, issue) {
  const buckets = telemetry.llm_usage_summary?.["by_" + dimension];
  if (!buckets || !Object.keys(buckets).length) return section(label,
    h("p", { class: "faint", text: "No " + dimension + " buckets recorded." }));
  const fields = ["logical_calls", "model_invocations", "successful_invocations", "failed_invocations",
    "retry_invocations", "input_tokens", "output_tokens", "total_tokens"];
  const headers = [label, "Logical starts", "Invocations", "Succeeded", "Failed", "Retries", "Input", "Output", "Total",
    "Usage reported / missing", "Known estimate", "Costed / retained", "Service sum / coverage", "Queue sum / coverage", "Cost coverage details"];
  const rows = Object.entries(buckets).sort(([a], [b]) => a.localeCompare(b, "en")).map(([name, bucket]) => {
    const selected = records.filter((record) => (record[dimension] || "unknown") === name);
    const cost = costSummary(selected, Pricing.catalog, issue), timing = timingSummary(selected, issue);
    const values = [name, ...fields.map((key) => metricCount(bucket[key])),
      metricCount(bucket.usage_reported_invocations) + " / " + metricCount(bucket.missing_usage_invocations),
      costText(cost.amount), cost.costed + " / " + selected.length,
      secondsText(timing.service_seconds.sum) + " / " + timing.service_seconds.count + " of " + selected.length,
      secondsText(timing.queue_seconds.sum) + " / " + timing.queue_seconds.count + " of " + selected.length];
    return h("tr", {}, values.map((value, index) => h(index === 0 ? "th" : "td", index === 0 ? { scope: "row", text: value } : { text: value })),
      h("td", {}, h("details", {}, h("summary", { text: cost.missing + " unpriceable" }),
        cost.issue ? h("p", { text: cost.issue }) : null,
        h("ul", {}, [...cost.reasons].map(([reason, count]) => h("li", { text: count + ": " + reason }))))));
  });
  return section(label, h("div", { class: "metrics-table-wrap", tabindex: "0", "aria-label": label + " breakdown, scroll horizontally" },
    h("table", { class: "metrics-table" },
      h("thead", {}, h("tr", {}, headers.map((text) => h("th", { scope: "col", text })))),
      h("tbody", {}, rows))));
}

function refreshMetrics() {
  if (!App.run.run_id) return;
  renderChrome();
  if (App.view === "run") renderRun();
  if (App.selection) renderDetail();
}

async function loadEndpointCatalog(file = null) {
  const generation = ++Pricing.generation;
  try {
    let raw;
    if (file) raw = JSON.parse(await readLocalText(file));
    else {
      const response = await fetch("endpoint_catalog.json", { cache: "no-store" });
      if (!response.ok) throw new Error("Catalog unavailable (HTTP " + response.status + ").");
      raw = await response.json();
    }
    const catalog = validateCatalog(raw);
    if (generation !== Pricing.generation) return;
    Object.assign(Pricing, { catalog, source: file?.name || "endpoint_catalog.json", error: null });
  } catch (error) {
    if (generation !== Pricing.generation) return;
    // A rejected replacement does not silently discard the visible active rates.
    Pricing.error = error.message;
  }
  refreshMetrics();
}

function renderRun() {
  const telemetry = App.observability, summary = telemetry.llm_usage_summary;
  const records = retainedInvocations(telemetry), issue = aggregateIssue(telemetry, records);
  const cost = costSummary(records, Pricing.catalog, issue), timing = timingSummary(records, issue);
  const complete = collectionStatus() === "complete" && summary?.missing_usage_invocations === 0;
  const countMatches = tokenCount(summary?.model_invocations) && summary.model_invocations === records.length;
  const attempts = Object.values(telemetry.variable_attempts || {}).flat();
  const catalogInput = h("input", { type: "file", accept: "application/json,.json", "aria-label": "Load pricing catalog",
    onchange: (event) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (file) void loadEndpointCatalog(file);
    } });
  const body = h("div", { class: "run-content" },
    h("div", { class: "run-heading" }, h("div", {}, h("h2", { text: "Observability" }),
      h("p", { class: "faint", text: "Recorded execution, not a billing statement. Collection health is separate from the clinical outcome." }))),
    h("div", { class: "metrics-grid" },
      metricCard("Recorded tokens", metricCount(summary?.total_tokens),
        "Input " + metricCount(summary?.input_tokens) + " / output " + metricCount(summary?.output_tokens)),
      metricCard("Model invocations", metricCount(summary?.model_invocations),
        "Logical starts " + metricCount(summary?.logical_calls) + " / retries " + metricCount(summary?.retry_invocations)),
      metricCard("Collection", collectionStatus(),
        "Provider usage: " + metricCount(summary?.usage_reported_invocations) + " reported / " + metricCount(summary?.missing_usage_invocations) + " missing", !complete),
      metricCard("Known estimated cost", costText(cost.amount),
        cost.costed + " of " + records.length + " retained invocations priceable", cost.amount == null || cost.missing > 0 || !complete || !countMatches)),
    !complete ? h("p", { class: "metric-banner", text:
      "Totals and estimates may be incomplete. Collection is " + collectionStatus() +
      "; missing callbacks and provider usage cannot be treated as zero." }) : null,
    !countMatches ? h("p", { class: "metric-banner", text:
      "Retained invocation count does not reconcile with an available recorded summary. Derived coverage is limited to retained records; the summary is not recomputed." }) : null,
    issue ? h("p", { class: "metric-banner", text: issue }) : null,
    section("Run", kv([["source", App.runSource || "not recorded"], ["run ID", App.run.run_id],
      ["schema", App.schemaVersion], ["status", App.run.status], ["started", App.run.started_at],
      ["finished", App.run.finished_at], ["envelope duration", secondsText(App.run.duration_seconds)]]),
      h("p", { class: "faint", text: "Envelope duration can include initialization, cleanup and an interactive summary pause; it is not model service time." })),
    telemetryDetail(),
    section("Capture settings", kv([
      ["prompt character limit", telemetry.max_content_chars === null ? "unbounded" : telemetry.max_content_chars ?? "not recorded"],
      ["any prompt truncated", telemetry.content_truncated == null ? "not recorded" : telemetry.content_truncated ? "yes" : "no"],
    ]), h("p", { class: "metric-banner", text: "PHI: disabling content capture or truncating prompts does not de-identify this artifact. Full notes and validation candidates remain present; responses are not truncated." })),
    section("Recorded validation attempts", kv([
      ["attempts", attempts.length], ["valid verdicts", attempts.filter((a) => a.is_valid === true).length],
      ["rejected verdicts", attempts.filter((a) => a.is_valid === false).length],
      ...["group", "individual", "repair"].map((mode) => [mode + " mode", attempts.filter((a) => a.mode === mode).length]),
    ]), h("p", { class: "faint", text: "A valid verdict can be not-found. One group call can feed multiple validations; these are not invocation or found-variable counts. Missing records do not establish that no validations occurred." })),
    section("Pricing", h("label", { class: "check file-pick" }, catalogInput, h("span", { text: "Load Pricing..." })),
      Pricing.error ? h("p", { class: "metric-banner", role: "status", text: Pricing.error + (Pricing.catalog ? " Previous catalog remains active." : " Estimates unavailable.") }) : null,
      Pricing.catalog ? kv([["catalog", Pricing.source], ["source", Pricing.catalog.source], ["version", Pricing.catalog.version], ["currency", Pricing.catalog.currency]]) :
        h("p", { class: "faint", text: "No pricing catalog loaded. Review and feedback remain available." }),
      kv([["known estimated cost", costText(cost.amount)], ["costed retained invocations", cost.costed], ["unpriceable retained invocations", cost.missing]]),
      cost.reasons.size ? h("ul", { class: "faint" }, [...cost.reasons].map(([reason, count]) => h("li", { text: count + ": " + reason }))) : null,
      h("p", { class: "faint", text: "View-time estimate using exact recorded model names. Unknown count provenance, missing rates and unsupported billing details withhold estimates. No cache detail means ordinary-input pricing, not proof of complete cache reporting. Coverage refers only to retained invocations, not the provider's invoice." })),
    section("Invocation timing", kv([
      ["summed invocation time", secondsText(timing.service_seconds.sum)],
      ["mean / maximum invocation time", secondsText(timing.service_seconds.mean) + " / " + secondsText(timing.service_seconds.max)],
      ["service timing coverage", timing.service_seconds.count + " / " + records.length + " retained invocations"],
      ["summed queue wait", secondsText(timing.queue_seconds.sum)],
      ["mean / maximum queue wait", secondsText(timing.queue_seconds.mean) + " / " + secondsText(timing.queue_seconds.max)],
      ["queue timing coverage", timing.queue_seconds.count + " / " + records.length + " retained invocations"],
      ["time with an observed model call active", secondsText(timing.interval_seconds)],
      ["interval coverage", timing.interval_covered + " / " + records.length + "; " + timing.interval_invalid + " invalid or inconsistent intervals excluded"],
    ]), h("p", { class: "faint", text: "Concurrent sums can exceed run duration. Callback time includes network and SDK retries, not pre-callback queueing or post-callback parsing. Queue wait measures only the local synchronous limiter. The wall-clock interval union is an observed estimate, not exclusive model-caused delay or endpoint headroom." })),
    h("p", { class: "faint", text: "Breakdown token and call counts are recorded, not rebuilt. Logical starts belong to the initial invocation's bucket; a resolved-model bucket can contain retries without starts. Timing/cost columns cover retained records only; the collection and aggregate warnings above apply to every row." }),
    usageBreakdown("agent", "By agent", telemetry, records, issue),
    usageBreakdown("node", "By node", telemetry, records, issue),
    usageBreakdown("model", "By model", telemetry, records, issue));
  clear($("#view-run")).append(body);
}
