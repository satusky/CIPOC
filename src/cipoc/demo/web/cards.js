/* Standalone snapshot-only entity cards. Load before app.js.
 * Catalog entries supply identities/names only; no captured messages are read.
 * Navigation uses raw entity IDs. A variable's instanceKey may also pin its
 * enclosing extract_branch (including a pass with no variable instance yet).
 * All sections remain in the document flow. Prose excerpts are deterministic;
 * checks, codes, IDs, reasons and roster entries are never count-limited.
 */
(function (root) {
  "use strict";

  const obj = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const list = (value) => Array.isArray(value) ? value : [];
  const own = (value, key) => Object.prototype.hasOwnProperty.call(obj(value), key);
  const str = (value) => value == null ? "" : String(value);
  const same = (a, b) => a != null && b != null && str(a) === str(b);
  const esc = (value) => str(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const slug = (value) => str(value).toLowerCase().replace(/[^a-z0-9]+/g, "-") || "unrecorded";
  const words = (value) => str(value).replace(/_/g, " ");
  const present = (value) => value != null && value !== "";
  const empty = (message = "Not recorded") => `<p class="entity-empty">${esc(message)}</p>`;
  const muted = (message) => `<p class="entity-muted">${esc(message)}</p>`;
  const labels = {
    unrecorded: "Not recorded", pending: "Pending", active: "Active", done: "Completed",
    extracted: "Extracted", not_found: "Not found", invalid: "Invalid", error: "Error",
    structured_data: "Structured data", skipped: "Skipped", blocked: "Blocked",
    not_applicable: "Not applicable", unavailable: "Instance unavailable",
  };
  const statusText = (status) => labels[status] || words(status);
  const badge = (status, css = "entity-status") =>
    `<span class="${css} status-${slug(status)}">${esc(statusText(status))}</span>`;
  const section = (title, html, extra = "") => `<section class="entity-section ${extra}">
    <h3 class="entity-section-title">${esc(title)}</h3>${html}</section>`;
  const sentences = new Intl.Segmenter("en", {granularity: "sentence"});

  function firstSentence(value) {
    const segments = sentences.segment(str(value).trim());
    return segments[Symbol.iterator]().next().value?.segment.trim() || "";
  }

  function confidenceBadge(value, compact = false) {
    const level = ["max", "high", "medium", "low"].includes(value) ? value : "unrecorded";
    const label = present(value) ? str(value) : "Not recorded";
    return `<span class="entity-confidence confidence-${level}" title="Confidence: ${esc(label)}" aria-label="Confidence: ${esc(label)}">` +
      `${esc(label)}${compact ? "" : " confidence"}</span>`;
  }

  function prose(value, limit = 600, missing = "Not recorded") {
    if (!present(value)) return empty(missing);
    const chars = Array.from(str(value));
    if (chars.length <= limit) return `<p class="entity-prose">${esc(value)}</p>`;
    return `<p class="entity-prose">${esc(chars.slice(0, limit).join(""))}…</p>` +
      muted(`Deterministic excerpt · first ${limit} of ${chars.length} characters`);
  }

  function navigation(kind, id, text, css, groupId, instanceKey) {
    return `<button type="button" class="${css}" data-entity-kind="${esc(kind)}" data-entity-id="${esc(id)}"` +
      (groupId != null ? ` data-group-id="${esc(groupId)}"` : "") +
      (instanceKey != null ? ` data-instance-key="${esc(instanceKey)}"` : "") + `>${text}</button>`;
  }

  function card(kind, title, eyebrow, meta, status, body, valueHtml = "") {
    return `<article class="entity-card entity-${kind}"><header class="entity-header">
      <div class="entity-eyebrow">${esc(eyebrow)}</div><h2 class="entity-title">${esc(title)}</h2>
      <div class="entity-meta">${esc(meta)}</div>${badge(status)}${valueHtml}</header>
      <div class="entity-body">${body}</div></article>`;
  }

  // Sorting is explicit, stable and independent of object-key insertion order
  // when the trace supplies started_t/index. Never sort by completion time.
  function instances(snapshot) {
    return Object.entries(obj(snapshot.instances)).map(([key, value], order) => ({
      ...obj(value), key: str(obj(value).key || key), order,
    }));
  }
  function latest(values) {
    const number = (value) => typeof value === "number" && Number.isFinite(value) ? value : -Infinity;
    return values.slice().sort((a, b) =>
      (number(a.started_t) - number(b.started_t) || 0) ||
      (number(a.index) - number(b.index) || 0) || a.order - b.order).pop() || null;
  }
  const requested = (instance) => {
    const result = obj(obj(instance).result).requested_variables;
    return obj(result || obj(obj(instance).input).requested_variables);
  };
  const task = (instance) => obj(obj(obj(instance).result).task || obj(obj(instance).input).task);
  const variableInfo = (instance) => obj(task(instance).variable || obj(obj(obj(instance).input).task).variable);
  const groupId = (instance) => requested(instance).group_id;
  const itemId = (instance) => variableInfo(instance).item_id ??
    (list(obj(obj(instance).result).variable_results).length === 1 ? obj(instance.result.variable_results[0]).item_id : undefined);
  const childOf = (child, parent) => !!child && !!parent && child.key.startsWith(`${parent.key}/`);
  const parentOf = (instance, all) => all.filter((g) => g.node === "extract_branch" && childOf(instance, g))
    .sort((a, b) => b.key.length - a.key.length)[0] || null;
  const groupVariables = (group) => list(requested(group).variables);
  const isActive = (instance) => !!instance && (instance.status === "active" || instance.active > 0);
  const isError = (instance) => !!instance && (instance.status === "error" || instance.errors > 0 || present(instance.error));

  function recordFrom(payload, id) {
    const results = obj(payload).variable_results;
    if (Array.isArray(results)) return results.find((v) => same(v.item_id, id)) || null;
    const entry = own(results, str(id)) ? results[str(id)] : null;
    if (entry && (!own(entry, "item_id") || same(entry.item_id, id))) return entry;
    return list(obj(obj(payload).extracted_values).variables).find((v) => same(v.item_id, id)) || null;
  }

  // Root details can supplement progress's shortened terminal reasons. Shared
  // scanner/retriever/extractor NodeDetails cannot identify a concurrent pass.
  function rootRecord(snapshot, id, gid) {
    const rootNodes = new Set(["initialize_case", "plan_extraction", "check_state", "merge_and_update", "finalize_case"]);
    const candidates = Object.entries(obj(snapshot.details)).filter(([key, d]) => rootNodes.has(key) || rootNodes.has(d.map_node_id))
      .map(([, d], order) => ({...d, order, started_t: d.finished_t ?? d.started_t}));
    for (const detail of candidates.sort((a, b) => (b.started_t ?? -1) - (a.started_t ?? -1) || b.order - a.order)) {
      for (const payload of [detail.result, detail.input]) {
        const groups = list(obj(payload).target_variables);
        if (gid != null && groups.length && !groups.some((g) => same(g.group_id, gid) && list(g.variables).some((v) => same(v.item_id, id)))) continue;
        const record = recordFrom(payload, id);
        if (record && record.status !== "pending") return record;
      }
    }
    return null;
  }

  function resolveVariable(selection, snapshot, all, catalog) {
    const id = str(selection.id);
    const progress = list(obj(snapshot.progress).variables);
    let gid = selection.groupId;
    let instance = null;
    let group = null;
    const explicit = selection.instanceKey != null;
    if (explicit) {
      const pin = all.find((i) => i.key === str(selection.instanceKey));
      if (pin && pin.node === "variable_branch" && same(itemId(pin), id)) {
        instance = pin;
        group = parentOf(pin, all);
        if (gid != null && !same(groupId(group), gid)) return {id, unavailable: true};
      } else if (pin && pin.node === "extract_branch" && (gid == null || same(groupId(pin), gid))) {
        group = pin;
        const member = groupVariables(group).some((v) => same(v.item_id, id));
        instance = latest(all.filter((i) => i.node === "variable_branch" && same(itemId(i), id) && childOf(i, group)));
        // A latest group pass can also contain planned structured/skipped items.
        const current = latest(all.filter((i) => i.node === "extract_branch" && same(groupId(i), groupId(group))));
        if (!member && !instance && !(current === group && progress.some((v) => same(v.item_id, id) && same(v.group_id, groupId(group))))) return {id, unavailable: true};
      } else return {id, unavailable: true};
    } else {
      const groups = all.filter((i) => i.node === "extract_branch" && (gid == null || same(groupId(i), gid)) &&
        (groupVariables(i).some((v) => same(v.item_id, id)) ||
         progress.some((v) => same(v.item_id, id) && same(v.group_id, groupId(i))) ||
         all.some((v) => v.node === "variable_branch" && same(itemId(v), id) && childOf(v, i))));
      group = latest(groups);
      instance = latest(all.filter((i) => i.node === "variable_branch" && same(itemId(i), id) &&
        (group ? childOf(i, group) : gid == null)));
      if (!group && instance) group = parentOf(instance, all);
    }
    gid = groupId(group) ?? gid;
    const rows = progress.filter((v) => same(v.item_id, id) && (gid == null || same(v.group_id, gid)));
    const row = rows.length === 1 ? rows[0] : null;
    gid = gid ?? obj(row).group_id;
    const currentGroup = group && latest(all.filter((i) => i.node === "extract_branch" && same(groupId(i), gid)));
    const memberOfPass = groupVariables(group).some((v) => same(v.item_id, id));
    const allowProgress = (!explicit || (!instance && group === currentGroup)) &&
      !(group && isActive(group) && memberOfPass && obj(row).terminal);
    const info = variableInfo(instance);
    const scoped = groupVariables(group).find((v) => same(v.item_id, id)) || {};
    const identities = list(catalog.groups).filter((g) => gid == null || same(g.id, gid))
      .flatMap((g) => list(g.variables)).filter((v) => same(v.itemId, id));
    const name = info.name || scoped.name || obj(row).name || obj(identities[0]).name || `Item ${id}`;
    const ownRecord = recordFrom(obj(instance).result, id);
    const branchRecord = recordFrom(obj(group).result, id);
    // A newer variable branch must not inherit a prior candidate/result from
    // the group reducer. Terminal progress fills only instance-free outcomes.
    const rootResult = !instance && allowProgress && row && row.terminal ? rootRecord(snapshot, id, gid) : null;
    const consistentRoot = rootResult && rootResult.status === row.status &&
      (!own(row, "value") || rootResult.value === row.value);
    const record = ownRecord || (!instance ? branchRecord : null) || (consistentRoot ? rootResult : null);
    const extraction = obj(record).extraction || (record && !own(record, "status") ? record : null);
    let candidate = extraction || obj(task(instance)).candidate || null;
    const lastAttempt = list(obj(instance).attempts).at(-1);
    if (!extraction && lastAttempt && own(lastAttempt, "value") &&
        (!candidate || candidate.value !== lastAttempt.value ||
         (lastAttempt.attempt != null && lastAttempt.attempt !== task(instance).extraction_attempts))) {
      // Condensed attempts retain the current scalar, not its explanation or
      // spans. Do not attach an earlier candidate's evidence to a repaired code.
      candidate = {item_id: id, value: lastAttempt.value};
    }
    const checks = list(obj(instance).attempts).filter((a) => a.node === "validate_extraction");
    let status = "unrecorded";
    if (isError(instance)) status = "error";
    else if (isActive(instance)) status = "active";
    else if (extraction && extraction.is_valid === false) status = "invalid";
    else if (record && present(record.status)) status = record.status;
    else if (extraction && extraction.is_valid === true && own(extraction, "value")) status = extraction.value == null ? "not_found" : "extracted";
    else if (instance) {
      const last = list(instance.attempts).at(-1);
      if (last && last.node === "validate_extraction" && typeof last.is_valid === "boolean") {
        status = !last.is_valid ? "invalid" : own(last, "value") ? (last.value == null ? "not_found" : "extracted") : "done";
      } else status = !last && instance.status === "invalid" ? "invalid" : "done";
    } else if (allowProgress && row) {
      status = row.status || "pending";
      if (status === "error" && row.flag === "!") status = "invalid";
      if (status === "pending" && ["retrieve", "extract", "validate"].includes(row.stage)) status = "active";
    } else if (group) status = "pending";
    const lastCheck = checks.at(-1);
    let value;
    if (candidate && own(candidate, "value")) value = candidate.value;
    else if (record && own(record, "value")) value = record.value;
    else if (instance && lastCheck && list(instance.attempts).at(-1) === lastCheck) value = lastCheck.value;
    else if (!instance && allowProgress && row && status !== "pending") value = row.value;
    const reasons = [];
    for (const reason of [obj(record).reason, !instance && allowProgress ? obj(row).detail : null, obj(instance).error]) {
      const text = typeof reason === "object" && reason ? reason.message : reason;
      if (present(text) && !reasons.includes(str(text))) reasons.push(str(text));
    }
    if (list(obj(record).blocking_item_ids).length) reasons.push(`Blocked by items: ${record.blocking_item_ids.join(", ")}`);
    return {id, gid, group, instance, name, status, value, candidate, checks, reasons,
      errors: list(obj(extraction).validation_errors),
      definition: info.description || scoped.description,
      confidence: obj(candidate).presence_confidence || (!instance && allowProgress ? obj(row).confidence : null)};
  }

  function valueText(value, status) {
    if (value === undefined) return "Not recorded";
    if (value === null) return status === "not_found" ? "null · no value found" : "null";
    return value === "" ? '"" (empty string)' : str(value);
  }

  function validation(view) {
    let html = view.checks.length ? `<div class="entity-attempts">${view.checks.map((check, index) => {
      const outcome = check.is_valid === true ? "Passed recorded checks" : check.is_valid === false ? "Failed recorded checks" : "Outcome not recorded";
      return `<div class="entity-attempt"><div class="entity-attempt-head"><strong>Check ${index + 1}</strong>` +
        (check.attempt != null ? ` <span>· extraction attempt ${esc(check.attempt)}</span>` : "") +
        ` <span>· Candidate: ${esc(valueText(check.value, ""))}</span> <span>${esc(outcome)}</span></div>` +
        (list(check.validation_errors).length ? `<ul class="entity-errors">${check.validation_errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` :
          check.is_valid === false ? muted("Failure reason not recorded") : "") + `</div>`;
    }).join("")}</div>` : empty("Not recorded · validation history unavailable");
    // A final verdict is useful even when condensed per-check history was not
    // captured. It is not an additional check and must not inflate the count.
    const historicalErrors = new Set(view.checks.flatMap((c) => list(c.validation_errors)));
    const finalErrors = view.errors.filter((e) => !historicalErrors.has(e));
    if (finalErrors.length) html += muted("Final recorded validation errors") +
      `<ul class="entity-errors">${finalErrors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>`;
    return html;
  }

  function rawNote(notes, id) {
    if (id == null || !own(notes, str(id))) return {};
    const note = obj(notes[str(id)]);
    return note.note_id != null && !same(note.note_id, id) ? {} : note;
  }
  function noteMeta(note) {
    return `${note.date || "Date not recorded"} · ${note.note_type || "Type not recorded"}`;
  }
  function addSpans(target, spans, label, fallbackId) {
    for (const span of list(spans)) {
      if (!span || typeof span.text !== "string") continue;
      target.push({noteId: span.note_id ?? fallbackId, text: span.text, label});
    }
  }

  // Exact Unicode/case/whitespace matching only. Repeated quotes choose the
  // first exact occurrence and disclose ambiguity (TextSpan carries no offset).
  // Merge contexts and mark unions so overlapping spans never duplicate text
  // or create nested marks. All associated concept/mention labels survive.
  function evidence(refs, notes) {
    if (!refs.length) return empty("Evidence not recorded");
    const unique = new Map();
    for (const ref of refs) {
      const key = JSON.stringify([ref.noteId == null ? null : str(ref.noteId), ref.text]);
      if (!unique.has(key)) unique.set(key, {...ref, labels: []});
      if (!unique.get(key).labels.includes(ref.label)) unique.get(key).labels.push(ref.label);
    }
    const context = unique.size > 6 ? 0 : unique.size > 3 ? 36 : 72;
    const budget = Math.max(240, Math.floor(2000 / unique.size));
    const byNote = new Map();
    for (const ref of unique.values()) {
      const id = ref.noteId == null ? null : str(ref.noteId);
      if (!byNote.has(id)) byNote.set(id, []);
      byNote.get(id).push(ref);
    }
    const items = [];
    for (const [id, citations] of byNote) {
      const note = rawNote(notes, id);
      const source = typeof note.content === "string" ? note.content : null;
      const ranges = [];
      const unmatched = [];
      for (const citation of citations) {
        const start = source != null && citation.text ? source.indexOf(citation.text) : -1;
        if (start < 0) { unmatched.push(citation); continue; }
        const end = start + citation.text.length;
        let count = 1;
        let position = source.indexOf(citation.text, start + 1);
        while (position !== -1) { count++; position = source.indexOf(citation.text, position + 1); }
        // Preserve the citation before spending room on surrounding context.
        const radius = citation.text.length > budget ? 0 : context;
        ranges.push({start: Math.max(0, start - radius), end: Math.min(source.length, end + radius),
          marks: [{start, end}], labels: citation.labels, repeated: count > 1 ? [`${count} exact occurrences; first shown`] : []});
      }
      ranges.sort((a, b) => a.start - b.start || a.end - b.end);
      const merged = [];
      for (const range of ranges) {
        const prior = merged.at(-1);
        if (prior && range.start <= prior.end) {
          prior.end = Math.max(prior.end, range.end);
          prior.marks.push(...range.marks);
          prior.labels = [...new Set([...prior.labels, ...range.labels])];
          prior.repeated = [...new Set([...prior.repeated, ...range.repeated])];
        } else merged.push({...range, marks: range.marks.slice()});
      }
      const sourceHeader = (labels) => `<div class="entity-source">` +
        (id == null ? "Note ID not recorded" : navigation("note", id, `Note ${esc(id)}`, "entity-source-link")) +
        ` <span>${esc(noteMeta(note))}</span><span class="entity-evidence-labels">${esc(labels.join(" · "))}</span></div>`;
      for (const range of merged) {
        const marks = [];
        for (const mark of range.marks.sort((a, b) => a.start - b.start)) {
          const prior = marks.at(-1);
          if (prior && mark.start <= prior.end) prior.end = Math.max(prior.end, mark.end);
          else marks.push({...mark});
        }
        let cursor = range.start;
        let text = range.start > 0 ? "…" : "";
        let shortened = false;
        for (const mark of marks) {
          text += esc(source.slice(cursor, mark.start));
          const cited = Array.from(source.slice(mark.start, mark.end));
          if (cited.length > budget) {
            const half = Math.floor(budget / 2);
            text += `<mark>${esc(cited.slice(0, half).join(""))}</mark>` +
              `<span class="entity-muted"> [… ${cited.length - 2 * half} cited characters omitted …] </span>` +
              `<mark>${esc(cited.slice(-half).join(""))}</mark>`;
            shortened = true;
          } else text += `<mark>${esc(cited.join(""))}</mark>`;
          cursor = mark.end;
        }
        text += esc(source.slice(cursor, range.end)) + (range.end < source.length ? "…" : "");
        items.push(`<div class="entity-evidence-item">${sourceHeader(range.labels)}` +
          muted(shortened ? "Deterministic cited-text excerpt · beginning and end shown" : "Source-note excerpt · exact cited text highlighted") +
          (range.repeated.length ? muted(range.repeated.join("; ")) : "") + `<p class="entity-note-text">${text}</p></div>`);
      }
      for (const citation of unmatched) {
        const reason = source == null ? "Source note unavailable" : !citation.text ? "Empty citation" : "Exact span not found in source note";
        items.push(`<div class="entity-evidence-item">${sourceHeader(citation.labels)}` +
          muted(`${reason} · cited quote, not highlighted`) +
          `<div class="entity-note-text">${prose(citation.text, budget, "Cited text is empty")}</div></div>`);
      }
    }
    return `<div class="entity-evidence">${items.join("")}</div>`;
  }

  function renderNote(selection, snapshot, notes, catalog, all) {
    const identity = list(catalog.notes).find((n) => same(n.noteId, selection.id) || same(n.id, selection.id));
    const id = str(identity ? identity.noteId : selection.id);
    const candidates = all.filter((i) => i.node === "note_branch" && same(obj(i.input).note_id ?? obj(obj(i.result).note).note_id, id));
    const instance = selection.instanceKey != null ? candidates.find((i) => i.key === str(selection.instanceKey)) : latest(candidates);
    const unavailable = selection.instanceKey != null && !instance;
    const result = obj(obj(instance).result);
    const note = {...obj(obj(instance).input), ...rawNote(notes, id)};
    const refs = [];
    const concepts = Object.entries(obj(result.concepts));
    const chips = concepts.map(([name, concept]) => {
      const value = obj(concept);
      const presence = value.presence === true ? "Present" : value.presence === false ? "Absent" : "Presence not recorded";
      const state = value.presence === true ? "present" : value.presence === false ? "absent" : "unknown";
      const icon = value.presence === true ? "✓" : value.presence === false ? "−" : "?";
      addSpans(refs, value.evidence, `${words(name)} · ${presence}`, id);
      return `<div class="entity-concept concept-${state}">` +
        `<span class="entity-concept-icon" role="img" aria-label="${presence}" title="${presence}"><span aria-hidden="true">${icon}</span></span>` +
        `<strong class="entity-concept-name">${esc(words(name))}</strong>${confidenceBadge(value.confidence, true)}</div>`;
    }).join("");
    const mentions = list(result.cancer_mentions).map((mention, index) => {
      const site = mention.affected_tissue ?? mention.site ?? mention.primary_site;
      const histology = mention.histology;
      const temporality = mention.status ?? mention.temporality;
      const label = `Cancer mention ${index + 1}${present(site) ? ` · ${site}` : ""}${present(temporality) ? ` · ${temporality}` : ""}`;
      addSpans(refs, mention.evidence, label, id);
      return `<div class="entity-mention"><strong>Mention ${index + 1}</strong>` +
        `<span>Site / tissue: ${esc(present(site) ? site : "Not recorded")}</span>` +
        `<span>Histology: ${esc(present(histology) ? histology : "Not recorded")}</span>` +
        `<span>Temporality: ${esc(present(temporality) ? temporality : "Not recorded")}</span>` +
        `<span>${confidenceBadge(mention.confidence)}</span>` +
        (typeof mention.metastasis === "boolean" ? `<span>Metastasis: ${mention.metastasis ? "Present" : "Absent"}</span>` : "") + `</div>`;
    }).join("");
    const status = unavailable ? "unavailable" : isError(instance) ? "error" : isActive(instance) ? "active" : instance ? "done" : "pending";
    return card("note", `Note ${id}`, "Clinical note", noteMeta(note), status,
      section("Summary", (unavailable ? empty("Selected instance is not available in this snapshot.") : prose(result.summary, 700)) +
        (present(obj(instance).error) ? prose(typeof instance.error === "object" ? instance.error.message : instance.error) : ""), "entity-summary") +
      section("Concepts", chips ? `<div class="entity-concept-grid">${chips}</div>` : empty("Concept assessment not recorded"), "entity-concepts") +
      section("Cancer mentions", mentions ? `<div class="entity-mentions">${mentions}</div>` : empty(Array.isArray(result.cancer_mentions) ? "No cancer mentions recorded" : "Cancer mentions not recorded"), "entity-cancer") +
      section("Evidence", evidence(refs, notes), "entity-evidence-section"));
  }

  function renderVariable(selection, snapshot, notes, catalog, all) {
    const view = resolveVariable(selection, snapshot, all, catalog);
    if (view.unavailable) return card("variable", `Item ${view.id}`, "NAACCR variable", "Selected instance is not available in this snapshot.", "unavailable",
      section("Definition", empty()) + section("Extraction explanation", empty()) + section("Evidence", empty()) + section("Validation checks", empty()));
    const refs = [];
    addSpans(refs, obj(view.candidate).spans, `Item ${view.id} · ${view.name}`);
    let explanation = prose(obj(view.candidate).explanation, 600, "Extraction explanation not recorded");
    if (view.status === "done") explanation += muted("Branch completed; extraction outcome not recorded.");
    if (view.reasons.length) explanation += `<ul class="entity-reasons">${view.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    const important = obj(view.candidate).most_important_note;
    if (important != null) explanation += `<div class="entity-source">Strongest cited note: ${navigation("note", important, `Note ${esc(important)}`, "entity-source-link")}</div>`;
    return card("variable", view.name, `NAACCR item ${view.id}`, view.gid != null ? `Group ${view.gid}` : "Group not recorded", view.status,
      section("Definition", prose(firstSentence(view.definition), 650, "Definition not recorded in this snapshot"), "entity-definition") +
      section("Extraction explanation", explanation, "entity-explanation") +
      section("Evidence", evidence(refs, notes), "entity-evidence-section") +
      section(view.checks.length ? `Validation checks · ${view.checks.length}` : "Validation checks", validation(view), "entity-validation"),
      `<div class="entity-result status-${slug(view.status)}"><div class="entity-value"><span>${view.status === "active" || view.status === "invalid" || view.status === "error" ? "Candidate value" : view.status === "extracted" ? "Extracted value" : "Value"}</span>` +
      `<strong>${esc(valueText(view.value, view.status))}</strong></div>${view.confidence ? confidenceBadge(view.confidence) : ""}</div>`);
  }

  function renderGroup(selection, snapshot, notes, catalog, all) {
    const id = str(selection.id);
    const candidates = all.filter((i) => i.node === "extract_branch" && same(groupId(i), id));
    const current = latest(candidates);
    const group = selection.instanceKey != null ? candidates.find((i) => i.key === str(selection.instanceKey)) : current;
    const unavailable = selection.instanceKey != null && !group;
    const historical = group && group !== current;
    const pg = !historical && !unavailable ? list(obj(snapshot.progress).groups).find((g) => same(g.group_id, id)) : null;
    const identity = list(catalog.groups).find((g) => same(g.id, id));
    const name = requested(group).name || obj(pg).name || obj(identity).name || id;
    const roster = new Map();
    const add = (item) => { if (item != null && !roster.has(str(item))) roster.set(str(item), true); };
    if (!unavailable) {
      groupVariables(group).forEach((v) => add(v.item_id));
      all.filter((v) => v.node === "variable_branch" && childOf(v, group)).forEach((v) => add(itemId(v)));
      if (!historical) {
        list(obj(pg).item_ids).forEach(add);
        list(obj(snapshot.progress).variables).filter((v) => same(v.group_id, id)).forEach((v) => add(v.item_id));
      }
    }
    const views = [...roster.keys()].map((item) => resolveVariable({kind: "variable", id: item, groupId: id,
      ...(group ? {instanceKey: group.key} : {})}, snapshot, all, catalog));
    const counts = new Map();
    for (const view of views) counts.set(view.status, (counts.get(view.status) || 0) + 1);
    let status = unavailable ? "unavailable" : isError(group) ? "error" : isActive(group) || obj(pg).active ? "active" : group ? "done" : views.length ? "pending" : "unrecorded";
    if (!group && views.length && views.every((v) => !["pending", "active", "unrecorded"].includes(v.status))) status = "done";
    const summary = unavailable ? empty("Selected instance is not available in this snapshot.") : views.length ?
      `<p class="entity-prose">${views.length} observed / planned variable${views.length === 1 ? "" : "s"}</p><div class="entity-chips">` +
        [...counts].map(([s, count]) => `<span class="entity-chip status-${slug(s)}">${esc(statusText(s))}: ${count}</span>`).join("") + `</div>` :
      empty("No variable roster observed in this snapshot");
    const tiles = views.map((view) => navigation("variable", view.id,
      `<span class="entity-tile-id">${esc(view.id)}</span><span class="entity-tile-name">${esc(view.name)}</span>` +
      badge(view.status, "entity-tile-status") + `<span class="entity-tile-value">${esc(valueText(view.value, view.status))}</span>` +
      (view.reasons.length ? `<span class="entity-tile-reason">${esc(view.reasons.join("; "))}</span>` : ""),
      "entity-tile", id, obj(view.instance).key || obj(group).key)).join("");
    return card("group", name, "Variable group", `Group ${id}`, status,
      section("Group summary", summary + (obj(pg).annotation ? prose(pg.annotation) : "") +
        (present(obj(group).error) ? prose(typeof group.error === "object" ? group.error.message : group.error) : ""), "entity-section-wide entity-summary") +
      section("Variables", tiles ? `<div class="entity-tiles">${tiles}</div>` : empty("Variable outcomes not recorded"), "entity-section-wide"));
  }

  function render(selection, snapshot, notesById = {}, catalog = {}) {
    if (!selection) return "";
    snapshot = obj(snapshot);
    catalog = obj(catalog);
    const all = instances(snapshot);
    if (selection.kind === "note") return renderNote(selection, snapshot, obj(notesById), catalog, all);
    if (selection.kind === "group") return renderGroup(selection, snapshot, obj(notesById), catalog, all);
    if (selection.kind === "variable") return renderVariable(selection, snapshot, obj(notesById), catalog, all);
    return "";
  }

  root.DemoCards = Object.freeze({render});
  if (typeof module !== "undefined" && module.exports) module.exports = root.DemoCards;
})(globalThis);
