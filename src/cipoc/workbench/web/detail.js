"use strict";
/* Detail pane — note, group and variable dossiers, plus the shared renderers
 * for prompts/responses, extraction attempts and evidence highlighting.
 *
 * Evidence spans carry no character offsets, so highlighting is a verbatim
 * substring search. A span that does not match is reported as unmatched rather
 * than dropped: a hallucinated or paraphrased quote is exactly the thing worth
 * seeing.
 */

/* --------------------------------------------------------------- fragments */

function kv(rows) {
  const dl = h("dl", { class: "kv" });
  for (const [key, value] of rows) {
    if (value === null || value === undefined) continue;
    dl.append(h("dt", { text: key }),
      value instanceof Node ? h("dd", {}, value) : h("dd", { text: String(value) }));
  }
  return dl;
}

const section = (title, ...body) =>
  h("div", { class: "section" }, h("h3", { text: title }), ...body);

function crossLink(kind, id, label) {
  return h("button", {
    type: "button",
    class: "link",
    text: label,
    onclick: () => show(kind, id),
  });
}

function rawBlock(label, value) {
  return h("details", {},
    h("summary", { text: label }),
    h("pre", { class: "raw", text: JSON.stringify(value, null, 2) }));
}

/* ------------------------------------------------------- evidence + notes */

/* Merge overlapping [start,end) ranges so nested quotes highlight once. */
function mergeRanges(ranges) {
  const sorted = ranges.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const [start, end] of sorted) {
    const last = out[out.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else out.push([start, end]);
  }
  return out;
}

function highlightedContent(content, spans) {
  const text = content || "";
  const ranges = [];
  const unmatched = [];
  for (const span of spans) {
    const needle = span.text || "";
    if (!needle) continue;
    let from = 0;
    let found = false;
    for (;;) {
      const at = text.indexOf(needle, from);
      if (at === -1) break;
      ranges.push([at, at + needle.length]);
      from = at + needle.length;
      found = true;
    }
    if (!found) unmatched.push(span);
  }

  const wrap = h("div", { class: "note-text" });
  let cursor = 0;
  for (const [start, end] of mergeRanges(ranges)) {
    if (start > cursor) wrap.append(document.createTextNode(text.slice(cursor, start)));
    wrap.append(h("mark", { text: text.slice(start, end) }));
    cursor = end;
  }
  wrap.append(document.createTextNode(text.slice(cursor)));
  return { node: wrap, unmatched };
}

function evidenceList(spans, { showNote = true } = {}) {
  if (!spans || !spans.length) return h("p", { class: "faint", text: "No evidence spans." });
  const list = h("ul", { class: "evidence" });
  for (const span of spans) {
    const note = App.notes.get(String(span.note_id));
    const matched = note && (note.content || "").includes(span.text || "");
    list.append(h("li", { class: matched ? "" : "unmatched" },
      showNote
        ? h("span", {}, crossLink("note", span.note_id, "note " + span.note_id), " ")
        : null,
      h("span", { text: "“" + (span.text || "") + "”" }),
      matched ? null : h("span", { class: "chip warn", style: "margin-left:6px", text: "not verbatim" })));
  }
  return list;
}

/* -------------------------------------------------------- LLM exchanges */

function exchangeCard(exchange, index) {
  const usage = exchange.usage || {};
  const head = h("h4", {},
    h("span", { text: exchange.node }),
    h("span", { class: "chip", text: exchange.agent }),
    exchange.attempt > 1 ? h("span", { class: "chip warn", text: "attempt " + exchange.attempt }) : null,
    exchange.error ? h("span", { class: "chip bad", text: "error" }) : null);

  const meta = [];
  if (exchange.model) meta.push("model " + exchange.model);
  if (usage.total_tokens) {
    meta.push(usage.input_tokens + " in / " + usage.output_tokens + " out");
  }

  const messages = h("div", {});
  for (const message of exchange.prompt_messages || []) {
    messages.append(h("div", { class: "msg " + (message.role || "human") },
      h("span", { class: "role", text: message.role || "message" }),
      h("pre", { text: message.content || "" })));
  }

  return h("div", { class: "card" + (exchange.error ? " bad" : "") },
    head,
    meta.length ? h("p", { class: "faint", style: "margin:0 0 6px;font-size:11.5px", text: meta.join(" · ") }) : null,
    h("details", { open: index === 0 ? true : null },
      h("summary", { text: "Prompt (" + (exchange.prompt_messages || []).length + " messages)" }),
      messages),
    exchange.error
      ? h("p", { class: "errors", text: exchange.error })
      : rawBlock("Response", exchange.response));
}

function exchangeSection(title, exchanges) {
  if (!exchanges.length) {
    if (!hasCapture()) {
      return section(title,
        h("p", { class: "faint", text: "This state dump carries no prompt capture." }));
    }
    return section(title, h("p", { class: "faint", text: "No model calls recorded." }));
  }
  return section(title + " (" + exchanges.length + ")", exchanges.map(exchangeCard));
}

/* ----------------------------------------------------------- attempts */

function attemptCard(attempt) {
  const candidate = attempt.candidate || {};
  const errors = attempt.validation_errors || [];
  return h("div", { class: "card " + (attempt.is_valid ? "good" : "bad") },
    h("h4", {},
      h("span", { text: "Attempt " + attempt.attempt }),
      h("span", { class: "chip", text: attempt.mode }),
      h("span", { class: attempt.is_valid ? "chip good" : "chip bad",
        text: attempt.is_valid ? "valid" : "rejected" })),
    kv([
      ["value", candidate.value == null ? "—" : candidate.value],
      ["confidence", candidate.presence_confidence || null],
      ["explanation", candidate.explanation || null],
    ]),
    errors.length
      ? h("ul", { class: "errors" }, errors.map((e) => h("li", { text: e })))
      : null,
    (candidate.spans || []).length
      ? h("div", { style: "margin-top:6px" }, evidenceList(candidate.spans))
      : null);
}

/* ------------------------------------------------------------ note view */

function noteDetail(noteId) {
  const note = App.notes.get(String(noteId));
  if (!note) return h("p", { class: "empty", text: "Unknown note " + noteId + "." });

  const body = h("div", {});
  const citing = variablesCitingNote(note.note_id);
  const citedSpans = citing.flatMap((v) =>
    (v.result.extraction.spans || []).filter((s) => String(s.note_id) === String(note.note_id)));
  const conceptSpans = Object.values(note.concepts || {}).flatMap((c) => c.evidence || []);
  const mentionSpans = (note.cancer_mentions || []).flatMap((m) => m.evidence || []);
  const { node: content, unmatched } =
    highlightedContent(note.content, [...citedSpans, ...conceptSpans, ...mentionSpans]);

  body.append(section("Metadata", kv([
    ["note id", note.note_id],
    ["date", note.date || "undated"],
    ["type", note.note_type || "—"],
    ["cancer status", (note.cancer_status || []).join(", ") || "none"],
    ["keywords", (note.flags || []).join(", ") || "—"],
  ])));

  body.append(section("Summary",
    h("p", { style: "margin:0", text: note.summary || "No summary recorded." })));

  body.append(section("Text",
    content,
    unmatched.length
      ? h("p", { class: "chip warn", style: "margin-top:8px", text: unmatched.length + " evidence span(s) did not match this text" })
      : null));

  const concepts = h("div", {});
  for (const [name, concept] of Object.entries(note.concepts || {})) {
    concepts.append(h("div", { class: "card" },
      h("h4", {},
        h("span", { text: name.replace(/_/g, " ") }),
        h("span", { class: concept.presence ? "chip good" : "chip", text: concept.presence ? "present" : "absent" }),
        concept.confidence ? h("span", { class: "chip", text: concept.confidence }) : null),
      concept.presence ? evidenceList(concept.evidence, { showNote: false }) : null));
  }
  body.append(section("Concepts", concepts));

  const mentions = note.cancer_mentions || [];
  body.append(section("Cancer mentions",
    mentions.length
      ? mentions.map((m) => h("div", { class: "card" },
          h("h4", {},
            h("span", { text: m.affected_tissue || "unspecified tissue" }),
            h("span", { class: "chip on", text: m.status }),
            h("span", { class: "chip", text: m.confidence }),
            m.metastasis ? h("span", { class: "chip warn", text: "metastatic" }) : null),
          evidenceList(m.evidence, { showNote: false })))
      : h("p", { class: "faint", text: "No cancer mentions recorded." })));

  const touching = groupsTouchingNote(note.note_id);
  body.append(section("Group selection",
    touching.length
      ? touching.map((t) => h("div", { class: "check-row" },
          h("span", { class: t.role === "selected" ? "mark-pass" : "mark-fail",
            text: t.role === "selected" ? "✓" : "✕" }),
          h("div", {},
            crossLink("group", t.group.group_id, t.group.name),
            h("span", { class: "chip", style: "margin-left:6px", text: t.role }),
            t.reason ? h("div", { class: "observed", text: t.reason }) : null)))
      : h("p", { class: "faint", text: "No note-selection decisions recorded." })));

  body.append(section("Cited by",
    citing.length
      ? h("div", { class: "pill-list" }, citing.map((v) =>
          crossLink("variable", v.item_id, v.item_id + " " + v.name)))
      : h("p", { class: "faint", text: "No variable cites this note as evidence." })));

  body.append(exchangeSection("Scanner calls", noteExchanges(note.note_id)));
  body.append(feedbackSection("note", noteId));
  body.append(section("Raw", rawBlock("ProcessedClinicalNote", note)));
  return body;
}

/* ----------------------------------------------------------- group view */

function checkRow(pass, label, observed) {
  return h("div", { class: "check-row" },
    h("span", { class: pass ? "mark-pass" : "mark-fail", text: pass ? "✓" : "✕" }),
    h("div", {},
      h("div", { text: label }),
      observed ? h("div", { class: "observed", text: observed }) : null));
}

function groupDetail(groupId) {
  const group = App.groupById.get(groupId);
  if (!group) return h("p", { class: "empty", text: "Unknown group " + groupId + "." });

  const state = groupState(group);
  const body = h("div", {});
  const descriptors = App.raw.note_corpus_descriptors || {};
  const facts = App.raw.case_facts || {};

  body.append(section("Configuration", kv([
    ["group id", group.group_id],
    ["stage", group.stage || "—"],
    ["extraction", group.extract_as_group === false ? "per variable" : "as a group"],
    ["depends on", (group.depends_on || []).join(", ") || null],
    ["outcome", state.label],
  ])));

  /* Gating. Configured conditions are shown against the corpus characteristics
   * and case facts recorded in the same state, alongside the reason the
   * orchestrator actually stamped on the group's variables. */
  const checks = h("div", {});
  const gates = group.gate || [];
  if (gates.length) {
    const concepts = descriptors.concepts || {};
    for (const gate of gates) {
      const related = {
        metastasis_present: ["metastasis"],
        lymph_nodes_removed: ["lymph_nodes_removed"],
        treatment_present: ["surgery", "chemotherapy", "radiation"],
      }[gate] || [];
      const observed = related
        .map((c) => c + ": " + ((concepts[c] || {}).presence ? "present" : "absent"))
        .join(" · ");
      checks.append(checkRow(state.kind !== "excluded", "corpus gate — " + gate, observed || null));
    }
  } else {
    checks.append(checkRow(true, "corpus gate — ungated", null));
  }

  if (group.applies_to) {
    const sites = (group.applies_to.gross_primary_sites || []).join(", ") || "—";
    const fams = (group.applies_to.histology_families || []).join(", ");
    checks.append(checkRow(state.kind !== "excluded",
      "site applicability — " + sites + (fams ? " / " + fams : ""),
      "case: gross_primary_site=" + (facts.gross_primary_site || "unknown") +
      ", histology=" + (facts.histology || "unknown")));
  } else {
    checks.append(checkRow(true, "site applicability — unrestricted", null));
  }

  if (group.note_filter) {
    const filter = group.note_filter;
    const bits = [];
    if ((filter.note_types || []).length) bits.push("types: " + filter.note_types.join(", "));
    if ((filter.keywords || []).length) bits.push("keywords: " + filter.keywords.join(", "));
    if ((filter.cancer_status || []).length) bits.push("status: " + filter.cancer_status.join(", "));
    if (filter.within_days != null) bits.push("within " + filter.within_days + " days");
    checks.append(checkRow(true, "note filter", bits.join(" · ")));
  }

  body.append(section("Gating", checks,
    state.reason
      ? h("p", { class: "muted", style: "margin:8px 0 0", text: "Recorded reason: " + state.reason })
      : null));

  /* The group is the only level at which a gating mistake is visible as one
     fact. Every variable below it will read `missed`, but they share a single
     upstream cause and fixing them one at a time is the wrong move. */
  if (hasTruth()) {
    const gv = groupVerdict(group);
    body.append(section("Ground truth",
      gv.tested
        ? kv([
            ["tested", gv.tested + " of " + gv.total + " variable(s)"],
            ["correct", gv.correct + " of " + gv.tested],
            ["disagreements", VERDICTS
              .filter((v) => v !== "match" && v !== "untested" && gv.counts[v])
              .map((v) => gv.counts[v] + " " + VERDICT_LABEL[v]).join(" · ") || "none"],
          ])
        : h("p", { class: "faint", style: "margin:0",
            text: "The reference file mentions none of this group's variables." }),
      gv.finding
        ? h("div", { class: "card bad", style: "margin-top:10px" },
            h("h4", {}, h("span", { class: "chip bad", text: "gate" })),
            h("p", { style: "margin:0", text: gv.finding === "wrongly_excluded"
              ? "This group was excluded, but the reference file carries values for its " +
                "variables — the gate or site rule above rejected a case it should have admitted."
              : "This group ran and coded values, but the reference file says none of its " +
                "variables should have one — the gate admitted a case it should have rejected." }))
        : null));
  }

  const sel = noteSelection(group.group_id);
  if (sel) {
    const rows = h("div", {});
    for (const noteId of sel.candidate_note_ids || []) {
      const chosen = (sel.selected_note_ids || []).some((n) => String(n) === String(noteId));
      const note = App.notes.get(String(noteId));
      rows.append(h("div", { class: "check-row" },
        h("span", { class: chosen ? "mark-pass" : "mark-fail", text: chosen ? "✓" : "✕" }),
        h("div", {},
          crossLink("note", noteId, "note " + noteId + (note ? " · " + note.note_type : "")),
          h("span", { class: "chip", style: "margin-left:6px", text: chosen ? "selected" : "not selected" }))));
    }
    for (const [noteId, reason] of Object.entries(sel.filtered_out || {})) {
      rows.append(h("div", { class: "check-row" },
        h("span", { class: "mark-fail", text: "✕" }),
        h("div", {},
          crossLink("note", noteId, "note " + noteId),
          h("span", { class: "chip", style: "margin-left:6px", text: "filtered out" }),
          h("div", { class: "observed", text: reason }))));
    }
    body.append(section(
      "Note selection — " + (sel.selected_note_ids || []).length + " of " +
      ((sel.candidate_note_ids || []).length + Object.keys(sel.filtered_out || {}).length),
      rows));
  } else {
    body.append(section("Note selection",
      h("p", { class: "faint", text: "No note-selection decisions recorded for this group." })));
  }

  const variables = App.variables.filter((v) => v.group_id === group.group_id);
  body.append(section("Variables (" + variables.length + ")",
    variables.map((v) => h("div", { class: "check-row" },
      h("span", { class: "dot " + statusDot(v.result), style: "margin-top:5px" }),
      h("div", {},
        crossLink("variable", v.item_id, v.item_id + " " + v.name),
        h("span", { class: "chip", style: "margin-left:6px", text: statusLabel(v.result.status) }),
        v.result.value ? h("span", { class: "chip on", style: "margin-left:4px", text: v.result.value }) : null,
        v.result.reason ? h("div", { class: "observed", text: v.result.reason }) : null)))));

  body.append(exchangeSection("Group-level calls", groupExchanges(group.group_id)));
  body.append(feedbackSection("group", groupId));
  body.append(section("Raw", rawBlock("TargetGroup", group)));
  return body;
}

/* -------------------------------------------------------- variable view */

function variableDetail(itemId) {
  const id = Number(itemId);
  const entry = App.variables.find((v) => v.item_id === id);
  if (!entry) return h("p", { class: "empty", text: "Unknown variable " + itemId + "." });

  const result = entry.result;
  const extraction = result.extraction;
  const body = h("div", {});
  const flags = ((App.raw.report || {}).flags || []).filter((f) => f.item_id === id);

  body.append(section("Outcome", kv([
    ["item id", id],
    ["group", crossLink("group", entry.group_id, entry.group_name)],
    ["status", statusLabel(result.status)],
    ["value", result.value == null ? "—" : result.value],
    ["confidence", extraction ? extraction.presence_confidence : null],
    ["valid", extraction ? (extraction.is_valid ? "yes" : "no") : null],
    ["attempts", extraction ? extraction.extraction_attempts : null],
    ["reason", result.reason || null],
    ["blocked by", (result.blocking_item_ids || []).length
      ? h("span", { class: "pill-list" }, result.blocking_item_ids.map((b) =>
          crossLink("variable", b, String(b))))
      : null],
  ])));

  /* Immediately after Outcome, so the recorded answer and the expected one read
     adjacently. Absent entirely when there is no reference file, and reduced to
     a single line when the reference does not mention this item. */
  if (hasTruth()) {
    const v = verdictFor(entry);
    if (v.verdict === "untested") {
      body.append(section("Ground truth",
        h("p", { class: "faint", style: "margin:0",
          text: "The reference file does not mention item " + id + "." })));
    } else {
      body.append(section("Ground truth",
        checkRow(CORRECT.has(v.verdict), VERDICT_LABEL[v.verdict],
          v.verdict === "match" ? null
            : "recorded " + (v.got || "no value") + " · expected " + (v.expected || "no value")),
        v.verdict === "near"
          ? h("p", { class: "muted", style: "margin:8px 0 0", text:
              "These agree once case and separators are ignored and leading zeros " +
              "are dropped — a formatting difference, not a different answer." })
          : null));
    }
  }

  if (flags.length) {
    body.append(section("Review flags", flags.map((f) =>
      h("div", { class: "card bad" },
        h("h4", {}, h("span", { class: "chip bad", text: f.flag_type })),
        h("p", { style: "margin:0", text: f.detail })))));
  }

  if (extraction) {
    body.append(section("Explanation",
      h("p", { style: "margin:0", text: extraction.explanation || "—" })));

    if ((extraction.validation_errors || []).length) {
      body.append(section("Validation errors",
        h("ul", { class: "errors" }, extraction.validation_errors.map((e) => h("li", { text: e })))));
    }

    body.append(section("Evidence",
      evidenceList(extraction.spans),
      extraction.most_important_note != null
        ? h("p", { class: "faint", style: "margin:8px 0 0" },
            "Most important note: ",
            crossLink("note", extraction.most_important_note, "#" + extraction.most_important_note))
        : null));

    /* Show the cited note text with this variable's own spans marked. */
    const primary = App.notes.get(String(extraction.most_important_note));
    if (primary) {
      const spans = (extraction.spans || []).filter(
        (s) => String(s.note_id) === String(primary.note_id));
      const { node } = highlightedContent(primary.content, spans);
      body.append(section("Cited note " + primary.note_id + " · " + (primary.note_type || ""), node));
    }
  }

  const attempts = variableAttempts(id);
  body.append(section("Attempts" + (attempts.length ? " (" + attempts.length + ")" : ""),
    attempts.length
      ? attempts.map(attemptCard)
      : h("p", { class: "faint", text: result.status === "structured_data"
          ? "Supplied as structured data; no extraction was attempted."
          : "No attempt records in this state dump." })));

  body.append(exchangeSection("Variable-level calls", variableExchanges(id)));

  const groupCalls = groupExchanges(entry.group_id).filter((e) => e.agent === "extractor");
  if (groupCalls.length) {
    body.append(section("Group extraction call",
      h("p", { class: "faint", style: "margin:0 0 8px", text:
        "This variable was extracted as part of its group; the group call is shown in full." }),
      groupCalls.map(exchangeCard)));
  }

  body.append(feedbackSection("variable", id));
  body.append(section("Raw", rawBlock("CaseVariableResult", result)));
  return body;
}

/* ------------------------------------------------------------ dispatch */

function renderDetail() {
  const panel = $("#detail");
  if (!App.selection) { panel.hidden = true; return; }

  const { kind, id } = App.selection;
  let title = "";
  let body;
  if (kind === "note") {
    const note = App.notes.get(String(id));
    title = note ? (note.note_type || "Note") + " · " + (note.date || "undated") : "Note " + id;
    body = noteDetail(id);
  } else if (kind === "group") {
    const group = App.groupById.get(id);
    title = group ? group.name : "Group " + id;
    body = groupDetail(id);
  } else {
    const entry = App.variables.find((v) => v.item_id === Number(id));
    title = entry ? entry.item_id + " · " + entry.name : "Variable " + id;
    body = variableDetail(id);
  }

  $("#detail-kind").textContent = kind;
  $("#detail-title").textContent = title;
  const target = clear($("#detail-body"));
  target.append(body);
  target.scrollTop = 0;
  panel.hidden = false;
}
