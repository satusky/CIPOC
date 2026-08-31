"use strict";
/* Control room — variables as bubbles inside their group's card, cards laid out
 * in wave order. A bubble's dot answers whichever question the result raises:
 * for a coded value, how far to trust it (confidence); for an empty one, why
 * it is empty (status), drawn hollow so the two never read alike.
 */

/* Statuses a variable can hold *without* a value. `extracted` and
 * `structured_data` are absent by construction — they always carry one, so
 * they are coloured by confidence and legended in the confidence row. */
const NO_VALUE_LEGEND = [
  ["not_found", "not found"],
  ["not_applicable", "not applicable"],
  ["blocked", "blocked"],
  ["error", "error"],
  ["pending", "pending"],
];

function bubbleTooltip(entry) {
  const r = entry.result;
  const e = r.extraction || {};
  const rows = [
    ["item", String(entry.item_id)],
    ["status", statusLabel(r.status)],
    ["value", r.value == null ? "—" : r.value],
  ];
  if (e.presence_confidence) rows.push(["confidence", e.presence_confidence]);
  if (e.extraction_attempts) rows.push(["attempts", String(e.extraction_attempts)]);
  if ((e.spans || []).length) rows.push(["evidence", e.spans.length + " span(s)"]);
  if (hasTruth()) {
    /* The bubble's mark says *that* a value disagrees; there is no room on the
     * line for the value it should have been. The tooltip carries it, so the
     * expected answer is one hover away rather than a pane away. */
    const v = verdictFor(entry);
    rows.push(["verdict", VERDICT_LABEL[v.verdict]]);
    if (v.verdict !== "untested" && v.expected !== v.got) {
      rows.push(["expected", v.expected || "no value"]);
    }
  }
  if (r.reason) rows.push(["reason", r.reason]);
  if ((e.validation_errors || []).length) rows.push(["invalid", e.validation_errors[0]]);

  const dl = h("dl", { class: "kv" });
  for (const [k, v] of rows) dl.append(h("dt", { text: k }), h("dd", { text: v }));
  return h("div", {}, h("h4", { text: entry.name }), dl);
}

function bubble(entry) {
  const r = entry.result;
  const level = indicatorConfidence(r);
  const value = hasValue(r) ? String(r.value) : null;

  const node = h("button", {
    type: "button",
    class: "bubble " + indicatorClass(r),
    dataset: { entity: "variable:" + entry.item_id,
               annotated: isAnnotated("variable", entry.item_id) ? "1" : null },
    "aria-label": entry.item_id + " " + entry.name + " — " +
      (value ? "value " + value + ", " + (level || "unrated") + " confidence"
             : statusLabel(r.status)) +
      (App.compare ? " \u2014 " + VERDICT_LABEL[verdictFor(entry).verdict] : ""),
    onclick: (ev) => { ev.stopPropagation(); select("variable", entry.item_id); },
  },
    h("span", { class: "bub-label" },
      h("span", { class: "bub-id", text: entry.item_id + ":" }),
      h("span", { class: "bub-name", text: entry.name })),
    /* An em dash rather than nothing when there is no value: the value sits in
     * its own right-hand column, and a blank cell there reads as a rendering
     * gap rather than as "this variable has no value". */
    h("span", { class: value ? "val" : "val none", text: value || "\u2014" }),
    /* The verdict rides in its own trailing track rather than recolouring the
     * dot, so confidence and correctness stay legible at the same time — a
     * max-confidence wrong answer is the thing worth finding, and it only
     * reads as one if both marks are present. */
    App.compare ? h("span", {
      class: "vmark m-" + verdictFor(entry).verdict,
      text: VERDICT_MARK[verdictFor(entry).verdict],
    }) : null
  );
  return hoverable(node, () => bubbleTooltip(entry));
}

function groupTooltip(group, state) {
  const rows = [
    ["group", group.group_id],
    ["stage", group.stage || "—"],
    ["outcome", state.label],
    ["variables", state.total + " · " + state.coded + " coded"],
  ];
  if ((group.gate || []).length) rows.push(["gate", group.gate.join(", ")]);
  if (group.applies_to) {
    const sites = (group.applies_to.gross_primary_sites || []).join(", ");
    const fams = (group.applies_to.histology_families || []).join(", ");
    rows.push(["applies to", [sites, fams].filter(Boolean).join(" / ") || "—"]);
  }
  if ((group.note_filter || {}).keywords) {
    rows.push(["note filter", group.note_filter.keywords.length + " keyword(s)"]);
  }
  const sel = noteSelection(group.group_id);
  if (sel) {
    rows.push(["notes", (sel.selected_note_ids || []).length + " of " +
      (sel.candidate_note_ids || []).length + " selected"]);
  }

  const dl = h("dl", { class: "kv" });
  for (const [k, v] of rows) dl.append(h("dt", { text: k }), h("dd", { text: v }));

  return h("div", {},
    h("h4", { text: group.name }),
    dl,
    state.reason ? h("p", { class: "muted", style: "margin:6px 0 0", text: state.reason }) : null
  );
}

function groupCard(group) {
  const state = groupState(group);
  const matcher = App.varFilter;

  const entries = App.variables.filter((v) => v.group_id === group.group_id);
  const visible = matcher
    ? entries.filter((v) =>
        String(v.item_id).includes(matcher) ||
        v.name.toLowerCase().includes(matcher) ||
        (v.result.value || "").toLowerCase().includes(matcher))
    : entries;
  if (matcher && visible.length === 0) return null;

  const gv = App.compare ? groupVerdict(group) : null;

  const tags = h("div", { class: "group-tags" },
    h("span", { class: "chip" + (state.kind === "excluded" ? "" : " on"), text: state.label }),
    gv && gv.tested
      ? h("span", { class: "chip " + (gv.correct === gv.tested ? "good" : "warn"),
          text: gv.correct + "/" + gv.tested + " correct" })
      : null,
    /* A gate that excluded a group the reference says should have run produces
     * a whole card of identical `missed` verdicts with one upstream cause.
     * Naming the cause here is what turns six wrong answers into one wrong
     * gate — the actionable form. */
    gv && gv.finding
      ? h("span", { class: "chip bad", text: FINDING_LABEL[gv.finding] })
      : null,
    (group.gate || []).map((g) => h("span", { class: "chip", text: "gate: " + g })),
    group.applies_to
      ? h("span", { class: "chip", text: "site: " + (group.applies_to.gross_primary_sites || []).join("/") })
      : null,
    group.note_filter ? h("span", { class: "chip", text: "note filter" }) : null,
    group.extract_as_group === false ? h("span", { class: "chip", text: "per-variable" }) : null,
    isAnnotated("group", group.group_id)
      ? h("span", { class: "chip on", text: "\u270e annotated" })
      : null
  );

  const card = h("div", {
    class: "group-card " + state.kind,
    role: "button",
    tabindex: "0",
    dataset: { entity: "group:" + group.group_id },
    onclick: () => select("group", group.group_id),
    onkeydown: (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select("group", group.group_id); }
    },
  },
    /* Stacked, not inline: side by side, a long name and a long group_id
     * squeeze each other and both wrap. */
    h("div", { class: "group-head" },
      h("h3", { text: group.name }),
      h("span", { class: "gid", text: group.group_id })),
    tags,
    h("div", { class: "bubbles" + (App.compare ? " compare" : "") }, visible.map(bubble)),
    state.reason && state.kind === "excluded"
      ? h("p", { class: "faint", style: "margin:0;font-size:11.5px", text: state.reason })
      : null
  );
  return hoverable(card, () => groupTooltip(group, state));
}

function legendRow(label, entries) {
  const row = h("div", { class: "legend" }, h("span", { class: "legend-label", text: label }));
  for (const [cls, text] of entries) {
    row.append(h("span", {}, h("i", { class: "dot " + cls }), text));
  }
  return row;
}

/* Like legendRow, but keyed by the trailing mark rather than a dot — the dot
 * column already means confidence, and repeating it here would say the marks
 * and the dots share a vocabulary when the whole point is that they do not. */
function verdictLegendRow() {
  const row = h("div", { class: "legend" },
    h("span", { class: "legend-label", text: "vs. ground truth" }));
  for (const verdict of VERDICTS) {
    row.append(h("span", {},
      h("i", { class: "vmark m-" + verdict, text: VERDICT_MARK[verdict] }),
      VERDICT_LABEL[verdict]));
  }
  return row;
}

function renderControlRoom() {
  const root = clear($("#control"));

  root.append(
    legendRow("value \u00b7 confidence",
      CONFIDENCE_LEVELS.map((level) => ["c-" + level, level])
        .concat([["s-structured_data", "structured data (unrated)"]])),
    legendRow("no value", NO_VALUE_LEGEND.map(([s, label]) => ["s-" + s + " hollow", label]))
  );
  /* Appended separately, not as a conditional argument: Element.append()
     stringifies null into a literal "null" text node, unlike h(), which skips
     falsy children. */
  if (App.compare) root.append(verdictLegendRow());

  let shown = 0;
  for (const wave of waves()) {
    const cards = wave.groups.map(groupCard).filter(Boolean);
    if (!cards.length) continue;
    shown += cards.length;
    root.append(
      h("div", { class: "wave-head", text: wave.title }),
      h("div", { class: "wave-grid" }, cards)
    );
  }
  if (!shown) root.append(h("p", { class: "empty", text: "No variables match this filter." }));
}
