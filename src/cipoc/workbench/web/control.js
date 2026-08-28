"use strict";
/* Control room — variables as bubbles inside their group's card, cards laid out
 * in wave order. Colour encodes recorded status; a dashed border marks a value
 * accepted at low confidence, and a hollow dot marks a variable with no value.
 */

const BUBBLE_LEGEND = [
  ["extracted", "extracted"],
  ["structured_data", "structured data"],
  ["not_found", "not found"],
  ["not_applicable", "not applicable"],
  ["blocked", "blocked"],
  ["error", "error"],
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
  if (r.reason) rows.push(["reason", r.reason]);
  if ((e.validation_errors || []).length) rows.push(["invalid", e.validation_errors[0]]);

  const dl = h("dl", { class: "kv" });
  for (const [k, v] of rows) dl.append(h("dt", { text: k }), h("dd", { text: v }));
  return h("div", {}, h("h4", { text: entry.name }), dl);
}

function bubble(entry) {
  const r = entry.result;
  const e = r.extraction || {};
  const low = e.presence_confidence === "low";
  const hollow = r.value == null;

  const node = h("button", {
    type: "button",
    class: "bubble s-" + r.status + (low ? " low-confidence" : "") + (hollow ? " hollow" : ""),
    dataset: { entity: "variable:" + entry.item_id },
    title: entry.name,
    "aria-label": entry.name + " — " + statusLabel(r.status) + (r.value ? ", value " + r.value : ""),
    onclick: (ev) => { ev.stopPropagation(); select("variable", entry.item_id); },
  },
    h("span", { text: String(entry.item_id) }),
    r.value != null ? h("span", { class: "val", text: r.value }) : null
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

  const tags = h("div", { class: "group-tags" },
    h("span", { class: "chip" + (state.kind === "excluded" ? "" : " on"), text: state.label }),
    (group.gate || []).map((g) => h("span", { class: "chip", text: "gate: " + g })),
    group.applies_to
      ? h("span", { class: "chip", text: "site: " + (group.applies_to.gross_primary_sites || []).join("/") })
      : null,
    group.note_filter ? h("span", { class: "chip", text: "note filter" }) : null,
    group.extract_as_group === false ? h("span", { class: "chip", text: "per-variable" }) : null
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
    h("div", { class: "group-head" },
      h("h3", { text: group.name }),
      h("span", { class: "gid", text: group.group_id })),
    tags,
    h("div", { class: "bubbles" }, visible.map(bubble)),
    state.reason && state.kind === "excluded"
      ? h("p", { class: "faint", style: "margin:0;font-size:11.5px", text: state.reason })
      : null
  );
  return hoverable(card, () => groupTooltip(group, state));
}

function renderControlRoom() {
  const root = clear($("#control"));

  const legend = h("div", { class: "legend" });
  for (const [status, label] of BUBBLE_LEGEND) {
    legend.append(h("span", {}, h("i", { class: "d-" + status }), label));
  }
  legend.append(
    h("span", {}, h("i", { class: "d-extracted", style: "box-shadow:inset 0 0 0 1.5px currentColor;background:transparent" }), "no value"),
    h("span", { style: "border:1px dashed var(--line);border-radius:999px;padding:1px 8px" }, "low confidence")
  );
  root.append(legend);

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
