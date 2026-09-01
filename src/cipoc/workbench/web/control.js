"use strict";
/* Control room — variables as bubbles inside their group's card, cards laid out
 * in wave order.
 *
 * A bubble carries exactly one reading: whichever metric the active lens
 * selects, painted on both of its channels (a dim ground and a matching
 * outline, both mixed from the class's own colour — see styles.css). The lens
 * vocabulary, its class order and its per-variable reading all live in the
 * LENSES table in app.js; nothing here knows what a status or a verdict is.
 */

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
    /* The bubble shows one metric; the tooltip is where the others stay
     * reachable. With the lens on confidence this is the only place the
     * verdict appears without switching tabs — and a max-confidence wrong
     * answer is exactly the pair worth reading together. */
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
  const lens = activeLens();
  const ind = indicatorFor(entry, App.lens);
  const value = hasValue(r) ? String(r.value) : null;

  const node = h("button", {
    type: "button",
    class: "bubble " + ind.cls + (ind.hollow ? " hollow" : ""),
    dataset: { entity: "variable:" + entry.item_id,
               annotated: isAnnotated("variable", entry.item_id) ? "1" : null },
    /* The spoken label states the same one reading the colour does, so the
       screen-reader pass tracks the lens instead of describing a fixed axis
       the sighted view may not be showing. */
    "aria-label": entry.item_id + " " + entry.name + " — " +
      (value ? "value " + value : "no value") + " \u2014 " + lens.aria(ind.key),
    onclick: (ev) => { ev.stopPropagation(); select("variable", entry.item_id); },
  },
    /* Occupies the indicator column, which styles.css reserves whenever a glyph
       lens is AVAILABLE rather than active — so the column is the same width
       under either lens and switching between them shifts nothing. */
    ind.mark ? h("i", { class: "vmark bub-mark " + ind.cls, text: ind.mark }) : null,
    h("span", { class: "bub-label" },
      h("span", { class: "bub-id", text: entry.item_id + ":" }),
      h("span", { class: "bub-name", text: entry.name })),
    /* An em dash rather than nothing when there is no value: the value sits in
     * its own right-hand column, and a blank cell there reads as a rendering
     * gap rather than as "this variable has no value". */
    h("span", { class: value ? "val" : "val none", text: value || "\u2014" })
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

/* The card's one roll-up chip, for the active lens — and only when it has
 * something to say. `complete` and `6/6 correct` are ink that reports the
 * absence of news; suppressing them is what lets the cards that DO carry news
 * stand out at a glance, and a card with no chip at all is the clean case.
 *
 * That silence is now the ONLY thing marking a clean card: the state colour on
 * the card's top border is gone (see .group-card in styles.css). Which is why
 * the fallback below still emits `pending` and `nothing coded` — those are not
 * clean, and with no border left to say so the chip is the whole signal.
 *
 * The group's static configuration — gate, site applicability, note filter,
 * per-variable extraction — used to sit here as four more chips of identical
 * weight. It belongs to no lens, never varies between runs, and is already in
 * this card's hover tooltip (groupTooltip, below) and in the detail pane's
 * Configuration and Gating sections.
 */
function groupChip(group, state) {
  if (App.lens === "accuracy" && hasTruth()) {
    const gv = groupVerdict(group);
    if (!gv.tested || gv.correct === gv.tested) return null;
    return h("span", { class: "chip warn", text: gv.correct + "/" + gv.tested + " correct" });
  }
  if (App.lens === "confidence") {
    const weak = App.variables.filter((v) =>
      v.group_id === group.group_id && ["medium", "low"].includes(indicatorFor(v, "confidence").key));
    if (!weak.length) return null;
    return h("span", { class: "chip warn", text: weak.length + " below high" });
  }
  if (state.kind === "ran" && state.label === "complete") return null;
  return h("span", { class: "chip" + (state.kind === "excluded" ? "" : " on"), text: state.label });
}

function groupCard(group) {
  const state = groupState(group);
  const matcher = App.varFilter;

  const entries = App.variables.filter((v) => v.group_id === group.group_id);
  const visible = entries.filter((v) => passesLens(v) && (!matcher ||
    String(v.item_id).includes(matcher) ||
    v.name.toLowerCase().includes(matcher) ||
    (v.result.value || "").toLowerCase().includes(matcher)));
  if ((matcher || App.classFilter.size) && visible.length === 0) return null;

  const tags = h("div", { class: "group-tags" },
    groupChip(group, state),
    /* A gate that excluded a group the reference says should have run produces
     * a whole card of identical `missed` verdicts with one upstream cause.
     * Naming the cause here is what turns six wrong answers into one wrong
     * gate — the actionable form. */
    App.lens === "accuracy" && hasTruth() && groupVerdict(group).finding
      ? h("span", { class: "chip bad", text: FINDING_LABEL[groupVerdict(group).finding] })
      : null,
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
    h("div", { class: "bubbles" }, visible.map(bubble)),
    state.reason && state.kind === "excluded"
      ? h("p", { class: "faint", style: "margin:0;font-size:11.5px", text: state.reason })
      : null
  );
  return hoverable(card, () => groupTooltip(group, state));
}

/* The one legend row — the key for the active lens, and the run's distribution
 * along it, and the filter over it, all in the same object.
 *
 * It replaced three static legend rows (16 entries) stacked above the first
 * card. Those had to spell out every vocabulary at once because all three were
 * live simultaneously; with one lens active there is one to spell out, and the
 * space that buys pays for the counts. Classes this run never produced are
 * dropped, so the row describes what happened rather than what could.
 *
 * The swatch mirrors the bubble it stands for, fill rule included — that is how
 * the fill rule gets taught without a sentence explaining it, and it is what
 * makes the row's two halves self-evident: the confidence classes carry a
 * washed ground, the no-value statuses sit bare. It is a rounded swatch rather
 * than a dot because a bubble is no longer drawn with one; a legend showing
 * dots would be teaching a vocabulary the grid below it does not use. `valued`
 * is counted off the data rather than assumed, so a class that ever spans both
 * still draws honestly.
 */
function lensLegend() {
  const rows = lensTally(App.lens);
  const row = h("div", { class: "legend" });

  for (const item of rows) {
    const active = App.classFilter.has(item.key);
    row.append(h("button", {
      type: "button",
      class: "legend-class" + (active ? " on" : ""),
      "aria-pressed": String(active),
      title: active ? "Stop filtering by " + item.label : "Show only " + item.label,
      onclick: () => {
        if (!App.classFilter.delete(item.key)) App.classFilter.add(item.key);
        render();
      },
    },
      item.mark
        ? h("i", { class: "vmark legend-mark " + item.cls, text: item.mark })
        : h("i", { class: "legend-swatch " + item.cls + (item.valued ? "" : " hollow") }),
      h("span", { text: item.label }),
      h("b", { text: String(item.count) })
    ));
  }

  if (App.classFilter.size) {
    row.append(h("button", {
      type: "button", class: "link legend-clear", text: "clear filter",
      onclick: () => { App.classFilter.clear(); render(); },
    }));
  }
  return row;
}

function renderControlRoom() {
  const root = clear($("#control"));

  root.append(lensLegend());

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
  if (!shown) {
    root.append(h("p", { class: "empty",
      text: App.classFilter.size
        ? "No variables match this filter in the " + activeLens().label.toLowerCase() + " lens."
        : "No variables match this filter." }));
  }
}
