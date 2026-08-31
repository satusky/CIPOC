"use strict";
/* Tabular view — one row per variable, sortable, filterable, grouped or flat.
 * "Derived" means the value came in as structured data rather than from the
 * extractor; that is the recorded status, not an inference. */

const COLUMNS = [
  { key: "item_id",    label: "Item",       cls: "num",  get: (v) => v.item_id },
  { key: "name",       label: "Variable",   get: (v) => v.name },
  { key: "group_name", label: "Group",      get: (v) => v.group_name },
  { key: "source",     label: "Source",     get: (v) => (v.result.status === "structured_data" ? "derived" : "extracted") },
  { key: "status",     label: "Status",     get: (v) => v.result.status },
  { key: "value",      label: "Value",      cls: "code", get: (v) => v.result.value || "" },
  { key: "confidence", label: "Confidence", get: (v) => (v.result.extraction || {}).presence_confidence || "" },
  { key: "valid",      label: "Valid",      get: (v) => {
      const e = v.result.extraction;
      return e ? (e.is_valid ? "yes" : "no") : "";
    } },
  { key: "attempts",   label: "Attempts",   cls: "num", get: (v) => (v.result.extraction || {}).extraction_attempts || 0 },
  { key: "evidence",   label: "Evidence",   cls: "num", get: (v) => (((v.result.extraction || {}).spans) || []).length },
];

/* Appended only while comparing. Kept out of COLUMNS rather than rendered
   empty: a table with no ground-truth file must be the table it was, and two
   blank columns would still take width and still be sortable. */
const COMPARE_COLUMNS = [
  { key: "expected", label: "Expected", cls: "code", get: (v) => verdictFor(v).expected || "" },
  { key: "verdict",  label: "Verdict",  get: (v) => verdictFor(v).verdict },
];

const activeColumns = () => (App.compare ? COLUMNS.concat(COMPARE_COLUMNS) : COLUMNS);

const CONFIDENCE_RANK = { low: 1, medium: 2, high: 3, max: 4 };

function sortValue(column, entry) {
  const raw = column.get(entry);
  if (column.key === "confidence") return CONFIDENCE_RANK[raw] || 0;
  /* Alphabetical would open on `match`; VERDICTS is ordered worst-first so one
     click puts the disagreements at the top, which is the only reason to sort
     by this column at all. */
  if (column.key === "verdict") return VERDICT_RANK[raw];
  return raw;
}

function filteredVariables() {
  const q = App.varFilter;
  if (!q) return App.variables.slice();
  return App.variables.filter((v) =>
    String(v.item_id).includes(q) ||
    v.name.toLowerCase().includes(q) ||
    v.group_name.toLowerCase().includes(q) ||
    v.result.status.includes(q) ||
    (v.result.value || "").toLowerCase().includes(q) ||
    (App.compare && verdictFor(v).verdict.includes(q)) ||
    (App.compare && (verdictFor(v).expected || "").toLowerCase().includes(q)));
}

function sortVariables(entries) {
  const column = activeColumns().find((c) => c.key === App.sort.key) || COLUMNS[0];
  return entries.sort((a, b) => {
    const av = sortValue(column, a);
    const bv = sortValue(column, b);
    if (av === bv) return a.item_id - b.item_id;
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * App.sort.dir;
    return String(av).localeCompare(String(bv)) * App.sort.dir;
  });
}

function headerCell(column) {
  const active = App.sort.key === column.key;
  const props = {
    scope: "col",
    role: "button",
    tabindex: "0",
    text: column.label,
    onclick: () => {
      if (App.sort.key === column.key) App.sort.dir *= -1;
      else App.sort = { key: column.key, dir: 1 };
      renderTable();
    },
  };
  if (active) props["aria-sort"] = App.sort.dir === 1 ? "ascending" : "descending";
  const th = h("th", props);
  th.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); th.click(); }
  });
  return th;
}

function bodyRow(entry) {
  const r = entry.result;
  const cells = activeColumns().map((column) => {
    if (column.key === "status") {
      return h("td", {},
        h("span", { class: "dot " + indicatorClass(r) }),
        statusLabel(r.status));
    }
    if (column.key === "verdict") {
      const verdict = verdictFor(entry).verdict;
      return h("td", {},
        h("i", { class: "vmark m-" + verdict, style: "margin-right:7px",
          text: VERDICT_MARK[verdict] }),
        VERDICT_LABEL[verdict]);
    }
    const value = column.get(entry);
    const cell = h("td", { class: column.cls || null, text: value === 0 ? "0" : String(value || "") });
    if (column.key === "valid" && value === "no") cell.className = "code";
    return cell;
  });

  const row = h("tr", {
    tabindex: "0",
    dataset: { entity: "variable:" + entry.item_id,
               annotated: isAnnotated("variable", entry.item_id) ? "1" : null },
    onclick: () => select("variable", entry.item_id),
    onkeydown: (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select("variable", entry.item_id); }
    },
  }, cells);
  return row;
}

function renderTable() {
  const root = clear($("#table-wrap"));
  const entries = sortVariables(filteredVariables());

  if (!entries.length) {
    root.append(h("p", { class: "empty", text: "No variables match this filter." }));
    return;
  }

  const body = h("tbody", {});
  if (App.grouped) {
    const order = App.groups.map((g) => g.group_id);
    const byGroup = new Map(order.map((id) => [id, []]));
    for (const entry of entries) byGroup.get(entry.group_id).push(entry);
    for (const groupId of order) {
      const rows = byGroup.get(groupId);
      if (!rows.length) continue;
      const group = App.groupById.get(groupId);
      const state = groupState(group);
      body.append(h("tr", { class: "group-sep" },
        h("td", { colspan: String(activeColumns().length) },
          group.name + " · " + rows.length + " of " + state.total + " · " + state.label)));
      for (const entry of rows) body.append(bodyRow(entry));
    }
  } else {
    for (const entry of entries) body.append(bodyRow(entry));
  }

  root.append(h("table", { class: "vtable" },
    h("thead", {}, h("tr", {}, activeColumns().map(headerCell))),
    body));
}
