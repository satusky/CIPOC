"use strict";
/* Ground truth — classify each recorded result against a `{item_id: value}`
 * reference file, and roll those verdicts up to the group.
 *
 * Everything here is pure: it reads `App.truth` and the already-indexed
 * results, and derives nothing that is not implied by the two. A ground-truth
 * file is always allowed to be partial — an item it does not mention is
 * `untested` and is excluded from every accuracy count, because the alternative
 * (scoring an unlisted item as wrong) makes a half-built reference file look
 * like a catastrophic run.
 */

/* Ordered worst-first: this is the sort order in the table and the order the
 * legend reads, so the verdicts worth acting on come first. */
const VERDICTS = ["mismatch", "missed", "spurious", "near", "match", "untested"];

const VERDICT_LABEL = {
  match: "correct",
  near: "correct, reformatted",
  mismatch: "wrong",
  missed: "missed",
  spurious: "spurious",
  untested: "untested",
};

/* The glyph shown in the bubble's trailing track. Chosen so the four that need
 * acting on are visually distinct from each other at 11px, and so `untested`
 * reads as "nothing was asserted" rather than as any kind of result. */
const VERDICT_MARK = {
  match: "\u2713",     // ✓
  near: "\u2248",      // ≈
  mismatch: "\u2715",  // ✕
  missed: "\u25cb",    // ○  — hollow: a value that should be there and is not
  spurious: "\u25cf",  // ●  — filled: a value that is there and should not be
  untested: "\u00b7",  // ·
};

/* Only these count as coded-correctly. `near` is included deliberately — see
 * looseValue() for why a formatting difference is not an extraction error. */
const CORRECT = new Set(["match", "near"]);

/* Ranked for sorting; matches VERDICTS order. */
const VERDICT_RANK = Object.fromEntries(VERDICTS.map((v, i) => [v, i]));

const hasTruth = () => App.truth.size > 0;

/* Comparison is on the trimmed string, because that is what the state holds
 * (`"2"`, `"20250220"`) and what `run_case_state.py:_load_structured_data`
 * already coerces this same file shape to. */
const exactValue = (value) => (value == null ? "" : String(value).trim());

/* A second, deliberately sloppy normalization: upper-cased, stripped of
 * everything but letters and digits, and stripped of leading zeros.
 *
 * Two values equal only under *this* are reported as `near` rather than
 * `mismatch`, and that distinction is the point. A hand-built reference file
 * writes `3` where the registry writes the fixed-width `03`, and a model that
 * answers `2025-03-18` for a YYYYMMDD item has understood the note perfectly
 * and formatted the answer wrongly. Both are real defects, but they are defects
 * of a different kind than reading the wrong date, and folding them together
 * hides the difference exactly where refinement needs it — item 1280 in the
 * committed state failed validation on precisely that dash-date shape.
 */
function looseValue(value) {
  const bare = exactValue(value).toUpperCase().replace(/[^A-Z0-9]/g, "");
  return bare.replace(/^0+(?=.)/, "");
}

/* undefined = the file does not mention this item at all; "" = it mentions it
 * and asserts there is no value. The two are different answers and the verdict
 * table treats them differently. */
const truthFor = (itemId) => App.truth.get(Number(itemId));

/* {verdict, got, expected} for one App.variables entry. */
function verdictFor(entry) {
  const expected = truthFor(entry.item_id);
  const got = exactValue(entry.result.value);

  if (expected === undefined) return { verdict: "untested", got, expected: null };

  const want = exactValue(expected);
  if (want === got) return { verdict: "match", got, expected: want };
  if (!want && !got) return { verdict: "match", got, expected: want };
  if (!want) return { verdict: "spurious", got, expected: want };
  if (!got) return { verdict: "missed", got, expected: want };
  if (looseValue(want) === looseValue(got)) return { verdict: "near", got, expected: want };
  return { verdict: "mismatch", got, expected: want };
}

const verdictClass = (verdict) => "m-" + verdict;

/* Roll-up over any set of entries: counts per verdict, plus tested/correct. */
function tally(entries) {
  const counts = Object.fromEntries(VERDICTS.map((v) => [v, 0]));
  for (const entry of entries) counts[verdictFor(entry).verdict] += 1;
  const tested = entries.length - counts.untested;
  const correct = [...CORRECT].reduce((sum, v) => sum + counts[v], 0);
  return { counts, tested, correct };
}

const caseAccuracy = () => tally(App.variables);

/* A group's roll-up, plus the one finding no per-variable verdict can carry.
 *
 * When a gate or a site rule excludes a whole group, every variable in it is
 * stamped `not_applicable` and each one compares as `missed` in isolation —
 * six identical verdicts that all have a single cause upstream. Naming that
 * cause at the group level is what turns "six wrong answers" into "one wrong
 * gate", which is the actionable form.
 */
function groupVerdict(group) {
  const entries = App.variables.filter((v) => v.group_id === group.group_id);
  const roll = tally(entries);
  const state = groupState(group);

  let finding = null;
  if (roll.tested > 0) {
    const expectsValues = entries.some((v) => exactValue(truthFor(v.item_id)) !== "");
    if (state.kind === "excluded" && expectsValues) finding = "wrongly_excluded";
    else if (state.kind !== "excluded" && state.coded > 0 && !expectsValues) {
      finding = "wrongly_admitted";
    }
  }

  return { ...roll, total: entries.length, finding, entries };
}

const FINDING_LABEL = {
  wrongly_excluded: "gate wrongly excluded this group",
  wrongly_admitted: "gate wrongly admitted this group",
};

/* --------------------------------------------------------------- loading */

/* Keys arrive as JSON strings; App.results is keyed by Number (app.js), and a
 * Map keyed both ways would miss every lookup. Values are coerced to string to
 * match the state, so a reference file written with a bare integer 2 still
 * compares equal to the recorded "2". */
function indexTruth(raw, source) {
  App.truth = new Map(
    Object.entries(raw || {}).map(([k, v]) => [Number(k), v == null ? "" : String(v)])
  );
  App.truthSource = App.truth.size ? source : null;
  if (!App.truth.size) App.compare = false;
}

/* Server first, then a sibling static file, then nothing. Absence is not an
 * error: with no reference file the workbench is exactly the report it was. */
async function loadTruth() {
  for (const [url, source] of [["api/ground-truth", "server"], ["ground_truth.json", "ground_truth.json"]]) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) continue;
      const raw = await response.json();
      if (raw && typeof raw === "object" && Object.keys(raw).length) {
        indexTruth(raw, source);
        return true;
      }
    } catch (err) { /* not served, or not JSON — try the next */ }
  }
  return false;
}

/* A reference file picked from disk, for the case where the workbench is served
 * without one (plain `http.server`, or a file you want to try without moving
 * it into place). Re-renders the chrome as well as the views: renderChrome()
 * runs once at boot, so the accuracy chip and the compare control would
 * otherwise never appear. */
function onTruthFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    let raw;
    try {
      raw = JSON.parse(reader.result);
    } catch (err) {
      alert("That file is not valid JSON: " + err.message);
      return;
    }
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      alert("Expected a JSON object of {item_id: value}.");
      return;
    }
    indexTruth(raw, file.name);
    App.compare = hasTruth();
    renderChrome();
    render();
    if (App.selection) renderDetail();
  };
  reader.readAsText(file);
}

function accuracyTooltip() {
  const acc = caseAccuracy();
  const dl = h("dl", { class: "kv" });
  for (const verdict of VERDICTS) {
    if (!acc.counts[verdict]) continue;
    dl.append(h("dt", {}, h("i", { class: "dot m-" + verdict }), VERDICT_LABEL[verdict]),
      h("dd", { text: String(acc.counts[verdict]) }));
  }
  return h("div", {},
    h("h4", { text: acc.correct + " of " + acc.tested + " tested values correct" }),
    dl,
    h("p", { class: "muted", style: "margin:6px 0 0",
      text: "reference: " + (App.truthSource || "—") + " · " +
        acc.counts.untested + " variable(s) it does not mention" }));
}
