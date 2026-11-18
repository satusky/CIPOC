"use strict";
/* Shared by every dashboard variant: scoring, run-to-run stability, chart
 * primitives, the filter row and the two drill-down drawers.
 *
 * Classic script on purpose. ES modules and fetch() are blocked on file://, and
 * the dashboard has to open by double-click inside the airgap.
 *
 * Everything is derived in the browser from the raw values in data.js, so any
 * filter or stratification re-scores from scratch and the numbers always agree. */
(function () {
  const RAW = window.CIPOC_EVAL;
  const V = RAW.variables, P = RAW.patients, R = RAW.runs;
  const nV = V.length, nP = P.length, nR = R.length;
  const ALL_V = V.map((_, i) => i);
  const LAST = nR - 1;

  /* ---- Scoring -----------------------------------------------------------
   * Values compare after a loose normalization (case, punctuation, leading
   * zeros), the same forgiveness the Workbench calls a "near" match: "01" vs
   * "1" is a formatting difference, not an extraction error.
   *
   * A blank ground-truth cell means "expected to stay uncoded". That gives five
   * outcomes per patient-variable cell:
   *   match     truth has a value, extracted the same value
   *   wrong     truth has a value, extracted a different one
   *   missed    truth has a value, nothing extracted
   *   spurious  truth is blank, something extracted
   *   tn        truth is blank, nothing extracted
   * A variable the run did not plan (status "-") or was given as structured
   * data (status "s") was never extracted, so it keeps the "none" verdict and
   * is not scored, the same as every cell of a failed patient.
   */
  const NONE = 0, MATCH = 1, WRONG = 2, MISSED = 3, SPURIOUS = 4, TN = 5;
  const VERDICT = ["none", "match", "wrong", "missed", "spurious", "tn"];
  const GLYPH = { none: "–", match: "✓", wrong: "✕", missed: "○", spurious: "●", tn: "·" };
  const VERDICT_LABEL = {
    none: "not run", match: "correct", wrong: "wrong value", missed: "missed",
    spurious: "spurious", tn: "correctly blank",
  };

  const norm = (value) => {
    if (value == null) return null;
    const text = String(value).trim().toUpperCase().replace(/[^A-Z0-9]/g, "").replace(/^0+(?=.)/, "");
    return text === "" ? null : text;
  };

  const UNSCORED_STATUS = new Set(["-", "s"]);
  const TRUTH = RAW.truth.map((row) => row.map(norm));
  R.forEach((run, r) => {
    run.norm = run.pred.map((row) => row && row.map(norm));
    run.verdict = new Uint8Array(nP * nV);
    for (let p = 0; p < nP; p++) {
      const got = run.norm[p];
      if (!got) continue;
      for (let v = 0; v < nV; v++) {
        if (UNSCORED_STATUS.has(run.status[p][v])) continue;
        const t = TRUTH[p][v], g = got[v];
        run.verdict[p * nV + v] = t == null ? (g == null ? TN : SPURIOUS) : g == null ? MISSED : g === t ? MATCH : WRONG;
      }
    }
    run.configChanged = r > 0 && run.config_digest !== R[r - 1].config_digest;
  });

  const scored = (r, p, v) => R[r].verdict[p * nV + v] !== NONE;

  function counts(r, vs, ps) {
    const c = [0, 0, 0, 0, 0, 0], verdict = R[r].verdict;
    for (const p of ps) for (const v of vs) c[verdict[p * nV + v]]++;
    return { match: c[1], wrong: c[2], missed: c[3], spurious: c[4], tn: c[5], n: c[1] + c[2] + c[3] + c[4] + c[5] };
  }

  /* Each metric is k/n. Precision and recall ignore correctly-blank cells, so a
   * variable that is blank for most patients cannot look good by staying blank;
   * accuracy counts them. F1 is not a simple proportion, so it gets no interval.
   * With no truth values at all (a prostate-only item in a breast stratum) the
   * three are undefined rather than zero; stray extractions there still show
   * up as spurious errors and in accuracy. */
  const METRICS = {
    f1: { label: "F1", parts: (c) => [2 * c.match, 2 * c.match + 2 * c.wrong + c.missed + c.spurious], ci: false },
    precision: { label: "Precision", parts: (c) => [c.match, c.match + c.wrong + c.spurious], ci: true },
    recall: { label: "Recall", parts: (c) => [c.match, c.match + c.wrong + c.missed], ci: true },
    accuracy: { label: "Accuracy", parts: (c) => [c.match + c.tn, c.n], ci: true },
  };
  const score = (c, key) => {
    key = key || state.metric;
    if (key !== "accuracy" && c.match + c.wrong + c.missed === 0) return null;
    const [k, n] = METRICS[key].parts(c);
    return n ? k / n : null;
  };
  /* Wilson 95% interval for a proportion; small strata need it to be read honestly. */
  function wilson(k, n) {
    if (!n) return null;
    const z = 1.96, p = k / n, d = 1 + (z * z) / n;
    const mid = (p + (z * z) / (2 * n)) / d;
    const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
    return [Math.max(0, mid - half), Math.min(1, mid + half)];
  }

  /* ---- State ------------------------------------------------------------- */
  const STORE = "cipoc-eval-dashboard";
  const state = Object.assign(
    { metric: "f1", stratify: "none", tissue: "all", year: "all", window: "config", view: "performance", theme: null },
    (() => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { return {}; } })()
  );
  let variant = null;
  function set(key, value) {
    state[key] = value;
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { /* private window */ }
    render();
  }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
  const applyTheme = () => {
    document.documentElement.dataset.theme = state.theme || (prefersDark.matches ? "dark" : "light");
  };
  applyTheme();
  prefersDark.addEventListener("change", () => { applyTheme(); render(); });
  const isDark = () => document.documentElement.dataset.theme === "dark";

  /* ---- Patients, strata, run windows ------------------------------------ */
  const TISSUES = (() => {
    const tally = new Map();
    P.forEach((p) => tally.set(p.tissue, (tally.get(p.tissue) || 0) + 1));
    // Colour follows the tissue, never its rank in a filtered view. Past seven
    // tissues the tail folds into "other" rather than inventing a ninth hue.
    const names = [...tally.keys()].sort((a, b) => tally.get(b) - tally.get(a) || a.localeCompare(b));
    const kept = names.length > 8 ? names.slice(0, 7) : names;
    const list = kept.slice().sort().map((name, i) => ({ key: name, label: name, color: `var(--series-${i + 1})`, has: (p) => p.tissue === name }));
    if (kept.length < names.length) list.push({ key: "other", label: "other", color: "var(--series-8)", has: (p) => !kept.includes(p.tissue) });
    return list;
  })();
  const YEARS = [...new Set(P.map((p) => p.year))].sort();
  const YEAR_BANDS = (() => {
    // At most five bands, so the ordinal ramp keeps visible steps. Bands are
    // counted back from the newest year.
    const size = Math.ceil(YEARS.length / 5), bands = [];
    for (let end = YEARS.length; end > 0; end -= size) bands.unshift(YEARS.slice(Math.max(0, end - size), end));
    return bands.map((years, i) => {
      const step = bands.length === 1 ? 3 : 1 + Math.round((i * 4) / (bands.length - 1));
      const label = years.length === 1 ? String(years[0]) : `${years[0]}–${String(years[years.length - 1]).slice(2)}`;
      return { key: label, label, color: `var(--ord-${step})`, has: (p) => years.includes(p.year) };
    });
  })();

  const patients = () =>
    P.map((_, i) => i).filter((i) =>
      (state.tissue === "all" || P[i].tissue === state.tissue) && (state.year === "all" || String(P[i].year) === state.year));
  function strata(ps, by) {
    by = by || state.stratify;
    const defs = by === "tissue" ? TISSUES : by === "year" ? YEAR_BANDS : null;
    if (!defs) return [{ key: "all", label: "All patients", color: "var(--accent)", patients: ps }];
    return defs.map((d) => ({ key: d.key, label: d.label, color: d.color, patients: ps.filter((p) => d.has(P[p])) }))
      .filter((s) => s.patients.length);
  }
  function windowRuns() {
    const all = R.map((_, i) => i);
    if (state.window === "all") return all;
    if (state.window === "last3") return all.slice(-3);
    if (state.window === "last5") return all.slice(-5);
    let first = LAST;
    while (first > 0 && R[first - 1].config_digest === R[LAST].config_digest) first--;
    return all.slice(first);
  }
  function windowLabel() {
    const runs = windowRuns();
    const span = runs.length === nR ? `all ${nR} runs` : runs.length === 1 ? `run ${runs[0] + 1} only` : `runs ${runs[0] + 1}–${runs[runs.length - 1] + 1}`;
    return state.window === "config" ? `${span}, same configuration` : span;
  }

  /* ---- Run-to-run stability ----------------------------------------------
   * Over a window of runs, one patient-variable cell is:
   *   right   the same value every run, and it is correct
   *   wrong   the same value every run, and it is not  (a systematic error)
   *   varies  the value changed between runs           (a stochastic one)
   *   na      blank truth, blank every run: nothing was ever at stake
   * `na` cells are left out of every share so site-limited variables do not pad
   * the consistency of patients they never applied to.
   *
   * A cell that varies also says how unsettled it is. `changes` counts the runs
   * whose value differs from the run before, `settled` the latest runs in a row
   * that agree, and `unrest` (0 to 1) weights each change by how recent it is:
   * one early change that has held since scores low, flipping every run scores 1. */
  function cellStability(p, v, runs) {
    const seen = new Map(), values = [];
    let ok = 0;
    for (const r of runs) {
      const verdict = R[r].verdict[p * nV + v];
      if (verdict === NONE) continue;
      const value = R[r].norm[p][v];
      values.push(value);
      seen.set(value, (seen.get(value) || 0) + 1);
      if (verdict === MATCH || verdict === TN) ok++;
    }
    const k = values.length;
    if (k < 2) return null;
    if (TRUTH[p][v] == null && seen.size === 1 && seen.has(null)) return { cls: "na", k, ok };
    if (seen.size === 1) return { cls: ok === k ? "right" : "wrong", k, ok, distinct: 1 };
    let changes = 0, weight = 0, settled = 1;
    for (let i = 1; i < k; i++) if (values[i] !== values[i - 1]) { changes++; weight += i; }
    while (settled < k && values[k - 1 - settled] === values[k - 1]) settled++;
    return { cls: "varies", k, ok, distinct: seen.size, changes, settled, unrest: weight / ((k * (k - 1)) / 2) };
  }
  function stability(ps, vs, runs) {
    const sum = { right: 0, wrong: 0, varies: 0, scored: 0, ok: 0, k: 0 };
    for (const p of ps) for (const v of vs) {
      const cell = cellStability(p, v, runs);
      if (!cell || cell.cls === "na") continue;
      sum[cell.cls]++; sum.scored++; sum.ok += cell.ok; sum.k += cell.k;
    }
    sum.consistency = sum.scored ? (sum.right + sum.wrong) / sum.scored : null;
    sum.accuracy = sum.k ? sum.ok / sum.k : null;
    return sum;
  }

  function usage(r, ps) {
    const out = { n: 0, failed: 0, missing: 0, input: 0, output: 0, seconds: 0 };
    for (const p of ps) {
      const u = R[r].usage[p];
      if (!u) continue;
      if (!R[r].pred[p]) { out.failed++; continue; }
      // Missing usage is unknown, not zero: leave the case out of the mean.
      if (u[0] == null || u[1] == null) { out.missing++; continue; }
      out.n++; out.input += u[0]; out.output += u[1]; out.seconds += u[3] || 0;
    }
    out.mean = out.n ? (out.input + out.output) / out.n : null;
    return out;
  }

  /* ---- DOM and formatting ------------------------------------------------ */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    fill(el, attrs, kids);
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    fill(el, attrs, kids);
    return el;
  }
  function fill(el, attrs, kids) {
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value == null || value === false) continue;
      if (key === "style") el.style.cssText = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : value);
    }
    // Labels come from data files: always text nodes, never markup.
    for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  const pct = (x, digits) => (x == null ? "–" : (x * 100).toFixed(digits == null ? 1 : digits) + "%");
  const compact = (n) => n == null ? "–" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + "K" : String(Math.round(n));
  const runTitle = (r) => `Run ${r + 1} · ${R[r].label || R[r].run_id} · ${String(R[r].started_at).slice(0, 10)}`;
  const variableLabel = (v) => `${V[v].item_id} ${V[v].name}`;

  /* A signed change with an arrow, so direction never rides on colour alone. */
  function delta(change, text, goodWhenUp) {
    if (change == null) return h("span", { class: "delta flat" }, "–");
    const flat = Math.abs(change) < 1e-9;
    const good = goodWhenUp === false ? change < 0 : change > 0;
    return h("span", { class: "delta " + (flat ? "flat" : good ? "up" : "down") }, (flat ? "■ " : change > 0 ? "▲ " : "▼ ") + text);
  }
  const deltaPP = (now, before) => (now == null || before == null ? delta(null) : delta(now - before, Math.abs((now - before) * 100).toFixed(1) + " pp"));

  const tipEl = h("div", { class: "tip", hidden: true });
  const tip = {
    /* rows: [{ color?, kind?: "line"|"box", value, label }] — value leads, label follows. */
    show(event, title, rows) {
      tipEl.replaceChildren();
      fill(tipEl, null, [
        title && h("div", { class: "t" }, title),
        (rows || []).map((row) => h("div", { class: "r" },
          row.color && h("i", { class: "key " + (row.kind || "line"), style: `background:${row.color}` }),
          h("b", null, row.value), h("span", null, row.label))),
      ]);
      tipEl.hidden = false;
      const box = tipEl.getBoundingClientRect();
      let x = event.clientX + 14, y = event.clientY + 14;
      if (x + box.width > innerWidth - 8) x = event.clientX - box.width - 14;
      if (y + box.height > innerHeight - 8) y = Math.max(8, innerHeight - box.height - 8);
      tipEl.style.left = Math.max(8, x) + "px";
      tipEl.style.top = y + "px";
    },
    hide() { tipEl.hidden = true; },
  };

  function seg(options, current, onPick, label) {
    return h("div", { class: "seg", role: "group", "aria-label": label },
      options.map(([key, text]) => h("button", { type: "button", "aria-pressed": String(key === current), onclick: () => onPick(key) }, text)));
  }
  function select(options, current, onPick, label) {
    const el = h("select", { "aria-label": label, onchange: (e) => onPick(e.target.value) },
      options.map(([key, text]) => h("option", { value: key, selected: key === current }, text)));
    el.value = current;
    return el;
  }
  const field = (label, control) => h("label", { class: "field" }, h("span", null, label), control);
  function legend(items, kind) {
    return h("div", { class: "legend" }, items.map((item) =>
      h("span", null, h("i", { class: "key " + (item.kind || kind || "box"), style: `background:${item.color}` }), item.label)));
  }

  /* ---- Charts ------------------------------------------------------------
   * A chart is drawn once its host is in the document and has a width. */
  let pending = [];
  function host(draw, height) {
    const el = h("div", { style: `height:${height}px` });
    pending.push([el, draw]);
    return el;
  }
  function flush() {
    const batch = pending;
    pending = [];
    for (const [el, draw] of batch) if (el.isConnected) el.replaceChildren(draw(Math.max(120, el.clientWidth)));
  }

  /* Shared y-domain for a set of series: floor to the nearest 10% below the
   * lowest value, so small multiples stay comparable without wasting the plot. */
  function domainOf(values) {
    const real = values.filter((x) => x != null);
    const low = real.length ? Math.min(...real) : 0;
    return [Math.max(0, Math.min(0.9, Math.floor(low * 10) / 10)), 1];
  }

  /* Gridline positions. A share axis ending at 100% steps in round percentages
   * down from the top; anything else gets five even steps. */
  function ticksFor(lo, hi) {
    if (hi !== 1) return [0, 1, 2, 3, 4].map((i) => lo + ((hi - lo) * i) / 4);
    const step = hi - lo <= 0.2 ? 0.05 : hi - lo <= 0.5 ? 0.1 : 0.2, out = [];
    for (let i = 0; hi - i * step >= lo - 1e-9; i++) out.push(Math.round((hi - i * step) * 100) / 100);
    return out;
  }

  /* Top of a count or token axis: four round steps that clear the largest value. */
  function niceMax(top) {
    const rough = Math.max(1, top) / 4, pow = Math.pow(10, Math.floor(Math.log10(rough)));
    return 4 * Math.max(1, [1, 2, 2.5, 5, 10].map((k) => k * pow).find((step) => step >= rough && Number.isInteger(step)));
  }

  /* opts: { series: [{label, color, values, lo?, hi?}], domain, compact, fmt,
   *         title(i), extra(i) -> rows, onclick, markers } — one value per run. */
  function lineChart(width, height, opts) {
    const compactMode = !!opts.compact, fmt = opts.fmt || pct;
    const m = compactMode ? { l: 4, r: 6, t: 6, b: 5 } : { l: 44, r: 16, t: 14, b: 24 };
    const [lo, hi] = opts.domain, n = opts.series[0].values.length;
    const iw = width - m.l - m.r, ih = height - m.t - m.b;
    const x = (i) => m.l + (n === 1 ? iw / 2 : (i * iw) / (n - 1));
    const y = (value) => m.t + ih * (1 - (Math.min(hi, Math.max(lo, value)) - lo) / (hi - lo || 1));
    const svg = s("svg", { width, height, role: "img", "aria-label": opts.label || "line chart" });

    const ticks = compactMode ? [lo, hi] : ticksFor(lo, hi);
    for (const tick of ticks) {
      svg.append(s("line", { class: Math.abs(tick - lo) < 1e-9 ? "axisline" : "gridline", x1: m.l, x2: width - m.r, y1: y(tick), y2: y(tick) }));
      if (!compactMode) svg.append(s("text", { x: m.l - 8, y: y(tick) + 4, "text-anchor": "end" }, fmt(tick, 0)));
    }
    svg.append(s("line", { class: "axisline", x1: m.l, x2: width - m.r, y1: y(lo), y2: y(lo) }));
    for (let i = 0; i < n; i++) {
      // A tick where the configuration changed: the first thing to check when a line moves.
      if (opts.markers !== false && R[i] && R[i].configChanged) {
        svg.append(compactMode
          ? s("line", { class: "axisline", x1: x(i), x2: x(i), y1: y(lo), y2: y(lo) - 4 })
          : s("line", { class: "gridline", x1: x(i), x2: x(i), y1: m.t, y2: y(lo) }));
      }
      if (!compactMode) svg.append(s("text", { x: x(i), y: height - 6, "text-anchor": "middle" }, i + 1));
    }

    for (const series of opts.series) {
      if (series.lo) {
        const upper = [], lower = [];
        series.values.forEach((value, i) => {
          if (value == null || series.lo[i] == null) return;
          upper.push(`${x(i)},${y(series.hi[i])}`);
          lower.unshift(`${x(i)},${y(series.lo[i])}`);
        });
        if (upper.length > 1) svg.append(s("polygon", { points: upper.concat(lower).join(" "), fill: series.color, opacity: 0.1 }));
      }
      let path = "", pen = false, end = -1;
      series.values.forEach((value, i) => {
        if (value == null) { pen = false; return; }
        path += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(value).toFixed(1)}`;
        pen = true; end = i;
      });
      svg.append(s("path", { d: path, fill: "none", stroke: series.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (end >= 0) svg.append(s("circle", { cx: x(end), cy: y(series.values[end]), r: compactMode ? 3 : 4, fill: series.color, stroke: "var(--surface)", "stroke-width": 2 }));
    }

    const hair = s("line", { class: "crosshair", y1: m.t, y2: y(lo), visibility: "hidden" });
    const dots = opts.series.map((series) => s("circle", { r: 4, fill: series.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
    const hit = s("rect", { x: 0, y: 0, width, height, fill: "transparent", style: opts.onclick ? "cursor:pointer" : "" });
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(n - 1, Math.round(((event.clientX - box.left - m.l) / (iw || 1)) * (n - 1))));
      hair.setAttribute("x1", x(i)); hair.setAttribute("x2", x(i)); hair.setAttribute("visibility", "visible");
      opts.series.forEach((series, k) => {
        const value = series.values[i];
        dots[k].setAttribute("visibility", value == null ? "hidden" : "visible");
        if (value != null) { dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(value)); }
      });
      const rows = opts.series.map((series) => ({
        color: series.color, value: fmt(series.values[i]),
        label: series.label + (series.lo && series.lo[i] != null ? `  (95% CI ${fmt(series.lo[i], 0)}–${fmt(series.hi[i], 0)})` : ""),
      }));
      tip.show(event, opts.title ? opts.title(i) : runTitle(i), rows.concat(opts.extra ? opts.extra(i) : []));
    });
    hit.addEventListener("pointerleave", () => {
      hair.setAttribute("visibility", "hidden");
      dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
      tip.hide();
    });
    if (opts.onclick) hit.addEventListener("click", opts.onclick);
    svg.append(hair, ...dots, hit);
    return svg;
  }

  function sparkline(values, width, height) {
    width = width || 84; height = height || 24;
    const real = values.filter((x) => x != null);
    if (real.length < 2) return h("span", { class: "muted" }, "–");
    const lo = Math.min(...real), hi = Math.max(...real), n = values.length;
    const x = (i) => 3 + (i * (width - 8)) / (n - 1);
    const y = (value) => 4 + (height - 8) * (1 - (value - lo) / (hi - lo || 1));
    let path = "", pen = false, end = 0;
    values.forEach((value, i) => {
      if (value == null) { pen = false; return; }
      path += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(value).toFixed(1)}`;
      pen = true; end = i;
    });
    return s("svg", { width, height, "aria-hidden": "true" },
      s("path", { d: path, fill: "none", stroke: "var(--muted)", "stroke-width": 1.5, "stroke-linejoin": "round" }),
      s("circle", { cx: x(end), cy: y(values[end]), r: 3, fill: "var(--accent)", stroke: "var(--surface)", "stroke-width": 1.5 }));
  }

  /* Stacked columns, one per run. cols: [{ parts: [{value, color, label}] }] */
  function columns(width, height, cols) {
    const m = { l: 44, r: 16, t: 10, b: 24 }, n = cols.length;
    const iw = width - m.l - m.r, ih = height - m.t - m.b;
    const max = niceMax(Math.max(...cols.map((col) => col.parts.reduce((a, part) => a + part.value, 0))));
    const band = iw / n, bar = Math.min(24, band - 6);
    const y = (value) => m.t + ih * (1 - value / max);
    const svg = s("svg", { width, height, role: "img", "aria-label": "errors per run" });
    for (let i = 0; i <= 4; i++) {
      const tick = (max * i) / 4;
      svg.append(s("line", { class: i ? "gridline" : "axisline", x1: m.l, x2: width - m.r, y1: y(tick), y2: y(tick) }),
        s("text", { x: m.l - 8, y: y(tick) + 4, "text-anchor": "end" }, tick));
    }
    cols.forEach((col, i) => {
      const cx = m.l + band * (i + 0.5), group = s("g");
      let base = 0;
      const live = col.parts.filter((part) => part.value > 0);
      live.forEach((part, k) => {
        const y0 = y(base), y1 = y(base + part.value), last = k === live.length - 1;
        const height = Math.max(1, y0 - y1 - (last ? 0 : 2)), r = last ? Math.min(4, height) : 0, left = cx - bar / 2;
        group.append(s("path", {
          fill: part.color,
          d: `M${left},${y0}V${y0 - height + r}Q${left},${y0 - height} ${left + r},${y0 - height}H${left + bar - r}Q${left + bar},${y0 - height} ${left + bar},${y0 - height + r}V${y0}Z`,
        }));
        base += part.value;
      });
      const hit = s("rect", { x: cx - band / 2, y: m.t, width: band, height: ih, fill: "transparent" });
      hit.addEventListener("pointermove", (event) => {
        group.setAttribute("opacity", 0.8);
        tip.show(event, runTitle(i), col.parts.map((part) => ({ color: part.color, kind: "box", value: part.value, label: part.label })));
      });
      hit.addEventListener("pointerleave", () => { group.setAttribute("opacity", 1); tip.hide(); });
      svg.append(group, hit, s("text", { x: cx, y: height - 6, "text-anchor": "middle" }, i + 1));
    });
    return svg;
  }

  /* A proportional bar with 2px surface gaps. parts: [{value, color, label}] */
  function stack(parts, total, cls) {
    total = total || parts.reduce((a, part) => a + part.value, 0) || 1;
    const el = h("div", { class: "stack " + (cls || ""), role: "img", "aria-label": parts.map((part) => `${part.label} ${part.value}`).join(", ") },
      parts.filter((part) => part.value > 0).map((part) => h("i", { style: `flex:${part.value} 0 0;background:${part.color}` })));
    el.style.width = Math.max(4, (100 * parts.reduce((a, part) => a + part.value, 0)) / total) + "%";
    el.addEventListener("pointermove", (event) => tip.show(event, "", parts.map((part) => ({ color: part.color, kind: "box", value: part.value, label: part.label }))));
    el.addEventListener("pointerleave", tip.hide);
    return el;
  }
  const ERROR_PARTS = (c) => [
    { value: c.wrong, color: "var(--wrong)", label: "wrong value" },
    { value: c.missed, color: "var(--missed)", label: "missed" },
    { value: c.spurious, color: "var(--spurious)", label: "spurious" },
  ];
  const STABILITY_PARTS = (sum) => [
    { value: sum.right, color: "var(--right)", label: "same every run, correct" },
    { value: sum.wrong, color: "var(--wrong)", label: "same every run, wrong" },
    { value: sum.varies, color: "var(--varies)", label: "varies between runs" },
  ];
  const errorLegend = () => legend(ERROR_PARTS({}).map((part) => ({ color: part.color, label: part.label })));
  const stabilityLegend = () => legend(STABILITY_PARTS({}).map((part) => ({ color: part.color, label: part.label })));

  /* ---- Heat colours ------------------------------------------------------ */
  const SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const inkOn = ([r, g, b]) => (0.2126 * r + 0.7152 * g + 0.0722 * b > 140 ? "#0b0b0b" : "#ffffff");
  /* Sequential: one hue, more is darker; in dark mode the low end recedes into the surface instead. */
  function seqColor(t) {
    const index = Math.round(Math.max(0, Math.min(1, t)) * 12), hex = SEQ[isDark() ? 12 - index : index];
    return { fill: hex, ink: inkOn(rgb(hex)) };
  }
  /* Diverging: blue for gains, red for losses, neutral gray at no change. */
  function divColor(change, max) {
    const t = Math.round(Math.min(1, Math.abs(change) / max) * 4) / 4;
    const mid = rgb(isDark() ? "#383835" : "#f0efec");
    const pole = rgb(change >= 0 ? (isDark() ? "#3987e5" : "#2a78d6") : (isDark() ? "#e66767" : "#e34948"));
    const mixed = mid.map((channel, i) => Math.round(channel + (pole[i] - channel) * t));
    return { fill: `rgb(${mixed.join(",")})`, ink: inkOn(mixed) };
  }

  /* ---- Shared cards ------------------------------------------------------ */
  const tile = (label, value, side, foot) => h("div", { class: "card kpi" },
    h("div", { class: "label" }, label), h("div", { class: "row" }, h("div", { class: "value" }, value), side), h("div", { class: "foot" }, foot));
  const variableName = (v) => [h("span", { class: "muted num" }, V[v].item_id + " "), V[v].name];
  /* A number with a bar beside it, for a table cell. */
  const bars = (text, bar) => h("div", { class: "bars" }, h("b", null, text), h("div", null, bar));

  /* A table that re-ranks when a column head is clicked.
   * cols: [{ key, label, title?, right?, num?, value?(row), cell(row) }]. A column
   * with no `value` is not sortable; blanks always sort last. The choice lives in
   * `store[sortKey]` and `store[dirKey]`, so it survives a re-render. */
  function sortable(cols, rows, store, sortKey, dirKey, onRow) {
    const col = cols.find((c) => c.key === store[sortKey] && c.value) || cols[0], dir = store[dirKey];
    const sorted = rows.slice().sort((a, b) => {
      const x = col.value(a), y = col.value(b);
      if (x == null || y == null) return (x == null) - (y == null);
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    });
    const pick = (c) => {
      if (c === col) store[dirKey] = -dir;
      else { store[sortKey] = c.key; store[dirKey] = c.num ? -1 : 1; }
      render();
    };
    return h("table", { class: "list" },
      h("thead", null, h("tr", null, cols.map((c) => (c.value
        ? h("th", {
          class: "sortable" + (c.right ? " r" : ""), tabindex: 0, title: c.title, "aria-sort": c === col ? (dir < 0 ? "descending" : "ascending") : null,
          onclick: () => pick(c), onkeydown: (event) => { if (event.key === "Enter") pick(c); },
        }, c.label, c === col ? (dir < 0 ? " ▼" : " ▲") : "")
        : h("th", { title: c.title }, c.label))))),
      h("tbody", null, sorted.map((row) => h("tr", { class: "click", onclick: () => onRow(row) }, cols.map((c) => c.cell(row))))));
  }

  function kpiRow(ps) {
    const trend = R.map((_, r) => score(counts(r, ALL_V, ps)));
    const done = ps.filter((p) => R[LAST].pred[p]).length;
    const sum = stability(ps, ALL_V, windowRuns());
    const tokens = R.map((_, r) => usage(r, ps));
    const now = tokens[LAST], before = LAST ? tokens[LAST - 1] : null;
    const share = (value) => (sum.scored ? pct(value / sum.scored, 0) : "–");
    return h("div", { class: "kpis" },
      tile(`${METRICS[state.metric].label}, all variables pooled`, pct(trend[LAST]), sparkline(trend),
        [LAST ? deltaPP(trend[LAST], trend[LAST - 1]) : null, ` vs run ${LAST} · run ${nR}, ${R[LAST].label || ""}`]),
      tile("Patients scored, latest run", `${done} / ${ps.length}`, null,
        `${ps.length - done} failed or not run · ${nV} variables`),
      tile("Run-to-run, patient-variable cells", share(sum.right + sum.wrong) + " stable", null,
        [stack(STABILITY_PARTS(sum), null, "tall"),
          h("div", { style: "margin-top:4px" }, `${share(sum.wrong)} always wrong · ${share(sum.varies)} vary · ${windowLabel()}`)]),
      tile("Tokens per patient, latest run", compact(now.mean), sparkline(tokens.map((u) => u.mean)),
        [before && now.mean != null && before.mean != null
          ? delta(now.mean - before.mean, Math.abs((now.mean / before.mean - 1) * 100).toFixed(1) + "%", false) : null,
          ` vs run ${LAST} · ${compact(now.n ? now.input / now.n : null)} in, ${compact(now.n ? now.output / now.n : null)} out`,
          now.missing ? ` · ${now.missing} without usage` : ""]));
  }

  function usageCard(ps) {
    const groups = strata(ps);
    const series = groups.map((g) => ({ label: g.label, color: g.color, values: R.map((_, r) => usage(r, g.patients).mean) }));
    const top = Math.max(1, ...series.flatMap((x) => x.values).filter((x) => x != null));
    return h("div", { class: "card" },
      h("h2", null, "Tokens per patient"),
      h("div", { class: "sub" }, "Mean input + output tokens per completed patient, by run. Cases without reported usage are left out, not counted as zero."),
      groups.length > 1 && legend(groups, "line"),
      host((width) => lineChart(width, 200, {
        series, domain: [0, niceMax(top)], fmt: compact, label: "tokens per patient by run",
        extra: (i) => { const u = usage(i, ps); return [{ value: R[i].model || "–", label: "model" }, { value: `${u.failed} failed, ${u.missing} no usage`, label: "" }]; },
      }), 200));
  }

  /* Variable x run table. The table twin of every trend chart; `heat` colours
   * it. mode: "value" | "prev" (change vs previous run) | "first". */
  function runMatrix(ps, opts) {
    opts = opts || {};
    const mode = opts.mode || "value";
    const rowsFor = (vs) => R.map((_, r) => score(counts(r, vs, ps)));
    const all = rowsFor(ALL_V), perVariable = ALL_V.map((v) => rowsFor([v]));
    const shown = (row, r) => mode === "value" ? row[r] : row[r] == null || row[mode === "prev" ? r - 1 : 0] == null ? null : row[r] - row[mode === "prev" ? r - 1 : 0];
    const [lo] = domainOf(perVariable.flat());
    const maxChange = Math.max(0.1, ...perVariable.flatMap((row) => row.map((_, r) => Math.abs(shown(row, r) || 0))));
    const cell = (row, r, title) => {
      const value = shown(row, r);
      const text = value == null ? "–" : mode === "value" ? (value * 100).toFixed(0) : (value > 0 ? "+" : value < 0 ? "−" : "") + Math.abs(value * 100).toFixed(0);
      const paint = !opts.heat || value == null ? null : mode === "value" ? seqColor((value - lo) / (1 - lo)) : divColor(value, maxChange);
      const td = h("td", { class: "r", style: (paint ? `background:${paint.fill};color:${paint.ink};` : "") + (R[r].configChanged ? "border-left:2px solid var(--surface);" : "") }, text);
      td.addEventListener("pointermove", (event) => tip.show(event, `${title} · ${runTitle(r)}`,
        [{ value: pct(row[r]), label: METRICS[state.metric].label }].concat(r ? [{ value: row[r] == null || row[r - 1] == null ? "–" : ((row[r] - row[r - 1]) * 100).toFixed(1) + " pp", label: "vs previous run" }] : [])));
      td.addEventListener("pointerleave", tip.hide);
      return td;
    };
    const body = [h("tr", null, h("td", null, h("b", null, "All variables, pooled")), R.map((_, r) => cell(all, r, "All variables")))];
    let group = null;
    for (const v of ALL_V) {
      if (V[v].group_id !== group) {
        group = V[v].group_id;
        body.push(h("tr", { class: "group-row" }, h("td", { colspan: nR + 1 }, V[v].group_name)));
      }
      body.push(h("tr", { class: "click", onclick: () => variableDrawer(v) },
        h("td", { class: "ellip" }, h("span", { class: "muted num" }, V[v].item_id + " "), V[v].name), R.map((_, r) => cell(perVariable[v], r, variableLabel(v)))));
    }
    const table = h("table", { class: "num" },
      h("thead", null, h("tr", null, h("th", null, "Variable"),
        R.map((run, r) => h("th", { class: "r", title: `${runTitle(r)}\n${run.notes || ""}` }, (run.configChanged ? "▾" : "") + (r + 1))))),
      h("tbody", null, body));
    return { table, lo, maxChange };
  }

  /* ---- Drawers ----------------------------------------------------------- */
  let drawerEl = null;
  function closeDrawer() {
    if (drawerEl) { drawerEl.forEach((el) => el.remove()); drawerEl = null; tip.hide(); }
  }
  function openDrawer(title, sub, body) {
    closeDrawer();
    const panel = h("aside", { class: "drawer", role: "dialog", "aria-label": title, tabindex: -1 },
      h("header", null, h("div", null, h("h2", null, title), h("div", { class: "sub" }, sub)),
        h("button", { class: "btn", type: "button", onclick: closeDrawer }, "Close")), body);
    drawerEl = [h("div", { class: "backdrop", onclick: closeDrawer }), panel];
    document.body.append(...drawerEl);
    panel.focus();
    flush();
  }
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });

  function verdictCell(p, v, r, extra) {
    const verdict = VERDICT[R[r].verdict[p * nV + v]];
    const raw = R[r].pred[p] ? R[r].pred[p][v] : null;
    const status = R[r].status[p] ? RAW.status_codes[R[r].status[p][v]] : null;
    const correct = verdict === "match" || verdict === "tn" || verdict === "none";
    return h("td", { class: `c mono v-${verdict} ${extra || ""}`, title: `${runTitle(r)}\n${VERDICT_LABEL[verdict]}${raw != null ? ": " + raw : ""}${status ? " (" + status + ")" : ""}` },
      h("span", { class: "glyph" }, GLYPH[verdict]), correct ? null : raw == null ? null : " " + raw);
  }
  const glyphLegend = () => h("div", { class: "legend" }, ["match", "tn", "wrong", "missed", "spurious", "none"].map((key) =>
    h("span", { class: "v-" + key }, h("span", { class: "glyph" }, GLYPH[key]), VERDICT_LABEL[key])));

  /* `at` is the run whose errors are itemised; the latest unless a page says otherwise. */
  function variableDrawer(v, at) {
    at = at == null ? LAST : at;
    const ps = patients(), groups = strata(ps), metric = METRICS[state.metric], windowed = windowRuns();
    const shownRuns = windowed.includes(at) ? windowed : windowed.concat(at).sort((a, b) => a - b);
    const series = groups.map((g) => {
      const cs = R.map((_, r) => counts(r, [v], g.patients));
      const one = { label: `${g.label} (n=${g.patients.length})`, color: g.color, values: cs.map((c) => score(c)) };
      if (groups.length === 1 && metric.ci) {
        const bands = cs.map((c) => wilson(...metric.parts(c)));
        one.lo = bands.map((b) => b && b[0]); one.hi = bands.map((b) => b && b[1]);
      }
      return one;
    });
    const perRun = R.map((_, r) => counts(r, [v], ps));
    const trend = h("div", { class: "card" },
      h("h2", null, `${metric.label} by run`),
      h("div", { class: "sub" }, groups.length > 1 ? "One line per stratum. Small strata move a lot between runs; check n before reading a gap."
        : metric.ci ? "Shaded band is the 95% Wilson interval. Vertical rules mark a configuration change." : "Vertical rules mark a configuration change."),
      groups.length > 1 && legend(groups, "line"),
      host((width) => lineChart(width, 220, {
        series, domain: domainOf(series.flatMap((x) => x.values.concat(x.lo || []))), label: `${metric.label} by run`,
        extra: (i) => [{ value: perRun[i].n, label: "cells scored" }, { value: R[i].notes || "", label: "" }],
      }), 220),
      h("h2", { style: "margin-top:14px" }, "Errors by run"), h("div", { class: "sub" }, "Count of patients, by kind of error."), errorLegend(),
      host((width) => columns(width, 150, perRun.map((c) => ({ parts: ERROR_PARTS(c) }))), 150));

    // Expected -> extracted pairs in one run: the fastest route to a prompt fix.
    const pairs = new Map();
    for (const p of ps) {
      const verdict = R[at].verdict[p * nV + v];
      if (verdict < WRONG || verdict > SPURIOUS) continue;
      const key = JSON.stringify([RAW.truth[p][v], R[at].pred[p][v], VERDICT[verdict]]);
      pairs.set(key, (pairs.get(key) || 0) + 1);
    }
    const ranked = [...pairs].sort((a, b) => b[1] - a[1]);
    const most = ranked.length ? ranked[0][1] : 1;
    const blank = (value) => (value == null ? h("span", { class: "muted" }, "blank") : value);
    const confusions = h("div", { class: "card" },
      h("h2", null, `Most common errors, run ${at + 1}`),
      h("div", { class: "sub" }, `${ranked.reduce((a, pair) => a + pair[1], 0)} errors in ${perRun[at].n} scored patients.`),
      ranked.length ? h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Expected"), h("th", null, "Extracted"), h("th", null, "Kind"), h("th", { class: "r" }, "Patients"))),
        h("tbody", null, ranked.slice(0, 10).map(([key, count]) => {
          const [expected, got, kind] = JSON.parse(key);
          return h("tr", { class: "v-" + kind }, h("td", { class: "mono" }, blank(expected)), h("td", { class: "mono" }, blank(got)),
            h("td", null, h("span", { class: "glyph" }, GLYPH[kind]), " ", VERDICT_LABEL[kind]),
            h("td", { class: "r" }, h("div", { class: "cell-bar" }, count, h("div", { class: "track" }, h("i", { style: `width:${(100 * count) / most}%` })))));
        }))) : h("div", { class: "muted" }, `No errors in run ${at + 1}.`),
      ranked.length > 10 && h("div", { class: "note", style: "margin-top:6px" }, `${ranked.length - 10} rarer pairs not shown.`));

    const order = { wrong: 0, varies: 1, right: 2, na: 3 };
    const rows = ps.map((p) => ({ p, cell: cellStability(p, v, windowed), verdict: R[at].verdict[p * nV + v] }))
      .filter((row) => (row.verdict >= WRONG && row.verdict <= SPURIOUS) || (row.cell && (row.cell.cls === "wrong" || row.cell.cls === "varies")))
      .sort((a, b) => (order[a.cell ? a.cell.cls : "na"] - order[b.cell ? b.cell.cls : "na"]) || a.p - b.p);
    const label = { wrong: "always wrong", varies: "varies", right: "stable", na: "–" };
    const failing = h("div", { class: "card" },
      h("h2", null, "Patients to look at"),
      h("div", { class: "sub" }, `Wrong in run ${at + 1}, or unstable over ${windowLabel()}. Click a patient for every variable.`),
      h("div", { class: "scroll-x" }, h("table", { class: "values" },
        h("thead", null, h("tr", null, h("th", null, "Patient"), h("th", null, "Tissue"), h("th", { class: "r" }, "Year"), h("th", null, "Expected"),
          shownRuns.map((r) => h("th", { class: "r" + (r === at ? " at" : ""), title: runTitle(r) }, r + 1)), h("th", null, "Over window"))),
        h("tbody", null, rows.slice(0, 60).map((row) => h("tr", { class: "click", onclick: () => patientDrawer(row.p, v, at) },
          h("td", null, P[row.p].patient_id), h("td", null, P[row.p].tissue), h("td", { class: "r" }, P[row.p].year),
          h("td", { class: "mono" }, blank(RAW.truth[row.p][v])), shownRuns.map((r) => verdictCell(row.p, v, r)),
          h("td", null, label[row.cell ? row.cell.cls : "na"])))))),
      rows.length > 60 && h("div", { class: "note", style: "margin-top:6px" }, `Showing 60 of ${rows.length}.`),
      h("div", { style: "margin-top:8px" }, glyphLegend()));

    openDrawer(variableLabel(v), `${V[v].group_name} · ${ps.length} patients in the current filter`, [trend, h("div", { class: "two" }, confusions, failing)]);
  }

  function patientDrawer(p, focus, at) {
    const patient = P[p], windowed = windowRuns(), sum = stability([p], ALL_V, windowed);
    let onlyInteresting = true;
    const tokens = R.map((run) => { const u = run.usage[p]; return u && u[0] != null && u[1] != null ? u[0] + u[1] : null; });
    const top = Math.max(1, ...tokens.filter((x) => x != null));
    const summary = h("div", { class: "two" },
      h("div", { class: "card" }, h("h2", null, "Run-to-run"), h("div", { class: "sub" }, `${sum.scored} scored variables over ${windowLabel()}.`),
        stack(STABILITY_PARTS(sum), null, "tall"),
        h("div", { style: "margin:8px 0" }, `${sum.right} stable and correct · ${sum.wrong} always wrong · ${sum.varies} vary`), stabilityLegend()),
      h("div", { class: "card" }, h("h2", null, "Tokens by run"), h("div", { class: "sub" }, "Input + output for this patient."),
        host((width) => lineChart(width, 120, { series: [{ label: "tokens", color: "var(--accent)", values: tokens }], domain: [0, niceMax(top)], fmt: compact }), 120)));

    const gridHost = h("div", { class: "scroll-x" });
    const drawGrid = () => {
      const body = [];
      let group = null;
      for (const v of ALL_V) {
        const cell = cellStability(p, v, windowed);
        const everWrong = R.some((run) => { const verdict = run.verdict[p * nV + v]; return verdict >= WRONG && verdict <= SPURIOUS; });
        if (onlyInteresting && !everWrong && !(cell && cell.cls === "varies") && v !== focus) continue;
        if (V[v].group_id !== group) {
          group = V[v].group_id;
          body.push(h("tr", { class: "group-row" }, h("td", { colspan: nR + 2 }, V[v].group_name)));
        }
        body.push(h("tr", { class: v === focus ? "hl" : "" },
          h("td", { class: "ellip" }, h("span", { class: "muted num" }, V[v].item_id + " "), V[v].name),
          h("td", { class: "mono" }, RAW.truth[p][v] == null ? h("span", { class: "muted" }, "blank") : RAW.truth[p][v]),
          R.map((_, r) => verdictCell(p, v, r))));
      }
      gridHost.replaceChildren(h("table", { class: "values" },
        h("thead", null, h("tr", null, h("th", null, "Variable"), h("th", null, "Expected"),
          R.map((run, r) => h("th", { class: "r" + (windowed.includes(r) ? " win" : "") + (r === at ? " at" : ""), title: `${runTitle(r)}\n${run.notes || ""}` }, (run.configChanged ? "▾" : "") + (r + 1))))),
        h("tbody", null, body.length ? body : h("tr", null, h("td", { colspan: nR + 2, class: "muted" }, "Every variable was correct in every run.")))));
    };
    const toggle = h("span", null);
    const drawToggle = () => toggle.replaceChildren(seg([[true, "Errors and variation"], [false, "All variables"]], onlyInteresting,
      (value) => { onlyInteresting = value; drawToggle(); drawGrid(); }, "Rows"));
    drawToggle(); drawGrid();
    const grid = h("div", { class: "card" },
      h("div", { class: "toolbar", style: "margin:0 0 8px" }, h("h2", null, "Every run, every variable"), h("span", { class: "spacer" }), toggle),
      h("div", { class: "sub" }, "A tick is a correct value; anything else shows what was extracted. Underlined runs are the variability window; ▾ marks a configuration change."
        + (at == null ? "" : ` The shaded column head is run ${at + 1}.`)),
      gridHost, h("div", { style: "margin-top:8px" }, glyphLegend()));
    openDrawer(patient.patient_id, `${patient.tissue} · diagnosed ${patient.year} · ${patient.n_notes} notes`, [summary, grid]);
  }

  /* ---- Page -------------------------------------------------------------- */
  const PAGES = [["index.html", "Overview"], ["run.html", "Single run"], ["trends.html", "Trends"], ["matrix.html", "Matrix"], ["triage.html", "Triage"]];
  let mainEl = null, filtersEl = null;

  /* A page may name the filters that apply to it (`filters`) and put a control
   * of its own at the head of the row (`lead`). A page that names none has no
   * filter row and is drawn whatever the stored filters select. */
  const unfiltered = () => !!variant.filters && !variant.filters.length && !variant.lead;
  function drawFilters() {
    const show = (key) => !variant.filters || variant.filters.includes(key);
    filtersEl.style.display = unfiltered() ? "none" : "";
    filtersEl.replaceChildren();
    fill(filtersEl, null, [
      variant.lead && variant.lead(),
      show("view") && seg([["performance", "Performance"], ["variability", "Variability"]], state.view, (x) => set("view", x), "View"),
      show("metric") && field("Metric", seg(Object.entries(METRICS).map(([key, metric]) => [key, metric.label]), state.metric, (x) => set("metric", x), "Metric")),
      show("stratify") && field("Stratify", seg([["none", "None"], ["tissue", "Tissue"], ["year", "Year"]], state.stratify, (x) => set("stratify", x), "Stratify")),
      show("tissue") && field("Tissue", select([["all", "All"]].concat([...new Set(P.map((p) => p.tissue))].sort().map((t) => [t, t])), state.tissue, (x) => set("tissue", x), "Tissue filter")),
      show("year") && field("Year", select([["all", "All"]].concat(YEARS.map((year) => [String(year), String(year)])), state.year, (x) => set("year", x), "Year filter")),
      show("window") && field("Variability window", select([["config", "Latest configuration"], ["last3", "Last 3 runs"], ["last5", "Last 5 runs"], ["all", "All runs"]],
        state.window, (x) => set("window", x), "Variability window")),
    ]);
  }
  function render() {
    if (!variant) return;
    drawFilters();
    const ps = patients();
    mainEl.replaceChildren();
    fill(mainEl, null, [ps.length || unfiltered() ? variant.render(ps) : h("div", { class: "card muted" }, "No patients match the current filters.")]);
    flush();
  }
  function topBar() {
    const here = location.pathname.split("/").pop() || "index.html";
    return h("div", { class: "top" },
      h("h1", null, "CIPOC test-set evaluation"),
      h("nav", null, PAGES.map(([href, text]) => h("a", { href, "aria-current": href === here ? "page" : null }, text))),
      h("div", { class: "meta" }, `${nR} runs · ${nP} patients · ${nV} variables · bundle ${String(RAW.generated_at).slice(0, 10)}`),
      h("button", { class: "btn", type: "button", onclick: () => { state.theme = isDark() ? "light" : "dark"; applyTheme(); set("theme", state.theme); } }, "Light / dark"));
  }
  function mount(spec) {
    variant = spec;
    filtersEl = h("div", { class: "filters" });
    mainEl = h("main");
    document.body.append(topBar(), filtersEl, mainEl, tipEl);
    let timer = null;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(render, 150); });
    render();
  }

  window.Eval = {
    V, P, R, RAW, ALL_V, LAST, nV, nP, nR, METRICS, VERDICT, GLYPH, VERDICT_LABEL,
    state, set, mount, render, patients, strata, windowRuns, windowLabel,
    counts, score, wilson, norm, scored, cellStability, stability, usage,
    h, s, pct, compact, delta, deltaPP, runTitle, variableLabel, tip, seg, select, field, legend,
    host, flush, domainOf, ticksFor, niceMax, topBar, tipEl, lineChart, sparkline, columns, stack,
    ERROR_PARTS, STABILITY_PARTS, errorLegend, stabilityLegend, seqColor, divColor,
    tile, variableName, bars, sortable, kpiRow, usageCard, runMatrix, variableDrawer, patientDrawer, applyTheme, isDark,
  };
})();
