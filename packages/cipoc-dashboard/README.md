# CIPOC evaluation dashboard

Tracks orchestrator performance on the test set across runs: per-variable metrics
over time, stratified by tissue or diagnosis year, token usage per patient, and
run-to-run variability.

The dashboard is static HTML, CSS and JavaScript with no dependencies and no
network access. Open `web/index.html` by double-clicking it. Nothing here imports
`cipoc`, and the Python side is stdlib only.

This is a prototype with three candidate layouts over one shared core. Pick one and
delete the others.

## Try it

```bash
python make_dummy_data.py     # writes data/ground_truth.csv, data/eval.sqlite, web/data.js
open web/index.html
```

`data/` and `web/data.js` are gitignored. With real runs they hold patient IDs and
extracted values, so treat them as PHI.

## Layout

| Path | What it is |
|---|---|
| `evalstore.py` | SQLite store, ingestion helpers, and the export to `web/data.js` |
| `make_dummy_data.py` | Synthetic test set and 14 sweeps with planted patterns |
| `web/data.js` | Exported bundle. One global, `window.CIPOC_EVAL` |
| `web/core.js`, `web/core.css` | Scoring, filters, charts, drawers. Shared by every page |
| `web/index.html` | Run list (each run links to its own page) and metric definitions |
| `web/run.html` | One run on its own: header, pooled metrics, by tissue and year, per-variable and per-patient tables. `run.html#run=<run_id>` picks the run; without it, the newest |
| `web/trends.html` | One trend chart per variable |
| `web/matrix.html` | Variable × run heatmap; patient × variable grid for variability |
| `web/triage.html` | Variables ranked by errors; accuracy-against-consistency scatter for variability |

## Storage

SQLite is the system of record. The browser cannot open it without a WASM build,
and `fetch()` is blocked on `file://`, so `evalstore.py export` flattens the store
into `web/data.js`, a plain script the pages load with a `<script>` tag. Metrics
are computed in the browser from raw values, so changing a filter or a metric
definition needs no re-export.

Tables: `variables`, `patients`, `ground_truth` (long), `runs`, `run_patients`
(status and usage per patient per run), `predictions` (long). Usage columns are
`NULL` when the provider did not report them, never zero.

The store lives wherever the sweeps run; the dashboard lives wherever the browser
is. Only `web/data.js` crosses between them: export on the cluster, copy that one
file into a local copy of `web/`, and reload the page. It is the only file in
`web/` that carries patient data.

SQLite needs real file locking. If the sweeps run on Databricks, write the database
on local disk (`/local_disk0` or `/tmp`) and copy it to the Volume afterwards; don't
open it directly on a `/Volumes` or `/dbfs` mount.

## Recording a sweep

A run here is one sweep over the test set. The orchestrator's own `run_id` is per
case and is kept as `case_run_id`.

```python
import json
import evalstore

conn = evalstore.connect("/local_disk0/eval.sqlite")
evalstore.load_ground_truth_dir(                  # once, or whenever the truth changes
    conn, "ground_truth/", evalstore.variable_names("config/variable_groups.json")
)

first = json.load(open(artifact_paths[patient_ids[0]]))
evalstore.record_run(
    conn, "sweep-2026-11-13", first["run"]["started_at"],
    label="date prompt v2", model="llm-large",
    cipoc_version=first["run"]["config_fingerprint"]["cipoc_version"],
    config_digest=evalstore.config_digest(first),
)
for patient_id in patient_ids:
    artifact = json.load(open(artifact_paths[patient_id]))    # OrchestratorRunResult JSON
    evalstore.record_case_artifact(conn, "sweep-2026-11-13", patient_id, artifact)
# A case that raised: evalstore.record_patient(conn, run_id, patient_id, {}, status="failed")
conn.commit()
evalstore.export_bundle(conn, "web/data.js")
```

`record_patient` takes plain values if you are not starting from an artifact. Sweeps
with the same `config_digest` are treated as replicates: the default variability
window is the trailing block of runs that share the latest digest.

Ground truth is either of these. `python evalstore.py load-truth DB PATH` takes
both, with `--names config/variable_groups.json` for variable names and groups.

- A directory of `<patient_id>.json` files, the same files the Workbench loads.
  Each is `{item_id: value}`. `null` or `""` means the variable should stay
  uncoded; an item missing from a file is untested for that patient. Keys that
  aren't item IDs, such as `mrn`, are ignored. Reloading replaces each patient's
  truth with its file. The diagnosis year comes from item 390, and tissue and
  note count from the patient's first recorded artifact. Tissue is the run's
  `gross_primary_site`, which a run leaves unset when it already knows the
  primary site, so such patients show as "unknown".
- A CSV with `patient_id`, `tissue`, `year`, `n_notes`, then one column per
  NAACCR item ID. A blank cell means the variable should stay uncoded.

## Scoring

Values are compared after trimming, upper-casing, and dropping punctuation and
leading zeros. Each patient × variable cell gets one outcome:

| Outcome | Truth | Extracted |
|---|---|---|
| correct | value | same value |
| wrong value | value | different value |
| missed | value | blank |
| spurious | blank | value |
| correctly blank | blank | blank |

- Precision = correct ÷ (correct + wrong + spurious)
- Recall = correct ÷ (correct + wrong + missed)
- F1 = harmonic mean of the two. The default, because it ignores correctly-blank cells
- Accuracy = (correct + correctly blank) ÷ all cells
- Headline figures pool counts over variables (micro-average)
- Where no patient in a selection has a truth value, precision, recall and F1 are
  undefined and shown as –
- Intervals are 95% Wilson

Over the variability window each cell is stable and correct, always wrong (the same
miss every run, so systematic), or varies (noise under one configuration). Cells
blank in the truth and in every run are left out.

Failed patients are excluded from that run's metrics and counted in the "Patients
scored" tile. Variables a run did not plan, and values given as structured data,
are left out of its metrics too, since neither was extracted. Both show as "not run".
Untested cells are left out of every run.
