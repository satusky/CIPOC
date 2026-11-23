# CIPOC evaluation dashboard

Tracks orchestrator performance on the test set across sweeps: per-variable metrics
over time, stratified by tissue or diagnosis year, token usage per patient, and
run-to-run variability.

It reads run artifacts and ground-truth files straight from their folders. Nothing
here imports `cipoc`, and the Python side is stdlib only.

This is a prototype with three candidate layouts over one shared core. Pick one and
delete the others.

## Try it

```bash
python make_dummy_data.py     # writes data/ground_truth/, data/sweeps/ and web/data.js
python dashboard.py serve     # http://127.0.0.1:8001
```

`data/` and `web/data.js` are gitignored. With real runs they hold patient IDs and
extracted values, so treat them as PHI.

## Run it on your data

Set your folders once in `dashboard.json`, next to `dashboard.py` (gitignored).
Relative paths are relative to that file.

```json
{
  "sweeps": "/path/to/sweeps",
  "truth": "/path/to/ground_truth",
  "names": "/path/to/CIPOC/config/variable_groups.json"
}
```

Then run `python dashboard.py serve`. `--sweeps`, `--truth` and `--names` override
the file; without either, the defaults are `data/sweeps`, `data/ground_truth` and
this repository's `config/variable_groups.json`. The port defaults to 8001, so it
can run beside the Workbench.

Every page load rebuilds the data from the folders, so a new sweep shows up on the
next refresh. Each file is parsed once, and again only when it changes. A file that
won't parse, such as an artifact still being written, is skipped with a warning
until it changes.

```
sweeps/
  2026-11-13-date-prompt/     # one folder per sweep; the name is its run ID
    sweep.json                # optional: {"label": "...", "notes": "...", "started_at": "2026-11-13"}
    P0001.json                # one artifact per patient, named like its ground-truth file
    P0002.json
ground_truth/
  P0001.json                  # {item_id: value}, the files the Workbench loads
```

- **Artifacts** are `OrchestratorRunResult` JSON, or the failure artifact of a case
  that raised, which counts as failed. Sweeps are ordered by `started_at` in
  `sweep.json`, else by their earliest artifact. Model, CIPOC version and config
  digest come from the config fingerprint. Sweeps with the same digest are
  replicates: the default variability window is the trailing block that shares
  the latest one.
- **Older files** from before schema 1.0, which hold the orchestrator's final state
  at the top level, are read as they are. They record no date, so set `started_at`
  in the sweep's `sweep.json`. They also have no duration or config fingerprint:
  the model comes from their model calls, and each such sweep counts as its own
  configuration.
- **Ground truth** defines the test set: one patient per file, and every item the
  files mention is a variable. `null` or `""` means the variable should stay
  uncoded; an item missing from a file is untested for that patient. Keys that
  aren't item IDs, such as `mrn`, are ignored. Artifacts for patients without a
  ground-truth file are ignored with a warning.
- **Patient details:** the diagnosis year is item 390, else the artifact's. Tissue
  and note count come from the earliest artifact that has them. Tissue is the run's
  `gross_primary_site`, which a run leaves unset when it already knows the primary
  site, so such patients show as "unknown".

`run_case_state` writes artifacts in this layout, including the failure artifact
when a case raises:

```python
from pathlib import Path

from cipoc.models import OrchestratorRunError
from scripts.run_case_state import run_case_state

sweep = Path("sweeps/2026-11-13-date-prompt")
for patient_id, notes in notes_by_patient.items():
    try:
        run_case_state(notes, sweep / f"{patient_id}.json")
    except OrchestratorRunError:
        pass  # its failure artifact is already written
```

To open the dashboard without a server, for example on a machine that can't see
the folders, write the bundle once and copy it into `web/`:

```bash
python dashboard.py export web/data.js
open web/index.html
```

## Layout

The pages load the bundle as `data.js`, a plain script that assigns one global, so
they also work from `file://`. Metrics are computed in the browser from raw values,
so changing a filter or a metric definition needs no rebuild. Token counts are
`null` when the provider did not report them, never zero.

| Path | What it is |
|---|---|
| `dashboard.py` | Reads the folders, builds the bundle, serves the pages |
| `make_dummy_data.py` | Synthetic test set and 14 sweeps with planted patterns |
| `web/data.js` | Exported bundle, for `file://`. One global, `window.CIPOC_EVAL` |
| `web/core.js`, `web/core.css` | Scoring, filters, charts, drawers. Shared by every page |
| `web/index.html` | Run list (each run links to its own page) and metric definitions |
| `web/run.html` | One run on its own: header, pooled metrics, by tissue and year, per-variable and per-patient tables. `run.html#run=<run_id>` picks the run; without it, the newest |
| `web/trends.html` | One trend chart per variable |
| `web/matrix.html` | Variable × run heatmap; patient × variable grid for variability |
| `web/triage.html` | Variables ranked by errors; accuracy-against-consistency scatter for variability |

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
