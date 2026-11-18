# CIPOC Workbench

A browser app for reviewing one CIPOC extraction run: the coded values and the
evidence behind them, the clinical notes, every model call, a comparison against
ground truth, and reviewer feedback. It is a separate package and does not import
the CIPOC runtime.

> **PHI:** run files contain the full clinical notes, and feedback can quote them.
> The server has no authentication, so keep it on `127.0.0.1` or another trusted
> interface.

![Variables view](docs/images/variables.png)

## Install

From the repository root:

```bash
python -m pip install ./packages/cipoc-workbench
```

To run from a checkout without installing, replace `cipoc-workbench` in the
commands below with `PYTHONPATH=packages/cipoc-workbench/src python -m cipoc_workbench`.

For an airgapped machine, build wheels on a connected one and install from them:

```bash
python -m pip wheel --wheel-dir wheelhouse ./packages/cipoc-workbench
python -m pip install --no-index --find-links=wheelhouse cipoc-workbench
```

## Run

```bash
cipoc-workbench serve --feedback-dir reviews
```

Open <http://127.0.0.1:8000> and choose **Load Run...** to open a run file.

To open a run at startup, with ground truth:

```bash
cipoc-workbench serve \
    --state path/to/run.json \
    --ground-truth path/to/ground_truth.json \
    --feedback-dir reviews
```

| Option | |
|---|---|
| `--state FILE` | Run to open at startup. Without it, the page starts empty. |
| `--ground-truth FILE` | Reference values for the startup run. Requires `--state`. |
| `--feedback-dir DIR` | Folder for saved feedback. Without it, feedback is read-only. |
| `--feedback FILE` | Older single-file feedback for the startup run. Use `--feedback-dir` instead. |
| `--host`, `--port` | Defaults to `127.0.0.1` and `8000`. |

To try it with the bundled example, from the repository root:

```bash
cipoc-workbench serve \
    --state packages/cipoc-workbench/src/cipoc_workbench/example/case_state.json \
    --ground-truth packages/cipoc-workbench/src/cipoc_workbench/example/ground_truth.json \
    --feedback-dir reviews
```

## Files

### Run file

The JSON artifact from a completed CIPOC run (`OrchestratorRunResult`,
`schema_version` 1.0, 1.1 or 1.2). Failed-run artifacts can't be opened. To make
one from the repository root:

```bash
PYTHONPATH=src python -m scripts.run_case_state \
    --notes tests/fixtures/note_bundle.json \
    --output tests/test_outputs/case_state.json
```

A file opened with **Load Run...** is read in the browser and never sent to the
server.

### Ground truth (optional)

A JSON object of NAACCR item ID to expected value:

```json
{ "400": "C504", "522": "8500", "690": "" }
```

- `""` means the variable should stay blank.
- An item that isn't in the file is untested and left out of the scores.
- Values that differ only in case, punctuation or leading zeros count as
  correct, reformatted.

Pass it with `--ground-truth`, or load it with **Ground truth...** in the
Variables view.

### Feedback

With `--feedback-dir`, each run's feedback is saved to `DIR/<run_id>.json` on
the first save. Reopening a run restores its feedback by run ID, whatever the
file is called.

### Pricing (optional)

Cost estimates need a JSON price list loaded with **Load Pricing...** in the
Observability view. The format is in [docs/reference.md](docs/reference.md#optional-pricing).

## Usage guide

The header shows the run at a glance: note and variable counts, run ID and
duration, recorded tokens, whether feedback can be saved, and, with ground
truth, how many tested variables are correct. Below it, **Case facts** are the
site, histology and other facts that scoped the run. The moon icon switches
between light and dark themes.

### Variables

Variables are grouped as the orchestrator ran them: initial groups first, then
the dependent groups they released. The **Confidence** lens colors each variable
by the model's confidence. A note under a group explains why it was skipped,
such as a corpus gate that wasn't met.

With ground truth loaded, the **Accuracy** lens shows whether each variable is
correct (✓), correct but reformatted (≈), wrong (✕), missed (○), spurious (●) or
untested (·). Click a variable to see its value, the ground-truth comparison, the
model's explanation, and the evidence quotes highlighted in the cited note.
**Esc** closes the panel.

![Accuracy lens with a variable open](docs/images/variable-detail.png)

### Table

**Tabular** lists every variable with its status, value, confidence, attempts,
evidence count, expected value and verdict. Click a column header to sort.
Uncheck **Grouped** for one flat list.

![Tabular view](docs/images/table.png)

### Groups

Click a group card, outside its variables, for the group's settings, each gate
and whether it passed, its ground-truth score, and which notes were selected for
it.

![Group detail](docs/images/group-detail.png)

### Model calls

Further down a variable's panel are its attempts and the model calls behind
them: model, tokens, duration, and the full prompt and response. Prompts and
responses appear only if the run captured them.

![A model call](docs/images/model-call.png)

### Notes

**Notes** lists the clinical notes with their summaries. Click one for its full
text, with the evidence that variables cited and the scanner's findings
highlighted, followed by its concepts, cancer mentions, the groups that selected
it, and the variables that cite it.

![Notes view](docs/images/notes.png)

### Observability

**Observability** covers the whole run: tokens, model calls, whether telemetry
was complete, a cost estimate if a price list is loaded, breakdowns by agent,
node and model, and timing.

![Observability view](docs/images/observability.png)

### Feedback

Every variable, group and note panel ends with a feedback form. Tick what went
wrong, give the expected value, add a comment, and click **Save**. Each item is
saved on its own; an **unsaved** tag marks a draft you haven't saved yet.
Variables, groups and notes with saved feedback are marked in the main views.

![Feedback form](docs/images/feedback.png)

### Opening another run

**Load Run...** switches to another run without restarting the server. You'll be
asked before unsaved feedback is discarded. Ground truth passed with
`--ground-truth` applies only to the startup run, so load the new run's ground
truth with **Ground truth...**. Refreshing the page goes back to the startup run,
or to an empty page if there isn't one.

## Development

From this package directory:

```bash
node --test tests/*.test.js
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

[docs/reference.md](docs/reference.md) covers the feedback API, how runs and
feedback are matched, the artifact fields the Workbench reads, telemetry and
timing definitions, and pricing rules.
