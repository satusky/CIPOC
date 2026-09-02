# CIPOC Workbench

The CIPOC Workbench is a standalone browser interface for reviewing a CIPOC
case-state JSON file, comparing extracted values with optional ground truth,
and saving reviewer feedback.

Install it independently of the CIPOC runtime:

```bash
python -m pip install ./packages/cipoc-workbench
```

Serve a case:

```bash
cipoc-workbench serve \
    --state path/to/case_state.json \
    --ground-truth path/to/ground_truth.json \
    --feedback path/to/feedback.json
```

Every path is optional. Run `cipoc-workbench serve` without arguments to view
the bundled example. By default, the server listens on `127.0.0.1:8000`.
The bundled input files are under `src/cipoc_workbench/example/`.

For an offline installation, build a wheelhouse on a compatible connected
machine:

```bash
python -m pip wheel --wheel-dir wheelhouse ./packages/cipoc-workbench
python -m pip install --no-index --find-links=wheelhouse cipoc-workbench
```
