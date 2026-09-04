# CIPOC Workbench

The CIPOC Workbench is a standalone browser interface for reviewing a canonical
CIPOC `OrchestratorRunResult` JSON artifact, comparing extracted values with
optional ground truth, and saving reviewer feedback. It accepts
`schema_version: "1.0"` and does not import the CIPOC runtime.

Install it independently of the CIPOC runtime:

```bash
python -m pip install ./packages/cipoc-workbench
```

Serve a case:

```bash
cipoc-workbench serve \
    --state path/to/run_result.json \
    --ground-truth path/to/ground_truth.json \
    --feedback path/to/feedback.json
```

Every path is optional. Run `cipoc-workbench serve` without arguments to view
the bundled example. By default, the server listens on `127.0.0.1:8000`.
The bundled input files are under `src/cipoc_workbench/example/`.

Generate an artifact from the repository root with:

```bash
PYTHONPATH=src python -m scripts.run_case_state \
    --notes tests/fixtures/note_bundle.json \
    --output tests/test_outputs/case_state.json
```

The canonical schema has top-level `run`, `case`, `inputs`, `corpus`, and
`observability` domains. The Workbench reads clinical values and note-selection
provenance from `case`, targets from `inputs`, processed notes and descriptors
from `corpus`, model exchanges and attempts from `observability`, and identity
and timing from `run`. `case` is the durable clinical output; the other domains
describe this particular execution. The artifact does not contain a raw graph
event timeline. The Workbench expects a completed `OrchestratorRunResult`, not
the diagnostic `OrchestratorRunFailure` carried by an `OrchestratorRunError`.

Prompt and parsed-response capture is enabled by default. Use
`--no-llm-content-capture` to omit those bodies while retaining exchange
metadata, errors, retries, variable attempts, and provider-reported token usage.
Use `--max-content-chars N` to truncate each retained prompt message to an
explicit bound; responses are not truncated, and the artifact records whether
any prompt was cut.

> **PHI:** Disabling LLM content capture does not de-identify an artifact.
> `corpus.note_corpus` contains the full processed clinical notes, and retained
> prompts and responses may also contain PHI. Keep the server bound to a trusted
> interface and handle result files accordingly.

Usage totals are limited to values reported through provider callbacks. Failed
calls and provider-SDK-internal retries may have missing usage; inspect the
reported/missing invocation counts. Token details are breakdowns of input and
output totals, not values to add to those totals.

Run the frontend tests with:

```bash
node --test tests/workbench.test.js
```

For an offline installation, build a wheelhouse on a compatible connected
machine:

```bash
python -m pip wheel --wheel-dir wheelhouse ./packages/cipoc-workbench
python -m pip install --no-index --find-links=wheelhouse cipoc-workbench
```
