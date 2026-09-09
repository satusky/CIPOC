# CIPOC Workbench

The CIPOC Workbench is a standalone browser interface for reviewing a canonical
CIPOC `OrchestratorRunResult` JSON artifact, comparing extracted values with
optional ground truth, and saving reviewer feedback. It accepts
`schema_version: "1.0"` and `"1.1"` and does not import the CIPOC runtime.
Historical artifacts are displayed as recorded, without rewriting or revalidating
their clinical results. The bundled example remains a `1.0` compatibility fixture.

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

The **telemetry** button in the case-facts rail opens collection status, typed
collection issues, provider usage, and unattributed invocations in the existing
detail pane. `complete`, `partial`, and `unavailable` describe collection, not the
clinical outcome. Missing legacy `collection_status` is shown as unknown, not
assumed complete. A null/missing `llm_usage_summary` is unavailable, never zero;
reported zero counts remain zero. Totals and dimension buckets come from the
artifact and are not recomputed by the Workbench.

Unattributed invocations are diagnostic records only. Their invocation ID,
namespace, agent, node, model, transport retry ordinal, usage, errors, and any
captured content remain inspectable without assigning them a semantic attempt or
inserting them into a variable/group call list. Raw invocation/exchange details
also preserve additional backend task-identity metadata. Prompt capture and
telemetry availability are independent.

Note IDs and primary citations retain their integer/string scalar types. Lookup
and navigation compare `String(id)` only: `1` and `"1"` identify the same note,
but `"001"` and `"1"` remain distinct. Primary citations appear in reverse note
links even when there are no evidence spans; a null citation never resolves to
a note named `"null"`.

Treatment-gate explanations include `hormonal_therapy` and `immunotherapy`
independently of chemotherapy. Missing concept keys in historical notes or
corpus descriptors mean **unknown (not recorded)**, not an absent finding.

### Backend Interface

The reader expects `observability.collection_issues` as a list of
`{code, message}` objects and `unattributed_exchanges` as a list with
`invocation_id`, `namespace` (array of strings), and the existing exchange fields
(`agent`, `node`, `model`, `usage`, `error`, `prompt_messages`, `response`,
`retry_ordinal`). These optional diagnostic collections default to empty when
omitted; no semantic attempts or usage totals are synthesized.

Applicability presentation supports recursive `any_of` and `all_of` **arrays of
restriction objects**, rendering parentheses and explicit OR/AND connectors.
For item 832 the expected structure is breast OR (cutaneous site AND melanoma
histology). The current backend uses `primary_sites` with codes and inclusive
ranges (for example, `C440-C449`) for the cutaneous alternative. Leaf field names
and values are displayed generically, including `gross_primary_sites`,
`primary_sites`, and `histology_families`; there is no dependency on the exact
name of the cutaneous-site leaf field or a Workbench-maintained site list. The
Workbench does not evaluate these conditions or derive cutaneous sites from
histology. Legacy flat restrictions remain readable. If the backend uses a
different compound-expression shape, the formatter and its tests need updating.

Run the offline tests from this package directory:

```bash
node --test tests/workbench.test.js
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

The Python tests exercise artifact serving and feedback round-trips using ASGI
directly, without the CIPOC runtime, live endpoints, or additional test dependencies.

For an offline installation, build a wheelhouse on a compatible connected
machine:

```bash
python -m pip wheel --wheel-dir wheelhouse ./packages/cipoc-workbench
python -m pip install --no-index --find-links=wheelhouse cipoc-workbench
```
