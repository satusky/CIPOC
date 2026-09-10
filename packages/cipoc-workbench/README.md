# CIPOC Workbench

The CIPOC Workbench is a standalone browser interface for reviewing a canonical
CIPOC `OrchestratorRunResult` JSON artifact, comparing extracted values with
optional ground truth, and saving reviewer feedback. It accepts
`schema_version: "1.0"`, `"1.1"`, and `"1.2"` and does not import the CIPOC runtime.
New runtime result and failure artifacts emit `1.2`; runtime readers accept all three.
Historical artifacts are displayed as recorded, without rewriting or revalidating
their clinical results. The bundled example remains a `1.0` compatibility fixture.

Install it independently of the CIPOC runtime:

```bash
python -m pip install ./packages/cipoc-workbench
```

Start with no artifact, then choose **Load Run...** in the browser:

```bash
cipoc-workbench serve --feedback-dir path/to/reviews
```

Or auto-load an explicit startup artifact, optionally with ground truth:

```bash
cipoc-workbench serve \
    --state path/to/run_result.json \
    --ground-truth path/to/ground_truth.json \
    --feedback-dir path/to/reviews
```

Run `cipoc-workbench serve` without arguments to start empty with read-only
annotations. `--state` is optional; `--ground-truth` and legacy `--feedback FILE`
require an explicit `--state`. `--feedback-dir` works with or without it.
By default, the server listens on `127.0.0.1:8000`.

The bundled `1.0` compatibility fixture is unchanged and never loaded implicitly.
Choose it with **Load Run...**, or explicitly serve it from the repository root:

```bash
cipoc-workbench serve --state packages/cipoc-workbench/src/cipoc_workbench/example/case_state.json
```

### Load Runs

Start once with `cipoc-workbench serve --feedback-dir path/to/reviews`, then use
**Load Run...** to switch completed run JSON files without restarting. Selected
artifacts stay only in the tab's memory: their contents are never uploaded or
stored in localStorage, IndexedDB, or server files. Refresh returns to empty
without `--state`, or reloads the explicitly configured startup artifact.
Reopening a run restores its server-saved feedback by `run.run_id`, not filename.
A new execution needs a new canonical run UUID.

Invalid files, failed-run artifacts, and cancellation leave the current run and
drafts intact. Unsaved feedback prompts **Cancel** or **Discard and Load**;
switching is blocked while a save is pending, and a failed save retains its draft.
Successful activation clears ground truth, selection, filters, and old feedback,
and resets the lens to confidence. Theme, view/layout preferences, and pricing
remain. Server-configured ground truth applies only at startup; load the new
run's reference explicitly. Late file/reference/feedback responses cannot replace
another activation's state. Loading remains usable if startup or feedback fails.

Automatic ground truth uses `GET /api/runs/{run_id}/ground-truth`, returning
`{run_id, values}` only for the UUID captured at server startup. Replacing the
startup artifact does not rebind that reference. Without a startup artifact,
ground truth cannot bind to a run, including the first locally loaded run.
The frontend checks response identity and never falls back to unbound
`/api/ground-truth` or `ground_truth.json`.
Use the local ground-truth picker when no bound reference is available.

### Feedback

`--feedback-dir DIRECTORY` stores one document at
`<DIRECTORY>/<canonical-run-uuid>.json`, creating it on first save. It remains
fully writable through the scoped APIs without a startup artifact. No feedback
destination means read-only annotations. The mutually exclusive legacy
`--feedback FILE` option requires `--state` and binds only to the startup run;
other locally opened runs remain viewable but cannot save into that file.

- `GET /api/runs/{run_id}/feedback` returns `run_id`, `annotations`, explicit
  `writable`, and `read_only_reason`. HTTP 200 alone does not enable editing.
- `PUT /api/runs/{run_id}/feedback/{kind}?entity_id=...` accepts an annotation
  (`flags`, `expected`, `note`) and returns `run_id`, `kind`, `id`, and `annotation`.
  This is the preferred browser route: URL-encode the query value to preserve
  dots and slashes in IDs. The older `.../{kind}/{entity_id}` route remains supported.
  Kinds are `variable`, `group`, and `note`; entity IDs remain strings.

Requests send only run identity and feedback, not the artifact. Each tab selects
its own run; there is no server-global active-run pointer. Documents verify their
`run_id`; corrupt or conflicting files are errors, not empty reviews to overwrite.
Legacy documents without `run_id` are accepted only with explicit startup
`--feedback` binding. Partial legacy annotations are normalized on copies; GET
never migrates files. An explicit save may add the bound identity while preserving
unrelated annotations and custom metadata, including metadata on the edited record.
An expected-only annotation is saved; clearing flags, note, and expected value
deletes it. Existing `/api/feedback` endpoints remain startup-bound, never
redirected by browser selection. Without a startup artifact, legacy feedback GET
returns an unknown (`null`) run ID and read-only empty annotations; legacy PUT
and `/api/ground-truth` return HTTP 409. Scoped ground truth also returns 409 for
any canonical run UUID when no startup identity was captured.

Deploy on a trusted interface in **one server process**. Feedback may contain PHI,
and there is no authentication or collaborative conflict resolution. A process-wide
lock covers the full read-modify-write operation with atomic file replacement,
preventing unrelated concurrent annotations from being lost. Same-annotation
edits are last-save-wins; multiple writer processes are not coordinated.

### Run Artifact

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

### Observability

The **Observability** tab (also opened by the **telemetry** button) shows run
identity/source, recorded tokens and invocation health, collection issues, capture
settings, validation attempts, agent/node/model breakdowns, timing, and optional
cost estimates. `web/run.js` renders this view and reuses the entity exchange cards.
Note, group, and variable dossiers retain their own exchanges.

`complete`, `partial`, and `unavailable` describe collection, not the clinical
outcome or provider-usage completeness. Missing legacy `collection_status` is
unknown. A null/missing usage summary is unavailable, never a zero-call run;
missing scalars in an empty summary are not zero. Recorded totals and dimension
buckets are not rebuilt from exchanges. `logical_calls` counts starts in each
bucket: a resolved-model bucket can contain retries without starts. Recorded
validation attempts are not invocation counts or found-variable counts; a valid
verdict can be not-found.

Unattributed invocations are diagnostic records only. Their invocation ID,
namespace, agent, node, model, transport retry ordinal, usage, errors, and any
captured content remain inspectable without assigning them a semantic attempt or
inserting them into a variable/group call list. Raw invocation/exchange details
also preserve additional backend task-identity metadata. Prompt capture and
telemetry availability are independent.

#### Timing and Provenance

Schema 1.2 adds optional fields shared by attributed and unattributed invocations.
Render them by field presence, not version; legacy artifacts are not upgraded.

- `started_at` / `finished_at` are UTC callback boundaries after local permit
  acquisition. `service_seconds` is monotonic callback duration including network
  and SDK-internal retries/backoff, not pure compute or total permit occupancy.
  Start precedes prompt serialization; finish is end/error callback entry before
  result parsing. Structured calls retain the permit through post-callback parsing.
- `queue_seconds` is only the local synchronous permit wait, excluding graph
  scheduling and LangGraph retry backoff. Instrumented unbounded calls can report
  zero; uninstrumented async/direct paths remain null.
- `usage_reported_fields` names retained scalars (`input_tokens`, `output_tokens`,
  `total_tokens`) verified against provider JSON before SDK coercion or adapter
  defaulting. Callback `token_usage` alone is insufficient: the SDK can coerce
  booleans, strings, and floats; adapter `usage_metadata` can also default missing
  counts or derive totals. Verification does not change normalized counts. Null
  means unverified/unrecorded; `[]` means none verified; an explicit list missing
  a required count is authoritative.

The runtime's `src/cipoc/llm/openai.py` hooks its wrapper-owned default synchronous
OpenAI HTTP client before SDK parsing. `llm/usage_evidence.py` carries only immutable
scalar evidence in a per-call `ContextVar` scope, bound to the invocation/request.
Non-streaming Chat Completions and Responses can certify real zero counts without
new configuration or dependencies. External clients, LangChain's proxy path,
async/direct calls, and streaming retain null provenance; callback metadata alone
cannot enable verification.

Timing sums, means, maxima, and coverage include diagnostic records once, as do
the runtime CLI's timing summaries. **Summed invocation time** and **summed queue
wait** are not wall time and can exceed run duration under concurrency. The
interval union estimates **time with an observed model call active**, excluding
missing, reversed, or inconsistent wall-clock intervals without discarding valid
monotonic durations. Envelope duration can include initialization, cleanup, and
the interactive summary pause. These metrics do not measure actual endpoint
headroom, saturation, or retry causation.

#### Optional Pricing

The Workbench tries same-origin `web/endpoint_catalog.json` at startup; the bundled
catalog has no model prices. Use **Load Pricing...** in Observability for a local
JSON catalog without editing the installed package or uploading the file:

```json
{
  "currency": "USD",
  "source": "replace with verified contract rates",
  "version": "example",
  "models": {
    "exact-recorded-model-name": {
      "input_per_million": null,
      "output_per_million": null,
      "cached_input_per_million": null
    }
  }
}
```

Rates must be finite and nonnegative; configured zero is valid, null is unknown.
Names must exactly match recorded models: no deployment aliases, external pricing
lookups, or guessed defaults. Catalog failure leaves review/feedback usable.
Pricing is view-time only and never changes runtime config or artifacts.
`token_limits.yaml` is reference material, not live quota configuration or a source
of model aliases.

Missing rates/usage, invalid or inconsistent counts, unsupported billing details,
and unverified zero counts withhold estimates rather than imply zero charge.
Legacy records with positive input/output counts can be estimated, but an explicit
`usage_reported_fields` list must verify both counts, including genuine zeros.
Cached input uses `input_token_details.cache_read`, subtracts from ordinary input,
and requires a cached rate when positive. Reasoning is already in output totals.
Absent cache details use ordinary-input pricing, not proof of complete billing.

**Known estimated cost** covers only priceable retained invocations, including
retries, failures, and diagnostics with sufficient usage. Coverage and unavailable
reasons remain visible; no priceable calls means unavailable, not a zero subtotal.
Collection, provider-usage, cost, and timing coverage are separate. Invalid
invocation identity/sequence withholds affected derived sums while preserving
individual records for inspection. Estimates are never a provider invoice.

### Clinical Presentation

Note IDs and primary citations retain their integer/string scalar types. Lookup
and navigation compare `String(id)` only: `1` and `"1"` identify the same note,
but `"001"` and `"1"` remain distinct. Primary citations appear in reverse note
links even when there are no evidence spans; a null citation never resolves to
a note named `"null"`.

Treatment-gate explanations include `hormonal_therapy` and `immunotherapy`
independently of chemotherapy. Missing concept keys in historical notes or
corpus descriptors mean **unknown (not recorded)**, not an absent finding.

### Backend Interface

`build_app(state_path=None)` does not read or fall back to the bundled example.
`GET /case_state.json` returns HTTP 204 with no body when no startup artifact is
configured. An explicitly configured missing file returns 404; an existing file
is served byte-for-byte without rewriting. Explicit malformed startup files are
still served so the browser can report the problem and offer **Load Run...** for
recovery. Startup identity is captured once, never rebound by local selection or
later file replacement. There is no global active run or first-local-run binding.
Both `build_app` and the CLI reject ground truth or legacy single-file feedback
without an explicit startup path (`ValueError` and a friendly nonzero exit,
respectively).

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
node --test tests/*.test.js
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
