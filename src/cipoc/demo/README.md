# Local CIPOC presenter

The demo presents the same scanner → planner → retriever → extractor workflow
as the historical `demo` branch: animated map, grouped presenter steps, evidence,
validation repairs, standalone entity cards, variable table, and OMOP staging
previews.
Live runs use CIPOC 1.1.0's public orchestrator and emit canonical schema 1.2
artifacts. The Workbench remains a separately installed completed-run reader.

## Install and serve

From the repository root:

```bash
uv sync --extra demo
uv run cipoc-demo serve --live --notes tests/fixtures/note_bundle.json
```

Alternatively, in an environment with the runtime and web dependencies installed:

```bash
PYTHONPATH=src python -m cipoc.demo serve --live --notes tests/fixtures/note_bundle.json
```

Open **http://127.0.0.1:8001/**. Preflight constructs the agents and checks note
identity, target metadata, environment references, and dictionaries without
calling a model. **Start Run** begins exactly one execution. Input files and
credentials stay server-configured. The initial screen shows model names,
per-model capacities, graph concurrency, and content/recording settings.

The source NAACCR dictionary under `documents/manuals/` is gitignored and must
be supplied separately on a fresh clone. Configure its path in `config.yaml`.
`--config FILE` selects configuration. Resource paths in it resolve against the
launch directory, or an explicit `--resource-root DIRECTORY`; paths are frozen
before execution, without changing CWD or environment per request.
`--structured-data FILE` accepts a JSON object of known item values.

The demo supplies an explicit 120-second request timeout and zero SDK retries
when configuration omits them. Existing configured values take precedence.
LangGraph's configured transient-node retry policy still applies. These are
per-request limits, not a whole-run deadline.

## Live presentation

- **Follow live / Jump to latest** tracks the advancing frontier.
- **Pause presentation**, previous/next, and step selection let the presenter
  inspect earlier work while extraction continues.
- **Play steps** uses one server-owned playback clock; opening a second browser
  does not accelerate it. Append `?viewer=1` for a viewer UI with shared cursor
  and disabled presenter controls. This is a UI convenience on the trusted local
  interface, not authentication.
- Elapsed time, active/completed model calls, transport retries, last-observation
  age, connection state, and distinct completion/failure states remain visible.
- Runtime observations include available service/local-wait timing and provider
  usage. Live observations are provisional and precede final entity attribution;
  canonical observability is authoritative. Validation checks/repairs, provider
  transport retries, and model invocations are different measures.
- Reconnects recover a consistent snapshot plus missing events. Refreshing or
  closing a browser tab does not rerun extraction. The local server must remain
  running. Graceful server shutdown waits for its owned worker; there is no
  cancellation or resumable-execution API. Force termination can leave a partial
  trace and no canonical artifact.

`--max-concurrency N` controls graph concurrency and must be at least 2.
`llm.max_concurrency` in configuration is the separate process-wide synchronous
capacity for a particular endpoint/model, and may be 1. The server runs one job
per process; separate server processes have separate model budgets.

## Replay entity cards

The presentation target is **Full HD (1920 × 1080)**. The orchestration map spans
the presentation board. Click the central **Case** bubble, in any phase, to open
the grouped variable table and the case facts recorded at the selected step.
A content-sized floating window, capped at **896px wide**, appears over the
clear right-hand portion of the map when an entity is selected. Workflow nodes
are fitted to the available space beside it, keeping their layout stable when
cards open or close. One case, note, group, or variable card is shown at a time,
with all its sections together rather than behind tabs or accordions. Live mode
uses the same frontend.

- **Note:** ID, date/type, scan status, summary, concept presence/confidence,
  cancer mentions and temporality, and evidence. Concepts occupy a uniform grid:
  ✓ indicates present, − absent, and ? unrecorded; present concepts have a teal fill.
- **Group:** summary and variable tiles with recorded status, value, and reasons.
- **Case:** recorded primary/gross site, histology, behavior, sex, and diagnosis
  date above a locally scrolling variable table. The table includes item IDs,
  values, confidence, statuses, review flags, and OMOP preview links. Facts and
  progress totals stay visible while the table scrolls.
- **Variable:** item/name, value or candidate value, the first sentence of the
  recorded definition where available, extraction explanation/confidence, evidence, and validation checks
  with candidate values, outcomes, and recorded failure reasons. These checks
  are distinct from provider retries. The value is emphasized in a full-width
  result band. Confidence badges use dark teal for max, green for high, amber for
  medium, and pink for low; missing or unrecognized levels remain neutral.

Note IDs and numeric NAACCR item IDs appear below their map bubbles. Select a
note/variable disc or its ID, a group container/gate on the map, or a group or
variable name in the Variables table. Group tiles open variable cards; evidence
note links open note cards. **Back** retraces this card navigation. **Close** or
**Escape** clears the card (Escape closes an open OMOP preview first); clicking
the map background also clears it. Card navigation is local to each browser and
leaves the playback cursor and map geometry unchanged.

Card results and the table reflect the **selected step-end snapshot**. The
within-step scrubber and replay button animate only the map. Across step changes,
the selected entity stays selected and is re-resolved against the new snapshot;
instance-specific bindings and Back history are cleared, including on rewind.

Evidence excerpts highlight exact source-note matches, preserving case,
whitespace, and punctuation. Repeated matches disclose that the first occurrence
is shown; long prose/citations use labeled deterministic excerpts. **Source note
unavailable** and **Exact span not found in source note** show the recorded quote
without highlighting. Read availability labels literally: **Pending**/**Active**
describe scan progress, **Not recorded** means missing assessment data, and
**No cancer mentions recorded** means a recorded empty mention list.

Raw prompts/responses are omitted from the main cards. The canonical artifact
retains captured model exchanges for review, subject to content-capture settings.

## Record, replay, and review

```bash
# Live presentation with an incrementally flushed recording (new path required).
uv run cipoc-demo serve --live --notes tests/fixtures/note_bundle.json \
    --record demo-runs/presentation.jsonl

# Headless recording, using the same canonical execution path.
uv run cipoc-demo record --notes tests/fixtures/note_bundle.json \
    --out demo-runs/rehearsal.jsonl

# Replay offline, with no endpoint credentials.
uv run cipoc-demo serve --replay demo-runs/rehearsal.jsonl

# Add the matching canonical artifact to enable review download and OMOP export.
uv run cipoc-demo serve --replay demo-runs/rehearsal.jsonl \
    --artifact demo-runs/JOB-UUID/run.json
```

Each execution atomically publishes `demo-runs/<job-id>/run.json` (or under
`--output-dir`). It is the exact canonical completed result or graph-failure
diagnostic. Initialization failures before a runtime envelope exists have a job
error, not a fabricated canonical artifact. Publication failure leaves the
in-memory artifact downloadable until the server closes. Recording/presentation
failures are reported independently and do not replace the clinical outcome.

JSONL recordings have independent `trace_version: "1.0"` records, contiguous
sequence numbers, monotonic observation times, a job identity at the start, and
the canonical run UUID/status at the end. Old unversioned recordings remain
readable. An incomplete last JSON line is ignored with a warning; malformed
interior records and unknown versions are rejected. A trace without a terminal
record displays as **incomplete**. Captured observation times are not model
service times or proof of causal ordering across concurrent branches.

The event queue is bounded. Overflow or reduction failure explicitly marks
presentation capture incomplete; the canonical result remains independent.
SSE queues coalesce refresh hints, while reconnect retrieves retained events.
Recordings remain in memory for scrubbing; replay checkpoints and snapshot
caches are bounded. This is a local single-case presenter, not an archival
stream-processing service.

For a dependable offline fallback, rehearse a saved recording and serve it on a
second port, e.g. `--port 8002`. A synthetic legacy compatibility recording is
available at `tests/fixtures/demo_trace.jsonl`. To generate a **current offline
rehearsal** with deterministic fake subagents and zero actual model calls:

```bash
PYTHONPATH=src python -m tests._demo_fixture --out demo-runs/offline.jsonl
```

Download a completed run JSON and use **Load Run…** in `cipoc-workbench serve`
(port 8000 by default). Failure diagnostics are downloadable but are not
completed Workbench inputs. No old feedback/ground-truth binding is redirected
when a live job finishes.

## OMOP and capture

Per-variable OMOP previews use the same pure row builder as file export, with
synthetic `person_id=1`. **Export OMOP…** asks for the actual numeric person ID
and downloads NOTE/NOTE_NLP CSVs plus row-validation errors in one ZIP, using the
completed case without another extraction. NLP date is the run's start date;
legacy traces without a recorded date use an explicit `1970-01-01` preview
placeholder. Concept columns are staging values requiring downstream mapping.

`--no-llm-content-capture` omits prompt/response bodies from both runtime exchange
capture and recorded runtime observations. `--max-content-chars N` limits each
captured prompt message. Raw graph message lists/group-response copies are
excluded from the presentation channel; clinical candidates, evidence, notes,
and errors remain.
All artifacts and recordings can contain PHI even with model content disabled.
Use the trusted local interface and handle downloaded files accordingly.

## Verification

```bash
PYTHONPATH=src python -m unittest tests.test_demo_state tests.test_demo_steps \
    tests.test_demo_server tests.test_demo_web tests.test_demo_live
node --test tests/demo_live.test.js tests/demo_cards.test.js \
    tests/demo_card_navigation.test.js tests/demo_map_viewport.test.js tests/demo_case_table.test.js
```

Tests use stdlib unittest and deterministic graphs; they do not call endpoints.
For an actual endpoint rehearsal, use the live command above with your chosen
notes/configuration, verify the final run JSON in Workbench, and preserve its
recording before presenting.
