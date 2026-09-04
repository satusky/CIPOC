# CIPOC

This repository contains code for the Cancer Identification and Precision
Oncology Center (CIPOC) at the University of North Carolina at Chapel Hill.
CIPOC extracts [NAACCR](https://www.naaccr.org/) (North American Association of
Central Cancer Registries) variables from free-text clinical notes using
LLM-backed agents coordinated by a deterministic orchestrator. It scans notes for
cancer evidence, selects the relevant notes for each requested variable, extracts
structured coded values against the NAACCR data dictionary, and rolls the results
up into an auditable, review-flagged case snapshot.

The design keeps LLM usage bounded to per-note/per-variable subagents while all
control flow, gating, filtering, and roll-up remain deterministic — so provenance,
confidence, and review reasons are preserved end to end.

> **Deployment target:** an airgapped Databricks Runtime 18.2 ML (CPU)
> environment. Dependencies are pinned to DBR 18.2–compatible versions. Do not
> introduce dependencies outside that set.

Development takes place in the airgapped UNC Health SHIRE environment, so this
repository may lag the internal code while changes are manually synchronized.
LLM workloads currently run in Azure Databricks; some Databricks-specific
components may remain while the project moves toward system-agnostic tooling.

## How it works

The `OrchestratorAgent` compiles a [LangGraph](https://langchain-ai.github.io/langgraph/)
state graph that drives the full extraction for one patient case:

```
raw notes
   │
   ▼
initialize ──► scan_notes ──► characterize_corpus ──► check_state ──┐
   (seed         (NoteScanner    (corpus descriptors      (loop hub) │
    plan &        per note,       + per-note digests)                │
    results)      fan-out)                                           │
                                                                     ▼
                                          ┌──────────────── plan_extraction
                                          │                  (fan out each
                                          │                   eligible group)
                                          ▼
                              extract_branch (per group):
                                retrieve_notes ──► extract
                                (hard NoteFilter +   (ExtractorAgent
                                 NoteRetriever        codes values,
                                 soft filter)         validates)
                                          │
                                          ▼
                                   merge_and_update ──► check_state
                                   (fold coded values
                                    into case facts)
                                          │
                              (no work left / fatal blocker)
                                          ▼
                                    finalize_case ──► Case snapshot
```

- **Scan** — `NoteScannerAgent` processes each `ClinicalNote` into a
  `ProcessedClinicalNote`: concept presence/evidence, cancer temporality status,
  a summary, and search keywords.
- **Characterize** — corpus-level descriptors and per-note digests are built
  deterministically from the processed notes.
- **Plan** — deterministic gating (`CorpusGate` predicates), site applicability,
  and variable dependencies decide which variable groups are eligible on each
  pass. Structured-data values supplied by the caller skip extraction entirely.
- **Retrieve** — for each group, a deterministic hard `NoteFilter` narrows the
  corpus, then `NoteRetrieverAgent` soft-filters the surviving note digests by
  relevance to the group's variables.
- **Extract** — `ExtractorAgent` codes the group's variables from the retrieved
  notes, validating each value against the data dictionary (with a repair loop
  for invalid extractions). Code descriptions are scoped by gross primary site.
- **Loop & finalize** — newly coded scoping facts feed the next planning pass;
  when nothing remains eligible (or a fatal blocker is hit), the graph finalizes
  a `Case` with per-variable results and a review report. The public run result
  wraps that durable clinical snapshot with inputs, corpus, and observability.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) with a pinned `uv.lock`:

```bash
uv sync
```

Requires Python ≥ 3.11 (see `.python-version`). Source lives under `src/`, so run
commands with `PYTHONPATH=src` unless the package is installed into the
environment.

### Workbench

The review workbench is a separate package and does not install CIPOC's runtime
dependencies. Install it with standard `pip`:

```bash
python -m pip install ./packages/cipoc-workbench
```

Then serve a canonical `OrchestratorRunResult` JSON artifact:

```bash
cipoc-workbench serve \
    --state tests/test_outputs/case_state.json \
    --ground-truth ground_truth.json \
    --feedback feedback.json
```

All three paths are optional. Without arguments, the workbench serves its
bundled example at `http://127.0.0.1:8000/`.

## Configuration

Runtime config is loaded from `config/config.yaml` via
`cipoc.utils.load_config()`. `${VAR}` placeholders are expanded from the
environment at load time.

- `llm:` — default LLM settings applied to every agent (model, provider,
  `base_url`, `api_key`, `max_concurrency`, reasoning effort, `retry`).
- `agents:` — optional per-agent overrides (e.g. `extractor`, `note_scanner`,
  `note_retriever`, `orchestrator`), merged on top of the `llm` defaults.
- `documents:` — paths to the NAACCR and tissue-keyed data dictionaries and the
  variable-group definitions.

Only the OpenAI-compatible provider is active (targeting Databricks/Azure
OpenAI-compatible endpoints). Set credentials via environment variables rather
than hardcoding them:

```bash
export AZURE_OPENAI_URL=...
export AZURE_OPENAI_API_KEY=...
```

The variables to extract are defined in `config/variable_groups.json` as ordered
groups with gating conditions, note filters, and NAACCR item IDs.

## Usage

### Run the full orchestrator

```bash
# End-to-end run over the shared note bundle fixture
PYTHONPATH=src python -m cipoc.agents.orchestrator

# Optionally seed already-known coded values (skip their extraction)
PYTHONPATH=src python -m cipoc.agents.orchestrator \
    --structured-data '{"400": "C509"}'
```

Programmatically:

```python
from cipoc.agents import OrchestratorAgent
from cipoc.models import OrchestratorRunError

agent = OrchestratorAgent()
try:
    result = agent.run(raw_notes)  # raw_notes: list[dict], each a ClinicalNote
except OrchestratorRunError as error:
    failure = error.failure        # partial inputs, corpus, and observability
    raise

case = result.case                 # durable clinical output
print(case.model_dump())
```

`run()` returns an `OrchestratorRunResult` only after the graph completes and the
clinical `Case` is finalized. A graph failure raises `OrchestratorRunError`; its
`failure` attribute is an `OrchestratorRunFailure` with the partial diagnostic
artifact and no `case` field.

### Write a Workbench artifact

Use the thin run-result CLI to execute a case and write the canonical JSON:

```bash
PYTHONPATH=src python -m scripts.run_case_state \
    --notes tests/fixtures/note_bundle.json \
    --output tests/test_outputs/case_state.json

# Retain exchange metadata and usage, but omit model prompts and responses.
PYTHONPATH=src python -m scripts.run_case_state \
    --notes tests/fixtures/note_bundle.json \
    --output tests/test_outputs/case_state.json \
    --no-llm-content-capture

# Alternatively, retain only a bounded prefix of each prompt message.
PYTHONPATH=src python -m scripts.run_case_state \
    --notes tests/fixtures/note_bundle.json \
    --output tests/test_outputs/case_state.json \
    --max-content-chars 20000
```

The versioned `schema_version: "1.0"` artifact has five domains:

- `run` - run identity, timing, completion status, configuration fingerprint,
  and `contains_phi`;
- `case` - the durable clinical output, including variable results, case facts,
  note-selection provenance, and review flags;
- `inputs` - configured target groups and caller-supplied structured values;
- `corpus` - full processed notes, note digests, and corpus descriptors;
- `observability` - variable attempts, LLM exchanges, capture settings, and
  provider-reported usage totals and breakdowns.

This is the JSON boundary consumed directly by the standalone Workbench. Failed
runs are serialized by this CLI as `OrchestratorRunFailure` diagnostics and exit
nonzero; they are not completed Workbench result artifacts.

LLM exchange metadata, retries, errors, variable attempts, and usage collection
remain enabled when `capture_llm_content=False` or
`--no-llm-content-capture` is used. Only prompt and parsed response bodies are
omitted. Prompt capture is unbounded by default. `max_content_chars` or
`--max-content-chars` optionally limits each retained prompt message and records
both per-message truncation metadata and the run-level `content_truncated` flag;
parsed responses are not truncated.

> **PHI:** Every run artifact is PHI-bearing. Disabling LLM content capture does
> not de-identify it because `corpus.note_corpus` still contains the full clinical
> notes. Retained prompts and responses may also contain PHI.

Token usage is provider-reported, not independently calculated. Totals cover the
invocations for which callbacks expose usage; failed calls and retries internal
to a provider SDK may not report usage or appear as separate invocations. Check
`usage_reported_invocations` and `missing_usage_invocations` before treating a
total as complete. Input/output detail counts, such as cached input or reasoning
output tokens, are breakdowns of the corresponding totals, not additional
tokens. The artifact does not estimate monetary cost, retain hidden reasoning,
or contain a raw graph-event timeline.

### Run individual agents

```bash
# Scanner demo against tests/fixtures/synthetic_note.json
PYTHONPATH=src python src/cipoc/agents/note_scanner.py

# Extractor demo against tests/fixtures/note_bundle.json
PYTHONPATH=src python src/cipoc/agents/extractor.py
```

Each agent's `draw(path=...)` renders its compiled graph to a PNG (falling back
to ASCII when no network is available); rendered diagrams live under
`src/cipoc/agents/visualization/`.

### Smoke checks

There is no formal CI or comprehensive test suite yet; validation is
fixture-based smoke checks plus deterministic unit tests
(`tests/test_progress_tracking.py`). A quick import/compile check:

```bash
PYTHONPATH=src python -m py_compile src/cipoc/agents/orchestrator.py
```

## Site-scoped data dictionary

Variable metadata comes from the NAACCR dictionary configured by
`documents.data_dictionary_path`. When case facts identify a supported gross
primary site, the corresponding entry in `documents/cipoc_data_dictionary.json`
replaces the variable's unscoped `allowed_codes`. Unknown sites and items not
present in the site dictionary retain their NAACCR values. Runtime extraction
does not load or compile rules from `documents/rules/`.

## Repository layout

```text
src/cipoc/
├── agents/
│   ├── base.py            # BaseAgent: config load, LLM init, graph compile, draw()
│   ├── note_scanner.py    # NoteScannerAgent: per-note concept/temporality/summary scan
│   ├── note_retriever.py  # NoteRetrieverAgent: soft-filter notes by relevance to a group
│   ├── extractor.py       # ExtractorAgent: code + validate a variable group's values
│   ├── orchestrator.py    # OrchestratorAgent: end-to-end case extraction graph
│   └── visualization/     # Rendered agent graph PNGs
├── llm/
│   ├── base.py            # LLMConfig / BaseAgentModel abstractions
│   ├── openai.py          # OpenAI-compatible ChatOpenAI wrapper
│   └── retry.py           # RetryPolicy for LLM-backed graph nodes
├── models/                # Pydantic contracts (see below)
├── prompts/               # Per-agent prompt strings
├── tools/
│   ├── extraction.py      # Data-dictionary lookup + variable value validation
│   ├── orchestration.py   # Deterministic gating/filtering/roll-up helpers
│   └── coding_context.py  # Legacy compiled-rule utilities (not used at runtime)
└── utils/
    ├── utils.py           # YAML config loader + CipocConfig
    ├── progress/          # shared graph stream runner + live dashboard
    ├── observability.py   # LLM exchange capture and usage aggregation
    └── databricks_utils.py

packages/
└── cipoc-workbench/       # Standalone browser-based result review package

config/          # config.yaml + variable_groups.json
documents/
├── manuals/     # NAACCR data dictionary + source manuals (gitignored, not in a clone)
├── markdown/    # markdown-converted manuals the rule compilers read
├── cipoc_data_dictionary.json  # tissue-keyed code descriptions used at runtime
└── rules/       # legacy compiled rule store (not used at runtime)
extract_rules/   # site/histology/coding-rule JSONs and conversion helpers
scripts/         # run-result, OMOP, graph-rendering, and offline utility CLIs
planning/        # design notes and MVP plans (planning/old/ = superseded reference)
tests/fixtures/     # synthetic notes and note bundles for smoke checks
tests/test_outputs/ # saved outputs from agent demo (__main__) runs
```

## Data models

Import models from `cipoc.models` rather than redefining schemas in agents or
tools. Key groups:

- **Notes** — `ClinicalNote`, `ProcessedClinicalNote`, `CancerMention`,
  `CancerStatus`, `NoteDigest`, `NoteCorpusDescriptors`, concept types
  (`ConceptWithEvidence`, `ConceptPresence`, `TextSpan`).
- **Variables** — `VariableInfo`, `VariableOutput`, `VariableGroupInfo`,
  `VariableGroupOutput`, validated variants, and targeting/gating types
  (`TargetGroup`, `CorpusGate`, `NoteFilter`, `SiteApplicability`).
- **Scoping** — `CaseFacts` supplies gross and coded primary-site context for
  selecting tissue-specific data-dictionary values.
- **Case** — `Case`, `CaseVariableResult`, `VariableStatus`, `CaseReport`,
  `ReviewFlag`/`ReviewFlagType`.
- **Observability** — typed model exchanges, variable attempts, normalized token
  usage, and capture metadata.
- **Run** — `OrchestratorRunResult`, `OrchestratorRunFailure`,
  `OrchestratorRunError`, and the versioned run/input/corpus contracts.
