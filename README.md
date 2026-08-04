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
  for invalid extractions) and case-scoped coding rules.
- **Loop & finalize** — newly coded scoping facts feed the next planning pass;
  when nothing remains eligible (or a fatal blocker is hit), the run finalizes
  into a `Case` with per-variable results and a review report flagging errors,
  invalid extractions, and low-confidence values.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) with a pinned `uv.lock`:

```bash
uv sync
```

Requires Python ≥ 3.11 (see `.python-version`). Source lives under `src/`, so run
commands with `PYTHONPATH=src` unless the package is installed into the
environment.

## Configuration

Runtime config is loaded from `config/config.yaml` via
`cipoc.utils.load_config()`. `${VAR}` placeholders are expanded from the
environment at load time.

- `llm:` — default LLM settings applied to every agent (model, provider,
  `base_url`, `api_key`, `max_concurrency`, reasoning effort, `retry`).
- `agents:` — optional per-agent overrides (e.g. `extractor`, `note_scanner`,
  `note_retriever`, `orchestrator`), merged on top of the `llm` defaults.
- `documents:` — paths to the NAACCR data dictionary, the compiled rule store,
  and the variable-group definitions.

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
PYTHONPATH=src python src/cipoc/agents/orchestrator.py

# Optionally seed already-known coded values (skip their extraction)
PYTHONPATH=src python src/cipoc/agents/orchestrator.py \
    --structured-data '{"400": "C509"}'
```

Programmatically:

```python
from cipoc.agents.orchestrator import OrchestratorAgent

agent = OrchestratorAgent()
case = agent.run(raw_notes)              # raw_notes: list[dict], each a ClinicalNote
print(case.model_dump())                 # per-variable results + review report
```

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

### Inspect rule applications

Show the final compiled rules that apply to one or more NAACCR item IDs and the
same case-scoped `VariableGroupInfo` used by the extractor:

```bash
PYTHONPATH=src python -m scripts.examine_rule_applications \
    522 523 \
    --gross-primary-site breast \
    --date-of-diagnosis 2021-06-01 \
    --sex female
```

The positional arguments are one or more integer item IDs. Optional case facts
are `--primary-site`, `--gross-primary-site`, `--histology`, `--behavior`,
`--sex`, and `--date-of-diagnosis`. Use `--rules-dir` or `--data-dictionary` to
override the default `documents/rules` and
`documents/manuals/naaccr_data_dictionary_v25.json` paths.

### Smoke checks

There is no formal CI or comprehensive test suite yet; validation is
fixture-based smoke checks plus deterministic unit tests
(`tests/test_progress_tracking.py`). A quick import/compile check:

```bash
PYTHONPATH=src python -m py_compile src/cipoc/agents/orchestrator.py
```

## Rule compilation

The extractor scopes coding to case-specific rules drawn from a committed rule
store under `documents/rules/`. That store is produced offline by the pipeline in
`scripts/rule_compilation/`, which compiles source manuals (SEER, STORE, Solid
Tumor Rules, FORDS, SPCSM) into validated `RuleUnit` records:

```
segment (pure) → tag (LLM) → validate (pure) → report + write
```

These scripts run **outside** the runtime package — the runtime never imports
them. Compilation is idempotent and resumable; only accepted units are written,
and quarantined units plus a spot-check sample go to a review report that must be
eyeballed before the output is trusted.

```bash
# Compile one chapter of one manual
PYTHONPATH=src python -m scripts.rule_compilation.compile_manual \
    --manual solid_tumor_rules --site-group breast \
    --root-heading "Breast Equivalent Terms"

# Run every pending compile listed in config/variable_groups.json
PYTHONPATH=src python -m scripts.rule_compilation.compile_pending --run
```

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
│   └── coding_context.py  # Rule store loading + case-scoped coding instructions
└── utils/
    ├── utils.py           # YAML config loader + CipocConfig
    ├── progress_tracking.py  # run_with_progress graph runner
    └── databricks_utils.py

config/          # config.yaml + variable_groups.json
documents/
├── manuals/     # NAACCR data dictionary + source manuals (gitignored, not in a clone)
├── markdown/    # markdown-converted manuals the rule compilers read
└── rules/       # compiled rule store (manifest.json + per-manual units)
extract_rules/   # site/histology/coding-rule JSONs and conversion helpers
scripts/         # offline rule-compilation pipeline
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
- **Rules** — `RuleUnit`, `RuleKind`, `RuleApplicability`, `CaseFacts`,
  `ScopedVariableContext`, `RuleStoreManifest`.
- **Case** — `Case`, `CaseVariableResult`, `VariableStatus`, `CaseReport`,
  `ReviewFlag`/`ReviewFlagType`.
