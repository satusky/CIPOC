# CIPOC Foundry Python transform

These files run the CIPOC orchestrator as a lightweight Foundry Python transform
(`@transform.using`), one patient per input row. Model calls go to the
language model Foundry injects into the transform, not to an OpenAI endpoint:

1. `cipoc.llm.FoundryModelsTransport` wraps the injected
   `OpenAiGptChatLanguageModel` and answers Chat Completions requests from it.
2. An `httpx.Client` using that transport is shared with every agent through
   `CipocConfig(model_kwargs={"http_client": client})`.
3. `foundry_models.py` translates each OpenAI request into a
   `GptChatCompletionRequest` and each `GptChatCompletionResponse` back into
   OpenAI JSON.

| File here | Goes to (in the Foundry repository) | Purpose |
| --- | --- | --- |
| `cipoc_extract.py` | `transforms-python/src/myproject/datasets/cipoc_extract.py` | The transform |
| `foundry_models.py` | `transforms-python/src/myproject/foundry_models.py` | OpenAI ↔ Foundry request/response translation |
| `config.yaml` | `transforms-python/src/myproject/resources/config.yaml` | CIPOC runtime config |

These files only run inside Foundry: they import `transforms`,
`palantir_models`, `language_model_service_api`, and `myproject`, none of which
exist locally. Nothing in `src/cipoc` imports them. After editing either copy,
update the other by hand.

## Repository layout

Start from a Python transforms repository with the standard `myproject`
template. When it is set up, it looks like this:

```text
transforms-python/
├── conda_recipe/
│   └── meta.yaml
├── setup.py                         # add package_data (see below)
└── src/
    ├── cipoc/                       # this repository's src/cipoc (see below)
    └── myproject/
        ├── __init__.py
        ├── pipeline.py              # template; discovers transforms in myproject.datasets
        ├── foundry_models.py        # ← deploy/foundry_transforms/foundry_models.py
        ├── datasets/
        │   ├── __init__.py
        │   └── cipoc_extract.py     # ← deploy/foundry_transforms/cipoc_extract.py
        └── resources/
            ├── __init__.py          # empty; the transform finds this directory through it
            ├── config.yaml          # ← deploy/foundry_transforms/config.yaml
            ├── naaccr_data_dictionary_v25.json   # ← documents/manuals/naaccr_data_dictionary_v25.json
            ├── cipoc_data_dictionary.json        # ← documents/cipoc_data_dictionary.json
            └── variable_groups.json              # ← config/variable_groups.json
```

Both data dictionaries are gitignored, so a fresh clone of this repository
doesn't contain them. Copy them from a machine that has them. The site data
dictionary must be the scoping file that canonical-naaccr generates; runs fail
on the older tissue-keyed file.

### The `cipoc` package

Copy `src/cipoc/` to `transforms-python/src/cipoc/`, leaving out:

- `agents/visualization/`: rendered graph images and HTML;
- `__pycache__/` directories.

Keep `utils/progress/`. The orchestrator streams its graph through it even
with `progress=False`.

From a local clone of the Foundry repository:

```bash
rsync -a --prune-empty-dirs \
    --exclude '__pycache__' --exclude '.DS_Store' \
    --exclude 'agents/visualization' \
    src/cipoc/ /path/to/foundry-repo/transforms-python/src/cipoc/
```

### `setup.py`

The template's `find_packages()` already picks up `cipoc` and
`myproject.resources`, but setuptools ships only `.py` files unless told
otherwise. Add `package_data` to the existing `setup(...)` call so the
YAML and JSON resources are included in the build:

```python
setup(
    ...
    packages=find_packages(exclude=['contrib', 'docs', 'test']),
    package_data={'': ['*.yaml', '*.json']},
    ...
)
```

## Dependencies

Add these libraries in the repository's **Libraries** panel:

- `palantir_models` and `language-model-service-api`, for the model input and
  the request/response types
  ([Palantir docs](https://www.palantir.com/docs/foundry/transforms-python/palantir-provided-models));
- CIPOC's runtime dependencies. Pin these to the versions in `pyproject.toml`
  when Foundry offers them; those are the versions CIPOC is tested against:
  `pydantic`, `langchain`, `langchain-core`, `langchain-openai`, `langgraph`,
  `openai`, `pyyaml`, `pandas`. `httpx` comes with `openai`.

The remaining `pyproject.toml` dependencies aren't needed in Foundry:
`databricks-sdk`, `mlflow-skinny`, `tqdm`, `ipython`, `grandalf`, `graphviz`,
and `matplotlib`. They serve Databricks, terminal and notebook display, or
graph drawing.

## Wiring the transform

Fill in the three constants at the top of `cipoc_extract.py`:

- `MODEL_RID`: the language model's RID, e.g.
  `ri.language-model-service..language-model.<name>`. Import the model into
  the repository first.
- `NOTES_DATASET_RID`: the input dataset, with one row per patient:
  - `mspi`: the patient identifier, copied to the output unchanged;
  - `notes`: an array of structs, each with `note_id` (int or string), `date`
    (date, timestamp, or `YYYY-MM-DD` string), `note_type` (string), and
    `content` (string). Extra struct fields are ignored.
- `OUTPUT_DATASET_RID`: the output dataset, written with one row per input
  row:

  | Column | Contents |
  | --- | --- |
  | `mspi` | From the input row |
  | `status` | `completed`, `failed`, or `invalid_input` |
  | `error` | Error text for `failed` and `invalid_input`; null for `completed` |
  | `result_json` | `OrchestratorRunResult` JSON for `completed`, `OrchestratorRunFailure` JSON for `failed`, null for `invalid_input` |

  - `invalid_input` means the notes were rejected before any model call:
    empty, malformed, or with duplicate note IDs.
  - `failed` means the run raised `OrchestratorRunError`.
  - Any other exception fails the build. That includes config, resource, and
    dependency errors.

## `config.yaml`

This file has the same format as `config/config.yaml`. Several settings
matter specifically in Foundry:

- **`base_url` and `api_key`** are placeholders. The transport answers
  every request itself, so nothing is ever sent to `base_url`. Don't put real
  credentials here.
- **`model`** names the run. It is recorded in the result artifact and is part
  of the concurrency key. The transport sends every request to the injected
  model, whatever this name is.
- **`use_responses_api`** must be `false`. The transport serves only Chat
  Completions; anything else returns 404.
- **`max_concurrency`** caps concurrent model calls in the process. Preview
  reuses one Python process across runs, so after you change this value
  Preview can fail with `Conflicting max_concurrency ... active capacity is N`
  because the previous run's agents haven't been garbage-collected yet.
  Restart the preview session. Full builds start a fresh process and aren't
  affected.
- **The `documents:` block is not used.** `build_config` in
  `cipoc_extract.py` replaces it with the fixed file names under
  `resources/`. To use a different variable groups file, change
  `build_config`, not the YAML.

## Troubleshooting

Model-call errors reach the output's `error` column as the OpenAI SDK's
exception text. The `type` field identifies the source:

| Error type | Meaning |
| --- | --- |
| `request_translation_error` (400 or 422) | `to_foundry_request` failed or rejected the request, e.g. legacy `functions` (422) or a translator bug. Not retried. |
| `upstream_error` (status carried from Foundry, else 502) | `create_chat_completion` raised. Retried only for 408, 409, 429, and 5xx. |
| `response_translation_error` (400) | `to_openai_response` failed on Foundry's reply. Not retried. |
| `unsupported_endpoint` (404) | `use_responses_api` is not `false`. |

## PHI

The output dataset contains PHI. `capture_llm_content=False` keeps prompts and
model responses out of `result_json`, but the artifact still contains the full
processed notes (`corpus.note_corpus`). Error text can also quote note content.
Restrict access to the output dataset accordingly. Test changes in Preview on
synthetic notes first.
