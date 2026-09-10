"""Offline regressions for the orchestrator producer and versioned run contract.

The dashboard fixture hosts the production orchestrator graph. Its extractor is
only a scripted branch result, never evidence that production validation passed.
Scanner calls use the production scanner graph and inherited model callbacks.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.pregel._loop import SyncPregelLoop
from pydantic import Field

from cipoc.agents.note_scanner import NoteScannerAgent
from cipoc.agents.orchestrator import OrchestratorAgent
from cipoc.llm import OpenAIConfig
from cipoc.models import (
    CaseFacts,
    ClinicalNote,
    CONCEPT_DESCRIPTIONS,
    OrchestratorRunError,
    OrchestratorRunFailure,
    OrchestratorRunResult,
    ProcessedClinicalNote,
    RunObservability,
    SiteApplicability,
    TargetGroup,
    TextSpan,
    VariableInfo,
    VariableStatus,
)
from cipoc.models.observability import ObservabilityIssue
from cipoc.utils import ObservabilityCollector

from tests.fake_orchestrator import Outcome, Script, build_fake_orchestrator


REPO_ROOT = Path(__file__).resolve().parents[1]


def _note(note_id="note-A"):
    return ClinicalNote(
        note_id=note_id,
        date="2026-09-01",
        note_type="Pathology",
        content=f"Synthetic carcinoma evidence for note {note_id}.",
    )


def _group(group_id="initial", item_ids=(400,), **kwargs):
    return TargetGroup(
        group_id=group_id,
        variables=[VariableInfo(item_id=item_id) for item_id in item_ids],
        stage=kwargs.pop("stage", "initial"),
        **kwargs,
    )


def _wait(event, description):
    if not event.wait(timeout=10):
        raise AssertionError(f"Timed out waiting for {description}.")


class _ScannerModel(BaseChatModel):
    """Return complete structured answers while exercising real LLM callbacks."""

    model_name: str = "offline-scanner"
    before_call: Callable | None = Field(default=None, exclude=True, repr=False)
    tissues: dict = Field(default_factory=dict, exclude=True)
    calls: list = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self):
        return "orchestrator-remediation-test"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    def structured(self, schema, messages):
        response = self.invoke(messages, response_schema=schema)
        return schema.model_validate_json(response.content)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        schema = kwargs["response_schema"]
        raw_note = next(
            message.content.partition("Clinical note:\n")[2]
            for message in messages
            if isinstance(message.content, str)
            and message.content.startswith("Clinical note:\n")
        )
        note = ClinicalNote.model_validate_json(raw_note)
        self.calls.append((note.note_id, schema.__name__))
        if self.before_call is not None:
            self.before_call(note, schema.__name__)
        evidence = [{"note_id": note.note_id, "text": note.content}]
        if schema.__name__ == "NoteSummary":
            answer = {"summary": f"Scanned: {note.content}", "keywords": ["carcinoma", "pathology"]}
        elif schema.__name__ == "CancerMentions":
            answer = {"mentions": [
                {
                    "presence": True, "confidence": "high", "evidence": evidence,
                    "status": status, "affected_tissue": tissue, "metastasis": False,
                }
                for status, tissue in self.tissues.get(note.note_id, [("current", "breast")])
            ]}
        else:
            answer = {
                name: {
                    "presence": name == "cancer", "confidence": "high",
                    "evidence": evidence if name == "cancer" else [],
                }
                for name in CONCEPT_DESCRIPTIONS
            }
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=json.dumps(answer),
            response_metadata={"model_name": self.model_name},
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        ))])


def _agent(*, model=None, groups=None):
    agent = build_fake_orchestrator(Script(outcomes={400: Outcome(value="C504")}))
    agent._target_variables = groups if groups is not None else [_group()]
    agent._target_group_hierarchy = []
    scanner = object.__new__(NoteScannerAgent)
    scanner._llm_config = agent._llm_config
    scanner._retry_policy = agent._retry_policy
    scanner.agent = model if model is not None else _ScannerModel()
    scanner._graph = scanner._build_graph()
    agent._scanner = scanner
    agent._graph = agent._build_graph()
    return agent


class _Completions(BaseCallbackHandler):
    """Observe actual branch callbacks independently of the production collector."""

    raise_error = True

    def __init__(self, completed, teardown=None, joined=None):
        self.completed = completed
        self.teardown = teardown or threading.Event()
        self.joined = joined or threading.Event()
        self.notes = []
        self.during_teardown = []
        self.model_starts = []
        self._branches = set()
        self._lock = threading.Lock()

    def on_chain_start(self, serialized, inputs, *, run_id, name=None, tags=None, **kwargs):
        if name == "note_branch" and any(tag.startswith("graph:step:") for tag in tags or ()):
            with self._lock:
                self._branches.add(run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self._lock:
            if run_id not in self._branches:
                return
            self._branches.remove(run_id)
            self.notes.extend(outputs["note_corpus"].values())
            self.during_teardown.append(self.teardown.is_set() and not self.joined.is_set())
            self.completed.set()

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        with self._lock:
            self.model_starts.append(metadata)


class _UndumpableConfig(OpenAIConfig):
    def model_dump(self, *args, **kwargs):
        raise AssertionError("Fingerprint must select safe fields before serialization.")

    def model_dump_json(self, *args, **kwargs):
        raise AssertionError("Fingerprint must not serialize the source config.")


def _secret_config(marker):
    return _UndumpableConfig(
        model="safe-model",
        api_key=f"{marker}-api-key",
        base_url=f"https://{marker}-user:{marker}-password@{marker}.invalid/{marker}-path?token={marker}-query#{marker}-fragment",
        default_headers={"Authorization": f"Bearer {marker}-header"},
        extra_body={"nested": [{"credential": f"{marker}-nested", "client": object()}]},
        http_client=object(),
        reasoning=None,
        temperature=0.2,
        top_p=0.9,
        max_tokens=100,
        max_concurrency=1,
        use_responses_api=False,
    ).model_copy(update={"reasoning": {
        "effort": "medium", "summary": "auto",
        "nested_secret": marker, "client": object(),
    }})


def _set_config(agent, settings):
    for component in (agent, agent._scanner, agent._retriever, agent._extractor):
        component._llm_config = settings


class OrchestratorRemediationTests(unittest.TestCase):
    def _subprocess_guard(self):
        """Bound deadlock-sensitive real-graph tests, including interpreter exit."""
        test_id = f"tests.test_orchestrator_remediation.{type(self).__name__}.{self._testMethodName}"
        if os.environ.get("CIPOC_REMEDIATION_CHILD") == test_id:
            return False
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", test_id],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CIPOC_REMEDIATION_CHILD": test_id},
            capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def _dictionaries(self, agent):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = {
            "400": {"item_name": "Primary Site", "item_length": 4,
                    "allowed_codes": {"C504": "Breast", "C349": "Lung", "C693": "Choroid"}},
            "410": {"item_name": "Laterality", "item_length": 1,
                    "allowed_codes": {"0": "Base only", "1": "Breast", "2": "Lung"}},
        }
        site = {
            "breast": {"400": {"allowed_codes": {"C504": "Breast"}},
                       "410": {"allowed_codes": {"1": "Breast"}}},
            "lung": {"400": {"allowed_codes": {"C349": "Lung"}},
                     "410": {"allowed_codes": {"2": "Lung"}}},
        }
        agent._data_dictionary_path = directory / "base.json"
        agent._site_data_dictionary_path = directory / "site.json"
        agent._data_dictionary_path.write_text(json.dumps(base), encoding="utf-8")
        agent._site_data_dictionary_path.write_text(json.dumps(site), encoding="utf-8")
        # Restore the real scoping method stubbed by the dashboard fixture.
        del agent._scope_group

    def test_public_empty_and_duplicate_ids_fail_before_execution(self):
        for notes, structured, message in (
            ([], None, "At least one"),
            ([], {400: "C349"}, "At least one"),
            ([_note(1), _note(1)], None, "Duplicate canonical"),
            ([_note(1), _note("1")], None, "Duplicate canonical"),
            ([_note("note-A"), _note("note-A")], None, "Duplicate canonical"),
        ):
            with self.subTest(ids=[note.note_id for note in notes], structured=structured):
                agent = _agent()
                with patch.object(agent._graph, "stream") as stream, patch.object(agent._scanner, "run") as scan:
                    with self.assertRaisesRegex(ValueError, message):
                        agent.run([note.model_dump() for note in notes], structured, progress=False)
                stream.assert_not_called()
                scan.assert_not_called()
                self.assertEqual(agent._scanner.agent.calls, [])

    def test_direct_graph_rejects_empty_colliding_and_mismatched_note_keys(self):
        for corpus, structured, message in (
            ({}, {}, "At least one"),
            ({}, {400: "C349"}, "At least one"),
            ({1: _note(1), "1": _note("1")}, {}, "duplicate canonical"),
            ({"outer": _note("inner")}, {}, "key does not match"),
            ({1: _note(2).model_dump()}, {}, "key does not match"),
        ):
            with self.subTest(corpus=corpus, structured=structured):
                agent = _agent()
                with patch.object(agent._scanner, "run") as scan:
                    with self.assertRaisesRegex(ValueError, message):
                        agent.compiled_graph.invoke(
                            {"note_corpus": corpus, "structured_data": structured},
                            config={"max_concurrency": 2},
                        )
                scan.assert_not_called()
                self.assertEqual(agent._scanner.agent.calls, [])

    def test_distinct_native_ids_survive_scanning_and_json(self):
        ids = [1, "01", "001", "note-A", "NOTE-A", " note-A "]
        result = _agent().run(
            [_note(note_id).model_dump() for note_id in ids], {400: "C349"}, progress=False,
        )
        self.assertEqual(list(result.corpus.note_corpus), ids)
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())
        for expected, note in zip(ids, restored.corpus.note_corpus.values()):
            self.assertEqual(note.note_id, expected)
            self.assertIs(type(note.note_id), type(expected))
            span = note.concepts["cancer"].evidence[0]
            self.assertEqual(span.note_id, expected)
            self.assertIs(type(span.note_id), type(expected))
        self.assertEqual(len(restored.corpus.note_corpus), len(ids))

    def test_concurrency_one_preflights_explicit_config_and_bound_settings(self):
        if self._subprocess_guard():
            return
        for source in ("explicit", "config", "bound"):
            for progress in (False, True):
                with self.subTest(source=source, progress=progress):
                    agent = _agent()
                    options = {}
                    if source == "explicit":
                        options["max_concurrency"] = 1
                    elif source == "config":
                        options["config"] = {"max_concurrency": 1}
                    else:
                        agent._graph = agent._graph.with_config({"max_concurrency": 1})
                    with patch.object(agent._graph, "stream") as stream, patch(
                        "cipoc.utils.progress.runner._select_renderer"
                    ) as renderer, patch.object(agent, "_config_fingerprint") as fingerprint:
                        with self.assertRaisesRegex(ValueError, "max_concurrency=1.*unsafe"):
                            agent.run([_note().model_dump()], progress=progress, **options)
                    stream.assert_not_called()
                    renderer.assert_not_called()
                    fingerprint.assert_not_called()
                    self.assertEqual(agent._scanner.agent.calls, [])

    def test_concurrency_two_runs_nested_graphs_and_overrides_unsafe_lower_layers(self):
        if self._subprocess_guard():
            return
        for source in ("explicit", "config", "bound", "explicit_override", "config_override"):
            with self.subTest(source=source):
                agent = _agent()
                options = {}
                if source in {"bound", "explicit_override", "config_override"}:
                    agent._graph = agent._graph.with_config({"max_concurrency": 2 if source == "bound" else 1})
                if source in {"explicit", "explicit_override"}:
                    options["max_concurrency"] = 2
                if source in {"config", "config_override", "explicit_override"}:
                    options["config"] = {"max_concurrency": 1 if source == "explicit_override" else 2}
                # The independent per-model LLM capacity remains valid at one.
                _set_config(agent, agent._llm_config.model_copy(update={"max_concurrency": 1}))
                result = agent.run([_note().model_dump()], {400: "C349"}, progress=False, **options)
                self.assertEqual(result.run.status, "completed")
                self.assertEqual(result.observability.llm_usage_summary.model_invocations, 3)
                self.assertEqual(result.observability.collection_status, "complete")
                self.assertEqual(result.run.config_fingerprint.agent_llm_config["note_scanner"]["max_concurrency"], 1)

    def test_retriever_aliases_map_to_offered_native_ids_and_deduplicate(self):
        agent = _agent()
        ids = [1, "001", "01", "note-A"]
        proposals = ["1", 1, "001", "note-A", "01", "001", "1 ", "NOTE-A", 999]
        with patch.object(agent._retriever, "run", return_value=proposals) as retrieve, patch.object(
            agent._extractor, "run", wraps=agent._extractor.run
        ) as extract:
            result = agent.run([_note(note_id).model_dump() for note_id in ids], progress=False)
        offered = retrieve.call_args.args[0].available_digests
        self.assertEqual(list(offered), ids)
        selection = result.case.note_selection["group:initial"]
        self.assertEqual(selection.candidate_note_ids, ids)
        self.assertEqual(selection.selected_note_ids, [1, "001", "note-A", "01"])
        self.assertIs(type(selection.selected_note_ids[0]), int)
        self.assertEqual(selection.discarded_note_ids, ["1 ", "NOTE-A", 999])
        self.assertEqual(
            [note.note_id for note in extract.call_args.args[0].notes],
            selection.selected_note_ids,
        )
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())
        self.assertEqual(restored.case.note_selection, result.case.note_selection)

    def test_coded_primary_beats_breast_mentions_in_characterization_scope_and_planning(self):
        agent = _agent(groups=[
            _group(), _group("laterality", (410,), stage="dependent"),
            _group("breast-only", (420,), stage="dependent",
                   applies_to=SiteApplicability(gross_primary_sites=["breast"])),
        ])
        self._dictionaries(agent)
        agent._extractor._script.outcomes[410] = Outcome(value="2")
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run([_note().model_dump()], {400: "C349"}, progress=False)
        self.assertEqual(result.case.case_facts.primary_site, "C349")
        self.assertIsNone(result.case.case_facts.gross_primary_site)
        self.assertEqual(result.corpus.note_corpus_descriptors.affected_tissues, {"current": {"breast"}})
        self.assertEqual(result.case.variable_results[420].status, VariableStatus.NOT_APPLICABLE)
        extract.assert_called_once()
        self.assertEqual(extract.call_args.args[0].requested_variables.variables[0].valid_codes, {"2": "Lung"})

    def test_unresolved_current_tissue_does_not_narrow_or_fall_back_to_history(self):
        model = _ScannerModel(tissues={"note-A": [
            ("current", "left breast"), ("current", "unclassified integument"),
            ("historical", "breast"),
        ]})
        agent = _agent(model=model)
        self._dictionaries(agent)
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run([_note().model_dump()], progress=False)
        self.assertIsNone(result.case.case_facts.gross_primary_site)
        self.assertEqual(
            set(extract.call_args.args[0].requested_variables.variables[0].valid_codes),
            {"C504", "C349", "C693"},
        )
        self.assertEqual(result.corpus.note_corpus_descriptors.affected_tissues["current"],
                         {"left breast", "unclassified integument"})

    def test_recognized_breast_aliases_still_narrow(self):
        model = _ScannerModel(tissues={"note-A": [("current", "left breast"), ("current", "right breast")]})
        agent = _agent(model=model)
        self._dictionaries(agent)
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run([_note().model_dump()], progress=False)
        self.assertEqual(result.case.case_facts.gross_primary_site, "breast")
        self.assertEqual(extract.call_args.args[0].requested_variables.variables[0].valid_codes, {"C504": "Breast"})

    def test_known_primary_without_site_dictionary_never_falls_back_to_gross_breast(self):
        agent = _agent()
        self._dictionaries(agent)
        # A later coded primary can coexist with an earlier gross-site inference.
        for primary in ("C693", "C69.3"):
            with self.subTest(primary=primary):
                scoped = agent._scope_group(
                    _group(item_ids=(410,)),
                    CaseFacts(primary_site=primary, gross_primary_site="breast"),
                )
                self.assertEqual(scoped.variables[0].valid_codes,
                                 {"0": "Base only", "1": "Breast", "2": "Lung"})

    def test_fingerprint_projects_before_serialization_and_ignores_auth_only_changes(self):
        agent = _agent()
        _set_config(agent, _secret_config("SYNTHETIC_SECRET_A"))
        first = agent._config_fingerprint()
        _set_config(agent, _secret_config("SYNTHETIC_SECRET_B"))
        second = agent._config_fingerprint()
        self.assertEqual(first, second)
        serialized = first.model_dump_json()
        for excluded in ("SYNTHETIC_SECRET", "default_headers", "extra_body", "http_client", "base_url", "api_key", "nested_secret"):
            self.assertNotIn(excluded, serialized)
        self.assertEqual(first.agent_llm_config["note_scanner"]["reasoning"],
                         {"effort": "medium", "summary": "auto"})
        self.assertEqual(first.agent_llm_config["note_scanner"]["temperature"], 0.2)

    def test_safe_settings_changes_are_reflected_in_fingerprint(self):
        agent = _agent()
        settings = _secret_config("SYNTHETIC_SECRET")
        _set_config(agent, settings)
        baseline = agent._config_fingerprint()
        for field, value in (
            ("model", "other-safe-model"), ("temperature", 0.4), ("top_p", 0.8),
            ("max_tokens", 200), ("max_concurrency", 2),
            ("endpoint_compatibility", "databricks"), ("structured_output_method", "json_mode"),
            ("use_responses_api", True), ("reasoning", {"effort": "high", "summary": "detailed"}),
        ):
            with self.subTest(field=field):
                _set_config(agent, settings.model_copy(update={field: value}))
                fingerprint = agent._config_fingerprint()
                self.assertNotEqual(fingerprint, baseline)
                for config in fingerprint.agent_llm_config.values():
                    self.assertEqual(config[field], value)

    def test_completed_and_failed_artifacts_never_serialize_source_config_secrets(self):
        agent = _agent()
        _set_config(agent, _secret_config("SYNTHETIC_SECRET_ARTIFACT"))
        result = agent.run([_note().model_dump()], {400: "C349"}, progress=False)
        original = RuntimeError("offline scanner failure")

        def fail_scan(note, schema):
            raise original

        agent._scanner.agent.before_call = fail_scan
        with self.assertRaises(OrchestratorRunError) as raised:
            agent.run([_note("failed").model_dump()], progress=False)
        self.assertIs(raised.exception.__cause__, original)
        for artifact in (result, raised.exception.failure):
            self.assertNotIn("SYNTHETIC_SECRET", artifact.model_dump_json())
            self.assertNotIn("http_client", artifact.model_dump_json())
            self.assertNotIn("default_headers", artifact.model_dump_json())

    def test_actual_result_is_1_1_and_legacy_1_0_remains_readable(self):
        result = _agent().run([_note().model_dump()], {400: "C349"}, progress=False)
        self.assertEqual(result.schema_version, "1.2")
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())
        self.assertEqual(restored, result)
        legacy = result.model_dump(mode="json")
        legacy["schema_version"] = "1.0"
        for name in ("collection_status", "collection_issues", "unattributed_exchanges"):
            legacy["observability"].pop(name)
        for exchanges in legacy["observability"]["llm_exchanges"].values():
            for exchange in exchanges:
                exchange.pop("invocation_id", None)
                exchange.pop("namespace", None)
        # Producer completion checks must not become historical-reader validators.
        legacy["case"]["report"] = None
        legacy["case"]["variable_results"] = {}
        historical = OrchestratorRunResult.model_validate_json(json.dumps(legacy))
        self.assertEqual(historical.schema_version, "1.0")
        self.assertIsNone(historical.case.report)
        self.assertEqual(historical.case.variable_results, {})
        self.assertEqual(historical.corpus, restored.corpus)
        self.assertIsNone(historical.observability.collection_status)
        self.assertEqual(historical.observability.llm_usage_summary.total_tokens, 30)

    def test_premature_real_graph_cannot_claim_completion(self):
        for defect in ("missing_item", "pending_item", "missing_report", "raw_note", "extra_note"):
            with self.subTest(defect=defect):
                agent = _agent(groups=[_group(item_ids=(400, 410))])
                structured = {400: "C349", 410: "2"}
                if defect == "missing_item":
                    initialize = agent.initialize

                    def omit_item(state):
                        update = initialize(state)
                        update["variable_results"].pop(410)
                        return update

                    agent.initialize = omit_item
                elif defect == "pending_item":
                    structured.pop(410)
                    agent.route_from_check = lambda state: "finalize_case"
                elif defect == "missing_report":
                    agent.finalize_case = lambda state: {}
                else:
                    characterize = agent.characterize_corpus

                    def corrupt_corpus(state):
                        update = characterize(state)
                        note = _note() if defect == "raw_note" else ProcessedClinicalNote(**_note("extra").model_dump())
                        update["note_corpus"] = {note.note_id: note}
                        return update

                    agent.characterize_corpus = corrupt_corpus
                agent._graph = agent._build_graph()
                with self.assertRaises(OrchestratorRunError) as raised:
                    agent.run([_note().model_dump()], structured, progress=False)
                error = raised.exception
                self.assertIsInstance(error.__cause__, RuntimeError)
                self.assertIn(
                    "processed input note" if defect in {"raw_note", "extra_note"} else "requested variables were finalized",
                    str(error.__cause__),
                )
                self.assertEqual(error.failure.run.status, "failed")
                self.assertNotIn("case", error.failure.model_dump())

    def _fanout_failure(self, *, during_teardown=False, snapshot_failure=False):
        completed = threading.Event()
        started = threading.Event()
        teardown = threading.Event()
        joined = threading.Event()
        original = RuntimeError("original fanout scanner failure")

        def synchronized_scan(note, schema):
            if schema != "NoteSummary":
                return
            if note.note_id == 7:
                started.set()
                if during_teardown:
                    _wait(teardown, "root loop teardown")
            elif note.note_id == "failed":
                _wait(started if during_teardown else completed, "successful sibling")
                raise original

        model = _ScannerModel(before_call=synchronized_scan)
        agent = _agent(model=model)
        callback = _Completions(completed, teardown, joined)
        observed = []
        original_exit = SyncPregelLoop.__exit__

        def exit_loop(loop, exc_type, exc_value, tb):
            if not loop.is_nested and exc_value is original:
                if during_teardown:
                    self.assertFalse(completed.is_set())
                teardown.set()
                try:
                    return original_exit(loop, exc_type, exc_value, tb)
                finally:
                    joined.set()
            return original_exit(loop, exc_type, exc_value, tb)

        snapshot_error = ValueError("injected telemetry finalization failure")
        original_snapshot = ObservabilityCollector.snapshot

        def snapshot(collector):
            self.assertTrue(joined.is_set())
            self.assertTrue(completed.is_set())
            if snapshot_failure:
                raise snapshot_error
            return original_snapshot(collector)

        with patch.object(SyncPregelLoop, "__exit__", exit_loop), patch.object(
            ObservabilityCollector, "snapshot", autospec=True, side_effect=snapshot
        ) as snapshot_call:
            with self.assertRaises(OrchestratorRunError) as raised:
                # Three workers allow two siblings plus the nested stream waiter.
                agent.run(
                    [_note(7).model_dump(), _note("failed").model_dump()],
                    progress=False, max_concurrency=3,
                    config={"callbacks": [callback]}, event_observer=observed.append,
                )
            snapshot_call.assert_called_once()

        error = raised.exception
        self.assertIs(error.__cause__, original)
        self.assertIn("synchronized_scan", [frame.name for frame in traceback.extract_tb(original.__traceback__)])
        failure = error.failure
        self.assertEqual(failure.schema_version, "1.2")
        self.assertEqual(failure.error, "RuntimeError: original fanout scanner failure")
        self.assertEqual(failure.run.status, "failed")
        self.assertNotIn("case", failure.model_dump())
        self.assertIsNotNone(failure.corpus)
        self.assertEqual(list(failure.corpus.note_corpus), [7])
        note = failure.corpus.note_corpus[7]
        self.assertIsInstance(note, ProcessedClinicalNote)
        self.assertIs(type(note.note_id), int)
        self.assertEqual(callback.notes, [note])
        self.assertEqual(callback.during_teardown, [during_teardown])
        self.assertEqual(note.summary, f"Scanned: {_note(7).content}")
        self.assertEqual(note.flags, ["carcinoma", "pathology"])
        self.assertTrue(note.concepts["cancer"].presence)
        self.assertEqual(note.concepts["cancer"].confidence, "high")
        self.assertEqual(note.concepts["cancer"].evidence, [TextSpan(note_id=7, text=_note(7).content)])
        self.assertEqual(note.cancer_status, {"current"})
        self.assertEqual(note.cancer_mentions[0].evidence, note.concepts["cancer"].evidence)
        self.assertEqual(failure.corpus.note_digests, {})
        self.assertIsNone(failure.corpus.note_corpus_descriptors)
        roots = [event.payload for event in observed if event.kind == "values" and event.is_root]
        self.assertTrue(roots)
        self.assertTrue(all(
            not isinstance(note, ProcessedClinicalNote)
            for state in roots for note in state.get("note_corpus", {}).values()
        ), "The successful scan must not have committed at root fan-in.")
        self.assertFalse(any(event.node == "characterize_corpus" for event in observed))
        self.assertEqual(len(callback.model_starts), 4)
        self.assertTrue(all(metadata["langgraph_checkpoint_ns"] for metadata in callback.model_starts))
        self.assertEqual(model.calls.count(("failed", "NoteSummary")), 1)
        restored = OrchestratorRunFailure.model_validate_json(failure.model_dump_json())
        self.assertEqual(list(restored.corpus.note_corpus.values()), [note])
        self.assertNotIn("case", restored.model_dump())
        if snapshot_failure:
            self.assertIsInstance(failure.observability, RunObservability)
            self.assertEqual(failure.observability.collection_status, "unavailable")
            self.assertIsNone(failure.observability.llm_usage_summary)
            self.assertIsNone(restored.observability.llm_usage_summary)
            self.assertEqual(failure.observability.llm_exchanges, {})
            self.assertEqual(failure.observability.variable_attempts, {})
            self.assertEqual(len(failure.observability.collection_issues), 1)
            issue = failure.observability.collection_issues[0]
            self.assertIsInstance(issue, ObservabilityIssue)
            self.assertEqual(issue.code, "telemetry_finalization_error")
            self.assertIn("ValueError", issue.message)
            self.assertNotIn(str(snapshot_error), failure.error)
        else:
            observation = failure.observability
            self.assertEqual(observation.collection_status, "complete")
            self.assertEqual(observation.collection_issues, [])
            self.assertEqual(observation.unattributed_exchanges, [])
            self.assertEqual(set(observation.llm_exchanges), {"note:7", "note:failed"})
            self.assertEqual([exchange.node for exchange in observation.llm_exchanges["note:7"]],
                             ["summarize_note", "detect_concepts", "get_cancer_mentions"])
            for exchanges in observation.llm_exchanges.values():
                for exchange in exchanges:
                    self.assertEqual(exchange.attempt, 1)
                    self.assertIsNone(exchange.retry_ordinal)
            usage = observation.llm_usage_summary
            self.assertEqual((usage.model_invocations, usage.successful_invocations, usage.failed_invocations), (4, 3, 1))
            self.assertEqual((usage.usage_reported_invocations, usage.missing_usage_invocations, usage.total_tokens), (3, 1, 30))
            self.assertIn(str(original), observation.llm_exchanges["note:failed"][0].error)
        return agent, failure

    def test_successful_sibling_before_root_fanin_is_retained_without_state_leak(self):
        if self._subprocess_guard():
            return
        agent, failure = self._fanout_failure()
        agent._scanner.agent.before_call = None
        result = agent.run([_note("fresh").model_dump()], {400: "C349"}, progress=False)
        self.assertNotEqual(result.run.run_id, failure.run.run_id)
        self.assertEqual(list(result.corpus.note_corpus), ["fresh"])
        self.assertEqual(set(result.observability.llm_exchanges), {"note:fresh"})
        self.assertEqual(result.observability.llm_usage_summary.model_invocations, 3)
        self.assertEqual(result.observability.llm_usage_summary.failed_invocations, 0)
        self.assertEqual(result.case.case_facts.primary_site, "C349")

        def fail_again(note, schema):
            raise RuntimeError("new run failure")

        agent._scanner.agent.before_call = fail_again
        with self.assertRaises(OrchestratorRunError) as raised:
            agent.run([_note("new-failure").model_dump()], progress=False)
        self.assertIsNone(raised.exception.failure.corpus)
        self.assertEqual(set(raised.exception.failure.observability.llm_exchanges), {"note:new-failure"})
        self.assertEqual(raised.exception.failure.observability.llm_usage_summary.model_invocations, 1)

    def test_successful_sibling_completing_during_root_teardown_is_retained(self):
        if self._subprocess_guard():
            return
        self._fanout_failure(during_teardown=True)

    def test_snapshot_failure_is_finalized_once_without_masking_graph_cause(self):
        if self._subprocess_guard():
            return
        self._fanout_failure(during_teardown=True, snapshot_failure=True)


if __name__ == "__main__":
    unittest.main()
