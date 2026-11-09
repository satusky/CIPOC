import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from cipoc.agents.orchestrator import CaseState, OrchestratorAgent
from cipoc.models import (
    CancerMention,
    CaseFacts,
    ConfidenceLevel,
    OrchestratorRunError,
    OrchestratorRunResult,
    ProcessedClinicalNote,
    TargetGroup,
    TextSpan,
    VariableInfo,
    VariableOutput,
)
from cipoc.tools import build_corpus_descriptors, candidate_gross_sites, load_variable_groups, site_applies
from cipoc.utils.progress.events import normalize

from tests.fake_orchestrator import (
    SCOPING_DICTIONARY,
    TISSUE_CODES,
    build_fake_orchestrator,
    graph_input,
    load_notes,
    primary_site_codes,
)


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"
BASE_DICTIONARY = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "manuals"
    / "naaccr_data_dictionary_v25.json"
)


class SiteApplicabilityTests(unittest.TestCase):
    def test_item_832_accepts_coded_breast_primary_site(self):
        group = next(
            group
            for group in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in group.variables)
        )

        self.assertTrue(site_applies(
            group.applies_to, CaseFacts(primary_site="C50.4"), tissue_codes=TISSUE_CODES
        ))
        self.assertFalse(site_applies(
            group.applies_to, CaseFacts(primary_site="C34.9"), tissue_codes=TISSUE_CODES
        ))


class VariableCitationTests(unittest.TestCase):
    def test_primary_citation_preserves_original_scalar_through_json(self):
        for note_id in (1, "001", "note-A", None):
            with self.subTest(note_id=note_id):
                candidate = VariableOutput(
                    item_id=400, value="C504", explanation="Synthetic evidence",
                    most_important_note=note_id, spans=[], presence_confidence="max",
                )
                restored = VariableOutput.model_validate_json(candidate.model_dump_json())
                self.assertEqual(restored.most_important_note, note_id)
                self.assertIs(type(restored.most_important_note), type(note_id))


class CorpusCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.note = ProcessedClinicalNote(
            note_id=1,
            date="2025-02-20",
            note_type="Pathology",
            content="Left breast core biopsy.",
            cancer_mentions=[
                CancerMention(
                    presence=True,
                    confidence=ConfidenceLevel.HIGH,
                    evidence=[TextSpan(note_id=1, text="Left breast core biopsy.")],
                    status="current",
                    affected_tissue="breast",
                    metastasis=False,
                )
            ],
            cancer_status={"current"},
        )
        self.breast_note = self.note.model_copy(update={"note_id": 2})
        self.free_text_note = self.note.model_copy(
            update={
                "note_id": 3,
                "cancer_mentions": [
                    self.note.cancer_mentions[0].model_copy(
                        update={"affected_tissue": "left breast"}
                    )
                ],
            }
        )

    def agent(self, base_dictionary=None):
        if base_dictionary is None:
            directory = TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            base_dictionary = Path(directory.name) / "base.json"
            base_dictionary.write_text(json.dumps({"400": {
                "item_name": "Primary Site", "item_length": 4,
                "allowed_codes": primary_site_codes(),
            }}))
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = base_dictionary
        agent._site_data_dictionary_path = SCOPING_DICTIONARY
        return agent

    def test_affected_tissue_is_kept_as_a_complete_name(self):
        descriptors = build_corpus_descriptors({1: self.note, 3: self.free_text_note})

        self.assertEqual(
            descriptors.affected_tissues, {"current": {"breast", "left breast"}}
        )

    def test_characterization_sets_gross_primary_site_before_planning(self):
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_single_cancer_mention_sets_gross_primary_site(self):
        state = CaseState(note_corpus={1: self.note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_a_name_outside_the_vocabulary_counts_as_unspecified(self):
        for corpus in ({3: self.free_text_note}, {1: self.note, 3: self.free_text_note}):
            with self.subTest(notes=list(corpus)):
                update = self.agent().characterize_corpus(CaseState(note_corpus=corpus))

                self.assertIsNone(update["case_facts"])

    def corpus(self, *mentions):
        """One note holding (status, tissue, behavior, diagnostic_certainty) mentions."""
        template = self.note.cancer_mentions[0]
        return {1: self.note.model_copy(update={"cancer_mentions": [
            template.model_copy(update={
                "status": status, "affected_tissue": tissue,
                "behavior": behavior, "diagnostic_certainty": certainty,
            })
            for status, tissue, behavior, certainty in mentions
        ]})}

    def test_several_tissues_become_candidates_without_a_gross_site(self):
        corpus = self.corpus(
            ("current", "ovary", "malignant", "definite"),
            ("current", "breast", "malignant", "probable"),
        )
        self.assertEqual(candidate_gross_sites(corpus), ["breast", "ovary"])

        facts = self.agent().characterize_corpus(CaseState(note_corpus=corpus))["case_facts"]
        self.assertEqual(facts.candidate_gross_sites, ["breast", "ovary"])
        self.assertIsNone(facts.gross_primary_site)

    def test_one_candidate_also_sets_the_gross_site(self):
        facts = self.agent().characterize_corpus(CaseState(note_corpus={1: self.note}))["case_facts"]
        self.assertEqual(facts.candidate_gross_sites, ["breast"])
        self.assertEqual(facts.gross_primary_site, "breast")

    def test_a_current_benign_tumor_blocks_an_older_cancer(self):
        for behavior in ("benign", "unknown"):
            with self.subTest(behavior=behavior):
                corpus = self.corpus(
                    ("current", "meninges", behavior, "definite"),
                    ("historical", "breast", "malignant", "definite"),
                )
                self.assertEqual(candidate_gross_sites(corpus), [])
                self.assertIsNone(self.agent().characterize_corpus(CaseState(note_corpus=corpus))["case_facts"])

    def test_a_current_possible_mention_doesnt_block_history(self):
        corpus = self.corpus(
            ("current", "lung", "malignant", "possible"),
            ("recent", "breast", "malignant", "definite"),
            ("historical", "ovary", "malignant", "definite"),
        )
        self.assertEqual(candidate_gross_sites(corpus), ["breast"])

    def test_an_unspecified_tissue_or_no_tissue_gives_no_candidates(self):
        self.assertEqual(candidate_gross_sites(self.corpus(
            ("current", "breast", "malignant", "definite"),
            ("current", "unspecified", "malignant", "definite"),
            ("historical", "lung", "malignant", "definite"),
        )), [])
        self.assertEqual(candidate_gross_sites({}), [])

    def test_known_site_facts_are_kept(self):
        for facts in (
            CaseFacts(primary_site="C341"),
            CaseFacts(gross_primary_site="lung"),
            CaseFacts(candidate_gross_sites=["lung", "ovary"]),
        ):
            with self.subTest(facts=facts):
                state = CaseState(note_corpus={1: self.note}, case_facts=facts)
                self.assertIs(self.agent().characterize_corpus(state)["case_facts"], facts)

    def test_characterization_does_not_need_the_site_dictionary(self):
        agent = object.__new__(OrchestratorAgent)
        agent._site_data_dictionary_path = None
        state = CaseState(note_corpus={1: self.note})

        update = agent.characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_initial_group_uses_breast_codes_after_characterization(self):
        if not BASE_DICTIONARY.exists():
            self.skipTest(f"{BASE_DICTIONARY} not present")
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})
        agent = self.agent(BASE_DICTIONARY)
        facts = agent.characterize_corpus(state)["case_facts"]
        initial_group = load_variable_groups(VARIABLE_GROUPS)[0]

        scoped = agent._scope_group(initial_group, facts)
        primary_site = next(
            variable for variable in scoped.variables if variable.item_id == 400
        )

        # Breast's nine codes, plus C809 for an unknown primary.
        self.assertEqual(len(primary_site.valid_codes), 10)
        self.assertIn("C504", primary_site.valid_codes)
        self.assertIn("C809", primary_site.valid_codes)
        self.assertNotIn("C341", primary_site.valid_codes)

    def test_characterization_and_scoping_read_each_dictionary_once(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base_path = Path(directory.name) / "base.json"
        site_path = Path(directory.name) / "site.json"
        base_path.write_text(json.dumps({"400": {
            "item_name": "Primary Site", "item_length": 4,
            "allowed_codes": primary_site_codes(),
        }}))
        site_path.write_text(SCOPING_DICTIONARY.read_text())
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = base_path
        agent._site_data_dictionary_path = site_path
        group = TargetGroup(group_id="initial", variables=[VariableInfo(item_id=400)])

        with patch("builtins.open", wraps=open) as opened:
            for _ in range(2):
                state = CaseState(note_corpus={2: self.breast_note})
                facts = agent.characterize_corpus(state)["case_facts"]
                scoped = agent._scope_group(group, facts)

        for path in (base_path, site_path):
            reads = [call for call in opened.call_args_list if Path(call.args[0]) == path]
            self.assertEqual(len(reads), 1, path.name)
        self.assertEqual(facts.gross_primary_site, "breast")
        self.assertEqual(set(scoped.variables[0].valid_codes), TISSUE_CODES["breast"] | {"C809"})


class OrchestratorRunTests(unittest.TestCase):
    class _Renderer:
        min_interval = 0

        def paint(self, snapshot, **kwargs):
            return True

        def close(self):
            pass

    @staticmethod
    def raw_notes():
        return [note.model_dump(mode="json") for note in load_notes()]

    @staticmethod
    def stable_dump(result):
        value = result.model_dump(mode="json")
        for field in ("run_id", "started_at", "finished_at", "duration_seconds"):
            value["run"].pop(field)
        return value

    def test_run_returns_result_from_full_root_state(self):
        agent = build_fake_orchestrator()

        result = agent.run(self.raw_notes(), structured_data={390: "20250101"}, progress=False)

        self.assertIsInstance(result, OrchestratorRunResult)
        # Re-run the independent full-state stream with the same structured seed.
        expected_with_structured = None
        expected_agent = build_fake_orchestrator()
        for raw_event in expected_agent._graph.stream(
            graph_input(structured_data={390: "20250101"}),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(raw_event, subgraphs=True)
            if event is not None and event.kind == "values" and event.is_root:
                expected_with_structured = event.payload
        self.assertEqual(result.case, CaseState(**expected_with_structured).to_case())
        self.assertEqual(result.inputs.structured_data, {390: "20250101"})
        self.assertEqual(result.inputs.target_variables, agent._target_variables)
        self.assertEqual(len(result.corpus.note_corpus), len(load_notes()))
        self.assertTrue(
            all(
                isinstance(note, ProcessedClinicalNote)
                and note.summary is not None
                and note.concepts
                for note in result.corpus.note_corpus.values()
            )
        )
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())
        restored_note = next(iter(restored.corpus.note_corpus.values()))
        self.assertIsInstance(restored_note, ProcessedClinicalNote)
        self.assertIsNotNone(restored_note.summary)
        self.assertTrue(restored_note.concepts)

    def test_run_info_and_fingerprint_have_identity_and_no_secret(self):
        result = build_fake_orchestrator().run(self.raw_notes(), progress=False)

        self.assertEqual(result.run.run_id.version, 4)
        self.assertLessEqual(result.run.started_at, result.run.finished_at)
        self.assertGreaterEqual(result.run.duration_seconds, 0)
        fingerprint = result.run.config_fingerprint
        self.assertTrue(fingerprint.variable_groups_digest.startswith("sha256:"))
        self.assertTrue(fingerprint.prompt_digests)
        self.assertEqual(fingerprint.max_extraction_attempts, 3)
        serialized = json.dumps(fingerprint.model_dump(mode="json"))
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("fake-api-key", serialized)
        self.assertEqual(
            fingerprint.retry["extractor"]["retry_on"],
            "cipoc.llm.retry.retry_on_transient",
        )

    def test_progress_modes_have_identical_non_identity_result(self):
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=self._Renderer(),
        ):
            displayed = build_fake_orchestrator().run(
                self.raw_notes(), progress=True, pause_before_summary=False
            )
        headless = build_fake_orchestrator().run(self.raw_notes(), progress=False)

        self.assertEqual(self.stable_dump(displayed), self.stable_dump(headless))

    def test_existing_callback_and_concurrency_survive_collector_attachment(self):
        agent = build_fake_orchestrator()
        original_graph = agent._graph

        class RecordingGraph:
            def __init__(self):
                self.kwargs = None

            def stream(self, graph_input, **kwargs):
                self.kwargs = kwargs
                yield from original_graph.stream(graph_input, **kwargs)

        graph = RecordingGraph()
        agent._graph = graph
        existing_callback = BaseCallbackHandler()
        observed = []

        agent.run(
            self.raw_notes(),
            progress=False,
            config={
                "callbacks": [existing_callback],
                "max_concurrency": 7,
                "tags": ["caller"],
            },
            event_observer=observed.append,
        )

        forwarded = graph.kwargs["config"]
        self.assertEqual(forwarded["max_concurrency"], 7)
        self.assertEqual(forwarded["tags"], ["caller"])
        self.assertIs(forwarded["callbacks"][0], existing_callback)
        self.assertEqual(len(forwarded["callbacks"]), 2)
        self.assertTrue(observed)
        self.assertEqual(graph.kwargs["stream_mode"], ["values", "tasks"])
        self.assertTrue(graph.kwargs["subgraphs"])

    def test_content_disabled_and_no_llm_runs_are_explicit(self):
        no_llm_result = build_fake_orchestrator().run(
            self.raw_notes(),
            progress=False,
        )

        self.assertEqual(no_llm_result.observability.llm_exchanges, {})
        self.assertTrue(no_llm_result.observability.variable_attempts)
        self.assertEqual(
            no_llm_result.observability.llm_usage_summary.model_invocations, 0
        )
        self.assertEqual(no_llm_result.observability.llm_usage_summary.total_tokens, 0)

        agent = build_fake_orchestrator()
        final_state = None
        for raw_event in agent._graph.stream(
            graph_input(), stream_mode=["values", "tasks"], subgraphs=True
        ):
            event = normalize(raw_event, subgraphs=True)
            if event is not None and event.kind == "values" and event.is_root:
                final_state = event.payload

        class ObservedGraph:
            def stream(self, graph_input, **kwargs):
                group_scope = ("extract_branch:group",)
                retrieve_scope = (*group_scope, "retrieve_notes:retrieve")
                yield (), "tasks", {
                    "id": "group",
                    "name": "extract_branch",
                    "input": {"requested_variables": {"group_id": "initial"}},
                }
                yield group_scope, "tasks", {
                    "id": "retrieve",
                    "name": "retrieve_notes",
                    "input": {},
                }
                yield retrieve_scope, "tasks", {
                    "id": "llm",
                    "name": "identify_relevant_notes",
                    "input": {},
                }
                callback = kwargs["config"]["callbacks"][-1]
                callback_run_id = uuid4()
                callback.on_chat_model_start(
                    {},
                    [[HumanMessage(content="sensitive prompt")]],
                    run_id=callback_run_id,
                    metadata={
                        "langgraph_node": "identify_relevant_notes",
                        "langgraph_checkpoint_ns": "|".join(
                            (*retrieve_scope, "identify_relevant_notes:llm")
                        ),
                        "ls_model_name": "fake-model",
                    },
                )
                callback.on_llm_end(
                    {
                        "generations": [[{
                            "message": {
                                "content": '{"relevant_note_ids":[50]}',
                                "usage_metadata": {
                                    "input_tokens": 4,
                                    "output_tokens": 2,
                                    "total_tokens": 6,
                                },
                            }
                        }]]
                    },
                    run_id=callback_run_id,
                )
                yield (), "values", final_state

        agent._graph = ObservedGraph()
        result = agent.run(
            self.raw_notes(),
            progress=False,
            capture_llm_content=False,
            max_content_chars=12,
        )

        exchange = result.observability.llm_exchanges["group:initial"][0]
        self.assertFalse(result.observability.llm_content_captured)
        self.assertEqual(result.observability.max_content_chars, 12)
        self.assertIsNone(exchange.prompt_messages)
        self.assertIsNone(exchange.response)
        self.assertEqual(exchange.usage.total_tokens, 6)
        self.assertEqual(result.observability.llm_usage_summary.total_tokens, 6)

    def test_graph_error_raises_with_partial_corpus_and_chained_cause(self):
        agent = build_fake_orchestrator()
        original_graph = agent._graph

        class FailingGraph:
            def stream(self, graph_input, **kwargs):
                for raw_event in original_graph.stream(graph_input, **kwargs):
                    yield raw_event
                    event = normalize(raw_event, subgraphs=True)
                    if (
                        event is not None
                        and event.kind == "values"
                        and event.is_root
                        and event.payload.get("note_corpus_descriptors") is not None
                    ):
                        group_scope = ("extract_branch:group",)
                        retrieve_scope = (*group_scope, "retrieve_notes:retrieve")
                        yield (), "tasks", {
                            "id": "group",
                            "name": "extract_branch",
                            "input": {
                                "requested_variables": {"group_id": "initial"}
                            },
                        }
                        yield group_scope, "tasks", {
                            "id": "retrieve",
                            "name": "retrieve_notes",
                            "input": {},
                        }
                        yield retrieve_scope, "tasks", {
                            "id": "llm",
                            "name": "identify_relevant_notes",
                            "input": {},
                        }
                        callback = kwargs["config"]["callbacks"][-1]
                        callback_run_id = uuid4()
                        callback.on_chat_model_start(
                            {},
                            [[HumanMessage(content="select notes")]],
                            run_id=callback_run_id,
                            metadata={
                                "langgraph_node": "identify_relevant_notes",
                                "langgraph_checkpoint_ns": "|".join(
                                    (*retrieve_scope, "identify_relevant_notes:llm")
                                ),
                            },
                        )
                        callback.on_llm_error(
                            TimeoutError("endpoint failed"), run_id=callback_run_id
                        )
                        raise RuntimeError("scripted graph failure")

        agent._graph = FailingGraph()
        with self.assertRaises(OrchestratorRunError) as raised:
            agent.run(self.raw_notes(), progress=False)

        failure = raised.exception.failure
        self.assertEqual(failure.run.status, "failed")
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn("scripted graph failure", failure.error)
        self.assertIsNotNone(failure.corpus)
        self.assertEqual(len(failure.corpus.note_corpus), len(load_notes()))
        self.assertNotIn("case", failure.model_dump())
        self.assertEqual(failure.observability.llm_usage_summary.model_invocations, 1)
        self.assertEqual(failure.observability.llm_usage_summary.failed_invocations, 1)
        self.assertIn("group:initial", failure.observability.llm_exchanges)

    def test_result_assembly_error_raises_with_failure_artifact(self):
        agent = build_fake_orchestrator()

        with patch.object(
            agent,
            "_corpus_from_state",
            side_effect=ValueError("invalid completed corpus"),
        ):
            with self.assertRaises(OrchestratorRunError) as raised:
                agent.run(self.raw_notes(), progress=False)

        failure = raised.exception.failure
        self.assertEqual(failure.run.status, "failed")
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("invalid completed corpus", failure.error)
        self.assertIsNotNone(failure.corpus)
        self.assertEqual(len(failure.corpus.note_corpus), len(load_notes()))

    def test_run_validates_stream_options(self):
        agent = build_fake_orchestrator()
        with self.assertRaises(ValueError):
            agent.run([], progress=False, max_concurrency=0)
        with self.assertRaises(TypeError):
            agent.run([], progress=False, max_concurrency=True)
        with self.assertRaises(ValueError):
            agent.run([], progress=False, max_content_chars=-1)
        with self.assertRaises(TypeError):
            agent.run([], progress=False, capture_llm_content="yes")
        with self.assertRaises(TypeError):
            agent.run([], structured_data=[], progress=False)


if __name__ == "__main__":
    unittest.main()
