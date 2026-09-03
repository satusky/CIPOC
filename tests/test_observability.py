import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from scripts import run_case_state as exporter
from cipoc.models import ProcessedClinicalNote
from cipoc.utils.observability import (
    LLMCaptureHandler,
    ObservabilityCollector,
    merge_callback_config,
)
from cipoc.utils.progress.events import ProgressEvent, normalize
from cipoc.utils.progress.model import ProgressModel
from cipoc.utils.progress.runner import run_with_progress
from tests.fake_orchestrator import (
    Outcome,
    Script,
    build_fake_orchestrator,
    load_notes,
    record_events,
)


def start(namespace, node, task_id, payload):
    return ProgressEvent(
        kind="task_start",
        namespace=namespace,
        node=node,
        task_id=task_id,
        payload=payload,
    )


def end(namespace, node, task_id, payload):
    return ProgressEvent(
        kind="task_end",
        namespace=namespace,
        node=node,
        task_id=task_id,
        payload=payload,
    )


def llm_result(message, *, llm_output=None):
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output=llm_output,
    )


def metadata(event, *, model="test-model"):
    return {
        "langgraph_node": event.node,
        "langgraph_checkpoint_ns": "|".join(event.scope),
        "ls_model_name": model,
    }


class _Graph:
    def stream(self, graph_input, **kwargs):
        yield "tasks", {"id": "one", "name": "initialize", "input": {}}
        yield "tasks", {
            "id": "one",
            "name": "initialize",
            "result": {},
            "error": None,
        }
        yield "values", {"answer": 1}


class _Renderer:
    min_interval = 0.01

    def paint(self, snapshot, **kwargs):
        return True

    def close(self):
        pass


class _ObservedGraph:
    def __init__(self):
        self.calls = []

    def stream(self, graph_input, **kwargs):
        self.calls.append(kwargs)
        group_scope = ("extract_branch:group",)
        retrieve_scope = (*group_scope, "retrieve_notes:retrieve")
        extract_scope = (*group_scope, "extract:extract")
        variable_scope = (*extract_scope, "variable_branch:variable")
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
            "id": "retriever",
            "name": "identify_relevant_notes",
            "input": {"requested_variables": {"group_id": "initial"}},
        }
        callbacks = kwargs.get("config", {}).get("callbacks", [])
        callback = next(
            (item for item in callbacks if isinstance(item, LLMCaptureHandler)), None
        )
        if callback is not None:
            run_id = uuid4()
            callback.on_chat_model_start(
                {},
                [[HumanMessage(content="select notes")]],
                run_id=run_id,
                metadata={
                    "langgraph_node": "identify_relevant_notes",
                    "langgraph_checkpoint_ns": "|".join(
                        (*retrieve_scope, "identify_relevant_notes:retriever")
                    ),
                    "ls_model_name": "retriever-test",
                },
            )
            callback.on_llm_end(
                llm_result(AIMessage(content='{"note_ids":[1,999]}')),
                run_id=run_id,
            )
        yield retrieve_scope, "tasks", {
            "id": "retriever",
            "name": "identify_relevant_notes",
            "result": {"relevant_note_ids": [1]},
            "error": None,
        }
        yield group_scope, "tasks", {
            "id": "extract",
            "name": "extract",
            "input": {},
        }
        yield extract_scope, "tasks", {
            "id": "variable",
            "name": "variable_branch",
            "input": {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "individual",
                    "extraction_attempts": 0,
                }
            },
        }
        yield variable_scope, "tasks", {
            "id": "validate",
            "name": "validate_extraction",
            "result": {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "individual",
                    "extraction_attempts": 1,
                    "candidate": None,
                    "validation_errors": ["missing"],
                    "is_valid": False,
                }
            },
            "error": None,
        }
        yield (), "values", {
            "answer": 1,
            "note_selection": {
                "group:initial": {
                    "group_id": "initial",
                    "requested_item_ids": [390],
                    "candidate_note_ids": [1],
                    "rejected_note_ids": {
                        2: ["note_type_mismatch", "cancer_status_mismatch"]
                    },
                    "selected_note_ids": [1],
                    "unevaluated_checks": ["keyword_filter_disabled"],
                }
            },
        }
        yield ("late-child:child",), "values", {"answer": "not-root"}


class _ObservedAgent:
    def __init__(self, graph):
        self.compiled_graph = graph
        self._config = SimpleNamespace(
            documents=lambda: SimpleNamespace(variable_groups_path="unused.json")
        )


class ObserverTests(unittest.TestCase):
    def test_runner_observes_each_event_before_model_ingestion(self):
        order = []
        original_ingest = ProgressModel.ingest

        def ingest(model, event, now):
            order.append(("model", event.kind))
            return original_ingest(model, event, now)

        with (
            patch(
                "cipoc.utils.progress.runner._select_renderer",
                return_value=_Renderer(),
            ),
            patch.object(ProgressModel, "ingest", ingest),
        ):
            result = run_with_progress(
                _Graph(),
                {},
                event_observer=lambda event: order.append(("observer", event.kind)),
            )

        self.assertEqual(result, {"answer": 1})
        self.assertEqual(
            order,
            [
                ("observer", "task_start"),
                ("model", "task_start"),
                ("observer", "task_end"),
                ("model", "task_end"),
                ("observer", "values"),
                ("model", "values"),
            ],
        )

    def test_exporter_feeds_identical_events_with_and_without_progress(self):
        note = {
            "note_id": 1,
            "date": "2026-01-01",
            "note_type": "test",
            "content": "",
        }
        results = []
        graphs = []
        groups = [
            {
                "group_id": "initial",
                "name": "Initial",
                "variables": [{"item_id": 390, "name": "Diagnosis Date"}],
            }
        ]
        for progress in (True, False):
            graph = _ObservedGraph()
            graphs.append(graph)
            with (
                patch.object(exporter, "OrchestratorAgent", return_value=_ObservedAgent(graph)),
                patch.object(exporter, "load_variable_groups", return_value=groups),
                patch.object(exporter, "load_group_hierarchy", return_value=[]),
                patch(
                    "cipoc.utils.progress.runner._select_renderer",
                    return_value=_Renderer(),
                ),
            ):
                results.append(
                    exporter.run_case_state(
                        [note], progress=progress, max_concurrency=4
                    )
                )

        self.assertEqual(results[0], results[1])
        self.assertEqual(
            results[0]["variable_attempts"]["group:initial/variable:390"][0]["mode"],
            "individual",
        )
        self.assertEqual(results[0]["answer"], 1)
        self.assertEqual(
            results[0]["llm_exchanges"]["group:initial"][0]["response"],
            {"note_ids": [1, 999]},
        )
        self.assertEqual(
            results[0]["note_selection"]["group:initial"],
            {
                "group_id": "initial",
                "candidate_note_ids": [1],
                "filtered_out": {
                    "2": (
                        "Note type did not match the configured note filter. "
                        "Cancer status did not match the configured note filter."
                    )
                },
                "selected_note_ids": [1],
                "retriever_offered": [1, 999],
            },
        )
        for graph in graphs:
            self.assertEqual(graph.calls[0]["stream_mode"], ["values", "tasks"])
            self.assertTrue(graph.calls[0]["subgraphs"])
            self.assertIn("callbacks", graph.calls[0]["config"])
            self.assertEqual(graph.calls[0]["config"]["max_concurrency"], 4)

    def test_exporter_can_disable_llm_capture_without_other_observability(self):
        note = {
            "note_id": 1,
            "date": "2026-01-01",
            "note_type": "test",
            "content": "",
        }
        graph = _ObservedGraph()
        groups = [
            {
                "group_id": "initial",
                "name": "Initial",
                "variables": [{"item_id": 390, "name": "Diagnosis Date"}],
            }
        ]
        with (
            patch.object(exporter, "OrchestratorAgent", return_value=_ObservedAgent(graph)),
            patch.object(exporter, "load_variable_groups", return_value=groups),
            patch.object(exporter, "load_group_hierarchy", return_value=[]),
        ):
            result = exporter.run_case_state(
                [note], progress=False, capture_llm=False, max_concurrency=3
            )

        self.assertNotIn("llm_exchanges", result)
        self.assertIn("variable_attempts", result)
        self.assertNotIn(
            "retriever_offered", result["note_selection"]["group:initial"]
        )
        self.assertEqual(graph.calls[0]["config"], {"max_concurrency": 3})

    def test_exporter_parser_enables_llm_capture_unless_disabled(self):
        parser = exporter.build_parser()

        self.assertFalse(parser.parse_args([]).no_llm_capture)
        self.assertTrue(parser.parse_args(["--no-llm-capture"]).no_llm_capture)

    def test_serializer_preserves_live_models_object_keys_sets_and_scan_fields(self):
        processed = ProcessedClinicalNote(
            **load_notes()[0].model_dump(), summary="scanned", flags=["flag"]
        )

        serialized = exporter.to_jsonable(
            {7: {"values": {"b", "a"}}, "note_corpus": {processed.note_id: processed}}
        )

        self.assertEqual(serialized["7"]["values"], ["a", "b"])
        self.assertIn("summary", serialized["note_corpus"][str(processed.note_id)])
        self.assertEqual(
            serialized["note_corpus"][str(processed.note_id)]["summary"], "scanned"
        )

    def test_fake_orchestrator_export_matches_complete_workbench_contract(self):
        raw_notes = [note.model_dump(mode="json") for note in load_notes()]
        results = []
        for progress in (True, False):
            agent = build_fake_orchestrator(
                Script(outcomes={390: Outcome(repairs=1)})
            )
            with (
                patch.object(exporter, "OrchestratorAgent", return_value=agent),
                patch(
                    "cipoc.utils.progress.runner._select_renderer",
                    return_value=_Renderer(),
                ),
            ):
                results.append(
                    exporter.run_case_state(
                        raw_notes, progress=progress, capture_llm=True
                    )
                )

        for channel in (
            "case_facts",
            "variable_results",
            "note_selection",
            "variable_attempts",
            "llm_exchanges",
        ):
            self.assertEqual(results[0][channel], results[1][channel])
        result = results[0]

        self.assertIn("note_corpus", result)
        self.assertIn("llm_exchanges", result)
        self.assertIn("note_selection", result)
        self.assertIn("variable_attempts", result)
        # The deterministic fake executes LLM-named graph nodes without calling a
        # model, so the channel exists but correctly contains no exchanges.
        self.assertEqual(result["llm_exchanges"], {})
        self.assertTrue(result["note_selection"])
        for selection in result["note_selection"].values():
            self.assertEqual(
                set(selection),
                {
                    "group_id",
                    "candidate_note_ids",
                    "filtered_out",
                    "selected_note_ids",
                },
            )
            self.assertTrue(
                all(
                    isinstance(reason, str)
                    for reason in selection["filtered_out"].values()
                )
            )
        attempts = result["variable_attempts"][
            "group:initial_llm_extraction/variable:390"
        ]
        self.assertEqual([attempt["mode"] for attempt in attempts], ["group", "repair"])
        self.assertEqual(
            set(attempts[0]),
            {"attempt", "mode", "candidate", "validation_errors", "is_valid"},
        )


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.collector = ObservabilityCollector()

    def observe(self, *events):
        for event in events:
            self.collector.observe(event)

    def test_namespace_bindings_cover_scanner_retriever_and_extraction_modes(self):
        note = start((), "note_branch", "note-branch", {"note_id": 50})
        scanner = start(
            note.scope,
            "summarize_note",
            "summary",
            {"note": {"note_id": 50}},
        )

        group = start(
            (),
            "extract_branch",
            "group-branch",
            {"requested_variables": {"group_id": "initial"}},
        )
        retrieve = start(group.scope, "retrieve_notes", "retrieve", {})
        retriever = start(
            retrieve.scope,
            "identify_relevant_notes",
            "identify",
            {"requested_variables": {"group_id": "initial"}},
        )
        extract = start(group.scope, "extract", "extract", {})
        grouped = start(
            extract.scope,
            "extract_group_values",
            "grouped",
            {"requested_variables": {"group_id": "initial"}},
        )
        variable = start(
            extract.scope,
            "variable_branch",
            "variable",
            {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "group",
                    "extraction_attempts": 1,
                }
            },
        )
        repair = start(
            variable.scope,
            "repair_invalid_extraction",
            "repair",
            {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "group",
                    "extraction_attempts": 1,
                }
            },
        )
        individual_variable = start(
            extract.scope,
            "variable_branch",
            "individual-variable",
            {
                "task": {
                    "variable": {"item_id": 400},
                    "extraction_mode": "individual",
                    "extraction_attempts": 0,
                }
            },
        )
        individual = start(
            individual_variable.scope,
            "extract_individual_value",
            "individual",
            {
                "task": {
                    "variable": {"item_id": 400},
                    "extraction_mode": "individual",
                    "extraction_attempts": 0,
                }
            },
        )
        self.observe(
            note,
            scanner,
            group,
            retrieve,
            retriever,
            extract,
            grouped,
            variable,
            repair,
            individual_variable,
            individual,
        )

        self.assertEqual(
            self.collector.binding_for(scanner.scope).entity_key, "note:50"
        )
        self.assertEqual(
            self.collector.binding_for(retriever.scope).entity_key, "group:initial"
        )
        self.assertEqual(
            self.collector.binding_for(grouped.scope).entity_key, "group:initial"
        )
        self.assertEqual(
            self.collector.binding_for(grouped.scope).semantic_attempt, 1
        )
        self.assertEqual(
            self.collector.binding_for(repair.scope).entity_key,
            "group:initial/variable:390",
        )
        self.assertEqual(
            self.collector.binding_for(repair.scope).semantic_attempt, 2
        )
        self.assertEqual(
            self.collector.binding_for(individual.scope).entity_key,
            "group:initial/variable:400",
        )
        self.assertEqual(
            self.collector.binding_for(individual.scope).semantic_attempt, 1
        )

        retry = self.collector.binding_for(repair.scope, transport_retry_ordinal=1)
        self.assertEqual(retry.transport_retry_ordinal, 1)
        self.assertEqual(retry.semantic_attempt, 2)

    def test_each_validation_end_records_one_attempt_with_correct_mode(self):
        group = start(
            (),
            "extract_branch",
            "group",
            {"requested_variables": {"group_id": "initial"}},
        )
        extract = start(group.scope, "extract", "extract", {})
        grouped_variable = start(
            extract.scope,
            "variable_branch",
            "grouped-variable",
            {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "group",
                    "extraction_attempts": 1,
                }
            },
        )
        individual_variable = start(
            extract.scope,
            "variable_branch",
            "individual-variable",
            {
                "task": {
                    "variable": {"item_id": 400},
                    "extraction_mode": "individual",
                    "extraction_attempts": 0,
                }
            },
        )
        self.observe(group, extract, grouped_variable, individual_variable)

        def result(item_id, extraction_mode, attempt, candidate, errors, is_valid):
            return {
                "task": {
                    "variable": {"item_id": item_id},
                    "extraction_mode": extraction_mode,
                    "extraction_attempts": attempt,
                    "candidate": candidate,
                    "validation_errors": errors,
                    "is_valid": is_valid,
                }
            }

        self.observe(
            end(
                grouped_variable.scope,
                "validate_extraction",
                "validate-1",
                result(390, "group", 1, None, ["missing"], False),
            ),
            end(
                grouped_variable.scope,
                "validate_extraction",
                "validate-2",
                result(390, "group", 2, {"item_id": 390, "value": "1"}, [], True),
            ),
            end(
                individual_variable.scope,
                "validate_extraction",
                "validate-3",
                result(
                    400,
                    "individual",
                    1,
                    {"item_id": 400, "value": "C509"},
                    [],
                    True,
                ),
            ),
        )

        attempts = self.collector.snapshot()["variable_attempts"]
        self.assertEqual(
            attempts["group:initial/variable:390"],
            [
                {
                    "attempt": 1,
                    "mode": "group",
                    "candidate": None,
                    "validation_errors": ["missing"],
                    "is_valid": False,
                },
                {
                    "attempt": 2,
                    "mode": "repair",
                    "candidate": {"item_id": 390, "value": "1"},
                    "validation_errors": [],
                    "is_valid": True,
                },
            ],
        )
        self.assertEqual(
            attempts["group:initial/variable:400"][0]["mode"], "individual"
        )
        self.assertEqual(sum(map(len, attempts.values())), 3)

    def test_real_nested_stream_records_each_validation_result_once(self):
        raw_events = record_events(
            Script(
                outcomes={
                    390: Outcome(repairs=1),
                    670: Outcome(repairs=1),
                }
            )
        )
        validation_results = 0
        for raw_event in raw_events:
            _, stream_mode, payload = raw_event
            if (
                stream_mode == "tasks"
                and payload.get("name") == "validate_extraction"
                and "input" not in payload
                and payload.get("result") is not None
            ):
                validation_results += 1
            event = normalize(raw_event, subgraphs=True)
            if event is not None:
                self.collector.observe(event)

        attempts = self.collector.snapshot()["variable_attempts"]
        self.assertEqual(sum(map(len, attempts.values())), validation_results)
        self.assertEqual(
            [
                attempt["mode"]
                for attempt in attempts[
                    "group:initial_llm_extraction/variable:390"
                ]
            ],
            ["group", "repair"],
        )
        self.assertEqual(
            [
                attempt["mode"]
                for attempt in attempts[
                    "group:site_specific_codes/variable:670"
                ]
            ],
            ["individual", "repair"],
        )


class LLMCaptureTests(unittest.TestCase):
    def setUp(self):
        self.collector = ObservabilityCollector(capture_llm=True)
        self.callback = self.collector.llm_callback
        self.assertIsInstance(self.callback, LLMCaptureHandler)

    def scanner_call(self, *, note_id=50, node="summarize_note", task_id="llm"):
        branch = start((), "note_branch", f"note-{note_id}", {"note_id": note_id})
        model = start(branch.scope, node, task_id, {})
        self.collector.observe(branch)
        self.collector.observe(model)
        return model

    def test_success_captures_prompt_parsed_response_model_and_normalized_usage(self):
        model = self.scanner_call()
        run_id = uuid4()
        self.callback.on_chat_model_start(
            {},
            [[SystemMessage(content="system"), HumanMessage(content="summarize")]],
            run_id=run_id,
            metadata=metadata(model, model="gpt-test"),
        )
        self.callback.on_llm_end(
            llm_result(
                AIMessage(
                    content='{"summary":"visible"}',
                    usage_metadata={
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "total_tokens": 11,
                    },
                )
            ),
            run_id=run_id,
        )

        exchange = self.collector.snapshot()["llm_exchanges"]["note:50"][0]
        self.assertEqual(exchange["agent"], "note_scanner")
        self.assertEqual(exchange["node"], "summarize_note")
        self.assertEqual(exchange["attempt"], 1)
        self.assertEqual(
            exchange["prompt_messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "human", "content": "summarize"},
            ],
        )
        self.assertEqual(exchange["response"], {"summary": "visible"})
        self.assertEqual(exchange["model"], "gpt-test")
        self.assertEqual(
            exchange["usage"],
            {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        )
        self.assertIsNone(exchange["error"])
        self.assertNotIn("retry_ordinal", exchange)

    def test_tool_call_arguments_are_the_structured_response(self):
        model = self.scanner_call(node="detect_concepts")
        run_id = uuid4()
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="detect")]], run_id=run_id, metadata=metadata(model)
        )
        self.callback.on_llm_end(
            llm_result(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ConceptFindings",
                            "args": {"cancer": {"presence": True}},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ),
            run_id=run_id,
        )

        exchange = self.collector.snapshot()["llm_exchanges"]["note:50"][0]
        self.assertEqual(exchange["response"], {"cancer": {"presence": True}})

    def test_raw_openai_and_databricks_shapes_are_supported_without_reasoning(self):
        model = self.scanner_call(node="detect_concepts")
        run_id = uuid4()
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="detect")]], run_id=run_id, metadata=metadata(model)
        )
        self.callback.on_llm_end(
            {
                "generations": [[{
                    "message": {
                        "content": [
                            {"type": "reasoning", "reasoning": "private"},
                            {"type": "text", "text": '{"cancer":true}'},
                        ],
                        "response_metadata": {
                            "model_name": "dbx-model",
                            "token_usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 4,
                            },
                        },
                    }
                }]]
            },
            run_id=run_id,
        )

        exchange = self.collector.snapshot()["llm_exchanges"]["note:50"][0]
        self.assertEqual(exchange["response"], {"cancer": True})
        self.assertNotIn("private", str(exchange))
        self.assertEqual(exchange["model"], "dbx-model")
        self.assertEqual(
            exchange["usage"],
            {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        )

    def test_raw_openai_tool_call_arguments_are_parsed(self):
        model = self.scanner_call(node="detect_concepts")
        run_id = uuid4()
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="detect")]], run_id=run_id, metadata=metadata(model)
        )
        self.callback.on_llm_end(
            {
                "generations": [[{
                    "message": {
                        "content": None,
                        "additional_kwargs": {
                            "tool_calls": [{
                                "function": {
                                    "name": "ConceptFindings",
                                    "arguments": '{"cancer":{"presence":true}}',
                                }
                            }]
                        },
                    }
                }]],
                "llm_output": {
                    "usage": {
                        "input_token_count": 9,
                        "output_token_count": 2,
                        "total_token_count": 11,
                    }
                },
            },
            run_id=run_id,
        )

        exchange = self.collector.snapshot()["llm_exchanges"]["note:50"][0]
        self.assertEqual(exchange["response"], {"cancer": {"presence": True}})
        self.assertEqual(exchange["usage"]["total_tokens"], 11)

    def test_failed_transport_retry_keeps_one_semantic_attempt(self):
        group = start(
            (),
            "extract_branch",
            "group",
            {"requested_variables": {"group_id": "initial"}},
        )
        extract = start(group.scope, "extract", "extract", {})
        variable = start(
            extract.scope,
            "variable_branch",
            "variable",
            {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "group",
                    "extraction_attempts": 1,
                }
            },
        )
        repair = start(
            variable.scope,
            "repair_invalid_extraction",
            "repair",
            {
                "task": {
                    "variable": {"item_id": 390},
                    "extraction_mode": "group",
                    "extraction_attempts": 1,
                }
            },
        )
        for event in (group, extract, variable, repair):
            self.collector.observe(event)

        first, second = uuid4(), uuid4()
        call_metadata = metadata(repair)
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="repair")]], run_id=first, metadata=call_metadata
        )
        self.callback.on_llm_error(TimeoutError("throttled"), run_id=first)
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="repair")]], run_id=second, metadata=call_metadata
        )
        self.callback.on_llm_end(
            llm_result(AIMessage(content='{"item_id":390,"value":"1"}')),
            run_id=second,
        )

        exchanges = self.collector.snapshot()["llm_exchanges"][
            "group:initial/variable:390"
        ]
        self.assertEqual([call["attempt"] for call in exchanges], [2, 2])
        self.assertNotIn("retry_ordinal", exchanges[0])
        self.assertEqual(exchanges[1]["retry_ordinal"], 1)
        self.assertEqual(exchanges[0]["error"], "TimeoutError: throttled")
        self.assertIsNone(exchanges[1]["error"])

    def test_concurrent_completions_preserve_invocation_start_order(self):
        branch = start((), "note_branch", "note", {"note_id": 50})
        first_event = start(branch.scope, "summarize_note", "first", {})
        second_event = start(branch.scope, "detect_concepts", "second", {})
        for event in (branch, first_event, second_event):
            self.collector.observe(event)

        first_started = threading.Event()
        second_finished = threading.Event()

        def first_call():
            run_id = uuid4()
            self.callback.on_chat_model_start(
                {}, [[HumanMessage(content="first")]], run_id=run_id,
                metadata=metadata(first_event),
            )
            first_started.set()
            second_finished.wait(1)
            self.callback.on_llm_end(
                llm_result(AIMessage(content='{"order":1}')), run_id=run_id
            )

        def second_call():
            first_started.wait(1)
            run_id = uuid4()
            self.callback.on_chat_model_start(
                {}, [[HumanMessage(content="second")]], run_id=run_id,
                metadata=metadata(second_event),
            )
            self.callback.on_llm_end(
                llm_result(AIMessage(content='{"order":2}')), run_id=run_id
            )
            second_finished.set()

        threads = [threading.Thread(target=first_call), threading.Thread(target=second_call)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)

        exchanges = self.collector.snapshot()["llm_exchanges"]["note:50"]
        self.assertEqual([call["response"]["order"] for call in exchanges], [1, 2])


class CallbackConfigTests(unittest.TestCase):
    def test_merge_preserves_callbacks_and_max_concurrency(self):
        existing = BaseCallbackHandler()
        capture = LLMCaptureHandler()
        original = {"callbacks": [existing], "max_concurrency": 7, "tags": ["case"]}

        merged = merge_callback_config(original, capture)

        self.assertEqual(merged["max_concurrency"], 7)
        self.assertEqual(merged["tags"], ["case"])
        self.assertEqual(merged["callbacks"], [existing, capture])
        self.assertEqual(original["callbacks"], [existing])

    def test_merge_copies_an_existing_callback_manager(self):
        existing = BaseCallbackHandler()
        capture = LLMCaptureHandler()
        manager = CallbackManager.configure(inheritable_callbacks=[existing])

        merged = merge_callback_config({"callbacks": manager}, capture)

        self.assertIsNot(merged["callbacks"], manager)
        self.assertIn(existing, merged["callbacks"].inheritable_handlers)
        self.assertIn(capture, merged["callbacks"].inheritable_handlers)
        self.assertNotIn(capture, manager.inheritable_handlers)

    def test_capture_is_opt_in(self):
        collector = ObservabilityCollector()
        config = {"max_concurrency": 3}

        self.assertIsNone(collector.llm_callback)
        self.assertEqual(collector.graph_config(config), config)
        self.assertNotIn("llm_exchanges", collector.snapshot())


class _CountingHandler(BaseCallbackHandler):
    def __init__(self):
        self.starts = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts += 1


class _DeterministicChatModel(BaseChatModel):
    model_name: str

    @property
    def _llm_type(self):
        return "deterministic-test"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content='{"ok":true}'))]
        )


class _FlakyChatModel(_DeterministicChatModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("transient")
        return super()._generate(messages, stop, run_manager, **kwargs)


class _ModelWrapper:
    def __init__(self, name):
        self.model = _DeterministicChatModel(model_name=name)

    def invoke(self):
        return self.model.invoke([HumanMessage(content="visible prompt")])


def _model_graph(node, wrapper):
    graph = StateGraph(dict)

    def call_model(state):
        wrapper.invoke()
        return state

    graph.add_node(node, call_model)
    graph.add_edge(START, node)
    graph.add_edge(node, END)
    return graph.compile()


class CallbackInheritanceTests(unittest.TestCase):
    def test_one_root_callback_reaches_nested_graphs_with_separate_wrappers(self):
        scanner = _model_graph("summarize_note", _ModelWrapper("scanner-model"))
        retriever = _model_graph(
            "identify_relevant_notes", _ModelWrapper("retriever-model")
        )
        extractor = _model_graph(
            "extract_group_values", _ModelWrapper("extractor-model")
        )

        branch = StateGraph(dict)

        def retrieve(state):
            retriever.invoke({"requested_variables": state["requested_variables"]})
            return state

        def extract(state):
            extractor.invoke({"requested_variables": state["requested_variables"]})
            return state

        branch.add_node("retrieve_notes", retrieve)
        branch.add_node("extract", extract)
        branch.add_edge(START, "retrieve_notes")
        branch.add_edge("retrieve_notes", "extract")
        branch.add_edge("extract", END)
        branch_graph = branch.compile()

        root = StateGraph(dict)

        def scan(state):
            scanner.invoke({"note_id": state["note_id"]})
            return state

        root.add_node("note_branch", scan)
        root.add_node("extract_branch", branch_graph)
        root.add_edge(START, "note_branch")
        root.add_edge("note_branch", "extract_branch")
        root.add_edge("extract_branch", END)
        graph = root.compile()

        collector = ObservabilityCollector(capture_llm=True)
        existing = _CountingHandler()
        config = collector.graph_config(
            {"callbacks": [existing], "max_concurrency": 4}
        )
        graph_input = {
            "note_id": 50,
            "requested_variables": {"group_id": "initial"},
        }
        for item in graph.stream(
            graph_input,
            config=config,
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(item, subgraphs=True)
            if event is not None:
                collector.observe(event)

        exchanges = collector.snapshot()["llm_exchanges"]
        self.assertEqual(existing.starts, 3)
        self.assertEqual(
            [call["agent"] for call in exchanges["note:50"]],
            ["note_scanner"],
        )
        self.assertEqual(
            [call["agent"] for call in exchanges["group:initial"]],
            ["note_retriever", "extractor"],
        )
        self.assertEqual(
            [call["model"] for calls in exchanges.values() for call in calls],
            ["scanner-model", "retriever-model", "extractor-model"],
        )

    def test_langgraph_retry_preserves_semantic_attempt_and_marks_transport_retry(self):
        model = _FlakyChatModel(model_name="flaky-model")

        def repair(state):
            model.invoke([HumanMessage(content="repair")])
            return state

        repair_builder = StateGraph(dict)
        repair_builder.add_node(
            "repair_invalid_extraction",
            repair,
            retry_policy=RetryPolicy(
                initial_interval=0.0,
                backoff_factor=1.0,
                max_interval=0.0,
                max_attempts=2,
                retry_on=lambda error: True,
            ),
        )
        repair_builder.add_edge(START, "repair_invalid_extraction")
        repair_builder.add_edge("repair_invalid_extraction", END)

        builder = StateGraph(dict)
        builder.add_node("variable_branch", repair_builder.compile())
        builder.add_edge(START, "variable_branch")
        builder.add_edge("variable_branch", END)
        graph = builder.compile()

        collector = ObservabilityCollector(capture_llm=True)
        graph_input = {
            "requested_variables": {"group_id": "initial"},
            "task": {
                "variable": {"item_id": 390},
                "extraction_mode": "group",
                "extraction_attempts": 1,
            },
        }
        for item in graph.stream(
            graph_input,
            config=collector.graph_config(),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(item, subgraphs=True)
            if event is not None:
                collector.observe(event)

        exchanges = collector.snapshot()["llm_exchanges"][
            "group:initial/variable:390"
        ]
        self.assertEqual(model.calls, 2)
        self.assertEqual([call["attempt"] for call in exchanges], [2, 2])
        self.assertNotIn("retry_ordinal", exchanges[0])
        self.assertEqual(exchanges[1]["retry_ordinal"], 1)
        self.assertEqual(exchanges[0]["error"], "TimeoutError: transient")
        self.assertEqual(exchanges[1]["response"], {"ok": True})


if __name__ == "__main__":
    unittest.main()
