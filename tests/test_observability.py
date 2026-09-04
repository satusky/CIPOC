import ast
import inspect
import threading
import textwrap
import unittest
from typing import TypedDict
from unittest.mock import patch
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from cipoc.agents.extractor import ExtractorAgent
from cipoc.agents.note_retriever import NoteRetrieverAgent
from cipoc.agents.note_scanner import NoteScannerAgent
from cipoc.models import LLMUsageSummary, RunObservability
from cipoc.utils.observability import (
    LLMCaptureHandler,
    ObservabilityCollector,
    aggregate_llm_usage,
    merge_callback_config,
    normalize_token_usage,
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
        self.assertTrue(self.collector.snapshot()["llm_content_captured"])

    def test_metadata_only_never_stores_prompt_or_response_bodies(self):
        collector = ObservabilityCollector(capture_llm_content=False)
        callback = collector.llm_callback
        branch = start((), "note_branch", "note-50", {"note_id": 50})
        model = start(branch.scope, "summarize_note", "llm", {})
        collector.observe(branch)
        collector.observe(model)
        call_metadata = metadata(model, model="metadata-model")

        failed, succeeded = uuid4(), uuid4()
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content="sensitive prompt")]],
            run_id=failed,
            metadata=call_metadata,
        )
        callback.on_llm_error(TimeoutError("throttled"), run_id=failed)
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content="another sensitive prompt")]],
            run_id=succeeded,
            metadata=call_metadata,
        )
        callback.on_llm_end(
            llm_result(
                AIMessage(
                    content='{"secret":"response body"}',
                    usage_metadata={
                        "input_tokens": 9,
                        "output_tokens": 4,
                        "total_tokens": 13,
                    },
                )
            ),
            run_id=succeeded,
        )

        self.assertNotIn("prompt_messages", callback._calls[0])
        self.assertNotIn("response", callback._calls[0])
        self.assertNotIn("prompt_messages", callback._calls[1])
        self.assertNotIn("response", callback._calls[1])
        self.assertNotIn("sensitive prompt", repr(callback._calls))
        self.assertNotIn("response body", repr(callback._calls))

        snapshot = collector.snapshot()
        exchanges = snapshot["llm_exchanges"]["note:50"]
        self.assertFalse(snapshot["llm_content_captured"])
        self.assertFalse(snapshot["content_truncated"])
        self.assertEqual(exchanges[0]["error"], "TimeoutError: throttled")
        self.assertEqual(exchanges[1]["retry_ordinal"], 1)
        self.assertEqual(exchanges[1]["model"], "metadata-model")
        self.assertEqual(
            exchanges[1]["usage"],
            {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
        )
        self.assertTrue(
            all(
                "prompt_messages" not in exchange and "response" not in exchange
                for exchange in exchanges
            )
        )
        RunObservability.model_validate(snapshot)

    def test_bounded_capture_truncates_prompts_only_with_explicit_metadata(self):
        collector = ObservabilityCollector(
            capture_llm_content=True, max_content_chars=5
        )
        callback = collector.llm_callback
        branch = start((), "note_branch", "note-50", {"note_id": 50})
        model = start(branch.scope, "summarize_note", "llm", {})
        collector.observe(branch)
        collector.observe(model)
        run_id = uuid4()
        response = {"summary": "this response remains exact"}

        callback.on_chat_model_start(
            {},
            [[SystemMessage(content="short"), HumanMessage(content="0123456789")]],
            run_id=run_id,
            metadata=metadata(model),
        )
        callback.on_llm_end(
            llm_result(AIMessage(content='{"summary":"this response remains exact"}')),
            run_id=run_id,
        )

        snapshot = collector.snapshot()
        exchange = snapshot["llm_exchanges"]["note:50"][0]
        self.assertEqual(snapshot["max_content_chars"], 5)
        self.assertTrue(snapshot["content_truncated"])
        self.assertEqual(
            exchange["prompt_messages"],
            [
                {
                    "role": "system",
                    "content": "short",
                    "truncated": False,
                    "original_char_count": 5,
                },
                {
                    "role": "human",
                    "content": "01234",
                    "truncated": True,
                    "original_char_count": 10,
                },
            ],
        )
        self.assertEqual(exchange["response"], response)
        RunObservability.model_validate(snapshot)

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

    def test_callback_prefers_usage_metadata_and_fills_only_missing_details(self):
        model = self.scanner_call(node="detect_concepts")
        run_id = uuid4()
        self.callback.on_chat_model_start(
            {}, [[HumanMessage(content="detect")]], run_id=run_id,
            metadata=metadata(model),
        )
        self.callback.on_llm_end(
            {
                "generations": [[{
                    "message": {
                        "content": '{"cancer":true}',
                        "usage_metadata": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                            "input_token_details": {"cache_read": 3},
                            "output_token_details": {"reasoning": 2},
                        },
                        "response_metadata": {
                            "token_usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 40,
                                "total_tokens": 140,
                                "prompt_tokens_details": {
                                    "cached_tokens": 3,
                                    "cache_creation_tokens": 1,
                                },
                                "completion_tokens_details": {
                                    "reasoning_tokens": 2,
                                    "accepted_prediction_tokens": 1,
                                },
                            }
                        },
                    }
                }]]
            },
            run_id=run_id,
        )

        exchange = self.collector.snapshot()["llm_exchanges"]["note:50"][0]
        self.assertEqual(exchange["usage"]["input_tokens"], 10)
        self.assertEqual(exchange["usage"]["output_tokens"], 4)
        self.assertEqual(
            exchange["usage"]["input_token_details"],
            {"cache_read": 3, "cache_creation": 1},
        )
        self.assertEqual(
            exchange["usage"]["output_token_details"],
            {"reasoning": 2, "accepted_prediction": 1},
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


class LLMUsageTests(unittest.TestCase):
    def test_normalizes_scalar_provider_shapes(self):
        cases = [
            (
                {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
                (8, 3, 11),
            ),
            (
                {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                (9, 4, 13),
            ),
            (
                {"input_token_count": 7, "output_token_count": 2},
                (7, 2, 9),
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                usage = normalize_token_usage(source)
                self.assertIsNotNone(usage)
                self.assertEqual(
                    (usage.input_tokens, usage.output_tokens, usage.total_tokens),
                    expected,
                )

    def test_details_are_recursive_canonical_and_unknown_fields_survive(self):
        usage = normalize_token_usage(
            {
                "input_tokens": 100,
                "output_tokens": 30,
                "total_tokens": 130,
                "input_token_details": {
                    "cache_read": 40,
                    "nested": {"cache_creation": 7, "vendor_input": 3},
                    "audio": 2,
                },
                "output_token_details": {
                    "reasoning": 20,
                    "audio": 1,
                    "accepted_prediction": 4,
                    "rejected_prediction": 2,
                    "nested": {"vendor_output": 5},
                },
            }
        )

        self.assertEqual(
            usage.input_token_details.root,
            {
                "cache_read": 40,
                "cache_creation": 7,
                "vendor_input": 3,
                "audio": 2,
            },
        )
        self.assertEqual(
            usage.output_token_details.root,
            {
                "reasoning": 20,
                "audio": 1,
                "accepted_prediction": 4,
                "rejected_prediction": 2,
                "vendor_output": 5,
            },
        )

    def test_authoritative_usage_avoids_duplicate_metadata_views(self):
        usage = normalize_token_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cache_read": 30},
                "output_token_details": {"reasoning": 10},
            },
            {
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "total_tokens": 1998,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "cache_creation_tokens": 6,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 10,
                    "accepted_prediction_tokens": 3,
                    "rejected_prediction_tokens": 1,
                },
            },
        )

        self.assertEqual(
            (usage.input_tokens, usage.output_tokens, usage.total_tokens),
            (100, 20, 120),
        )
        self.assertEqual(
            usage.input_token_details.root,
            {"cache_read": 30, "cache_creation": 6},
        )
        self.assertEqual(
            usage.output_token_details.root,
            {
                "reasoning": 10,
                "accepted_prediction": 3,
                "rejected_prediction": 1,
            },
        )

    def test_aggregates_counts_tokens_details_and_dimension_buckets(self):
        exchanges = {
            "group:initial/variable:390": [
                {
                    "entity_key": "group:initial/variable:390",
                    "agent": "extractor",
                    "node": "repair_invalid_extraction",
                    "attempt": 2,
                    "model": "gpt-test",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "input_token_details": {"cache_read": 3},
                        "output_token_details": {"reasoning": 2},
                    },
                    "error": "TimeoutError: transient",
                },
                {
                    "entity_key": "group:initial/variable:390",
                    "agent": "extractor",
                    "node": "repair_invalid_extraction",
                    "attempt": 2,
                    "retry_ordinal": 1,
                    "model": "gpt-test",
                    "usage": None,
                    "error": None,
                },
            ],
            "note:50": [
                {
                    "entity_key": "note:50",
                    "agent": "note_scanner",
                    "node": "summarize_note",
                    "attempt": 1,
                    "model": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "error": None,
                }
            ],
        }

        summary = aggregate_llm_usage(exchanges)

        self.assertIsInstance(summary, LLMUsageSummary)
        self.assertEqual(summary.logical_calls, 2)
        self.assertEqual(summary.model_invocations, 3)
        self.assertEqual(summary.retry_invocations, 1)
        self.assertEqual(summary.successful_invocations, 2)
        self.assertEqual(summary.failed_invocations, 1)
        self.assertEqual(summary.usage_reported_invocations, 2)
        self.assertEqual(summary.missing_usage_invocations, 1)
        self.assertEqual(
            (summary.input_tokens, summary.output_tokens, summary.total_tokens),
            (10, 4, 14),
        )
        self.assertEqual(summary.input_token_details.root, {"cache_read": 3})
        self.assertEqual(summary.output_token_details.root, {"reasoning": 2})
        self.assertEqual(summary.by_agent["extractor"].model_invocations, 2)
        self.assertEqual(
            summary.by_node["repair_invalid_extraction"].logical_calls, 1
        )
        self.assertEqual(summary.by_model["gpt-test"].retry_invocations, 1)
        self.assertEqual(summary.by_model["unknown"].model_invocations, 1)

    def test_snapshot_uses_one_call_snapshot_for_exchanges_and_summary(self):
        collector = ObservabilityCollector(capture_llm_content=False)
        branch = start((), "note_branch", "note-50", {"note_id": 50})
        model = start(branch.scope, "summarize_note", "llm", {})
        collector.observe(branch)
        collector.observe(model)
        run_id = uuid4()
        collector.llm_callback.on_chat_model_start(
            {}, [[HumanMessage(content="not retained")]], run_id=run_id,
            metadata=metadata(model),
        )
        collector.llm_callback.on_llm_end(
            llm_result(
                AIMessage(
                    content="not retained",
                    usage_metadata={
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                )
            ),
            run_id=run_id,
        )

        with patch.object(
            collector.llm_callback,
            "_snapshot",
            wraps=collector.llm_callback._snapshot,
        ) as captured_snapshot:
            snapshot = collector.snapshot()

        captured_snapshot.assert_called_once_with()
        self.assertEqual(len(snapshot["llm_exchanges"]["note:50"]), 1)
        self.assertEqual(snapshot["llm_usage_summary"]["model_invocations"], 1)
        self.assertEqual(snapshot["llm_usage_summary"]["total_tokens"], 3)
        self.assertNotIn(
            "prompt_messages", snapshot["llm_exchanges"]["note:50"][0]
        )
        RunObservability.model_validate(snapshot)


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

    def test_content_is_opt_in_but_callback_attachment_is_unconditional(self):
        collector = ObservabilityCollector()
        config = {"max_concurrency": 3}

        self.assertIsInstance(collector.llm_callback, LLMCaptureHandler)
        merged = collector.graph_config(config)
        self.assertEqual(merged["max_concurrency"], 3)
        self.assertEqual(merged["callbacks"], [collector.llm_callback])
        snapshot = collector.snapshot()
        self.assertFalse(snapshot["llm_content_captured"])
        self.assertEqual(snapshot["llm_exchanges"], {})


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
    fail_times: int = 1

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
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


class _RepairLoopState(TypedDict):
    requested_variables: dict
    task: dict
    remaining_repairs: int


class LLMNodeInvocationInvariantTests(unittest.TestCase):
    def test_each_production_llm_node_has_one_structured_call_site(self):
        llm_nodes = {
            "detect_concepts": NoteScannerAgent.detect_concepts,
            "summarize_note": NoteScannerAgent.summarize_note,
            "get_cancer_mentions": NoteScannerAgent.get_cancer_mentions,
            "identify_relevant_notes": NoteRetrieverAgent.identify_relevant_notes,
            "extract_group_values": ExtractorAgent.extract_group_values,
            "extract_individual_value": ExtractorAgent.extract_individual_value,
            "repair_invalid_extraction": ExtractorAgent.repair_invalid_extraction,
        }

        for node, method in llm_nodes.items():
            with self.subTest(node=node):
                tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
                structured_calls = [
                    candidate
                    for candidate in ast.walk(tree)
                    if isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and candidate.func.attr == "structured"
                ]
                self.assertEqual(
                    len(structured_calls),
                    1,
                    f"{node} must make exactly one model invocation per graph task",
                )


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

    def test_graph_loop_reentry_uses_a_new_checkpoint_namespace(self):
        model = _DeterministicChatModel(model_name="repair-model")

        def repair(state):
            model.invoke([HumanMessage(content="repair")])
            task = dict(state["task"])
            task["extraction_attempts"] += 1
            return {
                "task": task,
                "remaining_repairs": state["remaining_repairs"] - 1,
            }

        def route(state):
            if state["remaining_repairs"]:
                return "repair_invalid_extraction"
            return END

        repair_builder = StateGraph(_RepairLoopState)
        repair_builder.add_node("repair_invalid_extraction", repair)
        repair_builder.add_edge(START, "repair_invalid_extraction")
        repair_builder.add_conditional_edges("repair_invalid_extraction", route)

        builder = StateGraph(_RepairLoopState)
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
            "remaining_repairs": 3,
        }
        repair_starts = []
        for item in graph.stream(
            graph_input,
            config=collector.graph_config(),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(item, subgraphs=True)
            if event is None:
                continue
            collector.observe(event)
            if event.kind == "task_start" and event.node == "repair_invalid_extraction":
                repair_starts.append(event)

        captured_calls = collector.llm_callback._snapshot()
        namespaces = [call.namespace for call in captured_calls]
        self.assertEqual(len(repair_starts), 3)
        self.assertEqual(len(captured_calls), len(repair_starts))
        self.assertEqual(namespaces, [event.scope for event in repair_starts])
        self.assertEqual(len(set(namespaces)), 3)
        self.assertEqual(
            [call.transport_retry_ordinal for call in captured_calls],
            [None, None, None],
        )

        exchanges = collector.snapshot()["llm_exchanges"][
            "group:initial/variable:390"
        ]
        self.assertEqual([call["attempt"] for call in exchanges], [2, 3, 4])
        self.assertTrue(all("retry_ordinal" not in call for call in exchanges))
        summary = collector.snapshot()["llm_usage_summary"]
        self.assertEqual(summary["logical_calls"], 3)
        self.assertEqual(summary["model_invocations"], 3)
        self.assertEqual(summary["retry_invocations"], 0)

    def test_langgraph_retry_preserves_semantic_attempt_and_marks_transport_retry(self):
        model = _FlakyChatModel(model_name="flaky-model", fail_times=2)

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
                max_attempts=3,
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
        repair_starts = []
        for item in graph.stream(
            graph_input,
            config=collector.graph_config(),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(item, subgraphs=True)
            if event is not None:
                collector.observe(event)
                if (
                    event.kind == "task_start"
                    and event.node == "repair_invalid_extraction"
                ):
                    repair_starts.append(event)

        exchanges = collector.snapshot()["llm_exchanges"][
            "group:initial/variable:390"
        ]
        captured_calls = collector.llm_callback._snapshot()
        self.assertEqual(len(repair_starts), 1)
        self.assertEqual(model.calls, 3)
        self.assertEqual(len(captured_calls), 3)
        self.assertEqual(
            [call.namespace for call in captured_calls],
            [repair_starts[0].scope] * 3,
        )
        self.assertEqual(
            [call.transport_retry_ordinal for call in captured_calls],
            [None, 1, 2],
        )
        self.assertEqual([call["attempt"] for call in exchanges], [2, 2, 2])
        self.assertNotIn("retry_ordinal", exchanges[0])
        self.assertEqual(exchanges[1]["retry_ordinal"], 1)
        self.assertEqual(exchanges[2]["retry_ordinal"], 2)
        self.assertEqual(exchanges[0]["error"], "TimeoutError: transient")
        self.assertEqual(exchanges[1]["error"], "TimeoutError: transient")
        self.assertEqual(exchanges[2]["response"], {"ok": True})
        summary = collector.snapshot()["llm_usage_summary"]
        self.assertEqual(summary["logical_calls"], 1)
        self.assertEqual(summary["model_invocations"], 3)
        self.assertEqual(summary["retry_invocations"], 2)
        self.assertEqual(summary["failed_invocations"], 2)
        self.assertEqual(summary["successful_invocations"], 1)


if __name__ == "__main__":
    unittest.main()
