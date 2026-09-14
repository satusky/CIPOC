"""Offline regressions for remediation-plan fixes 9 and 17."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
from typing import Any, Callable, TypedDict
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._executor import BackgroundExecutor
from pydantic import Field, PrivateAttr, ValidationError

from cipoc.agents.extractor import ExtractorAgent, VariableExtractionTask
from cipoc.models import ClinicalNote, VariableInfo, VariableOutput
from cipoc.models.observability import LLMUsageBucket, RunObservability
from cipoc.utils import CipocConfig
from cipoc.utils.observability import ObservabilityCollector, aggregate_llm_usage
from cipoc.utils.progress.events import ProgressEvent, normalize


ENTITY = "group:remediation/variable:390"
ALIAS = "synthetic-deployment"
RESOLVED = "synthetic-model-2026-09"
PROMPT = "synthetic sensitive prompt"


class _Throttle(RuntimeError):
    status_code = 429


class _SiblingFailure(RuntimeError):
    pass


def _candidate(value):
    return {
        "item_id": 390,
        "value": value,
        "explanation": "synthetic parsed content",
        "most_important_note": "note-A",
        "spans": [{"note_id": "note-A", "text": "Synthetic evidence"}],
        "presence_confidence": "high",
    }


class _ScriptedChatModel(BaseChatModel):
    """Use real LangChain start/end/error callbacks, not hand-emitted lifecycles."""

    model_name: str = ALIAS
    replies: list[Any]
    calls: int = 0
    before_call: Callable[[int], None] | None = Field(default=None, exclude=True)
    _lock: Any = PrivateAttr(default_factory=Lock)

    @property
    def _llm_type(self):
        return "observability-remediation-fake"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._lock:
            index = self.calls
            self.calls += 1
        if self.before_call is not None:
            self.before_call(index)
        reply = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="",
            tool_calls=[{
                "name": "VariableOutput", "args": reply,
                "id": f"synthetic-tool-{index}", "type": "tool_call",
            }],
            response_metadata={"model_name": RESOLVED},
            usage_metadata={
                "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                "input_token_details": {"cache_read": 3},
                "output_token_details": {"reasoning": 2},
            },
        ))])


class _StructuredModel:
    def __init__(self, model):
        self.model = model

    def structured(self, schema, messages):
        response = self.model.invoke(messages)
        return schema.model_validate(response.tool_calls[0]["args"])


class _RepairState(TypedDict):
    requested_variables: dict
    task: VariableExtractionTask
    notes: list
    messages: list
    max_extraction_attempts: int
    variable_results: list


def _repair_graph(model, *, before_validate=None, sibling=None):
    agent = ExtractorAgent(llm=_StructuredModel(model), config=CipocConfig({
        "llm": {
            "model": ALIAS, "api_key": "synthetic",
            "base_url": "https://example.invalid/v1",
            "retry": {
                "initial_interval": 0.0, "max_interval": 0.0,
                "backoff_factor": 1.0, "jitter": False, "max_attempts": 2,
            },
        },
    }))
    validate = agent.validate_extraction

    def validate_with_gate(state):
        if before_validate is not None:
            before_validate(state)
        return validate(state)

    with patch.object(agent, "validate_extraction", validate_with_gate):
        branch = agent._build_variable_branch()
    graph = StateGraph(_RepairState)
    graph.add_node("variable_branch", branch)
    graph.add_edge(START, "variable_branch")
    graph.add_edge("variable_branch", END)
    if sibling is not None:
        graph.add_node("failing_sibling", sibling)
        graph.add_edge(START, "failing_sibling")
        graph.add_edge("failing_sibling", END)
    return graph.compile(), {
        "requested_variables": {"group_id": "remediation"},
        "task": VariableExtractionTask(
            variable=VariableInfo(item_id=390, valid_codes={"1": "Present"}),
            extraction_mode="group", extraction_attempts=1,
            candidate=VariableOutput.model_validate(_candidate("9")),
        ),
        "notes": [ClinicalNote(
            note_id="note-A", date="2026-01-01", note_type="pathology",
            content="Synthetic evidence in an offline note.",
        )],
        "messages": [HumanMessage(content=PROMPT)],
        "max_extraction_attempts": 4,
        "variable_results": [],
    }


def _unbound_call(collector):
    # A known ancestor is deliberately insufficient for binding this repair.
    ancestor = ProgressEvent(
        kind="task_start", namespace=(), node="extract_group_values",
        task_id="ancestor", payload={"requested_variables": {"group_id": "orphan"}},
    )
    collector.observe(ancestor)
    namespace = (*ancestor.scope, "repair_invalid_extraction:missing-task")
    model = _ScriptedChatModel(replies=[_candidate("1")])
    model.invoke([HumanMessage(content=PROMPT)], config=collector.graph_config({
        "metadata": {
            "langgraph_node": "repair_invalid_extraction",
            "langgraph_checkpoint_ns": "|".join(namespace),
        },
    }))
    return namespace


def _cleanup_probe():
    """Run in a deadline-limited child; the real executor still performs cleanup."""
    validation_entered = Event()
    cleanup_entered = Event()
    failure = _SiblingFailure("synthetic sibling failed")
    validations = []
    calls_in_cleanup = []
    streamed_repairs = []
    streamed_validations = []

    def before_validate(state):
        validations.append(state.task.extraction_attempts)
        if state.task.extraction_attempts == 2:
            validation_entered.set()
            if not cleanup_entered.wait(5):
                raise AssertionError("Root executor did not enter failure cleanup")

    def sibling(state):
        if not validation_entered.wait(5):
            raise AssertionError("Repair validation did not reach the gate")
        raise failure

    model = _ScriptedChatModel(
        replies=[_candidate("9"), _Throttle("retry in cleanup"),
                 _candidate("9"), _candidate("1")],
        before_call=lambda index: calls_in_cleanup.append(cleanup_entered.is_set()),
    )
    graph, graph_input = _repair_graph(
        model, before_validate=before_validate, sibling=sibling,
    )
    collector = ObservabilityCollector(capture_llm_content=True)
    original_exit = BackgroundExecutor.__exit__

    def enter_cleanup(executor, exc_type, exc_value, traceback):
        # Release only after stream iteration has raised into executor teardown.
        # Merely observing the sibling's error callback would still be too early.
        if exc_value is failure:
            cleanup_entered.set()
        return original_exit(executor, exc_type, exc_value, traceback)

    with patch.object(BackgroundExecutor, "__exit__", enter_cleanup):
        try:
            for raw in graph.stream(
                graph_input, config=collector.graph_config({"max_concurrency": 4}),
                stream_mode=["values", "tasks"], subgraphs=True,
            ):
                if cleanup_entered.is_set():
                    raise AssertionError("Stream delivery continued during cleanup")
                event = normalize(raw, subgraphs=True)
                if event is None:
                    continue
                collector.observe(event)
                if event.node == "repair_invalid_extraction" and event.kind == "task_start":
                    streamed_repairs.append(list(event.scope))
                if event.node == "validate_extraction" and event.kind == "task_end":
                    streamed_validations.append(list(event.scope))
        except _SiblingFailure as error:
            if error is not failure:
                raise AssertionError("Original graph error was replaced") from error
        else:
            raise AssertionError("Sibling failure was swallowed")
    return {
        "snapshot": collector.snapshot(), "repeated_snapshot": collector.snapshot(),
        "validations": validations, "calls_in_cleanup": calls_in_cleanup,
        "streamed_repairs": streamed_repairs,
        "streamed_validations": streamed_validations,
    }


class ObservabilityRemediationTests(unittest.TestCase):
    def assert_json_valid(self, snapshot):
        encoded = json.dumps(snapshot, allow_nan=False)
        parsed = RunObservability.model_validate_json(encoded)
        self.assertEqual(
            RunObservability.model_validate_json(parsed.model_dump_json()), parsed,
        )
        return parsed

    def assert_additive(self, summary):
        for dimension in ("by_agent", "by_node", "by_model"):
            buckets = summary[dimension].values()
            for name in LLMUsageBucket.model_fields:
                with self.subTest(dimension=dimension, metric=name):
                    if name.endswith("_token_details"):
                        total = Counter()
                        for bucket in buckets:
                            total.update(bucket[name])
                        self.assertEqual(dict(total), summary[name])
                    else:
                        self.assertEqual(sum(bucket[name] for bucket in buckets), summary[name])

    def test_alias_failure_resolved_version_retry_has_additive_retry_only_bucket(self):
        model = _ScriptedChatModel(replies=[_Throttle("deployment throttled"), _candidate("1")])
        graph, graph_input = _repair_graph(model)
        collector = ObservabilityCollector(capture_llm_content=True)
        graph.invoke(graph_input, config=collector.graph_config({"max_concurrency": 4}))

        snapshot = collector.snapshot()
        self.assert_json_valid(snapshot)
        self.assertEqual(snapshot["collection_status"], "complete")
        self.assertEqual(snapshot["collection_issues"], [])
        calls = snapshot["llm_exchanges"][ENTITY]
        self.assertEqual(model.calls, 2)
        self.assertEqual([call["model"] for call in calls], [ALIAS, RESOLVED])
        self.assertEqual([call["attempt"] for call in calls], [2, 2])
        self.assertEqual([call.get("retry_ordinal") for call in calls], [None, 1])
        self.assertEqual(calls[0]["namespace"], calls[1]["namespace"])
        self.assertNotEqual(calls[0]["invocation_id"], calls[1]["invocation_id"])
        self.assertEqual(calls[0]["error"], "_Throttle: deployment throttled")
        self.assertIsNone(calls[0]["usage"])
        self.assertEqual(calls[1]["response"], _candidate("1"))

        summary = snapshot["llm_usage_summary"]
        self.assertEqual(
            [summary[name] for name in (
                "logical_calls", "model_invocations", "retry_invocations",
                "failed_invocations", "successful_invocations",
                "missing_usage_invocations", "usage_reported_invocations",
                "input_tokens", "output_tokens", "total_tokens",
            )],
            [1, 2, 1, 1, 1, 1, 1, 10, 4, 14],
        )
        self.assertEqual(summary["by_model"][ALIAS]["logical_calls"], 1)
        resolved = summary["by_model"][RESOLVED]
        self.assertEqual(resolved["logical_calls"], 0)
        self.assertEqual(resolved["retry_invocations"], 1)
        self.assertEqual(resolved["model_invocations"], 1)
        self.assertEqual(resolved["input_token_details"], {"cache_read": 3})
        self.assertEqual(resolved["output_token_details"], {"reasoning": 2})
        self.assert_additive(summary)

    def test_callback_only_nested_repairs_preserve_exact_attempts_and_retries(self):
        model = _ScriptedChatModel(replies=[
            _candidate("9"), _Throttle("transport retry"), _candidate("9"), _candidate("1"),
        ])
        graph, graph_input = _repair_graph(model)
        collector = ObservabilityCollector(capture_llm_content=True)
        result = graph.invoke(graph_input, config=collector.graph_config({"max_concurrency": 4}))
        self.assertTrue(result["variable_results"][0].is_valid)
        self.assertEqual(result["variable_results"][0].extraction_attempts, 4)
        self.assert_repair_snapshot(collector.snapshot())

    def assert_repair_snapshot(self, snapshot):
        self.assert_json_valid(snapshot)
        self.assertEqual(snapshot["collection_status"], "complete")
        self.assertEqual(snapshot["collection_issues"], [])
        self.assertEqual(snapshot["unattributed_exchanges"], [])
        self.assertEqual(set(snapshot["llm_exchanges"]), {ENTITY})
        calls = snapshot["llm_exchanges"][ENTITY]
        self.assertEqual([call["attempt"] for call in calls], [2, 3, 3, 4])
        self.assertEqual([call.get("retry_ordinal") for call in calls], [None, None, 1, None])
        self.assertEqual(len({call["invocation_id"] for call in calls}), 4)
        self.assertEqual(len({tuple(call["namespace"]) for call in calls}), 3)
        self.assertEqual(calls[1]["namespace"], calls[2]["namespace"])
        self.assertEqual(set(snapshot["variable_attempts"]), {ENTITY})
        attempts = snapshot["variable_attempts"][ENTITY]
        self.assertEqual([attempt["attempt"] for attempt in attempts], [1, 2, 3, 4])
        self.assertEqual([attempt["mode"] for attempt in attempts], ["group", "repair", "repair", "repair"])
        self.assertEqual([attempt["is_valid"] for attempt in attempts], [False, False, False, True])
        self.assertEqual([attempt["candidate"]["value"] for attempt in attempts], ["9", "9", "9", "1"])
        self.assertTrue(all(attempt["validation_errors"] for attempt in attempts[:-1]))
        self.assertEqual(attempts[-1]["validation_errors"], [])
        summary = snapshot["llm_usage_summary"]
        self.assertEqual(summary["logical_calls"], 3)
        self.assertEqual(summary["model_invocations"], 4)
        self.assertEqual(summary["retry_invocations"], 1)
        self.assertEqual(summary["failed_invocations"], 1)
        self.assertEqual(summary["successful_invocations"], 3)
        self.assertEqual(summary["total_tokens"], 42)
        self.assert_additive(summary)

    def test_stream_and_callbacks_do_not_duplicate_variable_validations(self):
        model = _ScriptedChatModel(replies=[
            _candidate("9"), _Throttle("transport retry"), _candidate("9"), _candidate("1"),
        ])
        graph, graph_input = _repair_graph(model)
        collector = ObservabilityCollector(capture_llm_content=True)
        streamed = []
        with patch.object(collector.llm_callback, "_task_observer", wraps=collector.observe) as callback:
            for raw in graph.stream(
                graph_input, config=collector.graph_config({"max_concurrency": 4}),
                stream_mode=["values", "tasks"], subgraphs=True,
            ):
                event = normalize(raw, subgraphs=True)
                if event is not None:
                    collector.observe(event)
                    if event.node == "validate_extraction" and event.kind == "task_end":
                        streamed.append(event.scope)
        callback_scopes = [
            call.args[0].scope for call in callback.call_args_list
            if call.args[0].node == "validate_extraction" and call.args[0].kind == "task_end"
        ]
        self.assertEqual(len(streamed), 4)
        self.assertEqual(callback_scopes, streamed)
        self.assert_repair_snapshot(collector.snapshot())

    def test_sibling_failure_retains_two_later_repairs_during_cleanup(self):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "tests.test_observability_remediation", "--cleanup-worker"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"Nested failure-cleanup graph exceeded 30s deadline: {error}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        probe = json.loads(result.stdout)
        self.assertEqual(probe["calls_in_cleanup"], [False, True, True, True])
        self.assertEqual(probe["validations"], [1, 2, 3, 4])
        self.assertEqual(len(probe["streamed_repairs"]), 1)
        self.assertEqual(len(probe["streamed_validations"]), 1)
        snapshot = probe["snapshot"]
        self.assertEqual(snapshot, probe["repeated_snapshot"])
        self.assert_repair_snapshot(snapshot)
        calls = snapshot["llm_exchanges"][ENTITY]
        self.assertEqual(probe["streamed_repairs"], [calls[0]["namespace"]])
        for call in calls[1:]:
            self.assertNotIn(call["namespace"], probe["streamed_repairs"])

    def test_task_identity_is_distinct_from_semantic_identity(self):
        def invocation(task, suffix, ordinal=None):
            return {
                "entity_key": ENTITY, "agent": "extractor", "node": "repair_invalid_extraction",
                "attempt": 2, "namespace": [f"repair_invalid_extraction:{task}"],
                "invocation_id": f"{task}-{suffix}", "retry_ordinal": ordinal,
                "model": RESOLVED if ordinal else ALIAS,
                "error": None if ordinal else "throttled",
            }

        calls = [invocation("a", "initial"), invocation("b", "initial"),
                 invocation("b", "retry", 1), invocation("a", "retry", 1)]
        summary = aggregate_llm_usage(calls).model_dump(mode="json")
        self.assertEqual(summary["logical_calls"], 2)
        self.assertEqual(summary["retry_invocations"], 2)
        self.assertEqual(summary["by_model"][RESOLVED]["logical_calls"], 0)
        self.assertEqual(summary["by_model"][RESOLVED]["retry_invocations"], 2)
        self.assert_additive(summary)

        corruptions = {
            "duplicate invocation ID across tasks": [calls[0], {**calls[1], "invocation_id": calls[0]["invocation_id"]}],
            "missing initial invocation": [calls[3]],
            "missing retry": [calls[0], {**calls[3], "retry_ordinal": 2}],
            "duplicate initial in same task": [calls[0], {**calls[0], "invocation_id": "extra-initial"}],
            "repeated retry": [calls[0], calls[3], {**calls[3], "invocation_id": "extra-retry"}],
        }
        for reason, records in corruptions.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "[Ii]nvocation|sequence"):
                aggregate_llm_usage(records)

    def test_invalid_sequences_retain_exchanges_but_report_null_summary(self):
        model = _ScriptedChatModel(replies=[_Throttle("deployment throttled"), _candidate("1")])
        graph, graph_input = _repair_graph(model)
        collector = ObservabilityCollector(capture_llm_content=True)
        graph.invoke(graph_input, config=collector.graph_config({"max_concurrency": 4}))
        calls = collector.llm_callback._snapshot()
        corruptions = {
            "duplicate identity": [calls[0], replace(calls[1], run_id=calls[0].run_id)],
            "missing start": [calls[1]],
            "missing retry": [calls[0], replace(calls[1], transport_retry_ordinal=2)],
            "repeated retry": [*calls, replace(calls[1], run_id="extra-retry")],
        }
        for reason, records in corruptions.items():
            with self.subTest(reason=reason), patch.object(collector.llm_callback, "_snapshot", return_value=records):
                snapshot = collector.snapshot()
                self.assertEqual(snapshot["collection_status"], "partial")
                self.assertIsNone(snapshot["llm_usage_summary"])
                self.assertEqual(len(snapshot["llm_exchanges"][ENTITY]), len(records))
                self.assertIn("invalid_invocation_sequence", {issue["code"] for issue in snapshot["collection_issues"]})
                self.assertTrue(all(issue["message"] for issue in snapshot["collection_issues"]))
                self.assert_json_valid(snapshot)
                self.assertEqual(collector.snapshot(), snapshot)
        self.assertEqual(collector.snapshot()["collection_status"], "complete")

    def test_unbound_invocation_is_retained_without_fabricated_attempt(self):
        collector = ObservabilityCollector(capture_llm_content=True)
        namespace = _unbound_call(collector)
        snapshot = collector.snapshot()
        self.assert_json_valid(snapshot)
        self.assertEqual(snapshot["collection_status"], "partial")
        self.assertEqual(snapshot["llm_exchanges"], {})
        self.assertEqual(snapshot["variable_attempts"], {})
        self.assertEqual(len(snapshot["unattributed_exchanges"]), 1)
        call = snapshot["unattributed_exchanges"][0]
        self.assertEqual(call["namespace"], list(namespace))
        self.assertTrue(call["invocation_id"])
        self.assertNotIn("attempt", call)
        self.assertNotIn("entity_key", call)
        self.assertEqual(call["response"], _candidate("1"))
        self.assertEqual(call["model"], RESOLVED)
        self.assertEqual(snapshot["llm_usage_summary"]["model_invocations"], 1)
        self.assertEqual(snapshot["llm_usage_summary"]["total_tokens"], 14)
        self.assertEqual([issue["code"] for issue in snapshot["collection_issues"]], ["missing_task_binding"])
        self.assertIn(call["invocation_id"], snapshot["collection_issues"][0]["message"])

    def test_content_disabled_omits_bound_and_unbound_prompt_and_parsed_content(self):
        collector = ObservabilityCollector(capture_llm_content=False, max_content_chars=3)
        model = _ScriptedChatModel(replies=[_Throttle("deployment throttled"), _candidate("1")])
        graph, graph_input = _repair_graph(model)
        graph.invoke(graph_input, config=collector.graph_config({"max_concurrency": 4}))
        _unbound_call(collector)
        snapshot = collector.snapshot()
        self.assert_json_valid(snapshot)
        self.assertFalse(snapshot["content_truncated"])
        calls = snapshot["llm_exchanges"][ENTITY] + snapshot["unattributed_exchanges"]
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertNotIn("prompt_messages", call)
            self.assertNotIn("response", call)
        # Clinical validation candidates are still retained, so inspect callback
        # storage and exchanges rather than claiming the entire artifact is PHI-free.
        for retained in (collector.llm_callback._calls, calls):
            encoded = json.dumps(retained)
            self.assertNotIn(PROMPT, encoded)
            self.assertNotIn("synthetic parsed content", encoded)
        self.assertEqual(snapshot["llm_usage_summary"]["model_invocations"], 3)
        self.assertEqual(snapshot["llm_usage_summary"]["total_tokens"], 28)

    def test_unbound_truncation_is_counted_and_validated(self):
        collector = ObservabilityCollector(capture_llm_content=True, max_content_chars=3)
        _unbound_call(collector)
        snapshot = collector.snapshot()
        self.assert_json_valid(snapshot)
        self.assertTrue(snapshot["content_truncated"])
        call = snapshot["unattributed_exchanges"][0]
        self.assertEqual(call["prompt_messages"], [{
            "role": "human", "content": PROMPT[:3], "truncated": True,
            "original_char_count": len(PROMPT),
        }])
        self.assertEqual(call["response"], _candidate("1"))
        for field, value in (("content_truncated", False), ("max_content_chars", None),
                             ("max_content_chars", 2), ("llm_content_captured", False)):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                RunObservability.model_validate({**snapshot, field: value})
        for field, value in (("original_char_count", 3), ("truncated", False)):
            malformed = deepcopy(snapshot)
            malformed["unattributed_exchanges"][0]["prompt_messages"][0][field] = value
            with self.subTest(message_field=field), self.assertRaises(ValidationError):
                RunObservability.model_validate(malformed)

    def test_repeated_snapshots_are_stable_and_deeply_detached(self):
        collector = ObservabilityCollector(capture_llm_content=True, max_content_chars=3)
        graph, graph_input = _repair_graph(_ScriptedChatModel(replies=[_candidate("1")]))
        graph.invoke(graph_input, config=collector.graph_config({"max_concurrency": 4}))
        _unbound_call(collector)
        expected = collector.snapshot()
        detached = collector.snapshot()
        self.assertEqual(expected, detached)
        for call in (detached["llm_exchanges"][ENTITY][0], detached["unattributed_exchanges"][0]):
            call["namespace"].append("mutated")
            call["prompt_messages"][0]["content"] = "mutated"
            call["response"]["spans"][0]["text"] = "mutated"
            call["usage"]["input_token_details"]["cache_read"] = 999
        detached["variable_attempts"][ENTITY][0]["candidate"]["spans"].clear()
        detached["variable_attempts"][ENTITY][0]["validation_errors"].append("mutated")
        detached["collection_issues"][0]["message"] = "mutated"
        detached["llm_usage_summary"]["by_model"][RESOLVED]["total_tokens"] = 999
        self.assertEqual(collector.snapshot(), expected)
        self.assert_json_valid(collector.snapshot())


class ObservabilityRemediationModelTests(unittest.TestCase):
    def test_legacy_1_0_shapes_parse_without_new_identity_or_collection_fields(self):
        for captured in (False, True):
            with self.subTest(captured=captured):
                exchange = {"agent": "extractor", "node": "repair_invalid_extraction", "attempt": 2}
                if captured:
                    exchange.update(prompt_messages=[{"role": "human", "content": PROMPT}], response={"value": "1"})
                legacy = {
                    "llm_content_captured": captured,
                    "llm_exchanges": {ENTITY: [exchange]},
                    "variable_attempts": {ENTITY: [{"attempt": 2, "mode": "repair", "is_valid": True}]},
                }
                original = deepcopy(legacy)
                parsed = RunObservability.model_validate_json(json.dumps(legacy))
                self.assertEqual(legacy, original)
                self.assertIsNone(parsed.collection_status)
                self.assertEqual(parsed.collection_issues, [])
                self.assertEqual(parsed.unattributed_exchanges, [])
                call = parsed.llm_exchanges[ENTITY][0]
                self.assertEqual(call.entity_key, ENTITY)
                self.assertIsNone(call.invocation_id)
                self.assertEqual(call.namespace, [])
                self.assertEqual(call.attempt, 2)
                self.assertEqual(RunObservability.model_validate_json(parsed.model_dump_json()), parsed)

    def test_partial_and_unavailable_require_diagnostics_and_preserve_null_summary(self):
        for status in ("partial", "unavailable"):
            payload = {"llm_content_captured": False, "collection_status": status, "llm_usage_summary": None}
            with self.subTest(status=status):
                with self.assertRaises(ValidationError):
                    RunObservability.model_validate(payload)
                payload["collection_issues"] = [{"code": "capture_failed", "message": "Synthetic collection failure."}]
                parsed = RunObservability.model_validate(payload)
                restored = RunObservability.model_validate_json(parsed.model_dump_json())
                self.assertIsNone(restored.llm_usage_summary)
                self.assertIsNone(json.loads(restored.model_dump_json())["llm_usage_summary"])
                for diagnostic in ({"code": "", "message": "failure"}, {"code": "failure", "message": ""}):
                    with self.subTest(diagnostic=diagnostic), self.assertRaises(ValidationError):
                        RunObservability.model_validate({**payload, "collection_issues": [diagnostic]})
        with self.assertRaises(ValidationError):
            RunObservability.model_validate({**payload, "llm_usage_summary": {}})

    def test_complete_status_cannot_hide_issues_unbound_calls_or_unavailable_summary(self):
        baseline = {"llm_content_captured": False, "collection_status": "complete"}
        self.assertEqual(RunObservability.model_validate(baseline).llm_usage_summary.model_invocations, 0)
        for extra in (
            {"llm_usage_summary": None},
            {"collection_issues": [{"code": "failure", "message": "Synthetic diagnostic"}]},
            {"unattributed_exchanges": [{"invocation_id": "unbound", "agent": "extractor", "node": "repair_invalid_extraction"}]},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                RunObservability.model_validate({**baseline, **extra})


if __name__ == "__main__":
    if sys.argv[1:] == ["--cleanup-worker"]:
        print(json.dumps(_cleanup_probe(), allow_nan=False))
    else:
        unittest.main()
