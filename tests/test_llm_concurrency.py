"""Per-model endpoint budgets with synthetic models only; no live endpoint calls."""

import asyncio
import gc
import json
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.runnables.config import set_config_context
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from cipoc.llm import BaseAgentModel, LLMConfig, OpenAIAgentModel
from cipoc.utils.observability import ObservabilityCollector
from tests.test_observability import _DeterministicChatModel, _FlakyChatModel


class _Model:
    def __init__(self, invoke=None):
        self.call = invoke or (lambda messages: messages)

    def invoke(self, messages, config=None, stop=None, **kwargs):
        return self.call(messages)

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        return self.call(messages)

    def with_structured_output(self, schema, **kwargs):
        return self


class _Agent(BaseAgentModel):
    def _initialize_model(self, **kwargs):
        return kwargs["test_model"]


class EndpointConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = f"https://{uuid4().hex}.invalid/v1"

    def agent(self, capacity=1, *, endpoint=None, model=None, requested_model="test-model", **kwargs):
        return _Agent(
            LLMConfig(
                model=requested_model, api_key="test", base_url=endpoint or self.endpoint,
                max_concurrency=capacity,
            ),
            test_model=model or _Model(), **kwargs,
        )

    def test_capacities_are_strict_positive_in_config_and_overrides(self):
        for cap in (0, -1, True, False, 1.0, 1.5, "2"):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    self.agent(cap)
                with self.assertRaises(ValueError):
                    self.agent(max_concurrency=cap)
        for cap in (None, 1, 2):
            agent = self.agent(cap, endpoint=f"{self.endpoint}/{cap}")
            self.assertEqual(agent._endpoint_limiter.capacity, cap)

    def test_queue_measurement_covers_contention_without_mutating_caller_config(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                agent = self.agent()
                entered = threading.Event()
                reached_clock = threading.Event()
                clock = [10.0]
                received = []
                callback = BaseCallbackHandler()
                config = {
                    "metadata": {"caller": "keep", "cipoc_queue_seconds": 999},
                    "callbacks": [callback], "tags": ["caller"],
                    "configurable": {"custom": "keep"}, "max_concurrency": 4,
                }

                def monotonic():
                    value = clock[0]
                    reached_clock.set()
                    return value

                def invoke(messages, config=None, **kwargs):
                    received.append(config)
                    entered.set()
                    return messages

                agent._model.invoke = invoke
                with patch("cipoc.llm.base.monotonic", side_effect=monotonic):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        agent._semaphore.acquire()
                        try:
                            if structured:
                                pending = pool.submit(agent.structured, dict, "ok", config=config)
                            else:
                                pending = pool.submit(agent.invoke, "ok", config=config)
                            self.assertTrue(reached_clock.wait(3))
                            self.assertFalse(entered.is_set())
                            clock[0] = 12.5
                        finally:
                            agent._semaphore.release()
                        self.assertEqual(pending.result(timeout=3), "ok")
                self.assertEqual(received[0]["metadata"]["cipoc_queue_seconds"], 2.5)
                for key, value in config.items():
                    if key == "callbacks":
                        self.assertEqual(received[0][key].handlers, value)
                    elif key != "metadata":
                        self.assertEqual(received[0][key], value)
                self.assertEqual(received[0]["metadata"]["caller"], "keep")
                self.assertEqual(config["metadata"], {"caller": "keep", "cipoc_queue_seconds": 999})
                self.assertIsNot(received[0], config)
                self.assertIsNot(received[0]["metadata"], config["metadata"])

    def test_queue_metadata_preserves_context_callbacks_and_config(self):
        for callbacks in ([BaseCallbackHandler()], CallbackManager([BaseCallbackHandler()])):
            for explicit in (None, {"tags": ["explicit"]}):
                for structured in (False, True):
                    with self.subTest(callbacks=type(callbacks), explicit=explicit, structured=structured):
                        agent = self.agent(None)
                        parent_id = uuid4()
                        if isinstance(callbacks, CallbackManager):
                            callbacks.parent_run_id = parent_id
                        inherited = {
                            "callbacks": callbacks,
                            "metadata": {"langgraph_node": "summarize_note", "cipoc_queue_seconds": 999},
                            "configurable": {"thread_id": "synthetic"},
                            "recursion_limit": 42,
                        }
                        agent._model.invoke = Mock(return_value="ok")
                        with set_config_context(inherited) as context:
                            if structured:
                                context.run(agent.structured, dict, "ok", config=explicit)
                            else:
                                context.run(agent.invoke, "ok", config=explicit)
                        call = agent._model.invoke.call_args
                        config = call.kwargs["config"] if structured else call.args[1]
                        self.assertEqual(config["metadata"]["cipoc_queue_seconds"], 0.0)
                        self.assertEqual(config["metadata"]["langgraph_node"], "summarize_note")
                        self.assertEqual(config["configurable"], inherited["configurable"])
                        self.assertEqual(config["recursion_limit"], 42)
                        if isinstance(callbacks, list):
                            self.assertEqual(config["callbacks"].handlers, callbacks)
                        else:
                            self.assertEqual(config["callbacks"].handlers, callbacks.handlers)
                            self.assertEqual(config["callbacks"].parent_run_id, parent_id)
                        self.assertEqual(inherited["metadata"]["cipoc_queue_seconds"], 999)

    def test_real_tool_binding_does_not_duplicate_inherited_callbacks(self):
        def tool(value: str) -> str:
            """Return a synthetic value."""
            return value

        class Handler(BaseCallbackHandler):
            def __init__(self):
                self.starts = []
                self.ends = []

            def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
                self.starts.append((run_id, parent_run_id))

            def on_llm_end(self, response, *, run_id, **kwargs):
                self.ends.append(run_id)

        def respond(request):
            self.assertTrue(json.loads(request.content)["tools"])
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 1,
                "model": "offline", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }],
            })

        for manager in (False, True):
            for explicit in (None, {"tags": ["explicit"]}):
                with self.subTest(manager=manager, explicit=explicit):
                    handler = Handler()
                    collector = ObservabilityCollector(capture_llm_content=False)
                    handlers = [handler, collector.llm_callback]
                    parent_id = uuid4() if manager else None
                    callbacks = CallbackManager(
                        handlers=handlers.copy(), inheritable_handlers=handlers.copy(), parent_run_id=parent_id,
                    ) if manager else handlers
                    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
                        agent = self.agent(model=ChatOpenAI(
                            model="offline", api_key="synthetic", base_url=self.endpoint,
                            http_client=transport, max_retries=0, use_responses_api=False,
                        ), tools=[StructuredTool.from_function(tool)])
                        with set_config_context({"callbacks": callbacks}) as context:
                            self.assertEqual(context.run(agent.invoke, "offline", config=explicit).content, "ok")
                    self.assertEqual(len(handler.starts), 1)
                    self.assertEqual(handler.starts[0][1], parent_id)
                    self.assertEqual(handler.ends, [handler.starts[0][0]])
                    self.assertEqual(len(collector.llm_callback._snapshot()), 1)
                    self.assertEqual(collector.llm_callback.collection_issues(), [])
                    self.assertEqual(callbacks.handlers if manager else callbacks, handlers)
                    if manager:
                        self.assertEqual(callbacks.parent_run_id, parent_id)

    def test_queue_clock_errors_do_not_prevent_model_calls_or_release_permits_early(self):
        for structured in (False, True):
            for failure_at in (0, 1):
                with self.subTest(structured=structured, failure_at=failure_at):
                    agent = self.agent(model=_DeterministicChatModel(model_name="offline"))
                    collector = ObservabilityCollector(capture_llm_content=False)

                    def parse(schema, result):
                        self.assertFalse(agent._semaphore.acquire(blocking=False))
                        return result

                    clocks = [100, 101]
                    clocks[failure_at] = OSError("clock unavailable")
                    with (
                        patch("cipoc.llm.base.monotonic", side_effect=clocks),
                        patch.object(agent, "_structured_runnable", return_value=agent.model),
                        patch.object(agent, "_parse_structured_result", side_effect=parse),
                    ):
                        if structured:
                            agent.structured(dict, "offline", config=collector.graph_config())
                        else:
                            agent.invoke("offline", config=collector.graph_config())
                    self.assertTrue(agent._semaphore.acquire(blocking=False))
                    agent._semaphore.release()
                    snapshot = collector.snapshot()
                    self.assertEqual(len(snapshot["unattributed_exchanges"]), 1)
                    self.assertIsNone(snapshot["unattributed_exchanges"][0]["queue_seconds"])
                    self.assertIn("queue_timing_error", {issue["code"] for issue in snapshot["collection_issues"]})

    def test_real_structured_binding_with_inherited_callbacks_is_captured_once(self):
        def respond(request):
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 1,
                "model": "offline", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                }],
            })

        collector = ObservabilityCollector(capture_llm_content=False)
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            agent = OpenAIAgentModel({
                "model": "offline", "api_key": "synthetic", "base_url": self.endpoint,
                "reasoning": None, "use_responses_api": False, "max_retries": 0,
                "structured_output_method": "json_mode", "max_concurrency": 1,
            }, http_client=transport)
            with set_config_context({"callbacks": [collector.llm_callback]}) as context:
                self.assertEqual(context.run(agent.structured, dict, "offline"), {"ok": True})
        self.assertEqual(len(collector.llm_callback._snapshot()), 1)
        self.assertEqual(collector.llm_callback.collection_issues(), [])

    def test_queue_clock_failure_keeps_model_error_and_masks_inherited_timing(self):
        agent = self.agent(model=_FlakyChatModel(model_name="offline", fail_times=1))
        collector = ObservabilityCollector(capture_llm_content=False)
        inherited = collector.graph_config({"metadata": {"cipoc_queue_seconds": 999}})
        with (
            set_config_context(inherited) as context,
            patch("cipoc.llm.base.monotonic", side_effect=OSError("optional clock failure")),
        ):
            with self.assertRaisesRegex(TimeoutError, "transient"):
                context.run(agent.invoke, "offline")
        self.assertTrue(agent._semaphore.acquire(blocking=False))
        agent._semaphore.release()
        snapshot = collector.snapshot()
        call = snapshot["unattributed_exchanges"][0]
        self.assertEqual(call["error"], "TimeoutError: transient")
        self.assertIsNone(call["queue_seconds"])
        self.assertEqual(snapshot["llm_usage_summary"]["failed_invocations"], 1)
        self.assertIn("queue_timing_error", {issue["code"] for issue in snapshot["collection_issues"]})
        self.assertEqual(inherited["metadata"], {"cipoc_queue_seconds": 999})

    def test_sync_unbounded_reports_zero_but_async_and_direct_capture_queue_is_null(self):
        collector = ObservabilityCollector(capture_llm_content=False)
        agent = self.agent(None, model=_DeterministicChatModel(model_name="offline"))
        config = collector.graph_config()

        async def async_calls():
            await agent.ainvoke("async", config=config)
            await agent.astructured(dict, "async structured", config=config)

        with patch.object(agent, "_structured_runnable", return_value=agent.model):
            agent.invoke("sync", config=config)
            agent.structured(dict, "sync structured", config=config)
            asyncio.run(async_calls())
            agent.model.invoke("direct", config=config)
        calls = collector.snapshot()["unattributed_exchanges"]
        self.assertEqual([call["queue_seconds"] for call in calls], [0.0, 0.0, None, None, None])
        self.assertTrue(all(call["service_seconds"] >= 0 for call in calls))

    def test_langgraph_retries_release_permits_and_have_separate_service_and_queue_times(self):
        model = _FlakyChatModel(model_name="flaky", fail_times=1)
        agent = self.agent(model=model)
        collector = ObservabilityCollector(capture_llm_content=False)

        def retry_on(error):
            self.assertTrue(agent._semaphore.acquire(blocking=False))
            agent._semaphore.release()
            return isinstance(error, TimeoutError)

        def invoke(state):
            agent.invoke("offline")
            return state

        builder = StateGraph(dict)
        builder.add_node("summarize_note", invoke, retry_policy=RetryPolicy(
            initial_interval=0, max_interval=0, jitter=False, max_attempts=2, retry_on=retry_on,
        ))
        builder.add_edge(START, "summarize_note")
        builder.add_edge("summarize_note", END)
        with (
            patch("cipoc.llm.base.monotonic", side_effect=[10, 10.25, 20, 20.5]),
            patch("cipoc.utils.observability.monotonic", side_effect=[100, 102, 200, 203]),
        ):
            builder.compile().invoke({"note_id": "offline"}, config=collector.graph_config())
        calls = collector.snapshot()["llm_exchanges"]["note:offline"]
        self.assertEqual([call["queue_seconds"] for call in calls], [0.25, 0.5])
        self.assertEqual([call["service_seconds"] for call in calls], [2, 3])
        self.assertEqual([call.get("retry_ordinal") for call in calls], [None, 1])
        self.assertEqual([call["attempt"] for call in calls], [1, 1])
        self.assertNotEqual(calls[0]["invocation_id"], calls[1]["invocation_id"])

    def test_same_model_and_endpoint_share_budget_across_wrappers_and_credentials(self):
        with patch("cipoc.llm.openai.ChatOpenAI"):
            first = OpenAIAgentModel({
                "model": "model-one", "api_key": "key-one", "base_url": self.endpoint,
                "max_concurrency": 1,
            })
            second = OpenAIAgentModel({
                "model": "model-one", "api_key": "key-two", "base_url": f" {self.endpoint}/ ",
                "max_concurrency": 1,
            })
        self.assertIs(first._semaphore, second._semaphore)

    def test_different_models_have_independent_finite_and_unbounded_limits(self):
        first = self.agent(1, requested_model="model-one")
        second = self.agent(2, requested_model="model-two")
        unbounded = self.agent(None, requested_model="unbounded-model")
        self.assertIsNot(first._endpoint_limiter, second._endpoint_limiter)
        self.assertIsNone(unbounded._semaphore)
        with first._semaphore:
            self.assertTrue(second._semaphore.acquire(blocking=False))
            try:
                self.assertTrue(second._semaphore.acquire(blocking=False))
                try:
                    self.assertFalse(second._semaphore.acquire(blocking=False))
                    self.assertEqual(unbounded.invoke("unbounded"), "unbounded")
                finally:
                    second._semaphore.release()
            finally:
                second._semaphore.release()

    def test_different_models_can_fill_their_budgets_simultaneously(self):
        release = threading.Event()
        filled = threading.Event()
        lock = threading.Lock()
        active = {"one": 0, "two": 0}

        def call(model_name):
            def blocking(message):
                with lock:
                    active[model_name] += 1
                    if active == {"one": 1, "two": 2}:
                        filled.set()
                try:
                    if not release.wait(3):
                        raise TimeoutError("test did not release calls")
                    return message
                finally:
                    with lock:
                        active[model_name] -= 1
            return blocking

        first = self.agent(1, requested_model="one", model=_Model(call("one")))
        second = self.agent(2, requested_model="two", model=_Model(call("two")))
        peer = self.agent(2, requested_model="two", model=_Model(call("two")))
        self.assertIs(second._semaphore, peer._semaphore)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(first.invoke, "first"),
                pool.submit(second.structured, dict, "second"),
                pool.submit(peer.invoke, "peer"),
            ]
            try:
                self.assertTrue(filled.wait(3))
                self.assertFalse(first._semaphore.acquire(blocking=False))
                self.assertFalse(second._semaphore.acquire(blocking=False))
            finally:
                release.set()
            self.assertEqual([future.result(timeout=3) for future in futures], ["first", "second", "peer"])
        self.assertEqual(active, {"one": 0, "two": 0})

    def test_equivalent_endpoint_authorities_share_a_budget(self):
        host = self.endpoint.split("/")[2]
        for first_url, second_url in (
            (f"HTTPS://{host.upper()}:443/v1", f"https://{host}/v1/"),
            (f"HTTP://{host.upper()}:80/v1", f"http://{host}/v1/"),
            (f"https://{host}", f"https://{host}:443/"),
            (f"https://alice:secret@{host}/auth", f"https://bob:other@{host}/auth/"),
            ("https://[2001:DB8:0:0:0:0:0:1]:443/v1", "https://[2001:db8::1]/v1/"),
        ):
            with self.subTest(first_url=first_url, second_url=second_url):
                first = self.agent(endpoint=first_url)
                second = self.agent(endpoint=second_url)
                self.assertIs(first._semaphore, second._semaphore)
                with self.assertRaisesRegex(ValueError, "Conflicting max_concurrency"):
                    self.agent(2, endpoint=second_url)

    def test_canonicalization_preserves_distinct_endpoint_paths_ports_and_queries(self):
        first = self.agent(endpoint=self.endpoint + "/My%2FPath?route=one")
        for endpoint in (
            self.endpoint + "/my%2FPath?route=one",
            self.endpoint + "/My/Path?route=one",
            self.endpoint + "/My%2FPath?route=two",
            self.endpoint + "/My%2FPath?route=one/",
            self.endpoint + "/My%2FPath/?route=one",
            self.endpoint + "/My%2FPath//?route=one",
            self.endpoint.replace("/v1", ":8443/v1") + "/My%2FPath?route=one",
            self.endpoint.replace("https:", "http:") + "/My%2FPath?route=one",
        ):
            with self.subTest(endpoint=endpoint):
                other = self.agent(2, endpoint=endpoint)
                self.assertIsNot(first._semaphore, other._semaphore)
        equivalent = self.agent(endpoint=self.endpoint + "/My%2FPath?route=one#fragment")
        self.assertIs(first._semaphore, equivalent._semaphore)

    def test_an_explicit_empty_query_is_not_collapsed_into_a_query_free_endpoint(self):
        first = self.agent(endpoint=self.endpoint + "/")
        other = self.agent(2, endpoint=self.endpoint + "/?")
        self.assertIsNot(first._semaphore, other._semaphore)

    def test_registry_creation_is_locked_and_conflicts_are_explicit(self):
        barrier = threading.Barrier(12)

        def create(_):
            barrier.wait(timeout=3)
            return self.agent(2)

        with ThreadPoolExecutor(max_workers=12) as pool:
            agents = list(pool.map(create, range(12)))
        self.assertEqual(len({id(agent._endpoint_limiter) for agent in agents}), 1)
        for cap in (1, 3, None):
            with self.subTest(cap=cap), self.assertRaisesRegex(
                ValueError, "active capacity is 2, requested"
            ):
                self.agent(cap)

    def test_unbounded_wrapper_cannot_silently_bypass_a_later_cap(self):
        unbounded = self.agent(None)
        self.assertIsNone(unbounded._semaphore)
        self.assertEqual(unbounded.invoke("ok"), "ok")
        with self.assertRaisesRegex(ValueError, "Conflicting max_concurrency"):
            self.agent(1)

    def test_effective_endpoint_model_and_capacity_overrides_key_the_registry(self):
        first = self.agent(1)
        for overrides in (
            {"model": "test-model"},
            {"model_name": "test-model"},
            {"model": "test-model", "model_name": "ignored-alias"},
        ):
            with self.subTest(overrides=overrides), patch("cipoc.llm.openai.ChatOpenAI") as client:
                second = OpenAIAgentModel({
                    "model": "configured-model", "api_key": "test", "base_url": self.endpoint + "/other",
                    "max_concurrency": 3,
                }, openai_api_base=self.endpoint, max_concurrency=1, **overrides)
                self.assertIs(first._semaphore, second._semaphore)
                self.assertEqual(client.call_args.kwargs["base_url"], self.endpoint)
                self.assertEqual(client.call_args.kwargs["model"], "test-model")
                self.assertNotIn("model_name", client.call_args.kwargs)
                self.assertNotIn("max_concurrency", client.call_args.kwargs)
        other = self.agent(3, endpoint=self.endpoint + "/other")
        self.assertIsNot(first._semaphore, other._semaphore)

    def test_registry_releases_budget_after_last_wrapper_lifetime(self):
        first = self.agent()
        second = self.agent()
        limiter = weakref.ref(first._endpoint_limiter)
        del first
        gc.collect()
        self.assertIsNotNone(limiter())
        del second
        gc.collect()
        self.assertIsNone(limiter())
        replacement = self.agent(3)
        self.assertEqual(replacement._endpoint_limiter.capacity, 3)

    def test_failed_client_construction_does_not_leave_a_registered_budget(self):
        error = ValueError("client initialization failed")
        with patch.object(_Agent, "_initialize_model", side_effect=error):
            try:
                self.agent(1)
            except ValueError:
                pass
            else:
                self.fail("client construction should fail")
        # Keep the exception/traceback alive while constructing a different cap.
        self.assertIsNotNone(error.__traceback__)
        replacement = self.agent(3)
        self.assertEqual(replacement._endpoint_limiter.capacity, 3)

    def test_multiple_permits_are_usable_and_bound_total_work(self):
        agents = [self.agent(2) for _ in range(3)]
        self.assertTrue(agents[0]._semaphore.acquire(blocking=False))
        self.assertTrue(agents[1]._semaphore.acquire(blocking=False))
        self.assertFalse(agents[2]._semaphore.acquire(blocking=False))
        agents[0]._semaphore.release()
        self.assertTrue(agents[2]._semaphore.acquire(blocking=False))
        agents[1]._semaphore.release()
        agents[2]._semaphore.release()

    def test_inflight_call_keeps_the_weakly_registered_budget_alive(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking(messages):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release call")
            return messages

        agent = self.agent(model=_Model(blocking))
        limiter = weakref.ref(agent._endpoint_limiter)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(agent.invoke, "ok")
            del agent
            try:
                self.assertTrue(entered.wait(3))
                gc.collect()
                self.assertIsNotNone(limiter())
                with self.assertRaisesRegex(ValueError, "Conflicting max_concurrency"):
                    self.agent(2)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=3), "ok")
        gc.collect()
        self.assertIsNone(limiter())

    def test_invoke_and_structured_share_the_same_inflight_budget(self):
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()

        def blocking(messages):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release first call")
            return messages

        first = self.agent(model=_Model(blocking))
        second = self.agent(model=_Model(lambda message: second_entered.set() or message))

        def structured():
            second_started.set()
            return second.structured(dict, "second")

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending_first = pool.submit(first.invoke, "first")
            try:
                self.assertTrue(entered.wait(3))
                pending_second = pool.submit(structured)
                self.assertTrue(second_started.wait(3))
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release.set()
            self.assertEqual(pending_first.result(timeout=3), "first")
            self.assertEqual(pending_second.result(timeout=3), "second")

    def test_distinct_endpoints_do_not_block_each_other(self):
        first = self.agent()
        other = self.agent(endpoint=self.endpoint + "/other")
        with first._semaphore:
            self.assertTrue(other._semaphore.acquire(blocking=False))
            other._semaphore.release()

    def test_permit_covers_model_retries_and_structured_parsing(self):
        agent = self.agent()
        peer = self.agent()
        attempts = []

        def call(messages):
            for attempt in range(3):
                self.assertFalse(peer._semaphore.acquire(blocking=False))
                attempts.append(attempt)
            return messages

        def parse(schema, result):
            self.assertFalse(peer._semaphore.acquire(blocking=False))
            return result

        agent._model = _Model(call)
        with patch.object(agent, "_parse_structured_result", side_effect=parse):
            self.assertEqual(agent.structured(dict, "ok"), "ok")
        self.assertEqual(attempts, [0, 1, 2])
        self.assertTrue(peer._semaphore.acquire(blocking=False))
        peer._semaphore.release()

    def test_structured_parser_holds_permit_after_callback_service_time_finishes(self):
        agent = self.agent(model=_DeterministicChatModel(model_name="offline"))
        collector = ObservabilityCollector(capture_llm_content=False)

        def parse(schema, result):
            self.assertFalse(agent._semaphore.acquire(blocking=False))
            calls = collector.snapshot()["unattributed_exchanges"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["service_seconds"], 2)
            self.assertEqual(calls[0]["queue_seconds"], 0.25)
            return result

        with (
            patch.object(agent, "_structured_runnable", return_value=agent.model),
            patch.object(agent, "_parse_structured_result", side_effect=parse),
            patch("cipoc.llm.base.monotonic", side_effect=[10, 10.25]),
            patch("cipoc.utils.observability.monotonic", side_effect=[100, 102]),
        ):
            agent.structured(dict, "offline", config=collector.graph_config())
        self.assertTrue(agent._semaphore.acquire(blocking=False))
        agent._semaphore.release()

    def test_sdk_retry_and_backoff_retain_the_shared_permit(self):
        peer = self.agent()
        collector = ObservabilityCollector(capture_llm_content=False)
        requests = []

        def respond(request):
            self.assertFalse(peer._semaphore.acquire(blocking=False))
            self.assertEqual(json.loads(request.content)["model"], "test-model")
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, json={"error": {"message": "retry"}})
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 1,
                "model": "resolved-model-version", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }],
            })

        def backoff(seconds):
            self.assertFalse(peer._semaphore.acquire(blocking=False))

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            agent = OpenAIAgentModel({
                "model": "configured-model", "api_key": "test", "base_url": self.endpoint,
                "max_concurrency": 1, "max_retries": 1,
                "reasoning": None, "use_responses_api": False,
            }, model_name="test-model", http_client=transport)
            with (
                patch("openai._base_client.time.sleep", side_effect=backoff) as sleep,
                patch("cipoc.llm.base.monotonic", side_effect=[10, 10.25]),
                patch("cipoc.utils.observability.monotonic", side_effect=[100, 105]),
            ):
                self.assertEqual(agent.invoke("synthetic prompt", config=collector.graph_config()).content, "ok")
                sleep.assert_called_once()
        calls = collector.snapshot()["unattributed_exchanges"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["queue_seconds"], 0.25)
        self.assertEqual(calls[0]["service_seconds"], 5)
        self.assertNotIn("retry_ordinal", calls[0])
        self.assertEqual(len(requests), 2)
        self.assertIs(peer._semaphore, agent._semaphore)
        self.assertTrue(peer._semaphore.acquire(blocking=False))
        peer._semaphore.release()

    def test_model_and_parser_errors_release_permits_including_interrupts(self):
        for operation in ("invoke", "structured", "parse"):
            for error in (ValueError("failed"), KeyboardInterrupt("cancelled")):
                with self.subTest(operation=operation, error=type(error)):
                    agent = self.agent()
                    if operation == "parse":
                        agent._parse_structured_result = Mock(side_effect=error)
                    else:
                        agent._model.call = Mock(side_effect=error)
                    with self.assertRaises(type(error)):
                        if operation == "invoke":
                            agent.invoke("input")
                        else:
                            agent.structured(dict, "input")
                    self.assertTrue(agent._semaphore.acquire(blocking=False))
                    agent._semaphore.release()

    def test_async_and_direct_model_paths_remain_explicitly_unguarded(self):
        agent = self.agent()

        async def invoke_async():
            self.assertEqual(await agent.ainvoke("async"), "async")
            self.assertEqual(await agent.astructured(dict, "structured"), "structured")

        with agent._semaphore:
            asyncio.run(invoke_async())
            self.assertEqual(agent.model.invoke("direct"), "direct")


if __name__ == "__main__":
    unittest.main()
