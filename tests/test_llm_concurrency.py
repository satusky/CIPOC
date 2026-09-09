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

from cipoc.llm import BaseAgentModel, LLMConfig, OpenAIAgentModel


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

    def test_sdk_retry_and_backoff_retain_the_shared_permit(self):
        peer = self.agent()
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
            with patch("openai._base_client.time.sleep", side_effect=backoff) as sleep:
                self.assertEqual(agent.invoke("synthetic prompt").content, "ok")
                sleep.assert_called_once()
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
