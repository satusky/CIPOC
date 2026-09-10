"""Offline pre-SDK evidence regressions through the pinned HTTP/SDK/LC stack."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient
from pydantic import BaseModel

from cipoc.llm import OpenAIAgentModel
from cipoc.llm import usage_evidence as evidence
from cipoc.models import RunObservability
from cipoc.utils.observability import ObservabilityCollector, _read_llm_result
from tests.test_observability import llm_result, metadata, start
from langchain_core.messages import AIMessage


_MISSING = object()
_ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_ALL = ["input_tokens", "output_tokens", "total_tokens"]


def completion(usage=_MISSING, *, status=200):
    body = {
        "id": "offline", "object": "chat.completion", "created": 1,
        "model": "offline", "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"answer":"synthetic"}'},
        }],
    }
    if usage is not _MISSING:
        body["usage"] = usage
    return httpx.Response(status, json=body)


class UsageEvidenceTests(unittest.TestCase):
    def agent(self, respond, **config):
        def client(**kwargs):
            result = DefaultHttpxClient(transport=httpx.MockTransport(respond), **kwargs)
            self.addCleanup(result.close)
            return result

        # Replace only the test transport; production constructs and hooks its
        # own client, and the real SDK and adapter still perform every conversion.
        with patch("cipoc.llm.openai.DefaultHttpxClient", side_effect=client):
            return OpenAIAgentModel({
                "model": "offline", "api_key": "synthetic",
                "base_url": f"https://{uuid4().hex}.invalid/v1",
                "reasoning": None, "use_responses_api": False, "max_retries": 0,
                **config,
            })

    def invoke(self, agent, prompt="offline"):
        previous = evidence._capture.get()
        collector = ObservabilityCollector(capture_llm_content=False)
        message = agent.invoke(prompt, config=collector.graph_config())
        snapshot = RunObservability.model_validate(collector.snapshot())
        self.assertEqual(len(snapshot.unattributed_exchanges), 1)
        self.assertIs(evidence._capture.get(), previous)
        return message, snapshot.unattributed_exchanges[0]

    def test_real_wrapper_certifies_zero_but_not_sdk_coerced_invalid_values(self):
        for value in (0, False, "0", 0.0):
            with self.subTest(value=value, type=type(value)):
                raw = {**_ZERO, "completion_tokens": value}
                message, call = self.invoke(self.agent(lambda request: completion(raw)))
                self.assertIs(type(message.response_metadata["token_usage"]["completion_tokens"]), int)
                self.assertEqual(call.usage.output_tokens, 0)
                self.assertEqual(call.usage_reported_fields, _ALL if type(value) is int else ["input_tokens", "total_tokens"])

    def test_responses_api_and_bound_capture_retain_same_invocation_proof(self):
        def respond(request):
            self.assertTrue(request.url.path.endswith("/responses"))
            return httpx.Response(200, json={
                "id": "resp_offline", "object": "response", "created_at": 1,
                "status": "completed", "error": None, "model": "offline",
                "output": [{
                    "id": "msg_offline", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{
                        "type": "output_text", "text": "synthetic", "annotations": [],
                    }],
                }],
                "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            })

        agent = self.agent(respond, use_responses_api=True)
        collector = ObservabilityCollector(capture_llm_content=False)
        event = start((), "summarize_note", "offline", {"note_id": "offline"})
        collector.observe(event)
        agent.invoke("offline", config=collector.graph_config({"metadata": metadata(event)}))
        snapshot = RunObservability.model_validate(collector.snapshot())
        self.assertEqual(snapshot.collection_status, "complete")
        self.assertEqual(snapshot.llm_exchanges["note:offline"][0].usage_reported_fields, _ALL)
        self.assertEqual(snapshot.unattributed_exchanges, [])

    def test_omitted_details_only_and_conflicting_aliases_do_not_certify_scalars(self):
        for raw, fields, counts in (
            ({"prompt_tokens": 8}, ["input_tokens"], (8, 0, 8)),
            ({"completion_tokens": 8}, ["output_tokens"], (0, 8, 8)),
            ({"total_tokens": 8}, ["total_tokens"], (0, 0, 8)),
            ({"prompt_tokens_details": {"cached_tokens": 2}}, [], (0, 0, 0)),
            ({**_ZERO, "output_tokens": 1}, ["input_tokens", "total_tokens"], (0, 0, 0)),
            ({**_ZERO, "output_tokens": "0"}, ["input_tokens", "total_tokens"], (0, 0, 0)),
            ({**_ZERO, "output_tokens": 0}, _ALL, (0, 0, 0)),
        ):
            with self.subTest(raw=raw):
                _, call = self.invoke(self.agent(lambda request: completion(raw)))
                self.assertEqual(call.usage_reported_fields, fields)
                self.assertEqual((call.usage.input_tokens, call.usage.output_tokens, call.usage.total_tokens), counts)
                if "prompt_tokens_details" in raw:
                    self.assertEqual(call.usage.input_token_details.root, {"cache_read": 2})

    def test_evidence_never_changes_normalized_precedence_or_fills_derived_totals(self):
        response = llm_result(AIMessage(content="", usage_metadata={
            "input_tokens": 3, "output_tokens": 0, "total_tokens": 3,
        }, response_metadata={"token_usage": {"prompt_tokens": 999, "completion_tokens": 999}}))
        raw = (("prompt_tokens", 3), ("completion_tokens", 0))
        _, counts, _, fields = _read_llm_result(response, usage_evidence=raw)
        self.assertEqual(counts, {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3})
        self.assertEqual(fields, ["input_tokens", "output_tokens"])
        self.assertEqual(_read_llm_result(response, usage_evidence=(("prompt_tokens", 9),))[3], [])
        self.assertIsNone(_read_llm_result(response)[3])

    def test_sdk_retries_and_successive_calls_do_not_reuse_old_evidence(self):
        responses = iter([
            completion(_ZERO, status=429), completion({"prompt_tokens": 8}),
            completion(_ZERO), completion(), completion({"total_tokens": 4}),
        ])
        attempts = []

        def respond(request):
            attempts.append(request.headers["x-stainless-retry-count"])
            return next(responses)

        agent = self.agent(respond, max_retries=1)
        with patch("openai._base_client.time.sleep"):
            for expected in (["input_tokens"], _ALL, None, ["total_tokens"]):
                _, call = self.invoke(agent)
                self.assertEqual(call.usage_reported_fields, expected)
        self.assertEqual(attempts, ["0", "1", "0", "0", "0"])

    def test_read_timeout_retains_sdk_retry_semantics(self):
        class BrokenBody(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadTimeout("synthetic read timeout")
                yield b""  # Make this a stream generator.

        attempts = []

        def respond(request):
            attempts.append(request.headers["x-stainless-retry-count"])
            if len(attempts) == 1:
                return httpx.Response(200, headers={"content-type": "application/json"}, stream=BrokenBody())
            return completion(_ZERO)

        with patch("openai._base_client.time.sleep"):
            _, call = self.invoke(self.agent(respond, max_retries=1))
        self.assertEqual(attempts, ["0", "1"])
        self.assertEqual(call.usage_reported_fields, _ALL)

    def test_threaded_structured_invocations_isolate_evidence_on_shared_and_distinct_wrappers(self):
        class Answer(BaseModel):
            answer: str

        for shared in (False, True):
            with self.subTest(shared=shared):
                barrier = Barrier(2)

                def respond(request):
                    label = json.loads(request.content)["messages"][-1]["content"]
                    barrier.wait(timeout=5)
                    return completion(_ZERO if label == "zero" else {"prompt_tokens": 8})

                first = self.agent(respond, endpoint_compatibility="databricks")
                second = first if shared else self.agent(respond, endpoint_compatibility="databricks")

                def invoke(agent, label):
                    collector = ObservabilityCollector(capture_llm_content=False)
                    # Databricks include_raw=True uses a real RunnableParallel
                    # worker, so HTTP and callback capture see copied contexts.
                    result = agent.structured(Answer, label, config=collector.graph_config())
                    self.assertEqual(result.answer, "synthetic")
                    self.assertIsNone(evidence._capture.get())
                    return collector.llm_callback._snapshot()[0].usage_reported_fields

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(invoke, first, "zero"), pool.submit(invoke, second, "partial")]
                    self.assertEqual([future.result(timeout=10) for future in futures], [_ALL, ["input_tokens"]])

    def test_nested_wrappers_restore_outer_evidence_but_nested_direct_requests_invalidate_it(self):
        inner = self.agent(lambda request: completion(_ZERO))
        inner_calls = []

        def respond(request):
            _, call = self.invoke(inner)
            inner_calls.append(call)
            return completion({"prompt_tokens": 8})

        _, outer = self.invoke(self.agent(respond))
        self.assertEqual(inner_calls[0].usage_reported_fields, _ALL)
        self.assertEqual(outer.usage_reported_fields, ["input_tokens"])

        def nested_direct(request):
            label = json.loads(request.content)["messages"][-1]["content"]
            if label == "outer":
                agent.model.root_client.chat.completions.create(
                    model="offline", messages=[{"role": "user", "content": "inner"}],
                )
            return completion(_ZERO)

        agent = self.agent(nested_direct)
        _, call = self.invoke(agent, "outer")
        self.assertIsNone(call.usage_reported_fields)

    def test_only_sanitized_immutable_scalars_survive_until_parser_and_scope_always_cleans_up(self):
        class Answer(BaseModel):
            answer: str

        secret = "synthetic-sensitive-text"
        agent = self.agent(lambda request: completion({
            **_ZERO, "output_tokens": secret, "unexpected": secret,
        }), endpoint_compatibility="databricks", max_concurrency=1)
        states = []

        def parser(schema, result):
            state = evidence._capture.get()
            states.append(state)
            self.assertFalse(agent._semaphore.acquire(blocking=False))
            self.assertIsInstance(state.evidence, tuple)
            self.assertNotIn(secret, repr(state.evidence))
            self.assertEqual(dict(state.evidence)["output_tokens"], None)
            raise ValueError("synthetic parser failure")

        with patch.object(agent, "_parse_structured_result", side_effect=parser):
            with self.assertRaisesRegex(ValueError, "parser failure"):
                agent.structured(Answer, secret, config=ObservabilityCollector().graph_config())
        self.assertIsNone(evidence._capture.get())
        self.assertTrue(states[0].closed)
        self.assertIsNone(states[0].evidence)
        self.assertTrue(agent._semaphore.acquire(blocking=False))
        agent._semaphore.release()

    def test_external_client_and_direct_or_async_calls_remain_unverified(self):
        seen = []
        hooks = {"response": [lambda response: seen.append(response.status_code)]}
        with DefaultHttpxClient(transport=httpx.MockTransport(lambda request: completion(_ZERO)), event_hooks=hooks) as client:
            agent = OpenAIAgentModel({
                "model": "offline", "api_key": "synthetic", "base_url": "https://external.invalid/v1",
                "reasoning": None, "use_responses_api": False,
            }, http_client=client)
            _, call = self.invoke(agent)
            self.assertIsNone(call.usage_reported_fields)
            self.assertEqual(client.event_hooks["response"], hooks["response"])
            self.assertEqual(seen, [200])

        agent = self.agent(lambda request: completion(_ZERO))
        collector = ObservabilityCollector(capture_llm_content=False)
        agent.model.invoke("direct", config=collector.graph_config())
        self.assertIsNone(collector.llm_callback._snapshot()[0].usage_reported_fields)

        async def invoke_async():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: completion(_ZERO))) as client:
                agent = self.agent(lambda request: completion(_ZERO), http_async_client=client)
                collector = ObservabilityCollector(capture_llm_content=False)
                await agent.ainvoke("async", config=collector.graph_config())
                self.assertIsNone(collector.llm_callback._snapshot()[0].usage_reported_fields)
        asyncio.run(invoke_async())

    def test_default_client_preserves_timeout_aliases_sdk_options_and_closes_on_init_failure(self):
        for options in ({}, {"timeout": 2.5}, {"request_timeout": httpx.Timeout(7, connect=2)}):
            with self.subTest(options=options):
                seen = []

                def respond(request):
                    seen.append(request.extensions["timeout"])
                    self.assertEqual(request.headers["x-test"], "kept")
                    self.assertEqual(request.url.params["test"], "kept")
                    return completion(_ZERO)

                agent = self.agent(respond, **options, default_headers={"x-test": "kept"}, default_query={"test": "kept"})
                baseline = ChatOpenAI(model="offline", api_key="synthetic", base_url=agent._config.base_url, **options)
                self.addCleanup(baseline.root_client.close)
                self.assertEqual(agent.model.root_client.timeout, baseline.root_client.timeout)
                self.assertEqual(agent.model.http_client.timeout, baseline.root_client._client.timeout)
                self.assertEqual(agent.model.http_client.follow_redirects, baseline.root_client._client.follow_redirects)
                self.invoke(agent)
                self.assertEqual(seen, [baseline.root_client._client.timeout.as_dict()])

        clients = []

        def client(**kwargs):
            result = DefaultHttpxClient(**kwargs)
            clients.append(result)
            return result

        with patch("cipoc.llm.openai.DefaultHttpxClient", side_effect=client), patch("cipoc.llm.openai.ChatOpenAI", side_effect=ValueError("initialization")):
            with self.assertRaisesRegex(ValueError, "initialization"):
                OpenAIAgentModel({"model": "offline", "api_key": "synthetic", "base_url": "https://failure.invalid/v1"})
        self.assertTrue(clients[0].is_closed)

    def test_supplied_sdk_clients_and_proxy_paths_are_not_instrumented_or_mutated(self):
        for options in ({"client": object()}, {"root_client": object()}, {"openai_proxy": "http://proxy.invalid:8080"}):
            with self.subTest(options=options), patch("cipoc.llm.openai.ChatOpenAI") as constructor, patch("cipoc.llm.openai.DefaultHttpxClient") as factory:
                agent = OpenAIAgentModel({
                    "model": "offline", "api_key": "synthetic", "base_url": "https://fallback.invalid/v1",
                }, **options)
                self.assertIsNone(agent._usage_evidence_owner)
                factory.assert_not_called()
                for key, value in options.items():
                    self.assertIs(constructor.call_args.kwargs[key], value)
        with patch.dict(os.environ, {"OPENAI_PROXY": "http://proxy.invalid:8080"}), patch("cipoc.llm.openai.ChatOpenAI"), patch("cipoc.llm.openai.DefaultHttpxClient") as factory:
            OpenAIAgentModel({"model": "offline", "api_key": "synthetic", "base_url": "https://fallback.invalid/v1"})
            factory.assert_not_called()

    def test_model_errors_and_interrupts_clear_evidence_without_replacing_the_error(self):
        states = []

        def respond(request):
            states.append(evidence._capture.get())
            return completion(_ZERO, status=400)

        agent = self.agent(respond, max_concurrency=1)
        collector = ObservabilityCollector(capture_llm_content=False)
        with self.assertRaises(Exception) as caught:
            agent.invoke("offline", config=collector.graph_config())
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIsNone(evidence._capture.get())
        self.assertTrue(states[0].closed)
        self.assertIsNone(states[0].evidence)
        call = collector.llm_callback._snapshot()[0]
        self.assertIsNone(call.usage_reported_fields)
        self.assertIsNone(call.usage)
        self.assertTrue(agent._semaphore.acquire(blocking=False))
        agent._semaphore.release()
        owner = object()
        with self.assertRaises(KeyboardInterrupt):
            with evidence.usage_evidence_scope(owner):
                retained = evidence._capture.get()
                raise KeyboardInterrupt()
        self.assertTrue(retained.closed)
        self.assertIsNone(evidence._capture.get())

    def test_hook_replaces_evidence_and_skips_sse_ambiguous_json_and_unmatched_callbacks(self):
        owner = object()

        def receive(retry, response):
            request = httpx.Request("POST", "https://offline.invalid/v1/chat/completions",
                                    headers={"x-stainless-retry-count": str(retry)}, json={"stream": False})
            evidence.capture_usage_request(owner, request)
            response.request = request
            evidence.capture_usage_response(owner, response)

        with evidence.usage_evidence_scope(owner):
            evidence.start_usage_evidence("call")
            receive(0, completion(_ZERO))
            self.assertEqual(dict(evidence._capture.get().evidence), _ZERO)
            receive(1, completion({"prompt_tokens": 8}))
            self.assertEqual(dict(evidence._capture.get().evidence), {"prompt_tokens": 8})
            receive(2, completion())
            self.assertIsNone(evidence.finish_usage_evidence("call"))

        for response in (
            httpx.Response(200, headers={"content-type": "text/event-stream"}),
            httpx.Response(200, content='{"usage":{"prompt_tokens":0,"prompt_tokens":1}}', headers={"content-type": "application/json"}),
        ):
            with evidence.usage_evidence_scope(owner):
                evidence.start_usage_evidence("call")
                if response.headers["content-type"] == "text/event-stream":
                    with patch.object(response, "read", side_effect=AssertionError("must not read SSE")):
                        receive(0, response)
                else:
                    receive(0, response)
                self.assertIsNone(evidence.finish_usage_evidence("call"))
        with evidence.usage_evidence_scope(owner):
            evidence.start_usage_evidence("call")
            receive(0, completion(_ZERO))
            self.assertIsNone(evidence.finish_usage_evidence("different-call"))
            evidence.start_usage_evidence("nested-call")
            self.assertIsNone(evidence.finish_usage_evidence("call"))


if __name__ == "__main__":
    unittest.main()
