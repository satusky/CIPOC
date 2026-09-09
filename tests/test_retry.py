"""Pins the retry behaviour of the LLM-backed graph nodes.

Covers the predicate in isolation, the wiring (which nodes carry a policy and —
just as important — which do not), and the end-to-end behaviour through a real
agent graph driven by a stub model.
"""

import asyncio
import logging
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from cipoc.agents.note_retriever import NoteRetrieverAgent, RetrieverInput
from cipoc.llm import llm_retry_policy, retry_on_transient
from cipoc.models import NoteDigest, VariableGroupInfo, VariableInfo


def setUpModule():
    environment = patch.dict(os.environ, {
        "AZURE_OPENAI_URL": "https://retry-tests.invalid/v1",
        "RENCI_AZURE_API_KEY": "test",
    })
    environment.start()
    unittest.addModuleCleanup(environment.stop)


def _status_error(cls, status):
    request = httpx.Request("POST", "http://endpoint.invalid")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def _rate_limit():
    return _status_error(RateLimitError, 429)


class FakeLLM:
    """Stands in for a ``BaseAgentModel``: raises ``fail_times`` times, then answers."""

    def __init__(self, fail_times, exc_factory=_rate_limit):
        self.calls = 0
        self.fail_times = fail_times
        self.exc_factory = exc_factory

    def structured(self, schema, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return schema(note_ids=[1])


class RetryPredicateTests(unittest.TestCase):
    def test_transient_endpoint_failures_retry(self):
        for label, exc in (
            ("429", _rate_limit()),
            ("500", _status_error(InternalServerError, 500)),
            ("503", _status_error(InternalServerError, 503)),
            ("timeout", APITimeoutError(httpx.Request("POST", "http://endpoint.invalid"))),
        ):
            with self.subTest(label):
                self.assertTrue(retry_on_transient(exc))

    def test_client_errors_and_bugs_do_not_retry(self):
        """Retrying these burns `max_attempts` LLM calls on a request that can
        never succeed, which is what LangGraph's default predicate would do."""
        for label, exc in (
            ("400", _status_error(BadRequestError, 400)),
            ("401", _status_error(AuthenticationError, 401)),
            ("ValueError", ValueError("bad schema")),
            ("KeyError", KeyError("missing")),
            ("AttributeError", AttributeError("typo")),
        ):
            with self.subTest(label):
                self.assertFalse(retry_on_transient(exc))


class RetryWiringTests(unittest.TestCase):
    """Which nodes carry a policy. A node that invokes a subgraph must not, or a
    single throttled request replays the whole branch and multiplies attempts."""

    def _policies(self, graph):
        return {
            name: getattr(node, "retry_policy", None) is not None
            for name, node in graph.nodes.items()
            if not name.startswith("__")
        }

    def test_llm_nodes_retry_and_deterministic_nodes_do_not(self):
        from cipoc.agents import ExtractorAgent, NoteScannerAgent, OrchestratorAgent

        orchestrator = OrchestratorAgent()
        self.assertEqual(
            self._policies(NoteScannerAgent()._graph),
            {
                "initialize": False,
                "summarize_note": True,
                "detect_concepts": True,
                "get_cancer_mentions": True,
            },
        )
        self.assertEqual(
            self._policies(NoteRetrieverAgent()._graph),
            {"initialize": False, "identify_relevant_notes": True},
        )
        self.assertEqual(
            self._policies(ExtractorAgent()._graph),
            {
                "initialize": False,
                "load_notes": False,
                "extract_group_values": True,
                "variable_branch": False,  # subgraph: its own nodes retry
                "merge_variable_results": False,
            },
        )
        # Every orchestrator LLM call goes through a subagent graph.
        self.assertNotIn(True, set(self._policies(orchestrator._graph).values()))


class RetryThroughGraphTests(unittest.TestCase):
    GROUP = VariableGroupInfo(name="test", variables=[VariableInfo(item_id=400, name="Primary Site")])
    DIGESTS = {1: NoteDigest(note_id=1, note_type="pathology", summary="s")}

    def setUp(self):
        # LangGraph logs each retry at INFO with a traceback; keep the run quiet.
        logging.getLogger("langgraph.pregel._retry").setLevel(logging.CRITICAL)

    def _agent(self, llm, **policy):
        agent = NoteRetrieverAgent(llm=llm)
        agent._retry_policy = agent._retry_policy._replace(
            initial_interval=0.001, max_interval=0.002, **policy
        )
        agent._graph = agent._build_graph()
        return agent

    def _run(self, agent):
        return agent.run(
            RetrieverInput(requested_variables=self.GROUP, available_digests=self.DIGESTS),
            progress=False,
        )

    def test_rate_limits_are_retried_until_the_call_succeeds(self):
        llm = FakeLLM(fail_times=3)
        self.assertEqual(self._run(self._agent(llm)), [1])
        self.assertEqual(llm.calls, 4)

    def test_non_transient_error_fails_on_first_attempt(self):
        llm = FakeLLM(99, lambda: _status_error(BadRequestError, 400))
        with self.assertRaises(BadRequestError):
            self._run(self._agent(llm))
        self.assertEqual(llm.calls, 1)

    def test_exhausting_attempts_reraises_the_original_error(self):
        llm = FakeLLM(99)
        with self.assertRaises(RateLimitError):
            self._run(self._agent(llm, max_attempts=3))
        self.assertEqual(llm.calls, 3)


class RetryConfigTests(unittest.TestCase):
    def test_config_overrides_merge_onto_defaults(self):
        policy = llm_retry_policy(max_attempts=3)
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.max_interval, 60.0)
        self.assertIs(policy.retry_on, retry_on_transient)

    def test_retry_block_is_not_forwarded_to_the_model_client(self):
        """OpenAIConfig allows extra fields, so a stray `retry` key would reach
        ChatOpenAI as a model kwarg."""
        from cipoc.utils import load_config

        config = load_config()
        self.assertNotIn("retry", config.llm_config("extractor").model_dump())
        self.assertEqual(config.retry_policy("extractor").retry_on, retry_on_transient)

    def test_responses_reasoning_can_be_disabled_for_chat_completions(self):
        from cipoc.llm import OpenAIConfig

        config = OpenAIConfig(
            model="gpt-oss-120b",
            api_key="test",
            base_url="https://example.com/v1",
            reasoning=None,
            reasoning_effort="medium",
            use_responses_api=False,
            structured_output_method="function_calling",
        )

        self.assertIsNone(config.reasoning)
        self.assertFalse(config.model_dump()["use_responses_api"])
        self.assertEqual(config.structured_output_method, "function_calling")


class StructuredOutputTests(unittest.TestCase):
    def test_responses_failure_signals_survive_real_sync_and_async_mock_transports(self):
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        async def check_responses():
            requests = []
            payload = {}

            def respond(request):
                requests.append(str(request.url))
                return httpx.Response(200, json=payload)

            with httpx.Client(transport=httpx.MockTransport(respond)) as sync_client:
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
                    agent = OpenAIAgentModel({
                        "model": "test-model", "api_key": "test",
                        "base_url": "http://responses.internal:8080/custom",
                        "endpoint_compatibility": "databricks", "use_responses_api": True,
                        "reasoning": None, "max_retries": 0,
                    }, http_client=sync_client, http_async_client=async_client)
                    for status, refusal in (("completed", False), ("incomplete", False), ("completed", True)):
                        content = [{
                            "type": "output_text", "text": '{"answer":"complete"}',
                            "annotations": [], "parsed": {"answer": "complete"},
                        }]
                        if refusal:
                            content.append({"type": "refusal", "refusal": "sensitive-refusal"})
                        payload = {
                            "id": "resp_mock", "object": "response", "created_at": 1,
                            "status": status, "error": None, "model": "test-model",
                            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
                            "output": [{
                                "id": "msg_mock", "type": "message", "role": "assistant",
                                "status": "completed", "content": content,
                            }],
                            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                            "text": {"format": {"type": "json_schema", "name": "Answer", "schema": Answer.model_json_schema()}},
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        }
                        for asynchronous in (False, True):
                            with self.subTest(status=status, refusal=refusal, asynchronous=asynchronous):
                                if status == "completed" and not refusal:
                                    result = (
                                        await agent.astructured(Answer, ["synthetic"])
                                        if asynchronous else agent.structured(Answer, ["synthetic"])
                                    )
                                    self.assertEqual(result, Answer(answer="complete"))
                                else:
                                    with self.assertRaisesRegex(ValueError, "Databricks structured output") as caught:
                                        if asynchronous:
                                            await agent.astructured(Answer, ["synthetic"])
                                        else:
                                            agent.structured(Answer, ["synthetic"])
                                    self.assertNotIn("sensitive-refusal", str(caught.exception))
                                    self.assertFalse(retry_on_transient(caught.exception))
            self.assertEqual(requests, ["http://responses.internal:8080/custom/responses"] * 6)

        asyncio.run(check_responses())

    def test_databricks_rejects_explicit_failure_signals_sync_and_async(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"
        agent._semaphore = None
        agent._tools = None
        agent._model = Mock()
        runnable = agent._model.with_structured_output.return_value
        signals = [
            {location: metadata}
            for location in ("response_metadata", "additional_kwargs")
            for metadata in (
                {"finish_reason": "length"},
                {"finish_reason": "content_filter"},
                {"refusal": "sensitive-refusal"},
                {"status": "incomplete"},
                {"status": "failed"},
                {"status": "cancelled"},
                {"status": "in_progress"},
                {"status": "queued"},
                {"incomplete_details": {"reason": "max_output_tokens"}},
                {"incomplete_details": {"reason": "content_filter"}},
            )
        ]
        signals.extend([
            {"content": [{"type": "refusal", "refusal": "sensitive-refusal"}]},
            {"content": [{"type": "non_standard", "value": {
                "type": "refusal", "refusal": "sensitive-refusal",
            }}]},
        ])
        for signal in signals:
            content = [{"type": "text", "text": '{"answer":"complete"}'}]
            content.extend(signal.get("content", []))
            raw = AIMessage(content=content, **{k: v for k, v in signal.items() if k != "content"})
            before = raw.model_dump()
            for schema in (Answer, Answer.model_json_schema()):
                for parsed in (None, {"answer": "upstream"}):
                    result = {"raw": raw, "parsed": parsed, "parsing_error": None}
                    runnable.invoke.return_value = result
                    runnable.ainvoke = AsyncMock(return_value=result)
                    for asynchronous in (False, True):
                        with self.subTest(signal=signal, parsed=parsed, asynchronous=asynchronous):
                            with self.assertRaisesRegex(ValueError, "Databricks structured output") as caught:
                                if asynchronous:
                                    asyncio.run(agent.astructured(schema, ["synthetic"]))
                                else:
                                    agent.structured(schema, ["synthetic"])
                            self.assertNotIn("sensitive-refusal", str(caught.exception))
                            self.assertFalse(retry_on_transient(caught.exception))
                            self.assertEqual(raw.model_dump(), before)

    def test_databricks_accepts_completed_content_blocks_sync_and_async(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"
        agent._semaphore = None
        agent._tools = None
        agent._model = Mock()
        result = {
            "raw": AIMessage(
                content=[{"type": "reasoning", "reasoning": "thinking"},
                         {"type": "text", "text": '{"answer":"complete"}'}],
                response_metadata={"finish_reason": "stop", "status": "completed", "incomplete_details": None},
                additional_kwargs={"refusal": None},
            ),
            "parsed": {"answer": "upstream"}, "parsing_error": None,
        }
        runnable = agent._model.with_structured_output.return_value
        runnable.invoke.return_value = result
        runnable.ainvoke = AsyncMock(return_value=result)
        self.assertEqual(agent.structured(Answer, ["synthetic"]), Answer(answer="complete"))
        self.assertEqual(asyncio.run(agent.astructured(Answer, ["synthetic"])), Answer(answer="complete"))

    def test_standard_native_results_and_errors_are_unchanged_sync_and_async(self):
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "standard"
        agent._semaphore = None
        agent._tools = None
        agent._model = Mock()
        expected = Answer(answer="native")
        runnable = agent._model.with_structured_output.return_value
        runnable.invoke.return_value = expected
        runnable.ainvoke = AsyncMock(return_value=expected)
        self.assertIs(agent.structured(Answer, ["synthetic"]), expected)
        self.assertIs(asyncio.run(agent.astructured(Answer, ["synthetic"])), expected)
        agent._model.with_structured_output.assert_called_with(Answer, method="json_schema")
        failure = ValueError("native refusal")
        runnable.invoke.side_effect = failure
        runnable.ainvoke.side_effect = failure
        for asynchronous in (False, True):
            with self.assertRaises(ValueError) as caught:
                if asynchronous:
                    asyncio.run(agent.astructured(Answer, ["synthetic"]))
                else:
                    agent.structured(Answer, ["synthetic"])
            self.assertIs(caught.exception, failure)

    def test_databricks_complete_json_through_real_client_with_mock_transport(self):
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        contents = iter((
            [{"type": "reasoning", "reasoning": "thinking"},
             {"type": "text", "text": '{"answer":"complete"}'}],
            '{"answer":"missing closing brace"',
            '{"answer":"complete"} trailing',
        ))
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "mock-completion", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": next(contents)},
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            agent = OpenAIAgentModel({
                "model": "test-model", "api_key": "test",
                "base_url": "https://mock-transport.invalid/v1",
                "endpoint_compatibility": "databricks", "use_responses_api": False,
                "reasoning": None, "max_retries": 0,
            }, http_client=transport)
            messages = [{"role": "user", "content": "synthetic prompt"}]
            self.assertEqual(agent.structured(Answer, messages), Answer(answer="complete"))
            for _ in range(2):
                with self.assertRaises(ValueError):
                    agent.structured(Answer, messages)
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(request.url.host == "mock-transport.invalid" for request in requests))

    def test_databricks_requires_complete_raw_json_even_when_upstream_parsed(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"
        for schema in (Answer, Answer.model_json_schema()):
            for parsed in (None, {"answer": "repaired"}):
                for text in (
                    '{"answer":"truncated',
                    '{"answer":"missing brace"',
                    '{"answer":"complete"} trailing',
                    '{"answer":"first"}{"answer":"second"}',
                    '```json\n{"answer":"fenced"}\n```',
                    '{"answer":NaN}',
                    '{"answer":Infinity}',
                    '{"answer":-Infinity}',
                    '',
                ):
                    with self.subTest(schema=schema, parsed=parsed, text=text):
                        result = {
                            "raw": AIMessage(content=text),
                            "parsed": parsed,
                            "parsing_error": None,
                        }
                        with self.assertRaises(ValueError) as caught:
                            agent._parse_structured_result(schema, result)
                        self.assertFalse(retry_on_transient(caught.exception))

    def test_databricks_raw_text_blocks_are_authoritative_and_not_mutated(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"
        raw = AIMessage(content=[
            {"type": "reasoning", "reasoning": "Not JSON {"},
            {"type": "text", "text": ' \n{"answer":'},
            {"type": "text", "text": '"raw"}\t '},
        ])
        before = raw.model_dump()
        result = {"raw": raw, "parsed": {"answer": "upstream"}, "parsing_error": None}
        self.assertEqual(agent._parse_structured_result(Answer, result), Answer(answer="raw"))
        self.assertEqual(
            agent._parse_structured_result(Answer.model_json_schema(), result),
            {"answer": "raw"},
        )
        self.assertEqual(raw.model_dump(), before)
        self.assertEqual(result["parsed"], {"answer": "upstream"})

    def test_databricks_pydantic_validation_cannot_be_bypassed_by_parsed(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel, ValidationError
        from cipoc.llm import OpenAIAgentModel

        class Answer(BaseModel):
            answer: str

        agent = object.__new__(OpenAIAgentModel)
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"
        with self.assertRaises(ValidationError):
            agent._parse_structured_result(Answer, {
                "raw": AIMessage(content='{"wrong_field":"raw"}'),
                "parsed": {"answer": "upstream"},
                "parsing_error": None,
            })

    def test_standard_function_calling_uses_langchain_defaults(self):
        from cipoc.llm import OpenAIAgentModel

        class Runnable:
            def invoke(self, messages, **kwargs):
                return messages

        class Model:
            method = None

            def with_structured_output(self, schema, *, method):
                self.method = method
                return Runnable()

        agent = object.__new__(OpenAIAgentModel)
        agent._model = Model()
        agent._tools = None
        agent._semaphore = None
        agent._structured_output_method = "function_calling"
        agent._endpoint_compatibility = "standard"

        self.assertEqual(agent.structured(dict, ["message"]), ["message"])
        self.assertEqual(agent._model.method, "function_calling")

    def test_json_schema_parses_text_among_reasoning_blocks(self):
        from langchain_core.messages import AIMessage
        from cipoc.agents.note_scanner import NoteSummary
        from cipoc.llm import OpenAIAgentModel

        class Runnable:
            def invoke(self, messages, **kwargs):
                return {
                    "raw": AIMessage(content=[
                        {"type": "reasoning", "reasoning": "thinking"},
                        {
                            "type": "text",
                            "text": '{"summary":"summary","keywords":["one","two","three"]}',
                        },
                    ]),
                    "parsed": None,
                    "parsing_error": ValueError("content is a list"),
                }

        class Model:
            include_raw = None
            schema = None

            def with_structured_output(self, schema, *, method, include_raw=False):
                self.schema = schema
                self.include_raw = include_raw
                return Runnable()

        agent = object.__new__(OpenAIAgentModel)
        agent._model = Model()
        agent._tools = None
        agent._semaphore = None
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "databricks"

        response = agent.structured(NoteSummary, ["message"])

        self.assertEqual(response.summary, "summary")
        self.assertEqual(response.keywords, ["one", "two", "three"])
        self.assertTrue(agent._model.include_raw)
        self.assertIsInstance(agent._model.schema, dict)
        self.assertEqual(agent._model.schema["title"], "NoteSummary")

    def test_standard_json_schema_uses_native_pydantic_parsing(self):
        from cipoc.agents.note_scanner import NoteSummary
        from cipoc.llm import OpenAIAgentModel

        expected = NoteSummary(summary="summary", keywords=["one", "two", "three"])

        class Runnable:
            def invoke(self, messages, **kwargs):
                return expected

        class Model:
            schema = None
            kwargs = None

            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                self.kwargs = kwargs
                return Runnable()

        agent = object.__new__(OpenAIAgentModel)
        agent._model = Model()
        agent._tools = None
        agent._semaphore = None
        agent._structured_output_method = "json_schema"
        agent._endpoint_compatibility = "standard"

        response = agent.structured(NoteSummary, ["message"])

        self.assertIs(response, expected)
        self.assertIs(agent._model.schema, NoteSummary)
        self.assertEqual(agent._model.kwargs, {"method": "json_schema"})


class OpenAIReasoningConfigTests(unittest.TestCase):
    def test_standard_compatibility_is_default_and_not_forwarded(self):
        from cipoc.llm import OpenAIConfig, OpenAIAgentModel

        config = OpenAIConfig(
            model="gpt-5.5",
            api_key="test",
            base_url="https://api.openai.com/v1",
        )
        agent = OpenAIAgentModel(config)

        self.assertEqual(config.endpoint_compatibility, "standard")
        self.assertEqual(config.structured_output_method, "json_schema")
        self.assertNotIn("endpoint_compatibility", agent._model_kwargs())

    def test_databricks_compatibility_is_opt_in_and_not_forwarded(self):
        from cipoc.llm import OpenAIConfig, OpenAIAgentModel

        config = OpenAIConfig(
            model="gpt-oss-120b",
            api_key="test",
            base_url="https://example.com/v1",
            endpoint_compatibility="databricks",
            use_responses_api=False,
        )
        agent = OpenAIAgentModel(config)

        self.assertEqual(config.endpoint_compatibility, "databricks")
        self.assertNotIn("endpoint_compatibility", agent._model_kwargs())

    def test_standard_chat_completions_translates_reasoning_to_effort(self):
        from cipoc.llm import OpenAIConfig, OpenAIAgentModel

        agent = OpenAIAgentModel(OpenAIConfig(
            model="gpt-oss-120b",
            api_key="test",
            base_url="https://example.com/v1",
            use_responses_api=False,
            reasoning={"effort": "high", "summary": "detailed"},
        ))

        self.assertEqual(agent.model.reasoning_effort, "high")
        self.assertIsNone(agent.model.reasoning)

    def test_responses_preserves_nested_reasoning(self):
        from cipoc.llm import OpenAIConfig, OpenAIAgentModel

        agent = OpenAIAgentModel(OpenAIConfig(
            model="gpt-oss-120b",
            api_key="test",
            base_url="https://example.com/v1",
            use_responses_api=True,
            reasoning={"effort": "high", "summary": "detailed"},
        ))

        self.assertEqual(
            agent.model.reasoning,
            {"effort": "high", "summary": "detailed"},
        )
        self.assertIsNone(agent.model.reasoning_effort)


class EndpointValidationTests(unittest.TestCase):
    CONFIG = {"model": "test-model", "api_key": "test", "base_url": "https://endpoint.invalid/v1"}

    def test_malformed_endpoints_fail_before_client_creation_without_leaking_secrets(self):
        from cipoc.llm import LLMConfig, OpenAIAgentModel, OpenAIConfig

        invalid = (
            "endpoint.invalid/v1", "/relative/v1", "//endpoint.invalid/v1",
            "ftp://endpoint.invalid/v1", "file:///private/v1", "https:/endpoint.invalid/v1",
            "http://", "https:///v1", "https://:443/v1", "https://user:password-secret@/v1",
            "https://host:", "https://host:0/v1", "https://host:65536/v1", "https://host:-1/v1",
            "https://host:1.5/v1", "https://host:port-secret/v1", "https://host:443:80/v1",
            "https://user:password-secret@host:bad/v1?key=query-secret",
            "https://[::1", "https://[::1]junk/v1", "https://[::1]:bad/v1",
            "https://[not-an-ip]/v1", "https://[v1.future]/v1", "https://999.999.999.999/v1",
            "https://ho st/v1", "https://ho\nst/v1", "https://host/\x00path",
            "https://host\\other/v1", "https://host/{path with spaces}", "https://host/%zz",
            "https://ho%73t/v1", "https://host|other/v1", "https://host..internal/v1",
        )
        with patch("cipoc.llm.openai.ChatOpenAI") as client:
            for endpoint in invalid:
                for factory in (LLMConfig, OpenAIConfig, OpenAIAgentModel):
                    with self.subTest(endpoint=endpoint, factory=factory):
                        with self.assertRaises(ValueError) as caught:
                            config = {**self.CONFIG, "base_url": endpoint}
                            if factory is OpenAIAgentModel:
                                factory(config)
                            else:
                                factory(**config)
                        self.assertNotIn("secret", str(caught.exception))
                        self.assertNotIn(endpoint, str(caught.exception))
                for name in ("base_url", "openai_api_base"):
                    with self.subTest(endpoint=endpoint, override=name):
                        with self.assertRaises(ValueError) as caught:
                            OpenAIAgentModel(self.CONFIG, **{name: endpoint})
                        self.assertNotIn("secret", str(caught.exception))
                        self.assertNotIn(endpoint, repr(caught.exception))
            client.assert_not_called()

    def test_effective_custom_endpoint_is_the_actual_sync_and_async_destination(self):
        from cipoc.llm import OpenAIAgentModel

        async def check_destinations():
            for endpoint, expected in (
                ("http://localhost:8080/proxy/v1", "http://localhost:8080/proxy/v1/chat/completions"),
                ("http://127.0.0.1:9000/private", "http://127.0.0.1:9000/private/chat/completions"),
                ("http://[::1]:8081/serving-endpoints/", "http://[::1]:8081/serving-endpoints/chat/completions"),
                ("HTTPS://PRIVATE_HOST:443/team/My%2FModel", "https://private_host/team/My%2FModel/chat/completions"),
                ("https://gateway.internal:8443/Custom/v1", "https://gateway.internal:8443/Custom/v1/chat/completions"),
            ):
                requests = []

                def respond(request):
                    requests.append(str(request.url))
                    return httpx.Response(200, json={
                        "id": "mock", "object": "chat.completion", "created": 1,
                        "model": "test-model", "choices": [{
                            "index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }],
                    })

                with self.subTest(endpoint=endpoint), httpx.Client(transport=httpx.MockTransport(respond)) as sync_client:
                    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
                        agent = OpenAIAgentModel(
                            {**self.CONFIG, "reasoning": None, "use_responses_api": False},
                            openai_api_base=endpoint, http_client=sync_client, http_async_client=async_client,
                        )
                        self.assertEqual(agent.invoke("synthetic").content, "ok")
                        self.assertEqual((await agent.ainvoke("synthetic")).content, "ok")
                    self.assertEqual(requests, [expected, expected])

        asyncio.run(check_destinations())

    def test_endpoint_must_be_explicit_and_nonblank(self):
        from cipoc.llm import OpenAIAgentModel

        with patch.dict(os.environ, {
            "OPENAI_API_BASE": "https://fallback.invalid/v1",
            "OPENAI_BASE_URL": "https://fallback.invalid/v1",
        }), patch("cipoc.llm.openai.ChatOpenAI") as client:
            for endpoint in (None, "", "  \t\n"):
                for name in ("base_url", "openai_api_base"):
                    with self.subTest(endpoint=endpoint, name=name):
                        config = {k: v for k, v in self.CONFIG.items() if k != "base_url"}
                        config[name] = endpoint
                        with self.assertRaises(ValueError):
                            OpenAIAgentModel(config)
                        with self.assertRaises(ValueError):
                            OpenAIAgentModel(self.CONFIG, **{name: endpoint})
            with self.assertRaises(ValueError):
                OpenAIAgentModel({"model": "test-model", "api_key": "test"})
            client.assert_not_called()

    def test_effective_endpoint_alias_and_overrides_are_normalized_before_client(self):
        from cipoc.llm import OpenAIAgentModel

        for config, overrides, expected in (
            ({**self.CONFIG, "base_url": " https://endpoint.invalid/v1 "}, {}, "https://endpoint.invalid/v1"),
            ({"model": "test", "api_key": "test", "openai_api_base": "https://alias.invalid/v1"}, {}, "https://alias.invalid/v1"),
            (self.CONFIG, {"openai_api_base": " https://override.invalid/v1 "}, "https://override.invalid/v1"),
            (self.CONFIG, {"base_url": "https://override.invalid/v1"}, "https://override.invalid/v1"),
            ({**self.CONFIG, "openai_api_base": "https://ignored.invalid/v1"}, {}, self.CONFIG["base_url"]),
            (self.CONFIG, {"base_url": "https://canonical.invalid/v1", "openai_api_base": "https://ignored.invalid/v1"}, "https://canonical.invalid/v1"),
        ):
            with self.subTest(config=config, overrides=overrides):
                with patch("cipoc.llm.openai.ChatOpenAI") as client:
                    OpenAIAgentModel(config, **overrides)
                    self.assertEqual(client.call_args.kwargs["base_url"], expected)
                    self.assertNotIn("openai_api_base", client.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
