"""Pins the retry behaviour of the LLM-backed graph nodes.

Covers the predicate in isolation, the wiring (which nodes carry a policy and —
just as important — which do not), and the end-to-end behaviour through a real
agent graph driven by a stub model.
"""

import logging
import unittest

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


if __name__ == "__main__":
    unittest.main()
