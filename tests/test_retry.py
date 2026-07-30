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


if __name__ == "__main__":
    unittest.main()
