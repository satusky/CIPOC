"""Pins the Foundry models API transport and the shared runtime http_client plumbing.

Requests go through the real ``ChatOpenAI``/OpenAI SDK stack into a fake
Foundry model; nothing opens a socket.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from openai import BadRequestError, InternalServerError, NotFoundError, RateLimitError, UnprocessableEntityError
from pydantic import BaseModel, ValidationError

from cipoc.agents import NoteScannerAgent, OrchestratorAgent
from cipoc.agents.note_scanner import CancerMentions
from cipoc.llm import ChatCompletionStatusError, FoundryModelsTransport, OpenAIAgentModel, retry_on_transient
from cipoc.utils import CipocConfig, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class Answer(BaseModel):
    answer: str


class _FakeFoundryModel:
    """Stands in for an injected ``OpenAiGptChatLanguageModel``."""

    def __init__(self, content='{"answer":"complete"}', error=None):
        self.content = content
        self.error = error
        self.requests = []

    def create_chat_completion(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"text": self.content, "prompt": 7, "completion": 3}


def _to_request(body):
    return {"messages": body["messages"], "response_format": body.get("response_format")}


def _to_response(response):
    return {
        "id": "foundry-completion", "object": "chat.completion", "created": 1,
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": response["text"]},
        }],
        "usage": {
            "prompt_tokens": response["prompt"], "completion_tokens": response["completion"],
            "total_tokens": response["prompt"] + response["completion"],
        },
    }


def _transport(models, *, to_request=_to_request, to_response=_to_response):
    return FoundryModelsTransport(models, to_request=to_request, to_response=to_response)


def _settings(model="foundry-model", **overrides):
    return {
        "model": model, "api_key": "unused",
        "base_url": f"http://{uuid4().hex}.invalid/v1",
        "use_responses_api": False, "reasoning": None, "max_retries": 0,
        "structured_output_method": "json_schema",
        **overrides,
    }


class _HttpClientCase(unittest.TestCase):
    def client(self, transport):
        client = httpx.Client(transport=transport)
        self.addCleanup(client.close)
        return client

    def agent(self, transport, **overrides):
        return OpenAIAgentModel(_settings(**overrides), http_client=self.client(transport))


class ChatOpenAIRoundTripTests(_HttpClientCase):
    def test_structured_output_round_trips_through_the_fake_model(self):
        model = _FakeFoundryModel()
        agent = self.agent(_transport(model))

        result = agent.structured(Answer, [{"role": "user", "content": "synthetic prompt"}])

        self.assertEqual(result, Answer(answer="complete"))
        [request] = model.requests
        self.assertEqual(request["messages"], [{"role": "user", "content": "synthetic prompt"}])
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["response_format"]["json_schema"]["name"], "Answer")

    def test_mention_schema_requires_behavior_and_diagnostic_certainty_on_the_wire(self):
        mention = {
            "presence": True, "confidence": "max", "evidence": [{"note_id": 1, "text": "lung nodule"}],
            "status": "current", "affected_tissue": "lung", "metastasis": False,
            "behavior": "unknown", "diagnostic_certainty": "possible",
        }
        for compatibility in ("standard", "databricks"):
            with self.subTest(endpoint_compatibility=compatibility):
                model = _FakeFoundryModel(content=json.dumps({"mentions": [mention]}))
                agent = self.agent(_transport(model), endpoint_compatibility=compatibility)

                result = agent.structured(CancerMentions, [{"role": "user", "content": "synthetic prompt"}])

                self.assertEqual(result.mentions[0].diagnostic_certainty, "possible")
                [request] = model.requests
                schema = request["response_format"]["json_schema"]["schema"]
                sent = schema["properties"]["mentions"]["items"]
                if "$ref" in sent:
                    sent = schema["$defs"][sent["$ref"].rsplit("/", 1)[-1]]
                for field in ("behavior", "diagnostic_certainty"):
                    self.assertIn(field, sent["required"])
                    self.assertNotIn("default", sent["properties"][field])

                for field in ("behavior", "diagnostic_certainty"):
                    incomplete = {key: value for key, value in mention.items() if key != field}
                    model = _FakeFoundryModel(content=json.dumps({"mentions": [incomplete]}))
                    agent = self.agent(_transport(model), endpoint_compatibility=compatibility)
                    with self.assertRaises(ValidationError):
                        agent.structured(CancerMentions, [{"role": "user", "content": "synthetic prompt"}])

    def test_plain_invoke_reports_usage_and_defaults_the_model_name(self):
        agent = self.agent(_transport(_FakeFoundryModel(content="hello")))

        message = agent.invoke([{"role": "user", "content": "synthetic prompt"}])

        self.assertEqual(message.content, "hello")
        self.assertEqual(
            (message.usage_metadata["input_tokens"], message.usage_metadata["output_tokens"],
             message.usage_metadata["total_tokens"]),
            (7, 3, 10),
        )
        self.assertEqual(message.response_metadata["model_name"], "foundry-model")

    def test_pydantic_responses_are_serialized(self):
        from openai.types.chat import ChatCompletion

        agent = self.agent(_transport(
            _FakeFoundryModel(content="hello"),
            to_response=lambda response: ChatCompletion.model_validate(
                {**_to_response(response), "model": "resolved-model"}
            ),
        ))

        message = agent.invoke([{"role": "user", "content": "synthetic prompt"}])

        self.assertEqual(message.content, "hello")
        self.assertEqual(message.response_metadata["model_name"], "resolved-model")

    def test_responses_api_requests_are_rejected_without_calling_the_model(self):
        model = _FakeFoundryModel()
        agent = self.agent(_transport(model), use_responses_api=True)

        with self.assertRaises(NotFoundError) as caught:
            agent.structured(Answer, [{"role": "user", "content": "synthetic prompt"}])

        self.assertIn("use_responses_api: false", str(caught.exception))
        self.assertFalse(retry_on_transient(caught.exception))
        self.assertEqual(model.requests, [])

    def test_requests_route_by_model_name(self):
        first, second = _FakeFoundryModel(content="first"), _FakeFoundryModel(content="second")
        transport = _transport({"model-a": first, "model-b": second})

        message = self.agent(transport, model="model-b").invoke([{"role": "user", "content": "x"}])
        with self.assertRaises(NotFoundError) as caught:
            self.agent(transport, model="model-c").invoke([{"role": "user", "content": "x"}])

        self.assertEqual(message.content, "second")
        self.assertEqual((len(first.requests), len(second.requests)), (0, 1))
        self.assertFalse(retry_on_transient(caught.exception))


class _StatusCarrier(Exception):
    def __init__(self, status_code):
        super().__init__("upstream refused")
        self.status_code = status_code


class _ResponseCarrier(Exception):
    """Shaped like ``requests.HTTPError``: the status lives on ``response``."""

    def __init__(self, status_code):
        super().__init__("upstream refused")
        self.response = type("Response", (), {"status_code": status_code})()


def _raise(error):
    def fail(*args, **kwargs):
        raise error
    return fail


class ErrorStatusTests(_HttpClientCase):
    def test_failures_map_to_statuses_the_retry_predicate_understands(self):
        cases = [
            # (label, transport kwargs, model error, expected exception, transient, model called)
            ("request translation bug", {"to_request": _raise(KeyError("messages"))}, None,
             BadRequestError, False, False),
            ("request translation status", {"to_request": _raise(ChatCompletionStatusError(422, "no tools"))}, None,
             UnprocessableEntityError, False, False),
            ("model status_code", {}, _StatusCarrier(429), RateLimitError, True, True),
            ("model response.status_code", {}, _ResponseCarrier(503), InternalServerError, True, True),
            ("model client error", {}, _StatusCarrier(400), BadRequestError, False, True),
            ("model explicit status", {}, ChatCompletionStatusError(429, "throttled"), RateLimitError, True, True),
            ("model unknown failure", {}, RuntimeError("connection reset"), InternalServerError, True, True),
            ("response translation bug", {"to_response": _raise(TypeError("bad shape"))}, None,
             BadRequestError, False, True),
            ("response unserializable", {"to_response": lambda response: {"choices": object()}}, None,
             BadRequestError, False, True),
        ]
        for label, transport_kwargs, model_error, expected, transient, called in cases:
            with self.subTest(label):
                model = _FakeFoundryModel(error=model_error)
                agent = self.agent(_transport(model, **transport_kwargs))
                with self.assertRaises(expected) as caught:
                    agent.structured(Answer, [{"role": "user", "content": "synthetic prompt"}])
                self.assertEqual(retry_on_transient(caught.exception), transient)
                self.assertEqual(len(model.requests), int(called))

    def test_unknown_model_failure_reports_status_502_and_names_the_error(self):
        agent = self.agent(_transport(_FakeFoundryModel(error=RuntimeError("connection reset"))))

        with self.assertRaises(InternalServerError) as caught:
            agent.invoke([{"role": "user", "content": "x"}])

        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn("RuntimeError: connection reset", str(caught.exception))

    def test_status_error_requires_an_http_error_status(self):
        for status in (200, 399, 600, True, "429", 429.0):
            with self.subTest(status=status), self.assertRaises(ValueError):
                ChatCompletionStatusError(status, "x")


class RawHttpTests(_HttpClientCase):
    def setUp(self):
        self.model = _FakeFoundryModel()
        self.http = self.client(_transport(self.model))
        self.url = "http://foundry-models.invalid/v1/chat/completions"

    def test_streaming_non_json_and_non_post_requests_are_rejected(self):
        responses = [
            self.http.post(self.url, json={"model": "m", "messages": [], "stream": True}),
            self.http.post(self.url, content=b"not json"),
            self.http.post(self.url, json=["not", "an", "object"]),
            self.http.get(self.url),
            self.http.post("http://foundry-models.invalid/v1/responses", json={"model": "m"}),
        ]

        self.assertEqual([response.status_code for response in responses], [400, 400, 400, 404, 404])
        for response in responses:
            self.assertIn("message", response.json()["error"])
        self.assertEqual(self.model.requests, [])

    def test_successful_response_is_openai_json(self):
        response = self.http.post(self.url, json={"model": "m", "messages": [{"role": "user", "content": "x"}]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(json.loads(response.content)["model"], "m")

    def test_models_are_validated_at_construction(self):
        with self.assertRaises(ValueError):
            _transport({})
        with self.assertRaises(TypeError):
            _transport(object())
        with self.assertRaises(TypeError):
            _transport({"m": _FakeFoundryModel(), "n": object()})


def _raw_config():
    return {
        "llm": _settings(),
        "documents": {"variable_groups_path": str(REPO_ROOT / "config" / "variable_groups.json")},
    }


class RuntimeModelKwargsTests(_HttpClientCase):
    def test_only_http_client_is_accepted(self):
        for kwargs in ({"timeout": 5}, {"http_client": None, "model": "x"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CipocConfig(_raw_config(), model_kwargs=kwargs)

    def test_model_kwargs_are_shared_by_reference_and_not_mutable_through_the_view(self):
        client = self.client(_transport(_FakeFoundryModel()))
        config = CipocConfig(_raw_config(), model_kwargs={"http_client": client})

        view = config.model_kwargs
        view["http_client"] = None

        self.assertIs(config.model_kwargs["http_client"], client)
        self.assertEqual(CipocConfig(_raw_config()).model_kwargs, {})

    def test_load_config_passes_runtime_kwargs_through(self):
        client = self.client(_transport(_FakeFoundryModel()))
        with patch.dict(os.environ, {"AZURE_OPENAI_URL": "https://load.invalid/v1", "RENCI_AZURE_API_KEY": "test"}):
            config = load_config(REPO_ROOT / "config" / "config.yaml", model_kwargs={"http_client": client})

        self.assertIs(config.model_kwargs["http_client"], client)

    def test_orchestrator_shares_one_client_with_every_subagent(self):
        model = _FakeFoundryModel(content="hello")
        client = self.client(_transport(model))

        orchestrator = OrchestratorAgent(config=CipocConfig(_raw_config(), model_kwargs={"http_client": client}))

        components = (orchestrator, orchestrator._scanner, orchestrator._retriever, orchestrator._extractor)
        for component in components:
            self.assertIs(component.agent._model.http_client, client)
        orchestrator._extractor.agent.invoke([{"role": "user", "content": "x"}])
        self.assertEqual(len(model.requests), 1)

    def test_explicit_kwargs_win_and_a_supplied_llm_short_circuits(self):
        shared = self.client(_transport(_FakeFoundryModel()))
        explicit = self.client(_transport(_FakeFoundryModel()))
        config = CipocConfig(_raw_config(), model_kwargs={"http_client": shared})

        scanner = NoteScannerAgent(config=config, http_client=explicit)
        supplied = OpenAIAgentModel(_settings())
        injected = NoteScannerAgent(supplied, config=config)

        self.assertIs(scanner.agent._model.http_client, explicit)
        self.assertIs(injected.agent, supplied)


if __name__ == "__main__":
    unittest.main()
