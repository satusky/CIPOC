import json
import os
from functools import partial
from weakref import finalize

from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient
from pydantic import AliasChoices, BaseModel, Field, ConfigDict
from typing import ClassVar, Literal

from .base import BaseAgentModel, EndpointURL, LLMConfig
from .usage_evidence import capture_usage_request, capture_usage_response


class OpenAIReasoning(BaseModel):
    effort: Literal["low", "medium", "high"] = Field(default="medium", description="Reasoning effort")
    summary: Literal["detailed", "auto"] | None = Field(default="auto", description="Summarization of reasoning output")


class OpenAIConfig(LLMConfig):
    provider: str = "openai"
    base_url: EndpointURL = Field(
        validation_alias=AliasChoices("base_url", "openai_api_base"),
        description="Explicit HTTP(S) model endpoint URL with a host",
    )
    endpoint_compatibility: Literal["standard", "databricks"] = Field(
        default="standard",
        description="Opt-in endpoint compatibility behavior; standard preserves native LangChain/OpenAI handling.",
    )
    reasoning: OpenAIReasoning | None = Field(description="Responses API reasoning args", default_factory=OpenAIReasoning)
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class OpenAIAgentModel(BaseAgentModel):
    _non_model_fields: ClassVar[set[str]] = (
        BaseAgentModel._non_model_fields | {"endpoint_compatibility"}
    )

    def __init__(self, config: OpenAIConfig | dict, **kwargs):
        if isinstance(config, dict):
            config = OpenAIConfig(**config)
        self._endpoint_compatibility = config.endpoint_compatibility
        super().__init__(config, **kwargs)

    def _model_kwargs(self, **overrides) -> dict:
        # Resolve aliases before merging so the limiter and client use the same
        # effective endpoint/model. Canonical names win within an override layer.
        for alias, name in (("openai_api_base", "base_url"), ("model_name", "model")):
            if alias in overrides:
                value = overrides.pop(alias)
                overrides.setdefault(name, value)
        model_kwargs = super()._model_kwargs(**overrides)
        model_kwargs.pop("openai_api_base", None)
        model_kwargs.pop("model_name", None)
        return model_kwargs

    def _initialize_model(self, **model_kwargs) -> ChatOpenAI:
        if model_kwargs.get("use_responses_api") is False:
            reasoning = model_kwargs.pop("reasoning", None)
            if reasoning is not None:
                effort = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning.effort
                model_kwargs.setdefault("reasoning_effort", effort)
        # Never mutate caller clients or interfere with LangChain's proxy path.
        if (
            any(model_kwargs.get(key) is not None for key in ("http_client", "client", "root_client"))
            or model_kwargs.get("openai_proxy", os.getenv("OPENAI_PROXY"))
        ):
            return ChatOpenAI(**model_kwargs)
        owner = object()
        client = DefaultHttpxClient(
            base_url=model_kwargs["base_url"],
            timeout=model_kwargs.get("timeout", model_kwargs.get("request_timeout")),
            event_hooks={
                "request": [partial(capture_usage_request, owner)],
                "response": [partial(capture_usage_response, owner)],
            },
        )
        try:
            model = ChatOpenAI(**{**model_kwargs, "http_client": client})
        except BaseException:
            client.close()
            raise
        self._usage_evidence_owner = owner
        # Match the default clients' cleanup without tying an escaped model's
        # lifetime to its wrapper, or retaining the model in a finalizer closure.
        finalize(model.root_client if model.root_client is not None else model, client.close)
        return model

    def _structured_runnable(self, schema):
        if not (
            self._endpoint_compatibility == "databricks"
            and self._structured_output_method == "json_schema"
        ):
            return super()._structured_runnable(schema)

        # A Pydantic response_format makes the OpenAI SDK eagerly parse
        # message.content before LangChain can handle content block lists.
        request_schema = (
            schema.model_json_schema()
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else schema
        )
        return self.model.with_structured_output(
            request_schema,
            method="json_schema",
            include_raw=True,
        )

    def _parse_structured_result(self, schema, result):
        if not (
            self._endpoint_compatibility == "databricks"
            and self._structured_output_method == "json_schema"
        ):
            return super()._parse_structured_result(schema, result)
        # Upstream parsers can repair truncated JSON and still populate parsed.
        # Always require one complete raw JSON document, using only text blocks
        # (not the adjacent Databricks reasoning blocks).
        raw = result["raw"]
        for metadata in (raw.response_metadata, raw.additional_kwargs):
            if metadata.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("Databricks structured output was truncated or content-filtered.")
            if metadata.get("refusal"):
                raise ValueError("Databricks structured output was refused.")
            if metadata.get("status") in {
                "incomplete", "failed", "cancelled", "in_progress", "queued",
            } or metadata.get("incomplete_details"):
                raise ValueError("Databricks structured output is incomplete.")
        for block in raw.content if isinstance(raw.content, list) else []:
            if isinstance(block, dict) and block.get("type") == "non_standard":
                block = block.get("value")
            if isinstance(block, dict) and block.get("type") == "refusal":
                raise ValueError("Databricks structured output was refused.")

        def reject_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        parsed = json.loads(raw.text, parse_constant=reject_constant)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(parsed)
        return parsed


if __name__ == "__main__":
    import os
    import argparse
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What is the the deal with airline food?")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    prompt = args.prompt
    model = args.model
    endpoint = args.endpoint or os.environ.get("AZURE_OPENAI_URL")
    api_key = args.api_key or os.environ.get("RENCI_AZURE_API_KEY")

    messages = [{"role": "user", "content": prompt}]
    reasoning = {"effort": "medium", "summary": "detailed"}
    config = dict(
        model=model,
        api_key=api_key,
        base_url=endpoint,
        reasoning=reasoning
    )
    
    client = OpenAIAgentModel(config)
    completion = client.invoke(messages)
    print(completion)
