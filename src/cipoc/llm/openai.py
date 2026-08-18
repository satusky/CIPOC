from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, ConfigDict
from typing import ClassVar, Literal

from .base import BaseAgentModel, LLMConfig


class OpenAIReasoning(BaseModel):
    effort: Literal["low", "medium", "high"] = Field(default="medium", description="Reasoning effort")
    summary: Literal["detailed", "auto"] | None = Field(default="auto", description="Summarization of reasoning output")


class OpenAIConfig(LLMConfig):
    provider: str = "openai"
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

    def _initialize_model(self, **kwargs) -> ChatOpenAI:
        model_kwargs = self._model_kwargs(**kwargs)
        if model_kwargs.get("use_responses_api") is False:
            reasoning = model_kwargs.pop("reasoning", None)
            if reasoning is not None:
                effort = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning.effort
                model_kwargs.setdefault("reasoning_effort", effort)
        return ChatOpenAI(**model_kwargs)

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
        if result["parsed"] is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(result["parsed"])
            return result["parsed"]

        # Databricks returns JSON as a text block alongside reasoning blocks.
        raw = result["raw"]
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return PydanticOutputParser(pydantic_object=schema).parse(raw.text)
        return JsonOutputParser().parse(raw.text)


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
