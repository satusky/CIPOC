import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any

from .base import BaseAgentModel, LLMConfig


DEFAULT_LIMITER_SETTINGS = {
    "max_retries": 10,
    "initial_delay": 1.0,
    "exponential_base": 2.0,
    "max_delay": 60.0,
    "jitter": True,
    "retry_on": (RateLimitError,),
    "logger_name": None,
}


class OpenAIReasoning(BaseModel):
    effort: Literal["low", "medium", "high"] = Field(default="medium", description="Reasoning effort")
    summary: Literal["detailed", "auto"] | None = Field(default="auto", description="Summarization of reasoning output")


class OpenAIConfig(LLMConfig):
    provider: str = "openai"
    reasoning: OpenAIReasoning = Field(description="Reasoning args", default_factory=OpenAIReasoning)
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class OpenAIAgentModel(BaseAgentModel):
    def __init__(self, config: OpenAIConfig | dict, **kwargs):
        if isinstance(config, dict):
            config = OpenAIConfig(**config)
        super().__init__(config, **kwargs)

    def _initialize_model(self, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(**self._model_kwargs(**kwargs))


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