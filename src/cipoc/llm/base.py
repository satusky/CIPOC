from abc import ABC, abstractmethod
from typing import ClassVar
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool


class LLMConfig(BaseModel):
    model: str = Field(description="Name of LLM")
    api_key: SecretStr = Field(description="API key")
    base_url: str = Field(description="Base URL for model endpoint")
    provider: str | None = Field(default=None, description="Model provider (discriminator). Subclasses narrow this with a concrete default.")
    tools: list[StructuredTool] | None = Field(default=None, description="List of available tools")
    model_config = ConfigDict(protected_namespaces=())


class BaseAgentModel(ABC):
    _model: BaseChatModel
    _config: LLMConfig
    _tools: list[StructuredTool] | None
    _non_model_fields: ClassVar[set[str]] = {"tools", "provider"}

    def __init__(self, config: LLMConfig, **kwargs) -> None:
        self._config = config
        self._tools = kwargs.pop("tools") if "tools" in kwargs else self._config.tools
        self._model = self._initialize_model(**kwargs)

    @property
    def model(self) -> BaseChatModel:
        if self._tools is not None:
            return self._model.bind_tools(self._tools)
        return self._model

    def _model_kwargs(self, **overrides) -> dict:
        kwargs = self._config.model_dump(exclude=self._non_model_fields)
        kwargs.update(overrides)
        return kwargs

    @abstractmethod
    def _initialize_model(self, **kwargs) -> BaseChatModel:
        ...

    def invoke(self, messages, *, config=None, stop=None, **kwargs):
        result = self.model.invoke(
            messages,
            config,
            stop=stop,
            **kwargs
        )
        return result

    async def ainvoke(self, messages, *, config=None, stop=None, **kwargs):
        result = await self.model.ainvoke(
            messages,
            config,
            stop=stop,
            **kwargs
        )
        return result


