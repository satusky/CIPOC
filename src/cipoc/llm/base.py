from abc import ABC, abstractmethod
from threading import Semaphore

from typing import ClassVar, Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool


class LLMConfig(BaseModel):
    model: str = Field(description="Name of LLM")
    api_key: SecretStr = Field(description="API key")
    base_url: str = Field(description="Base URL for model endpoint")
    max_concurrency: int | None = Field(default=None, description="Max number of concurrent instances for specified endpoint.")
    provider: str | None = Field(default=None, description="Model provider (discriminator). Subclasses narrow this with a concrete default.")
    tools: list[StructuredTool] | None = Field(default=None, description="List of available tools")
    structured_output_method: Literal["function_calling", "json_mode", "json_schema"] = Field(
        default="json_schema",
        description="LangChain method used to request structured model output.",
    )
    model_config = ConfigDict(protected_namespaces=())


class BaseAgentModel(ABC):
    _model: BaseChatModel
    _config: LLMConfig
    _tools: list[StructuredTool] | None
    _non_model_fields: ClassVar[set[str]] = {
        "tools",
        "provider",
        "max_concurrency",
        "structured_output_method",
    }

    def __init__(self, config: LLMConfig, **kwargs) -> None:
        self._config = config
        self._tools = kwargs.pop("tools") if "tools" in kwargs else self._config.tools
        self._structured_output_method = self._config.structured_output_method
        self._model = self._initialize_model(**kwargs)
        self._semaphore = Semaphore(self._config.max_concurrency) if self._config.max_concurrency else None

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
        if self._semaphore is None:
            return self.model.invoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )

        with self._semaphore:
            return self.model.invoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )

    async def ainvoke(self, messages, *, config=None, stop=None, **kwargs):
        return await self.model.ainvoke(
            messages,
            config,
            stop=stop,
            **kwargs
        )

    def structured(self, schema, messages, **kwargs):
        """Invoke the model with structured output under the concurrency guard.

        The single endpoint permit is held across the whole call, including any
        ``ChatOpenAI`` retry/backoff inside ``.invoke`` — a throttled request keeps
        occupying a slot until it resolves. Node call sites should route through
        this rather than ``self.model.with_structured_output(...).invoke(...)``,
        which bypasses the semaphore.
        """
        runnable = self._structured_runnable(schema)
        if self._semaphore is None:
            result = runnable.invoke(messages, **kwargs)
            return self._parse_structured_result(schema, result)
        with self._semaphore:
            result = runnable.invoke(messages, **kwargs)
            return self._parse_structured_result(schema, result)

    async def astructured(self, schema, messages, **kwargs):
        """Async sibling of :meth:`structured` — currently an unguarded passthrough.

        The concurrency guard is intentionally omitted: acquiring the
        ``threading.Semaphore`` across an ``await`` would block the event loop
        whenever it had to wait for a permit. To bound the async path, swap
        ``self._semaphore`` to an ``asyncio.Semaphore`` and guard with
        ``async with self._semaphore:`` here. Until then the sync
        :meth:`structured` path is the bounded one.
        """
        runnable = self._structured_runnable(schema)
        result = await runnable.ainvoke(messages, **kwargs)
        return self._parse_structured_result(schema, result)

    def _structured_runnable(self, schema):
        return self.model.with_structured_output(
            schema, method=self._structured_output_method
        )

    def _parse_structured_result(self, schema, result):
        return result
