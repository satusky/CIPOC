import asyncio
from abc import ABC, abstractmethod
from threading import Semaphore

from typing import ClassVar
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
    model_config = ConfigDict(protected_namespaces=())


class BaseAgentModel(ABC):
    _model: BaseChatModel
    _config: LLMConfig
    _tools: list[StructuredTool] | None
    _non_model_fields: ClassVar[set[str]] = {"tools", "provider", "max_concurrency"}

    def __init__(self, config: LLMConfig, **kwargs) -> None:
        self._config = config
        self._tools = kwargs.pop("tools") if "tools" in kwargs else self._config.tools
        self._model = self._initialize_model(**kwargs)
        self._semaphore = Semaphore(self._config.max_concurrency) if self._config.max_concurrency else None
        # Built on first async use rather than here: an asyncio.Semaphore binds to
        # the loop it is first awaited on, and the model is constructed off-loop.
        self._async_semaphore: asyncio.Semaphore | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

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

    def _aguard(self) -> asyncio.Semaphore | None:
        """The endpoint permit for the async path, created on the running loop.

        Cached per loop rather than once: a semaphore is bound to the loop it is
        first awaited on, and a process can run ``asyncio.run`` more than once —
        a notebook cell re-run would otherwise await a semaphore tied to a closed
        loop. Falsy ``max_concurrency`` means unbounded, matching the sync path
        (an ``asyncio.Semaphore(0)`` would deadlock rather than run unbounded).
        """
        if not self._config.max_concurrency:
            return None
        loop = asyncio.get_running_loop()
        if self._async_semaphore is None or self._async_loop is not loop:
            self._async_semaphore = asyncio.Semaphore(self._config.max_concurrency)
            self._async_loop = loop
        return self._async_semaphore

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
        guard = self._aguard()
        if guard is None:
            return await self.model.ainvoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )

        async with guard:
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
        runnable = self.model.with_structured_output(schema)
        if self._semaphore is None:
            return runnable.invoke(messages, **kwargs)
        with self._semaphore:
            return runnable.invoke(messages, **kwargs)

    async def astructured(self, schema, messages, **kwargs):
        """Async sibling of :meth:`structured`, bounded by :meth:`_aguard`.

        Guards with an ``asyncio.Semaphore`` rather than the sync one: acquiring
        a ``threading.Semaphore`` across an ``await`` would block the whole event
        loop whenever it had to wait for a permit. Otherwise the contract is the
        same — one permit held for the duration of the call, and node call sites
        should route through this rather than
        ``self.model.with_structured_output(...).ainvoke(...)``, which bypasses it.
        """
        runnable = self.model.with_structured_output(schema)
        guard = self._aguard()
        if guard is None:
            return await runnable.ainvoke(messages, **kwargs)
        async with guard:
            return await runnable.ainvoke(messages, **kwargs)


