from abc import ABC, abstractmethod
from ipaddress import IPv4Address, IPv6Address
import re
from threading import BoundedSemaphore, Lock
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

from typing import Annotated, ClassVar, Literal
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr, StringConstraints, TypeAdapter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool


def _validate_endpoint_url(value: str) -> str:
    try:
        # urlsplit alone silently removes some controls and permits malformed
        # authorities. Validate before handing an explicit URL to the SDK.
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError
        if "\\" in value or re.search(r"%(?![0-9a-fA-F]{2})", value):
            raise ValueError
        url = urlsplit(value)
        host = url.hostname
        authority = url.netloc.rsplit("@", 1)[-1]
        if url.scheme not in {"http", "https"} or not host:
            raise ValueError
        if not re.fullmatch(r"(?:\[[^\[\]]+\]|[^:\[\]]+)(?::[0-9]+)?", authority):
            raise ValueError
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError
        # Allow private/single-label DNS hosts (including underscores), without
        # requiring public DNS. Bracketed addresses must be usable IPv6.
        if authority.startswith("["):
            IPv6Address(host)
        elif re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", host):
            IPv4Address(host)
        elif not re.fullmatch(
            r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*\.?", host.encode("idna").decode("ascii")
        ):
            raise ValueError
    except (ValueError, UnicodeError):
        raise ValueError(
            "base_url must be an explicit HTTP(S) URL with a valid host "
            "and optional port (1-65535)."
        ) from None
    return value


EndpointURL = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
    AfterValidator(_validate_endpoint_url),
]
_EndpointCapacity = Annotated[int, Field(strict=True, gt=0)] | None
_ENDPOINT_URL = TypeAdapter(EndpointURL, config=ConfigDict(hide_input_in_errors=True))
_ENDPOINT_CAPACITY = TypeAdapter(_EndpointCapacity)


class _EndpointLimiter:
    def __init__(self, capacity: int | None):
        self.capacity = capacity
        self.semaphore = BoundedSemaphore(capacity) if capacity is not None else None


_ENDPOINT_LIMITERS: WeakValueDictionary[
    tuple[str, str, int, str, str | None, str], _EndpointLimiter
] = WeakValueDictionary()
_ENDPOINT_LIMITERS_LOCK = Lock()


def _endpoint_limiter(endpoint: str, model: str, capacity: int | None) -> _EndpointLimiter:
    url = urlsplit(endpoint)
    # Match the SDK's appended slash only for query-free URLs: its raw-path
    # joining treats query-bearing URLs differently. Never collapse path case,
    # escapes, or queries. Credentials/fragments do not split a model's budget.
    # urlsplit lowercases schemes and DNS hostnames.
    query = url.query if "?" in endpoint.partition("#")[0] else None
    path = url.path
    if query is None and not path.endswith("/"):
        path += "/"
    host = str(IPv6Address(url.hostname)) if ":" in url.hostname else url.hostname
    key = (url.scheme, host, url.port or (443 if url.scheme == "https" else 80), path, query, model)
    with _ENDPOINT_LIMITERS_LOCK:
        limiter = _ENDPOINT_LIMITERS.get(key)
        if limiter is None:
            limiter = _EndpointLimiter(capacity)
            _ENDPOINT_LIMITERS[key] = limiter
        elif limiter.capacity != capacity:
            raise ValueError(
                "Conflicting max_concurrency for the same model at the same endpoint: "
                f"active capacity is {limiter.capacity!r}, requested {capacity!r}. "
                "All live wrappers for that endpoint/model pair must use the same capacity, including None."
            )
        return limiter


class LLMConfig(BaseModel):
    model: str = Field(description="Name of LLM")
    api_key: SecretStr = Field(description="API key")
    base_url: EndpointURL = Field(description="Explicit HTTP(S) model endpoint URL with a host")
    max_concurrency: _EndpointCapacity = Field(default=None, description="Process-wide synchronous capacity for this model at this endpoint; positive integer or None for unbounded.")
    provider: str | None = Field(default=None, description="Model provider (discriminator). Subclasses narrow this with a concrete default.")
    tools: list[StructuredTool] | None = Field(default=None, description="List of available tools")
    structured_output_method: Literal["function_calling", "json_mode", "json_schema"] = Field(
        default="json_schema",
        description="LangChain method used to request structured model output.",
    )
    model_config = ConfigDict(protected_namespaces=(), hide_input_in_errors=True)


class BaseAgentModel(ABC):
    """Model wrapper with a process-local, synchronous per-model budget.

    Separate live wrappers share a budget keyed by effective base URL and exact
    requested model name, regardless of credentials. URL scheme/host case, default
    ports, and the SDK's appended slash are normalized; paths and queries remain
    distinct. Different models at the same URL have independent budgets, with no
    aggregate endpoint cap. Provider-reported model names do not change the key.
    The weak registry releases a budget when its last wrapper/call is gone.
    Async methods and direct
    access through ``model`` remain unguarded; processes/Databricks workers do not
    share permits. This is not a distributed or mixed sync/async rate limiter.
    """

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
        capacity = _ENDPOINT_CAPACITY.validate_python(
            kwargs.pop("max_concurrency", self._config.max_concurrency)
        )
        model_kwargs = self._model_kwargs(**kwargs)
        limiter = _endpoint_limiter(model_kwargs["base_url"], model_kwargs["model"], capacity)
        try:
            self._model = self._initialize_model(**model_kwargs)
        except BaseException:
            # A retained initialization traceback must not reserve a budget.
            del limiter
            raise
        self._endpoint_limiter = limiter
        self._semaphore = limiter.semaphore

    @property
    def model(self) -> BaseChatModel:
        if self._tools is not None:
            return self._model.bind_tools(self._tools)
        return self._model

    def _model_kwargs(self, **overrides) -> dict:
        kwargs = self._config.model_dump(exclude=self._non_model_fields)
        kwargs.update(overrides)
        # Validate after constructor overrides, before a provider can silently
        # fall back to an environment/default public endpoint.
        kwargs["base_url"] = _ENDPOINT_URL.validate_python(kwargs.get("base_url"))
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
        """Async passthrough; does not acquire the synchronous per-model budget."""
        return await self.model.ainvoke(
            messages,
            config,
            stop=stop,
            **kwargs
        )

    def structured(self, schema, messages, **kwargs):
        """Invoke the model with structured output under the concurrency guard.

        The per-model permit is held across the whole call, including any
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
        whenever it had to wait for a permit. A separate async semaphore would
        not enforce a combined sync/async budget either. Until a coordinated
        limiter is implemented, only the sync path is bounded.
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
