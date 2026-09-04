"""Execution observability derived from normalized LangGraph events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import threading
from typing import Any, Callable, Iterable, Mapping
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from cipoc.models.observability import (
    AttemptMode,
    LLMExchange,
    LLMUsageBucket,
    LLMUsageSummary,
    NormalizedTokenUsage,
    VariableAttempt,
)

from .progress.events import ProgressEvent, field


@dataclass(frozen=True)
class TaskBinding:
    """Entity and extraction-attempt context for one graph task scope."""

    entity_key: str
    graph_node: str
    semantic_attempt: int | None = None
    transport_retry_ordinal: int | None = None


@dataclass(frozen=True)
class _CapturedLLMCall:
    """One completed callback lifecycle before graph-task correlation."""

    namespace: tuple[str, ...]
    graph_node: str
    run_id: str
    model: str | None
    prompt_messages: list[dict[str, Any]] | None = None
    response: Any = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    transport_retry_ordinal: int | None = None


_LLM_AGENT_BY_NODE = {
    "detect_concepts": "note_scanner",
    "summarize_note": "note_scanner",
    "get_cancer_mentions": "note_scanner",
    "identify_relevant_notes": "note_retriever",
    "extract_group_values": "extractor",
    "extract_individual_value": "extractor",
    "repair_invalid_extraction": "extractor",
}


def _value(obj: Any, name: str, default: Any = None) -> Any:
    """Read callback data from live objects or provider-shaped mappings."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        kwargs = obj.get("kwargs")
        if isinstance(kwargs, Mapping) and name in kwargs:
            return kwargs[name]
        return default
    return getattr(obj, name, default)


def _visible_content(content: Any) -> str:
    """Flatten visible text blocks while deliberately excluding reasoning."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        block_type = str(_value(block, "type", "")).lower()
        if "reason" in block_type or "thinking" in block_type:
            continue
        text = _value(block, "text")
        if text is None and block_type in {"text", "output_text", "input_text", ""}:
            text = _value(block, "content")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _prompt_message(
    message: Any, max_content_chars: int | None = None
) -> dict[str, Any]:
    role = _value(message, "type") or _value(message, "role") or "unknown"
    content = _visible_content(_value(message, "content", message))
    result = {
        "role": str(role),
        "content": content,
    }
    if max_content_chars is not None:
        original_char_count = len(content)
        result.update(
            content=content[:max_content_chars],
            truncated=original_char_count > max_content_chars,
            original_char_count=original_char_count,
        )
    return result


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _tool_arguments(message: Any) -> Any:
    tool_calls = _value(message, "tool_calls") or []
    additional = _value(message, "additional_kwargs", {}) or {}
    if not tool_calls and isinstance(additional, Mapping):
        tool_calls = additional.get("tool_calls") or []

    if not tool_calls:
        content = _value(message, "content")
        if isinstance(content, (list, tuple)):
            tool_calls = [
                block
                for block in content
                if str(_value(block, "type", "")).lower()
                in {"tool_call", "function_call"}
            ]

    arguments: list[Any] = []
    for call in tool_calls:
        function = _value(call, "function", {}) or {}
        args = _value(call, "args")
        if args is None:
            args = _value(function, "arguments")
        if args is not None:
            arguments.append(_parse_json(args))
    if len(arguments) == 1:
        return arguments[0]
    return arguments or None


def _first_generation(response: Any) -> Any:
    generations = _value(response, "generations", []) or []
    try:
        return generations[0][0]
    except (IndexError, KeyError, TypeError):
        choices = _value(response, "choices", []) or []
        return choices[0] if choices else response


_TOKEN_ALIASES = {
    "input_tokens": (
        "input_tokens",
        "prompt_tokens",
        "input_token_count",
        "prompt_token_count",
    ),
    "output_tokens": (
        "output_tokens",
        "completion_tokens",
        "output_token_count",
        "completion_token_count",
    ),
    "total_tokens": ("total_tokens", "total_token_count"),
}

_DETAIL_CONTAINERS = {
    "input_token_details": "input",
    "prompt_tokens_details": "input",
    "prompt_token_details": "input",
    "output_token_details": "output",
    "completion_tokens_details": "output",
    "completion_token_details": "output",
}

_INPUT_DETAIL_ALIASES = {
    "cached": "cache_read",
    "cached_tokens": "cache_read",
    "cache_read_tokens": "cache_read",
    "cached_input_tokens": "cache_read",
    "cache_creation_tokens": "cache_creation",
    "cache_creation_input_tokens": "cache_creation",
    "audio_tokens": "audio",
}

_OUTPUT_DETAIL_ALIASES = {
    "audio_tokens": "audio",
    "reasoning_tokens": "reasoning",
    "accepted_prediction_tokens": "accepted_prediction",
    "rejected_prediction_tokens": "rejected_prediction",
}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count >= 0 else None


def _detail_values(value: Any, side: str) -> dict[str, int]:
    """Flatten nested provider details while retaining unknown numeric leaves."""
    aliases = _INPUT_DETAIL_ALIASES if side == "input" else _OUTPUT_DETAIL_ALIASES
    details: dict[str, int] = {}

    def visit(candidate: Any) -> None:
        mapping = _mapping(candidate)
        if mapping is None:
            return
        for raw_key, raw_value in mapping.items():
            nested = _mapping(raw_value)
            if nested is not None:
                visit(nested)
                continue
            count = _token_count(raw_value)
            if count is None:
                continue
            key = str(raw_key)
            canonical = aliases.get(key, key)
            details.setdefault(canonical, count)

    visit(value)
    return details


def _usage_source(value: Any) -> tuple[dict[str, int | None], dict[str, int], dict[str, int]] | None:
    mapping = _mapping(value)
    if mapping is None:
        return None

    scalars: dict[str, int | None] = {}
    recognized = False
    for canonical, aliases in _TOKEN_ALIASES.items():
        scalars[canonical] = None
        for alias in aliases:
            if alias not in mapping:
                continue
            count = _token_count(mapping[alias])
            if count is not None:
                scalars[canonical] = count
                recognized = True
                break

    input_details: dict[str, int] = {}
    output_details: dict[str, int] = {}
    for container, side in _DETAIL_CONTAINERS.items():
        if container not in mapping:
            continue
        target = input_details if side == "input" else output_details
        for key, count in _detail_values(mapping[container], side).items():
            target.setdefault(key, count)
            recognized = True

    if not recognized:
        return None
    return scalars, input_details, output_details


def normalize_token_usage(*sources: Any) -> NormalizedTokenUsage | None:
    """Normalize one invocation's provider-reported usage without double counting.

    The first recognized source is authoritative. Later sources can only fill
    canonical token-detail keys absent from that source; their scalar totals are
    ignored. Detail values are breakdowns of, not additions to, scalar totals.
    """
    normalized_sources = [
        normalized
        for source in sources
        if (normalized := _usage_source(source)) is not None
    ]
    if not normalized_sources:
        return None

    scalars, input_details, output_details = normalized_sources[0]
    input_details = dict(input_details)
    output_details = dict(output_details)
    for _, secondary_input, secondary_output in normalized_sources[1:]:
        for key, count in secondary_input.items():
            input_details.setdefault(key, count)
        for key, count in secondary_output.items():
            output_details.setdefault(key, count)

    input_tokens = scalars["input_tokens"] or 0
    output_tokens = scalars["output_tokens"] or 0
    total_tokens = scalars["total_tokens"]
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return NormalizedTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=input_details,
        output_token_details=output_details,
    )


def _usage_dict(usage: NormalizedTokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    dumped = usage.model_dump(mode="json")
    if not dumped["input_token_details"]:
        dumped.pop("input_token_details")
    if not dumped["output_token_details"]:
        dumped.pop("output_token_details")
    return dumped


def _read_llm_result(
    response: Any,
    *,
    capture_content: bool = True,
) -> tuple[Any, dict[str, Any] | None, str | None]:
    """Extract visible structured output, normalized usage, and model name."""
    generation = _first_generation(response)
    message = _value(generation, "message") or generation

    parsed = None
    if capture_content:
        parsed = _tool_arguments(message)
        if parsed is None:
            content = _visible_content(_value(message, "content"))
            if not content:
                content = _visible_content(_value(generation, "text"))
            parsed = _parse_json(content) if content else None

    response_metadata = _value(message, "response_metadata", {}) or {}
    generation_info = _value(generation, "generation_info", {}) or {}
    llm_output = _value(response, "llm_output", {}) or {}
    usage_candidates = (
        _value(message, "usage_metadata"),
        _value(response_metadata, "token_usage") or _value(response_metadata, "usage"),
        _value(generation_info, "token_usage") or _value(generation_info, "usage"),
        _value(llm_output, "token_usage") or _value(llm_output, "usage"),
    )
    usage = _usage_dict(normalize_token_usage(*usage_candidates))

    model = None
    for source in (response_metadata, generation_info, llm_output):
        model = _value(source, "model_name") or _value(source, "model")
        if model:
            break
    return parsed, usage, str(model) if model else None


def _parse_namespace(metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    namespace = (metadata or {}).get("langgraph_checkpoint_ns", "")
    return _namespace_tuple(str(namespace)) if namespace else ()


def _model_name(
    serialized: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None
) -> str | None:
    for source in (metadata or {}, serialized or {}, _value(serialized, "kwargs", {})):
        model = (
            _value(source, "ls_model_name")
            or _value(source, "model_name")
            or _value(source, "model")
        )
        if model:
            return str(model)
    return None


class LLMCaptureHandler(BaseCallbackHandler):
    """Thread-safe model telemetry capture with optional visible content."""

    raise_error = False

    def __init__(
        self,
        *,
        capture_llm_content: bool = True,
        max_content_chars: int | None = None,
    ) -> None:
        if max_content_chars is not None and (
            isinstance(max_content_chars, bool)
            or not isinstance(max_content_chars, int)
        ):
            raise TypeError("max_content_chars must be an integer or None")
        if max_content_chars is not None and max_content_chars < 0:
            raise ValueError("max_content_chars must be non-negative")
        self._lock = threading.RLock()
        self._calls: list[dict[str, Any]] = []
        self._pending: dict[str, int] = {}
        self._starts: dict[tuple[tuple[str, ...], str], int] = {}
        self.capture_llm_content = capture_llm_content
        self.max_content_chars = max_content_chars

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompt_factory = None
        if self.capture_llm_content:
            prompt_factory = lambda: [
                _prompt_message(message, self.max_content_chars)
                for batch in messages
                for message in batch
            ]
        self._start(serialized, prompt_factory, run_id, metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompt_factory = None
        if self.capture_llm_content:
            prompt_factory = lambda: [
                _prompt_message(
                    {"role": "prompt", "content": prompt},
                    self.max_content_chars,
                )
                for prompt in prompts
            ]
        self._start(
            serialized,
            prompt_factory,
            run_id,
            metadata,
        )

    def _start(
        self,
        serialized: Mapping[str, Any] | None,
        prompt_factory: Callable[[], list[dict[str, Any]]] | None,
        run_id: UUID,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        namespace = _parse_namespace(metadata)
        graph_node = str((metadata or {}).get("langgraph_node", ""))
        retry_key = (namespace, graph_node)
        with self._lock:
            prior_starts = self._starts.get(retry_key, 0)
            self._starts[retry_key] = prior_starts + 1
            self._pending[str(run_id)] = len(self._calls)
            call = {
                "complete": False,
                "namespace": namespace,
                "graph_node": graph_node,
                "run_id": str(run_id),
                "model": _model_name(serialized, metadata),
                "transport_retry_ordinal": prior_starts or None,
            }
            if prompt_factory is not None:
                call["prompt_messages"] = prompt_factory()
            self._calls.append(call)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        parsed, usage, model = _read_llm_result(
            response, capture_content=self.capture_llm_content
        )
        fields = {"usage": usage, "model": model}
        if self.capture_llm_content:
            fields["response"] = parsed
        self._finish(run_id, **fields)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, error=f"{type(error).__name__}: {error}")

    def _finish(self, run_id: UUID, **fields: Any) -> None:
        with self._lock:
            index = self._pending.pop(str(run_id), None)
            if index is None:
                index = len(self._calls)
                self._calls.append(
                    {
                        "namespace": (),
                        "graph_node": "",
                        "run_id": str(run_id),
                        "model": None,
                        "transport_retry_ordinal": None,
                    }
                )
            call = self._calls[index]
            for name, value in fields.items():
                if value is not None or name != "model" or call.get("model") is None:
                    call[name] = value
            call["complete"] = True

    def _snapshot(self) -> list[_CapturedLLMCall]:
        """Return completed calls in invocation-start order."""
        with self._lock:
            return [
                _CapturedLLMCall(
                    namespace=tuple(call["namespace"]),
                    graph_node=call["graph_node"],
                    run_id=call["run_id"],
                    model=call.get("model"),
                    prompt_messages=deepcopy(call.get("prompt_messages")),
                    response=deepcopy(call.get("response")),
                    usage=deepcopy(call.get("usage")),
                    error=call.get("error"),
                    transport_retry_ordinal=call.get("transport_retry_ordinal"),
                )
                for call in self._calls
                if call.get("complete")
            ]


def merge_callback_config(
    config: Mapping[str, Any] | None, callback: BaseCallbackHandler
) -> dict[str, Any]:
    """Add an inheritable callback without dropping any existing graph config."""
    merged = dict(config or {})
    callbacks = merged.get("callbacks")
    if callbacks is None:
        merged["callbacks"] = [callback]
    elif isinstance(callbacks, (list, tuple)):
        if not any(existing is callback for existing in callbacks):
            merged["callbacks"] = [*callbacks, callback]
    else:
        manager = callbacks.copy()
        handlers = [
            *getattr(manager, "handlers", ()),
            *getattr(manager, "inheritable_handlers", ()),
        ]
        if not any(existing is callback for existing in handlers):
            manager.add_handler(callback, inherit=True)
        merged["callbacks"] = manager
    return merged


def _namespace_tuple(namespace: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(namespace, str):
        return tuple(segment for segment in namespace.split("|") if segment)
    return tuple(namespace)


def _candidate_value(candidate: Any) -> Any:
    if isinstance(candidate, BaseModel):
        return candidate.model_dump(mode="json")
    return deepcopy(candidate)


def _exchange_items(
    exchanges: Mapping[str, Any] | Iterable[Any],
) -> Iterable[tuple[str | None, Any]]:
    if isinstance(exchanges, Mapping):
        if "node" in exchanges or "entity_key" in exchanges:
            yield None, exchanges
            return
        for entity_key, values in exchanges.items():
            if isinstance(values, (list, tuple)):
                for value in values:
                    yield str(entity_key), value
            else:
                yield str(entity_key), values
        return
    for value in exchanges:
        yield None, value


def _exchange_value(exchange: Any, name: str, default: Any = None) -> Any:
    if isinstance(exchange, BaseModel):
        return getattr(exchange, name, default)
    return _value(exchange, name, default)


def aggregate_llm_usage(
    exchanges: Mapping[str, Any] | Iterable[Any],
) -> LLMUsageSummary:
    """Aggregate correlated model callback lifecycles into typed usage buckets."""
    bucket_data: dict[tuple[str, str], dict[str, Any]] = {}

    def accumulator(dimension: str, key: str) -> dict[str, Any]:
        return bucket_data.setdefault(
            (dimension, key),
            {
                "logical_keys": set(),
                "model_invocations": 0,
                "successful_invocations": 0,
                "failed_invocations": 0,
                "retry_invocations": 0,
                "usage_reported_invocations": 0,
                "missing_usage_invocations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_token_details": {},
                "output_token_details": {},
            },
        )

    def add(bucket: dict[str, Any], logical_key: tuple[str, str, int], exchange: Any) -> None:
        bucket["logical_keys"].add(logical_key)
        bucket["model_invocations"] += 1
        failed = _exchange_value(exchange, "error") is not None
        bucket["failed_invocations" if failed else "successful_invocations"] += 1
        retry_ordinal = _exchange_value(exchange, "retry_ordinal")
        if retry_ordinal is None:
            retry_ordinal = _exchange_value(exchange, "transport_retry_ordinal")
        if retry_ordinal is not None:
            bucket["retry_invocations"] += 1

        raw_usage = _exchange_value(exchange, "usage")
        if raw_usage is None:
            bucket["missing_usage_invocations"] += 1
            return
        usage = (
            raw_usage
            if isinstance(raw_usage, NormalizedTokenUsage)
            else normalize_token_usage(raw_usage)
        )
        if usage is None:
            bucket["missing_usage_invocations"] += 1
            return
        bucket["usage_reported_invocations"] += 1
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[name] += getattr(usage, name)
        for name in ("input_token_details", "output_token_details"):
            target = bucket[name]
            for key, count in getattr(usage, name).root.items():
                target[key] = target.get(key, 0) + count

    for entity_hint, exchange in _exchange_items(exchanges):
        entity = str(_exchange_value(exchange, "entity_key", entity_hint) or "unknown")
        node = str(_exchange_value(exchange, "node") or "unknown")
        agent = str(_exchange_value(exchange, "agent") or "unknown")
        model = str(_exchange_value(exchange, "model") or "unknown")
        attempt = int(_exchange_value(exchange, "attempt", 1) or 1)
        logical_key = (entity, node, attempt)
        for dimension, key in (
            ("total", "total"),
            ("agent", agent),
            ("node", node),
            ("model", model),
        ):
            add(accumulator(dimension, key), logical_key, exchange)

    def finish(data: dict[str, Any]) -> LLMUsageBucket:
        values = dict(data)
        values["logical_calls"] = len(values.pop("logical_keys"))
        return LLMUsageBucket(**values)

    total = finish(accumulator("total", "total"))
    return LLMUsageSummary(
        **total.model_dump(),
        by_agent={
            key: finish(data)
            for (dimension, key), data in bucket_data.items()
            if dimension == "agent"
        },
        by_node={
            key: finish(data)
            for (dimension, key), data in bucket_data.items()
            if dimension == "node"
        },
        by_model={
            key: finish(data)
            for (dimension, key), data in bucket_data.items()
            if dimension == "model"
        },
    )


class ObservabilityCollector:
    """Correlate nested graph tasks without depending on a display or frontend."""

    def __init__(
        self,
        *,
        capture_llm_content: bool | None = None,
        max_content_chars: int | None = None,
        capture_llm: bool | None = None,
    ) -> None:
        if capture_llm_content is None:
            capture_llm_content = capture_llm if capture_llm is not None else False
        elif capture_llm is not None and capture_llm != capture_llm_content:
            raise ValueError(
                "capture_llm and capture_llm_content cannot specify different values"
            )
        self._lock = threading.RLock()
        self._entity_scopes: dict[tuple[str, ...], str] = {}
        self._task_bindings: dict[tuple[str, ...], TaskBinding] = {}
        self._variable_attempts: dict[str, list[VariableAttempt]] = {}
        self.capture_llm_content = capture_llm_content
        self.max_content_chars = max_content_chars
        self._llm_callback = LLMCaptureHandler(
            capture_llm_content=capture_llm_content,
            max_content_chars=max_content_chars,
        )

    @property
    def llm_callback(self) -> LLMCaptureHandler:
        """The callback to attach for model metadata and optional content."""
        return self._llm_callback

    def graph_config(
        self, config: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Return graph config with this collector's callback attached."""
        return merge_callback_config(config, self._llm_callback)

    def observe(self, event: ProgressEvent) -> None:
        """Consume one event in stream order."""
        with self._lock:
            if event.kind == "task_start":
                self._observe_start(event)
            elif event.kind == "task_end" and event.node == "validate_extraction":
                self._observe_validation(event)

    def __call__(self, event: ProgressEvent) -> None:
        self.observe(event)

    def binding_for(
        self,
        namespace: str | Iterable[str],
        *,
        transport_retry_ordinal: int | None = None,
    ) -> TaskBinding | None:
        """Return the nearest task binding for a graph/callback namespace."""
        with self._lock:
            scope = self._resolve_scope(
                self._task_bindings, _namespace_tuple(namespace)
            )
            if scope is None:
                return None
            binding = self._task_bindings[scope]
            if transport_retry_ordinal is None:
                return binding
            return replace(
                binding, transport_retry_ordinal=transport_retry_ordinal
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-ready copy of the captured channels."""
        with self._lock:
            captured_calls = self._llm_callback._snapshot()
            snapshot = {
                "llm_content_captured": self.capture_llm_content,
                "max_content_chars": self.max_content_chars,
                "content_truncated": False,
                "variable_attempts": {
                    key: [attempt.model_dump(mode="json") for attempt in attempts]
                    for key, attempts in self._variable_attempts.items()
                },
                "llm_exchanges": {},
            }
            exchanges: dict[str, list[dict[str, Any]]] = {}
            for call in captured_calls:
                binding = self.binding_for(call.namespace)
                if binding is None:
                    continue
                node = call.graph_node or binding.graph_node
                exchange = {
                    "entity_key": binding.entity_key,
                    "agent": _LLM_AGENT_BY_NODE.get(node, "unknown"),
                    "node": node,
                    "attempt": binding.semantic_attempt or 1,
                    "model": call.model,
                    "usage": deepcopy(call.usage),
                    "error": call.error,
                }
                if self.capture_llm_content:
                    exchange["prompt_messages"] = deepcopy(call.prompt_messages or [])
                    exchange["response"] = deepcopy(call.response)
                    if any(
                        message.get("truncated", False)
                        for message in call.prompt_messages or ()
                    ):
                        snapshot["content_truncated"] = True
                if call.transport_retry_ordinal is not None:
                    exchange["retry_ordinal"] = call.transport_retry_ordinal
                exchanges.setdefault(binding.entity_key, []).append(exchange)
            snapshot["llm_exchanges"] = exchanges
            snapshot["llm_usage_summary"] = aggregate_llm_usage(exchanges).model_dump(
                mode="json"
            )
            return snapshot

    def _observe_start(self, event: ProgressEvent) -> None:
        inherited = self._resolve(self._entity_scopes, event.namespace)
        entity_key = self._entity_from_start(event, inherited)
        if entity_key is None:
            return

        semantic_attempt = self._semantic_attempt(event)
        self._entity_scopes[event.scope] = entity_key
        self._task_bindings[event.scope] = TaskBinding(
            entity_key=entity_key,
            graph_node=event.node,
            semantic_attempt=semantic_attempt,
        )

    def _observe_validation(self, event: ProgressEvent) -> None:
        task = field(event.payload, "task")
        if task is None:
            return
        variable = field(task, "variable")
        item_id = field(variable, "item_id")
        if item_id is None:
            return

        entity_key = self._resolve(self._entity_scopes, event.namespace)
        if entity_key is None or not entity_key.startswith("group:"):
            return
        group_key = entity_key.split("/variable:", 1)[0]
        variable_key = f"{group_key}/variable:{int(item_id)}"

        attempt = int(field(task, "extraction_attempts", 0) or 0)
        extraction_mode = str(field(task, "extraction_mode", "individual"))
        mode: AttemptMode = (
            "repair"
            if attempt > 1
            else "group"
            if extraction_mode == "group"
            else "individual"
        )
        record = VariableAttempt(
            attempt=attempt,
            mode=mode,
            candidate=_candidate_value(field(task, "candidate")),
            validation_errors=[
                str(error) for error in (field(task, "validation_errors", []) or [])
            ],
            is_valid=bool(field(task, "is_valid", False)),
        )
        self._variable_attempts.setdefault(variable_key, []).append(record)

    def _entity_from_start(
        self, event: ProgressEvent, inherited: str | None
    ) -> str | None:
        if event.node == "note_branch":
            note_id = field(event.payload, "note_id")
            if note_id is None:
                note_id = field(field(event.payload, "note"), "note_id")
            if note_id is not None:
                return f"note:{note_id}"

        requested = field(event.payload, "requested_variables")
        group_id = field(requested, "group_id")
        if group_id is not None:
            group_key = f"group:{group_id}"
            inherited_group = (
                inherited.split("/variable:", 1)[0] if inherited else None
            )
            if inherited_group != group_key:
                inherited = group_key

        if event.node == "variable_branch":
            item_id = field(field(field(event.payload, "task"), "variable"), "item_id")
            if item_id is not None and inherited is not None:
                group_key = inherited.split("/variable:", 1)[0]
                return f"{group_key}/variable:{int(item_id)}"
        return inherited

    @staticmethod
    def _semantic_attempt(event: ProgressEvent) -> int | None:
        if event.node == "extract_group_values":
            return 1
        if event.node not in {
            "extract_individual_value",
            "repair_invalid_extraction",
            "validate_extraction",
        }:
            return None
        attempts = int(
            field(field(event.payload, "task"), "extraction_attempts", 0) or 0
        )
        if event.node in {"extract_individual_value", "repair_invalid_extraction"}:
            return attempts + 1
        return attempts

    @classmethod
    def _resolve(
        cls, scopes: Mapping[tuple[str, ...], Any], namespace: tuple[str, ...]
    ) -> Any:
        scope = cls._resolve_scope(scopes, namespace)
        return None if scope is None else scopes[scope]

    @staticmethod
    def _resolve_scope(
        scopes: Mapping[tuple[str, ...], Any], namespace: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        for length in range(len(namespace), -1, -1):
            candidate = namespace[:length]
            if candidate in scopes:
                return candidate
        return None


__all__ = [
    "AttemptMode",
    "LLMCaptureHandler",
    "ObservabilityCollector",
    "TaskBinding",
    "VariableAttempt",
    "aggregate_llm_usage",
    "merge_callback_config",
    "normalize_token_usage",
]
