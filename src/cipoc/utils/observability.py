"""Execution observability derived from normalized LangGraph events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import threading
from typing import Any, Iterable, Literal, Mapping
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from .progress.events import ProgressEvent, field


AttemptMode = Literal["group", "individual", "repair"]


@dataclass(frozen=True)
class TaskBinding:
    """Entity and extraction-attempt context for one graph task scope."""

    entity_key: str
    graph_node: str
    semantic_attempt: int | None = None
    transport_retry_ordinal: int | None = None


@dataclass(frozen=True)
class VariableAttempt:
    """The candidate and verdict emitted by one validation task."""

    attempt: int
    mode: AttemptMode
    candidate: Any
    validation_errors: tuple[str, ...]
    is_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "mode": self.mode,
            "candidate": deepcopy(self.candidate),
            "validation_errors": list(self.validation_errors),
            "is_valid": self.is_valid,
        }


@dataclass(frozen=True)
class _CapturedLLMCall:
    """One completed callback lifecycle before graph-task correlation."""

    namespace: tuple[str, ...]
    graph_node: str
    run_id: str
    model: str | None
    prompt_messages: list[dict[str, Any]]
    response: Any = None
    usage: dict[str, int] | None = None
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


def _prompt_message(message: Any) -> dict[str, Any]:
    role = _value(message, "type") or _value(message, "role") or "unknown"
    return {
        "role": str(role),
        "content": _visible_content(_value(message, "content", message)),
    }


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


def _normalized_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None

    def token(*names: str) -> int | None:
        for name in names:
            raw = value.get(name)
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return None
        return None

    input_tokens = token(
        "input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count"
    )
    output_tokens = token(
        "output_tokens",
        "completion_tokens",
        "output_token_count",
        "completion_token_count",
    )
    total_tokens = token("total_tokens", "total_token_count")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _read_llm_result(
    response: Any,
) -> tuple[Any, dict[str, int] | None, str | None]:
    """Extract visible structured output, normalized usage, and model name."""
    generation = _first_generation(response)
    message = _value(generation, "message") or generation

    parsed = _tool_arguments(message)
    if parsed is None:
        content = _visible_content(_value(message, "content"))
        if not content:
            content = _visible_content(_value(generation, "text"))
        parsed = _parse_json(content) if content else None

    response_metadata = _value(message, "response_metadata", {}) or {}
    generation_info = _value(generation, "generation_info", {}) or {}
    llm_output = _value(response, "llm_output", {}) or {}
    usage = None
    for candidate in (
        _value(message, "usage_metadata"),
        _value(response_metadata, "token_usage") or _value(response_metadata, "usage"),
        _value(generation_info, "token_usage") or _value(generation_info, "usage"),
        _value(llm_output, "token_usage") or _value(llm_output, "usage"),
    ):
        usage = _normalized_usage(candidate)
        if usage is not None:
            break

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
        model = _value(source, "ls_model_name") or _value(source, "model_name") or _value(source, "model")
        if model:
            return str(model)
    return None


class LLMCaptureHandler(BaseCallbackHandler):
    """Thread-safe, opt-in capture of visible model request/response data."""

    raise_error = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._calls: list[dict[str, Any]] = []
        self._pending: dict[str, int] = {}
        self._starts: dict[tuple[tuple[str, ...], str], int] = {}

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
        prompt_messages = [
            _prompt_message(message)
            for batch in messages
            for message in batch
        ]
        self._start(serialized, prompt_messages, run_id, metadata)

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
        self._start(
            serialized,
            [{"role": "prompt", "content": prompt} for prompt in prompts],
            run_id,
            metadata,
        )

    def _start(
        self,
        serialized: Mapping[str, Any] | None,
        prompt_messages: list[dict[str, Any]],
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
            self._calls.append(
                {
                    "complete": False,
                    "namespace": namespace,
                    "graph_node": graph_node,
                    "run_id": str(run_id),
                    "model": _model_name(serialized, metadata),
                    "prompt_messages": deepcopy(prompt_messages),
                    "transport_retry_ordinal": prior_starts or None,
                }
            )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        parsed, usage, model = _read_llm_result(response)
        self._finish(run_id, response=parsed, usage=usage, model=model)

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
                        "prompt_messages": [],
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
                    prompt_messages=deepcopy(call["prompt_messages"]),
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


class ObservabilityCollector:
    """Correlate nested graph tasks without depending on a display or frontend."""

    def __init__(self, *, capture_llm: bool = False) -> None:
        self._lock = threading.RLock()
        self._entity_scopes: dict[tuple[str, ...], str] = {}
        self._task_bindings: dict[tuple[str, ...], TaskBinding] = {}
        self._variable_attempts: dict[str, list[VariableAttempt]] = {}
        self._llm_callback = LLMCaptureHandler() if capture_llm else None

    @property
    def llm_callback(self) -> LLMCaptureHandler | None:
        """The callback to attach to a graph, absent unless capture was enabled."""
        return self._llm_callback

    def graph_config(
        self, config: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None:
        """Return graph config with this collector's optional callback attached."""
        if self._llm_callback is None:
            return None if config is None else dict(config)
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

    def snapshot(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return a detached, JSON-ready copy of the captured channels."""
        with self._lock:
            snapshot = {
                "variable_attempts": {
                    key: [attempt.as_dict() for attempt in attempts]
                    for key, attempts in self._variable_attempts.items()
                }
            }
            if self._llm_callback is not None:
                exchanges: dict[str, list[dict[str, Any]]] = {}
                for call in self._llm_callback._snapshot():
                    binding = self.binding_for(call.namespace)
                    if binding is None:
                        continue
                    node = call.graph_node or binding.graph_node
                    exchange = {
                        "agent": _LLM_AGENT_BY_NODE.get(node, "orchestrator"),
                        "node": node,
                        "attempt": binding.semantic_attempt or 1,
                        "prompt_messages": deepcopy(call.prompt_messages),
                        "response": deepcopy(call.response),
                        "model": call.model,
                        "usage": deepcopy(call.usage),
                        "error": call.error,
                    }
                    if call.transport_retry_ordinal is not None:
                        exchange["retry_ordinal"] = call.transport_retry_ordinal
                    exchanges.setdefault(binding.entity_key, []).append(exchange)
                snapshot["llm_exchanges"] = exchanges
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
            validation_errors=tuple(
                str(error) for error in (field(task, "validation_errors", []) or [])
            ),
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
    "merge_callback_config",
]
