"""Tap 2 — capture raw LLM I/O off the running graph via a LangChain callback.

Attaching one :class:`LLMCaptureHandler` to the graph stream config
(``graph.stream(..., config={"callbacks": [handler]})``) records every nested
subagent model call — prompts in, reasoning, structured response, token usage —
without touching ``BaseAgentModel`` or any agent. LangGraph seeds each node's run
config into a contextvar that ``with_structured_output(...).invoke()`` inherits,
so the handler fires for calls the agents make with no explicit ``config=``.

Each call is correlated back to the graph node that issued it through the run
metadata LangGraph attaches: ``metadata["langgraph_node"]`` and
``metadata["langgraph_checkpoint_ns"]`` (the same namespace vocabulary Tap 1
uses). Nodes fan out across threads, so all mutation is guarded by a lock.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .events import LLMCall


# LangGraph joins checkpoint-namespace segments with this separator; each segment
# is ``f"{node}:{task_id}"`` — the same shape Tap 1's ``ProgressEvent.namespace``
# carries, so the two taps share one namespace vocabulary.
_NS_SEP = "|"


def _coerce_content(content: Any) -> str:
    """Flatten message content (str or a list of content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _parse_namespace(metadata: dict[str, Any] | None) -> tuple[str, ...]:
    if not metadata:
        return ()
    ns = metadata.get("langgraph_checkpoint_ns", "")
    if not ns:
        return ()
    return tuple(segment for segment in ns.split(_NS_SEP) if segment)


class LLMCaptureHandler(BaseCallbackHandler):
    """Collect one :class:`LLMCall` per model invocation on the graph.

    ``calls`` is the ordered list of completed captures. ``by_run`` holds the
    in-flight partials between ``on_chat_model_start`` and ``on_llm_end``.
    """

    raise_error = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[LLMCall] = []
        self._pending: dict[str, dict[str, Any]] = {}

    # --- start ---------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompt_messages = [
            {"role": getattr(message, "type", "unknown"),
             "content": _coerce_content(getattr(message, "content", message))}
            for message in (messages[0] if messages else [])
        ]
        model = None
        if metadata:
            model = metadata.get("ls_model_name") or metadata.get("model")
        with self._lock:
            self._pending[str(run_id)] = {
                "node": (metadata or {}).get("langgraph_node", ""),
                "namespace": _parse_namespace(metadata),
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "model": model,
                "prompt_messages": prompt_messages,
            }

    # --- end -----------------------------------------------------------------
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        text, reasoning, usage = _read_llm_result(response)
        self._finish(run_id, response=text, reasoning=reasoning, usage=usage)

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
            pending = self._pending.pop(str(run_id), None)
            if pending is None:
                # A model start we never saw (e.g. non-chat LLM); record what we
                # can so nothing is silently dropped.
                pending = {"node": "", "namespace": (), "run_id": str(run_id)}
            self.calls.append(LLMCall(**{**pending, **fields}))

    # --- reporting -----------------------------------------------------------
    def snapshot(self) -> list[LLMCall]:
        with self._lock:
            return list(self.calls)

    def by_agent(self) -> dict[str, int]:
        """Count captured calls per owning agent (for the propagation check)."""
        from .mapping import infer_agent

        counts: dict[str, int] = {}
        for call in self.snapshot():
            agent = infer_agent(call.namespace)
            counts[agent] = counts.get(agent, 0) + 1
        return counts


def _read_llm_result(response: Any) -> tuple[str, str | None, dict[str, Any] | None]:
    """Pull text, reasoning summary, and token usage out of an ``LLMResult``."""
    text = ""
    reasoning: str | None = None
    usage: dict[str, Any] | None = None
    try:
        generation = response.generations[0][0]
    except (AttributeError, IndexError, TypeError):
        return text, reasoning, usage

    text = getattr(generation, "text", "") or ""
    message = getattr(generation, "message", None)
    if message is not None:
        if not text:
            text = _coerce_content(getattr(message, "content", ""))
        usage = getattr(message, "usage_metadata", None) or usage
        extra = getattr(message, "additional_kwargs", {}) or {}
        meta = getattr(message, "response_metadata", {}) or {}
        reasoning = _extract_reasoning(extra) or _extract_reasoning(meta)

    if usage is None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
    return text, reasoning, usage


def _reasoning_text(item: Any) -> str:
    """Text of one reasoning-summary item, which may be a ``{"text": ...}`` block.

    The OpenAI reasoning shape is a list of ``{"type": "summary_text", "text":
    ...}`` dicts, so a plain ``str(item)`` would leak the dict repr; pull the text
    field when present and fall back to :func:`_coerce_content` otherwise.
    """
    if isinstance(item, dict):
        return _coerce_content(
            item.get("text") or item.get("content") or item.get("summary_text") or ""
        )
    return _coerce_content(item)


def _extract_reasoning(container: dict[str, Any]) -> str | None:
    """Best-effort reasoning-summary extraction across endpoint shapes."""
    value = container.get("reasoning") or container.get("reasoning_content")
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        summary = value.get("summary") or value.get("text")
        if isinstance(summary, list):
            return "\n".join(_reasoning_text(item) for item in summary)
        return _coerce_content(summary) if summary else None
    if isinstance(value, list):
        return "\n".join(_reasoning_text(item) for item in value)
    return str(value)


__all__ = ["LLMCaptureHandler"]
