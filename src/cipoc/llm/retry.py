"""Retry policy for the LLM-backed nodes of an agent graph.

Retries live on the graph node, not inside the model wrapper. LangGraph reruns
only the node that raised, so a throttled call replays one LLM request rather
than a whole branch, and the per-model concurrency permit held by
:meth:`BaseAgentModel.structured` is released for the duration of the backoff
instead of idling while it sleeps.

``retry_on_transient`` deliberately narrows LangGraph's default predicate, which
retries any exception it does not recognize. Only endpoint-level failures that a
second identical request could plausibly survive are retried; a bad prompt, a
schema mismatch, or a bug in a node fails on the first attempt instead of
costing ``max_attempts`` LLM calls.
"""

from __future__ import annotations

from openai import APIConnectionError

from langgraph.types import RetryPolicy


# 408/409/429 plus any 5xx. 429 is the rate limit this exists for; 408/409 and
# the 5xx band cover request timeouts and endpoint-side faults.
_TRANSIENT_STATUS = frozenset({408, 409, 429})

# Total attempts and the interval ceiling: 8 attempts backing off 1s, 2, 4, 8,
# 16, 32, 60 covers just over two minutes of sustained throttling before the run
# gives up. Azure rate limits reset on a per-minute window, so a cap well under
# the total is what actually lets a run ride one out.
DEFAULT_RETRY_SETTINGS: dict[str, object] = {
    "max_attempts": 8,
    "initial_interval": 1.0,
    "backoff_factor": 2.0,
    "max_interval": 60.0,
    "jitter": True,
}


def retry_on_transient(exc: BaseException) -> bool:
    """True when ``exc`` is an endpoint failure worth reissuing the request for.

    Matches on HTTP status rather than exception class so it holds for any
    OpenAI-compatible SDK error carrying ``status_code``
    (``RateLimitError``, ``InternalServerError``, ``APIStatusError``).
    ``APIConnectionError`` — which covers ``APITimeoutError`` — has no status.
    """
    if isinstance(exc, APIConnectionError):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return False
    return status in _TRANSIENT_STATUS or 500 <= status < 600


def llm_retry_policy(**overrides) -> RetryPolicy:
    """Build the :class:`RetryPolicy` for an LLM-backed node.

    Accepts any ``RetryPolicy`` field as an override; ``retry_on`` defaults to
    :func:`retry_on_transient` and can be replaced with a different predicate or
    a tuple of exception classes.
    """
    settings = {**DEFAULT_RETRY_SETTINGS, "retry_on": retry_on_transient}
    settings.update(overrides)
    return RetryPolicy(**settings)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_RETRY_SETTINGS",
    "RetryPolicy",
    "llm_retry_policy",
    "retry_on_transient",
]
