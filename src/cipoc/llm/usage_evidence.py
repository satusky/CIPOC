"""Invocation-local scalar evidence collected before OpenAI SDK coercion."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from threading import Lock


TOKEN_ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count"),
    "output_tokens": ("output_tokens", "completion_tokens", "output_token_count", "completion_token_count"),
    "total_tokens": ("total_tokens", "total_token_count"),
}
_SCALARS = frozenset(alias for aliases in TOKEN_ALIASES.values() for alias in aliases)


@dataclass
class _Capture:
    owner: object
    invocation_id: str | None = None
    evidence: tuple[tuple[str, int | None], ...] | None = None
    next_retry: int = 0
    request_id: int | None = None
    eligible: bool = True
    finished: bool = False
    closed: bool = False
    lock: object = field(default_factory=Lock)


_capture: ContextVar[_Capture | None] = ContextVar("cipoc_usage_evidence", default=None)


@contextmanager
def usage_evidence_scope(owner):
    # A mutable cell is shared with LangChain's copied worker contexts; the
    # retained evidence itself is immutable and never contains response text.
    capture = _Capture(owner) if owner is not None else None
    token = _capture.set(capture)
    try:
        yield
    finally:
        if capture is not None:
            with capture.lock:
                capture.closed = True
                capture.evidence = None
        _capture.reset(token)


def start_usage_evidence(run_id):
    capture = _capture.get()
    if capture is not None:
        with capture.lock:
            if capture.invocation_id is None:
                capture.invocation_id = str(run_id)
            elif capture.invocation_id != str(run_id):
                # More than one model lifecycle in this wrapper scope is not a
                # proven one-to-one binding. Nested wrapper scopes are separate.
                capture.eligible = False
                capture.evidence = None


def finish_usage_evidence(run_id):
    capture = _capture.get()
    if capture is not None:
        with capture.lock:
            if capture.invocation_id == str(run_id):
                capture.finished = True
                if capture.eligible and not capture.closed:
                    return capture.evidence
    return None


def capture_usage_request(owner, request):
    capture = _capture.get()
    if capture is None or capture.owner is not owner:
        return
    with capture.lock:
        capture.evidence = None
        if capture.closed or capture.finished or not capture.eligible:
            return
        # The pinned SDK numbers attempts starting at zero. A second initial
        # request (including nested direct SDK calls) must not cross-certify.
        if (
            capture.invocation_id is None
            or request.headers.get("x-stainless-retry-count") != str(capture.next_retry)
            or request.method != "POST"
            or not request.url.path.endswith(("/chat/completions", "/responses"))
        ):
            capture.eligible = False
            return
        try:
            streaming = json.loads(request.content).get("stream", False)
        except Exception:
            capture.eligible = False
            return
        if streaming is not False:
            capture.eligible = False
            return
        capture.next_retry += 1
        capture.request_id = id(request)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Ambiguous JSON object")
        result[key] = value
    return result


def capture_usage_response(owner, response):
    capture = _capture.get()
    if capture is None or capture.owner is not owner:
        return
    with capture.lock:
        capture.evidence = None
        if capture.closed or capture.finished or not capture.eligible:
            return
        if capture.request_id != id(response.request):
            capture.eligible = False
            return
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not response.is_success or not (content_type == "application/json" or content_type.endswith("+json")):
        return
    # Non-streaming SDK sends read this body anyway. Let transport read failures
    # reach the SDK's unchanged retry handling; parsing failures only lose proof.
    response.read()
    try:
        usage = response.json(object_pairs_hook=_unique_object).get("usage")
        if not isinstance(usage, dict):
            return
        # Invalid values become a presence sentinel, not retained strings/objects
        # that might carry PHI. Omitted aliases remain absent.
        evidence = tuple(
            (key, value if type(value) is int and value >= 0 else None)
            for key, value in usage.items() if key in _SCALARS
        )
    except Exception:
        return
    with capture.lock:
        if not capture.closed and not capture.finished and capture.eligible and capture.request_id == id(response.request):
            capture.evidence = evidence
