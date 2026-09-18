"""One canonical execution with a bounded, optional presentation side channel.

Callbacks enqueue detached observations only. A separate consumer owns replay
reduction and recording; neither browsers nor disk I/O run inside model callbacks.
Sequence/time are observation order, not a causal or provider-compute timeline.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError

from cipoc.models import ClinicalNote, OrchestratorRunError
from cipoc.utils.observability import CapturedLLMCall
from cipoc.utils.progress.events import ProgressEvent

from .events import DemoEvent
from .mapping import infer_agent, map_node_id
from .serialize import to_jsonable
from .trace import TraceWriter


class DemoSetupError(ValueError):
    """An actionable preflight diagnostic known to contain no input bodies."""


def progress_event(seq: int, elapsed: float, event: ProgressEvent) -> DemoEvent:
    return DemoEvent(
        seq=seq, t=elapsed, type=event.kind, node=event.node,
        task_id=event.task_id, namespace=event.namespace,
        map_node_id=map_node_id(event.node, event.namespace),
        agent=infer_agent(event.namespace), payload=_presentation_payload(to_jsonable(event.payload)),
        error=to_jsonable(event.error),
    )


def _presentation_payload(value: Any) -> Any:
    # Raw state messages bypass the collector's capture/truncation policy.
    # Clinical results stay in task state; exchange bodies come only from the
    # runtime invocation observer, under its configured content policy.
    if isinstance(value, dict):
        return {key: _presentation_payload(item) for key, item in value.items()
                if key not in {"messages", "group_context"}}
    if isinstance(value, list):
        return [_presentation_payload(item) for item in value]
    return value


def atomic_json(path: Path, data: Any) -> None:
    """Publish complete JSON on the same filesystem, never a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cipoc-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class DemoRun:
    """One local job, prepared without inference, executed exactly once."""

    def __init__(
        self, raw_notes: list[dict], *, structured_data: dict | None = None,
        agent_factory: Callable[[], Any], output_dir: Path = Path("demo-runs"),
        record_path: Path | None = None, max_concurrency: int | None = None,
        capture_llm_content: bool = True, max_content_chars: int | None = None,
        queue_capacity: int = 2048,
    ) -> None:
        self.job_id = str(uuid4())
        self.raw_notes = raw_notes
        self.structured_data = structured_data
        self.agent_factory = agent_factory
        self.agent = None
        self.output_path = output_dir.resolve() / self.job_id / "run.json"
        self.record_path = record_path.resolve() if record_path else None
        self.max_concurrency = max_concurrency
        self.capture_llm_content = capture_llm_content
        self.max_content_chars = max_content_chars
        self.status = "idle"
        self.error: str | None = None
        self.artifact: dict | None = None
        self.artifact_saved = False
        self.issues: list[str] = []
        self.summary: dict[str, Any] = {"notes": len(raw_notes) if isinstance(raw_notes, list) else None}
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self._start: float | None = None
        self._end: float | None = None
        self._last_event: float | None = None
        self._lock = threading.RLock()
        self._queue: queue.Queue[DemoEvent] = queue.Queue(maxsize=queue_capacity)
        self._seq = 0
        self._active_calls: set[str] = set()
        self.completed_calls = 0
        self.retry_calls = 0
        self._executed = False

    def issue(self, message: str) -> None:
        with self._lock:
            if message not in self.issues:
                self.issues.append(message)

    def prepare(self) -> bool:
        try:
            if not isinstance(self.raw_notes, list) or not self.raw_notes:
                raise DemoSetupError("A nonempty JSON list of clinical notes is required.")
            notes = [ClinicalNote.model_validate(note) for note in self.raw_notes]
            if len({str(note.note_id) for note in notes}) != len(notes):
                raise DemoSetupError("Duplicate canonical note IDs.")
            if self.max_concurrency is not None and (type(self.max_concurrency) is not int or self.max_concurrency < 2):
                raise DemoSetupError("Graph concurrency must be at least 2 or unset.")
            if self.max_content_chars is not None and self.max_content_chars < 0:
                raise DemoSetupError("Prompt content limit must be nonnegative.")
            if self.structured_data is not None and not isinstance(self.structured_data, dict):
                raise DemoSetupError("Structured data must be a JSON object.")
            from cipoc.agents.orchestrator import OrchestratorInput
            OrchestratorInput(note_corpus={note.note_id: note for note in notes}, structured_data=self.structured_data or {})
            if self.record_path and self.record_path.exists():
                raise DemoSetupError("Recording destination already exists; choose a new file.")
            self.agent = self.agent_factory()
            groups = self.agent._target_variables
            self.summary = {
                "notes": len(notes), "groups": [group.name for group in groups],
                "variables": sum(len(group.variables) for group in groups),
                "models": {
                    name: {
                        key: self.agent._config.agent_settings(name).get(key)
                        for key in ("model", "max_concurrency", "timeout", "max_retries")
                    }
                    for name in ("note_scanner", "note_retriever", "extractor")
                },
            }
            self.status, self.error = "ready", None
            return True
        except Exception as error:
            # Validation errors can include note content/credentials. Detailed
            # canonical graph diagnostics are a separate downloadable artifact.
            self.status = "initialization_failed"
            self.error = f"Preflight failed ({type(error).__name__}). Check notes, configuration, environment references, dictionaries and output paths."
            if isinstance(error, DemoSetupError):
                self.error = str(error)
            elif isinstance(error, ValidationError):
                fields = [".".join(map(str, issue["loc"])) for issue in error.errors(include_input=False, include_context=False)]
                self.error = "Preflight validation failed for: " + ", ".join(fields)
            return False

    def meta(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "job_id": self.job_id, "status": self.status, "error": self.error,
                "summary": self.summary, "issues": list(self.issues),
                "started_at": self.started_at, "finished_at": self.finished_at,
                "elapsed_seconds": (self._end or now) - self._start if self._start else 0,
                "last_update_seconds": now - self._last_event if self._last_event else None,
                "active_calls": len(self._active_calls), "completed_calls": self.completed_calls,
                "retry_calls": self.retry_calls,
                "graph_concurrency": self.max_concurrency,
                "capture_llm_content": self.capture_llm_content,
                "max_content_chars": self.max_content_chars,
                "recording": self.record_path is not None,
                "artifact_available": self.artifact is not None,
                "artifact_saved": self.artifact_saved,
                "run_id": self.artifact["run"]["run_id"] if self.artifact else None,
            }

    def _enqueue(self, make: Callable[[int, float], DemoEvent]) -> None:
        try:
            with self._lock:
                event = make(self._seq, round(time.monotonic() - self._start, 6))
                self._queue.put_nowait(event)
                self._seq += 1
                self._last_event = time.monotonic()
        except Exception:
            self.issue("Presentation capture incomplete: an observation could not be queued. Canonical output is independent.")

    def _observe(self, event: ProgressEvent) -> None:
        self._enqueue(lambda seq, t: progress_event(seq, t, event))

    def _invocation(self, phase: str, call: CapturedLLMCall) -> None:
        with self._lock:
            if phase == "started":
                self._active_calls.add(call.run_id)
                self.retry_calls += int(call.transport_retry_ordinal is not None)
            else:
                self._active_calls.discard(call.run_id)
                self.completed_calls += 1
        data = asdict(call)
        data["complete"] = phase == "finished"
        data["node"] = call.graph_node
        data["prompt_messages"] = data.get("prompt_messages") or []
        self._enqueue(lambda seq, t: DemoEvent(
            seq=seq, t=t, type="llm_start" if phase == "started" else "llm_call",
            node=call.graph_node, namespace=call.namespace,
            map_node_id=map_node_id(call.graph_node, call.namespace),
            agent=infer_agent(call.namespace), payload=to_jsonable(data), error=call.error,
        ))

    def execute(self, consume: Callable[[DemoEvent], None]) -> None:
        with self._lock:
            if self._executed:
                raise RuntimeError("This job has already been started.")
            self._executed = True
        if self.agent is None and not self.prepare():
            return
        self._start = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"
        writer = None
        if self.record_path:
            try:
                writer = TraceWriter(self.record_path)
            except Exception:
                self.issue("Recording could not be opened; canonical output remains enabled.")
        self._enqueue(lambda seq, t: DemoEvent(seq, t, "run_start", payload={
            "job_id": self.job_id, "started_at": self.started_at,
            "artifact_path": str(self.output_path), "contains_phi": True,
        }))
        done = threading.Event()

        def execute_graph() -> None:
            try:
                result = self.agent.run(
                    self.raw_notes, structured_data=self.structured_data,
                    progress=False, pause_before_summary=False,
                    max_concurrency=self.max_concurrency,
                    capture_llm_content=self.capture_llm_content,
                    max_content_chars=self.max_content_chars,
                    event_observer=self._observe, invocation_observer=self._invocation,
                )
                self.artifact = result.model_dump(mode="json")
                self.status = "completed"
            except OrchestratorRunError as error:
                self.artifact = error.failure.model_dump(mode="json")
                self.status = "failed"
                self.error = "Extraction failed. Download the failure diagnostic for details."
            except Exception as error:
                self.status = "initialization_failed"
                self.error = f"Execution could not initialize ({type(error).__name__})."
            finally:
                done.set()

        worker = threading.Thread(target=execute_graph, name="cipoc-demo-execution")
        worker.start()

        def deliver(event: DemoEvent) -> None:
            nonlocal writer
            try:
                consume(event)
            except Exception:
                self.issue("Presentation reduction failed; use the canonical artifact for results.")
            if writer:
                try:
                    writer.write(event)
                except Exception:
                    self.issue("Recording failed; the replay may be incomplete.")
                    try:
                        writer.close()
                    except Exception:
                        pass
                    writer = None

        while not done.is_set() or not self._queue.empty():
            try:
                deliver(self._queue.get(timeout=0.1))
            except queue.Empty:
                pass
        worker.join()
        if self.artifact:
            try:
                atomic_json(self.output_path, self.artifact)
                self.artifact_saved = True
            except Exception:
                self.issue("Artifact publication failed; download the in-memory artifact before closing the server.")
        self._end = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()
        deliver(DemoEvent(
            self._seq, round(self._end - self._start, 6),
            "run_end" if self.status == "completed" else "run_error",
            payload={"job_id": self.job_id, "run_id": self.meta()["run_id"],
                     "status": self.status, "issues": list(self.issues)}, error=self.error,
        ))
        if writer:
            try:
                writer.close()
            except Exception:
                self.issue("Recording could not be closed cleanly.")
