"""Offline live-run boundaries using actual orchestration and synthetic agents."""

import asyncio
import csv
import io
import json
import queue
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import warnings
import zipfile

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from cipoc.demo.events import DemoEvent
from cipoc.demo.server import Broadcaster, LiveDemoSession, build_app, load_replay_session
from cipoc.demo.state import replay
from cipoc.demo.steps import StepBuilder, build_steps
from cipoc.demo.stream import DemoRun, progress_event
from cipoc.demo.trace import read_trace, write_trace
from cipoc.export import OmopExporter
from cipoc.models import OrchestratorRunResult
from cipoc.utils.observability import LLMCaptureHandler
from cipoc.utils.progress.events import ProgressEvent
from tests.fake_orchestrator import build_fake_orchestrator, load_notes
from tests.test_observability import llm_result


def agent():
    value = build_fake_orchestrator()
    value._target_variables = value._target_variables[:1]
    return value


class LiveRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.agent = agent()
        self.job = DemoRun(
            [load_notes()[0].model_dump()], agent_factory=lambda: self.agent,
            output_dir=self.path, record_path=self.path / "trace.jsonl",
            max_concurrency=2,
        )

    def test_explicit_start_once_canonical_artifact_and_reconnect(self):
        session = LiveDemoSession(self.job)
        with patch.object(self.agent, "run", wraps=self.agent.run) as run:
            with TestClient(build_app(session)) as client:
                self.assertEqual(client.get("/api/meta").json()["status"], "idle")
                self.assertFalse(run.called)
                client.post("/api/start")
                client.post("/api/start")
                session.join()
                self.assertEqual(run.call_count, 1)
                self.assertFalse(run.call_args.kwargs["progress"])
                self.assertFalse(run.call_args.kwargs["pause_before_summary"])
                self.assertEqual(self.job.status, "completed", self.job.error)
                self.assertFalse(self.job.issues)
                downloaded = client.get("/api/artifact").json()
                result = OrchestratorRunResult.model_validate(downloaded)
                self.assertEqual(json.loads(self.job.output_path.read_text()), downloaded)
                last = client.get("/api/update?after=-1").json()
                self.assertEqual(last["view"]["snapshot"]["seq"], last["events"][-1]["seq"])
                self.assertEqual(client.get(f'/api/update?after={last["events"][-1]["seq"]}').json()["events"], [])
                # New browser client attaches to the existing job.
                with TestClient(build_app(session)) as second:
                    self.assertEqual(second.get("/api/meta").json()["run_id"], str(result.run.run_id))
                self.assertEqual(run.call_count, 1)
                trace = read_trace(self.path / "trace.jsonl")
                self.assertEqual(trace[-1].payload["run_id"], str(result.run.run_id))
                replayed = load_replay_session(self.path / "trace.jsonl", artifact_path=self.job.output_path)
                self.assertEqual(replayed.case_at_seq(trace[-1].seq)["variable_results"], result.case.model_dump(mode="json")["variable_results"])
                for seq in (0, len(trace) // 2, len(trace) - 1):
                    self.assertEqual(session.snapshot_at_seq(seq), replay(trace[:seq + 1], description=session.description).snapshot().to_dict())

    def test_graph_failure_is_diagnostic_not_completed_case(self):
        with patch.object(self.agent._scanner._graph, "invoke", side_effect=RuntimeError("synthetic failure")):
            events = []
            self.job.execute(events.append)
        self.assertEqual(self.job.status, "failed")
        self.assertNotIn("case", self.job.artifact)
        self.assertTrue(self.job.artifact_saved)
        self.assertEqual(events[-1].type, "run_error")
        self.assertFalse(replay(events).snapshot().finished)

    def test_full_bundle_keeps_every_note_and_exposes_individual_results_before_merge(self):
        notes = load_notes()
        self.job.raw_notes = [note.model_dump() for note in notes]
        session = LiveDemoSession(self.job)
        early_results = []

        def consume(event):
            session.append(event)
            if event.type != "task_end" or event.node != "variable_branch":
                return
            snapshot = session._latest.snapshot()
            key = "/".join((*event.namespace, f"variable_branch:{event.task_id}"))
            instance = snapshot.instances[key]
            self.assertEqual(instance.active, 0)
            for output in instance.result["variable_results"]:
                row = snapshot.progress.variables[output["item_id"]]
                if not row.terminal:
                    early_results.append(output)

        self.job.execute(consume)
        self.assertEqual(self.job.status, "completed", self.job.error)
        self.assertFalse(self.job.issues)
        self.assertTrue(early_results, "Individual results must be available before the root wave merges.")
        self.assertTrue(all(result["is_valid"] and result["value"] is not None for result in early_results))
        snapshot = session._latest.snapshot()
        instances = [instance for instance in snapshot.instances.values() if instance.node == "note_branch"]
        self.assertEqual({str(instance.input["note_id"]) for instance in instances}, {str(note.note_id) for note in notes})
        self.assertEqual(len(instances), len(notes))
        self.assertTrue(all(instance.status == "done" for instance in instances))
        self.assertEqual(snapshot.progress.notes_done, len(notes))

    def test_preflight_rejects_canonical_note_collisions_before_agent_factory(self):
        note = self.job.raw_notes[0]
        self.job.raw_notes = [{**note, "note_id": 1}, {**note, "note_id": "1"}]
        with patch.object(self.job, "agent_factory") as factory:
            self.job.execute(lambda event: None)
            factory.assert_not_called()
        self.assertEqual(self.job.status, "initialization_failed")
        self.assertIsNone(self.job.artifact)

    def test_optional_recording_and_reduction_failures_preserve_output(self):
        with patch("cipoc.demo.stream.TraceWriter.write", side_effect=OSError("disk")):
            self.job.execute(lambda event: (_ for _ in ()).throw(RuntimeError("view")))
        self.assertEqual(self.job.status, "completed")
        self.assertTrue(self.job.artifact_saved)
        self.assertTrue(self.job.issues)

    def test_publication_failure_leaves_downloadable_artifact(self):
        session = LiveDemoSession(self.job)
        with patch("cipoc.demo.stream.atomic_json", side_effect=OSError("disk")):
            session.start()
            session.join()
        self.assertEqual(self.job.status, "completed")
        self.assertFalse(self.job.artifact_saved)
        with TestClient(build_app(session)) as client:
            self.assertEqual(client.get("/api/artifact").status_code, 200)

    def test_bounded_capture_reports_overflow(self):
        self.job._queue = queue.Queue(maxsize=1)
        self.job._start = time.monotonic()
        self.job._observe(ProgressEvent(kind="values", namespace=(), payload={}))
        self.job._observe(ProgressEvent(kind="values", namespace=(), payload={}))
        self.assertEqual(self.job._queue.qsize(), 1)
        self.assertIn("incomplete", self.job.issues[0])

    def test_export_uses_finished_case_without_another_model_run(self):
        session = LiveDemoSession(self.job)
        session.start()
        session.join()
        result = OrchestratorRunResult.model_validate(self.job.artifact)
        expected = OmopExporter(person_id=42, nlp_date=result.run.started_at.date()).build(
            notes=result.corpus.note_corpus.values(), case=result.case,
        )
        with TestClient(build_app(session)) as client, patch.object(self.agent, "run") as run:
            response = client.get("/api/export?person_id=42")
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
                rows = list(csv.DictReader(io.StringIO(bundle.read("note_nlp.csv").decode())))
            self.assertEqual(len(rows), len(expected.note_nlp_rows))
            for row, expected_row in zip(rows, expected.note_nlp_rows):
                self.assertEqual(row, {key: str(value) for key, value in expected_row.model_dump().items()})
            run.assert_not_called()
        original_items = set(result.case.variable_results)
        subset = OmopExporter(person_id=42, nlp_date=result.run.started_at.date()).build(
            notes=result.corpus.note_corpus.values(), case=result.case, item_ids=[next(iter(original_items))],
        )
        self.assertEqual(set(result.case.variable_results), original_items)
        self.assertTrue(all(row.note_nlp_source_concept_id == next(iter(original_items)) for row in subset.note_nlp_rows))


class ObserverAndReplayTests(unittest.TestCase):
    def test_live_observer_is_detached_capture_aware_and_failure_isolated(self):
        for capture in (False, True):
            observed = []
            def observer(phase, call):
                observed.append((phase, call))
                if call.prompt_messages:
                    call.prompt_messages[0]["content"] = "mutated observer copy"
                raise RuntimeError("presentation offline")
            handler = LLMCaptureHandler(capture_llm_content=capture, max_content_chars=3, invocation_observer=observer)
            run_id = uuid4()
            handler.on_chat_model_start({}, [[HumanMessage("abcdef")]], run_id=run_id,
                                        metadata={"langgraph_node": "summarize_note", "langgraph_checkpoint_ns": "note_branch:n|summarize_note:s"})
            handler.on_llm_end(llm_result(AIMessage(content="answer")), run_id=run_id)
            self.assertEqual([phase for phase, _ in observed], ["started", "finished"])
            call = handler._snapshot()[0]
            self.assertEqual(call.prompt_messages[0]["content"] if capture else call.prompt_messages, "abc" if capture else None)
            self.assertEqual(call.response, "answer" if capture else None)
            self.assertEqual(handler.collection_issues(), [])

    def test_raw_graph_messages_do_not_bypass_capture_policy(self):
        event = progress_event(0, 0, ProgressEvent(kind="values", namespace=(), payload={
            "messages": [HumanMessage("secret prompt")],
            "task": {"candidate": {"value": "1"}, "messages": ["prompt"]},
            "group_context": {"raw": "response"},
        }))
        self.assertNotIn("secret prompt", json.dumps(event.to_dict()))
        self.assertNotIn("group_context", event.payload)
        self.assertEqual(event.payload["task"]["candidate"]["value"], "1")

    def test_incremental_steps_equal_batch_for_every_legacy_prefix(self):
        events = read_trace(Path(__file__).parent / "fixtures/demo_trace.jsonl")
        builder = StepBuilder()
        for index, event in enumerate(events):
            builder.ingest(event)
            self.assertEqual(builder.steps, build_steps(events[:index + 1]))

    def test_truncated_tail_recovery_and_exclusive_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path, [DemoEvent(0, 0, "run_start")])
            with self.assertRaises(FileExistsError):
                write_trace(path, [])
            with path.open("ab") as stream:
                stream.write(b'{"seq": 1, "payload": "\xe2')
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                self.assertEqual(len(read_trace(path)), 1)
                self.assertTrue(captured)
            path.write_text('{"seq": 0, broken}\n')
            with self.assertRaises(ValueError):
                read_trace(path)

    def test_slow_subscriber_receives_latest_hint_with_bounded_memory(self):
        async def exercise():
            broadcaster = Broadcaster()
            queue = broadcaster.subscribe()
            for revision in range(1000):
                broadcaster.publish({"revision": revision})
            self.assertEqual(queue.qsize(), 1)
            self.assertEqual((await queue.get())["revision"], 999)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
