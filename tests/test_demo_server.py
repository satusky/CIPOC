"""Phase 2 — the FastAPI demo server: REST state, controls, and live session.

SSE (``/api/stream``) is exercised at the :class:`Broadcaster` level rather than
through the test client, whose blocking streaming reader does not compose with a
long-lived keep-alive generator.
"""

import asyncio
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from cipoc.demo.events import DemoEvent  # noqa: E402
from cipoc.demo.server import (  # noqa: E402
    Broadcaster,
    DemoSession,
    LiveDemoSession,
    build_app,
    load_replay_session,
)
from cipoc.demo.trace import read_trace  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"


def _client(session):
    from fastapi.testclient import TestClient

    return TestClient(build_app(session))


class ReplayApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = load_replay_session(FIXTURE)
        cls.client = _client(cls.session)

    def test_meta_reports_run_shape(self):
        meta = self.client.get("/api/meta").json()
        self.assertEqual(meta["mode"], "replay")
        self.assertEqual(meta["num_events"], len(read_trace(FIXTURE)))
        self.assertGreater(meta["num_steps"], 0)

    def test_events_and_steps_endpoints(self):
        events = self.client.get("/api/events").json()
        steps = self.client.get("/api/steps").json()
        self.assertEqual(len(events), self.session.meta()["num_events"])
        self.assertEqual(len(steps), self.session.meta()["num_steps"])
        self.assertEqual(steps[0]["start_seq"], 0)

    def test_step_snapshot_endpoint(self):
        snap = self.client.get("/api/step/0").json()
        self.assertIn("progress", snap)
        self.assertIn("active_map_nodes", snap)

    def test_step_index_out_of_range_is_404(self):
        self.assertEqual(self.client.get("/api/step/9999").status_code, 404)

    def test_snapshot_by_seq_defaults_to_last_event(self):
        default = self.client.get("/api/snapshot").json()
        self.assertTrue(default["finished"])
        mid = self.client.get("/api/snapshot", params={"seq": 60}).json()
        self.assertFalse(mid["finished"])

    def test_case_endpoint_returns_hydrated_state(self):
        case = self.client.get("/api/case").json()
        self.assertIn("variable_results", case)

    def test_notes_endpoint_returns_note_text_for_highlighting(self):
        notes = self.client.get("/api/notes").json()
        self.assertTrue(notes, "expected a scanned note corpus")
        note = next(iter(notes.values()))
        self.assertIn("content", note)
        self.assertIn("note_type", note)
        self.assertTrue(note["content"])

    def test_notes_keys_survive_json_round_trip_as_strings(self):
        notes = self.session.notes()
        self.assertTrue(all(isinstance(key, str) for key in notes))

    def test_index_page_served(self):
        self.assertEqual(self.client.get("/").status_code, 200)


class CursorControlTests(unittest.TestCase):
    def setUp(self):
        self.session = load_replay_session(FIXTURE)
        self.client = _client(self.session)

    def test_next_and_prev_move_the_cursor(self):
        self.assertEqual(self.client.get("/api/cursor").json()["cursor"], 0)
        self.assertEqual(self.client.post("/api/next").json()["cursor"], 1)
        self.assertEqual(self.client.post("/api/next").json()["cursor"], 2)
        self.assertEqual(self.client.post("/api/prev").json()["cursor"], 1)

    def test_prev_clamps_at_zero(self):
        self.assertEqual(self.client.post("/api/prev").json()["cursor"], 0)

    def test_goto_clamps_and_reports_at_end(self):
        view = self.client.post("/api/goto/9999").json()
        self.assertEqual(view["cursor"], self.session.meta()["num_steps"] - 1)
        self.assertTrue(view["at_end"])

    def test_view_carries_the_step_and_its_snapshot(self):
        view = self.client.post("/api/goto/1").json()
        self.assertEqual(view["step"]["index"], 1)
        self.assertIn("progress", view["snapshot"])

    def test_play_then_pause_toggles_state(self):
        self.assertTrue(self.client.post("/api/play").json()["playing"])
        self.assertFalse(self.client.post("/api/pause").json()["playing"])

    def test_play_at_end_does_not_start(self):
        self.client.post("/api/goto/9999")
        self.assertFalse(self.client.post("/api/play").json()["playing"])


class BroadcasterTests(unittest.TestCase):
    def test_publish_fans_out_to_all_subscribers(self):
        async def run():
            broadcaster = Broadcaster()
            a, b = broadcaster.subscribe(), broadcaster.subscribe()
            broadcaster.publish({"type": "cursor", "cursor": 5})
            return (await a.get())["cursor"], (await b.get())["cursor"]

        self.assertEqual(asyncio.run(run()), (5, 5))

    def test_unsubscribe_stops_delivery(self):
        async def run():
            broadcaster = Broadcaster()
            a, b = broadcaster.subscribe(), broadcaster.subscribe()
            broadcaster.unsubscribe(a)
            broadcaster.publish({"cursor": 1})
            return a.empty(), (await b.get())["cursor"]

        empty_a, got_b = asyncio.run(run())
        self.assertTrue(empty_a)
        self.assertEqual(got_b, 1)

    def test_stream_route_is_registered(self):
        app = build_app(load_replay_session(FIXTURE))
        paths = {route.path for route in app.routes}
        self.assertIn("/api/stream", paths)

    def test_notes_route_is_registered(self):
        app = build_app(load_replay_session(FIXTURE))
        paths = {route.path for route in app.routes}
        self.assertIn("/api/notes", paths)

    def test_publish_marshals_onto_a_bound_loop(self):
        async def run():
            broadcaster = Broadcaster()
            broadcaster.bind_loop(asyncio.get_running_loop())
            queue = broadcaster.subscribe()
            broadcaster.publish({"type": "live", "num_steps": 3})
            # A loop-marshalled deliver runs on the next loop tick, not inline.
            return (await asyncio.wait_for(queue.get(), 1.0))["num_steps"]

        self.assertEqual(asyncio.run(run()), 3)


class LiveSessionTests(unittest.TestCase):
    def test_appending_events_grows_steps_and_gates_cursor(self):
        events = read_trace(FIXTURE)
        live = LiveDemoSession(iter([]))
        self.assertEqual(live.mode, "live")
        self.assertEqual(len(live.steps), 0)
        for event in events[:7]:
            live.append(event)
        partial_steps = len(live.steps)
        self.assertGreater(partial_steps, 0)
        # The cursor cannot advance past what has been produced.
        self.assertEqual(live.goto(9999)["cursor"], partial_steps - 1)
        for event in events[7:]:
            live.append(event)
        self.assertGreater(len(live.steps), partial_steps)

    def test_start_consumes_the_source_to_completion(self):
        events = read_trace(FIXTURE)
        live = LiveDemoSession(iter(events))
        live.start()
        live._thread.join(timeout=15)
        self.assertEqual(len(live.events), len(events))
        self.assertTrue(live.meta()["done"])

    def test_live_meta_reports_live_mode(self):
        live = LiveDemoSession(iter([]))
        self.assertEqual(live.meta()["mode"], "live")

    def test_listener_is_notified_as_content_streams_in(self):
        events = read_trace(FIXTURE)
        live = LiveDemoSession(iter([]))
        seen: list[dict] = []
        live.set_listener(seen.append)
        for event in events:
            live.append(event)
        self.assertTrue(seen, "expected live notifications as events arrived")
        self.assertTrue(all(m["type"] == "live" for m in seen))
        self.assertEqual(seen[-1]["num_steps"], len(live.steps))
        # Finer-grained than step boundaries: content-bearing events (values /
        # task_end / llm_call) notify too, so a presenter on the in-progress
        # frontier step sees per-note results fill in — hence far more
        # notifications than there are steps.
        self.assertGreater(len(seen), len(live.steps))

    def test_content_event_notifies_without_opening_a_step(self):
        # A nested values event advances the frontier step's snapshot but opens no
        # new step; it must still notify so the frontier can re-render.
        live = LiveDemoSession(iter([]))
        seen: list[dict] = []
        live.set_listener(seen.append)
        live.append(DemoEvent(seq=0, t=0.0, type="task_start", node="note_branch",
                              task_id="a", namespace=(), map_node_id="scanner_initialize",
                              agent="scanner", payload={"note_id": 1}))
        steps_before = len(live.steps)
        seen.clear()
        live.append(DemoEvent(seq=1, t=0.1, type="values", namespace=("note_branch:a",),
                              payload={"summary": "s"}))
        self.assertEqual(len(live.steps), steps_before)  # no new step opened
        self.assertEqual(len(seen), 1)                    # but it still notified

    def test_listener_notified_on_completion(self):
        events = read_trace(FIXTURE)
        live = LiveDemoSession(iter(events))
        seen: list[dict] = []
        live.set_listener(seen.append)
        live.start()
        live._thread.join(timeout=15)
        self.assertTrue(seen[-1]["done"])

    def test_failing_listener_does_not_break_append(self):
        live = LiveDemoSession(iter([]))
        live.set_listener(lambda _msg: (_ for _ in ()).throw(RuntimeError("boom")))
        for event in read_trace(FIXTURE)[:10]:
            live.append(event)  # must not raise
        self.assertGreater(len(live.steps), 0)


if __name__ == "__main__":
    unittest.main()
