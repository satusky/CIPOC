"""Offline artifact/feedback boundary checks, without the CIPOC runtime or httpx."""

import json
import tempfile
import unittest
from pathlib import Path

from cipoc_workbench import EXAMPLE_DIR
from cipoc_workbench.server import build_app


class WorkbenchServerTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, app, path, *, method="GET", payload=None):
        messages = []

        async def receive():
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode() if payload is not None else b"",
                "more_body": False,
            }

        async def send(message):
            messages.append(message)

        await app({
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "method": method, "scheme": "http", "path": path,
            "root_path": "", "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "server": ("localhost", 8000), "client": ("localhost", 12345),
        }, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return status, body

    async def test_versions_and_unavailable_usage_are_served_without_rewriting(self):
        for version in ("1.0", "1.1"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                artifact = json.loads((EXAMPLE_DIR / "case_state.json").read_text())
                artifact["schema_version"] = version
                artifact["case"]["variable_results"]["400"]["extraction"]["most_important_note"] = "001"
                artifact["corpus"]["note_corpus"] = {
                    "1": {"note_id": 1}, "001": {"note_id": "001"}, "note-A": {"note_id": "note-A"},
                }
                if version == "1.1":
                    artifact["observability"].update(
                        collection_status="unavailable",
                        collection_issues=[{"code": "snapshot_failed", "message": "Collection failed."}],
                        unattributed_exchanges=[], llm_usage_summary=None,
                    )
                path = Path(directory) / "run.json"
                original = json.dumps(artifact).encode()
                path.write_bytes(original)
                status, body = await self.request(build_app(state_path=path), "/case_state.json")
                self.assertEqual(status, 200)
                self.assertEqual(body, original)
                restored = json.loads(body)
                self.assertEqual(restored["case"]["variable_results"]["400"]["extraction"]["most_important_note"], "001")
                self.assertEqual(restored["corpus"]["note_corpus"]["1"]["note_id"], 1)
                self.assertEqual(restored["corpus"]["note_corpus"]["001"]["note_id"], "001")
                if version == "1.1":
                    self.assertIsNone(restored["observability"]["llm_usage_summary"])

    async def test_feedback_keeps_string_note_ids_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.json"
            app = build_app(feedback_path=path)
            for note_id in ("1", "001", "note-A"):
                status, body = await self.request(app, "/api/feedback/note/" + note_id,
                                                  method="PUT", payload={"note": "Review " + note_id})
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["id"], note_id)
            status, body = await self.request(app, "/api/feedback")
            self.assertEqual(status, 200)
            notes = json.loads(body)["annotations"]["note"]
            self.assertEqual(set(notes), {"1", "001", "note-A"})
            self.assertEqual(notes["001"]["note"], "Review 001")

    async def test_bundled_example_and_static_frontend(self):
        app = build_app()
        status, body = await self.request(app, "/case_state.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["schema_version"], "1.0")
        for path in ("/", "/app.js", "/detail.js", "/styles.css"):
            with self.subTest(path=path):
                status, body = await self.request(app, path)
                self.assertEqual(status, 200)
                self.assertTrue(body)


if __name__ == "__main__":
    unittest.main()
