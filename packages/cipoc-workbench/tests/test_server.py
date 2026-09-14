"""Offline artifact/feedback boundary checks, without the CIPOC runtime or httpx."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlencode, urlsplit

from cipoc_workbench import EXAMPLE_DIR
from cipoc_workbench import server
from cipoc_workbench.server import build_app

RUN_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
RUN_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
EMPTY_ANNOTATIONS = {"variable": {}, "group": {}, "note": {}}


class WorkbenchServerTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, app, path, *, method="GET", payload=None):
        messages = []
        url = urlsplit(path)

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
            "http_version": "1.1", "method": method, "scheme": "http", "path": unquote(url.path),
            "raw_path": url.path.encode("utf-8"), "root_path": "", "query_string": url.query.encode("ascii"),
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
            app = build_app(state_path=EXAMPLE_DIR / "case_state.json", feedback_path=path)
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

    async def test_default_has_no_artifact_reads_or_response_body_and_serves_frontend(self):
        with patch.object(server, "_read_json", side_effect=AssertionError("No startup artifact may be read")) as read:
            app = build_app()
            for _ in range(2):
                self.assertEqual(await self.request(app, "/case_state.json"), (204, b""))
            read.assert_not_called()
        for path in ("/", "/app.js", "/detail.js", "/styles.css"):
            with self.subTest(path=path):
                status, body = await self.request(app, path)
                self.assertEqual(status, 200)
                self.assertTrue(body)

    async def test_explicit_bundled_example_is_served_unchanged(self):
        path = EXAMPLE_DIR / "case_state.json"
        original = path.read_bytes()
        self.assertEqual(json.loads(original)["schema_version"], "1.0")
        app = build_app(state_path=path)
        for _ in range(2):
            self.assertEqual(await self.request(app, "/case_state.json"), (200, original))

    async def test_no_destination_explicitly_reports_read_only(self):
        app = build_app(state_path=EXAMPLE_DIR / "case_state.json")
        startup_id = json.loads((EXAMPLE_DIR / "case_state.json").read_text())["run"]["run_id"]
        for url, run_id in (("/api/feedback", startup_id), (f"/api/runs/{RUN_A}/feedback", RUN_A)):
            with self.subTest(url=url):
                status, body = await self.request(app, url)
                self.assertEqual(status, 200)
                document = json.loads(body)
                self.assertEqual(document["run_id"], run_id)
                self.assertIs(document["writable"], False)
                self.assertIn("--feedback-dir", document["read_only_reason"])
                self.assertEqual(document["annotations"], EMPTY_ANNOTATIONS)
                status, _ = await self.request(app, url + "/note/001", method="PUT", payload={"note": "No save"})
                self.assertEqual(status, 409)

    async def test_absent_startup_routes_never_bind_to_scoped_feedback_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "reviews"
            for destination in (None, reviews):
                with self.subTest(feedback_dir=destination), patch.object(
                    server, "_read_json", side_effect=AssertionError("No startup artifact or ground truth may be read"),
                ) as read:
                    app = build_app(feedback_dir=destination)
                    for scoped_run in (None, RUN_A, RUN_B):
                        if scoped_run is not None:
                            status, _ = await self.request(
                                app, f"/api/runs/{scoped_run}/feedback/note?entity_id=001",
                                method="PUT", payload={"note": "Locally selected run"},
                            )
                            self.assertEqual(status, 200 if destination is not None else 409)
                        self.assertEqual(await self.request(app, "/case_state.json"), (204, b""))
                        status, body = await self.request(app, "/api/feedback")
                        self.assertEqual(status, 200)
                        document = json.loads(body)
                        self.assertIsNone(document["run_id"])
                        self.assertIsNone(document["state_file"])
                        self.assertIs(document["writable"], False)
                        self.assertTrue(document["read_only_reason"])
                        self.assertEqual(document["annotations"], EMPTY_ANNOTATIONS)
                        status, _ = await self.request(app, "/api/feedback/note/001", method="PUT", payload={"note": "No binding"})
                        self.assertEqual(status, 409)
                        status, body = await self.request(app, "/api/ground-truth")
                        self.assertEqual(status, 409)
                        self.assertIn("cannot be bound", json.loads(body)["detail"])
                        for run_id in (RUN_A, RUN_B):
                            status, body = await self.request(app, f"/api/runs/{run_id}/ground-truth")
                            self.assertEqual(status, 409)
                            self.assertEqual(json.loads(body)["run_id"], run_id)
                            self.assertNotIn("values", json.loads(body))
                    read.assert_not_called()
            self.assertEqual({path.name for path in reviews.iterdir()}, {f"{RUN_A}.json", f"{RUN_B}.json"})

    async def test_directory_restores_isolated_runs_and_string_entity_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "new" / "reviews"
            app = build_app(feedback_dir=reviews)
            for run_id in (RUN_A, RUN_B):
                status, body = await self.request(app, f"/api/runs/{run_id}/feedback")
                self.assertEqual(status, 200)
                document = json.loads(body)
                self.assertEqual(document["run_id"], run_id)
                self.assertIs(document["writable"], True)
                self.assertIsNone(document["read_only_reason"])
                self.assertIsNone(document["state_file"])
                self.assertEqual(document["annotations"], EMPTY_ANNOTATIONS)
            self.assertFalse(reviews.parent.exists(), "GET must not create directories or files")

            saves = [
                (run_id, kind, entity_id)
                for run_id in (RUN_A, RUN_B)
                for kind in EMPTY_ANNOTATIONS
                for entity_id in ("1", "001", "note-A")
            ]
            responses = await asyncio.gather(*(
                self.request(app, f"/api/runs/{run_id}/feedback/{kind}/{entity_id}", method="PUT",
                             payload={"note": f"{run_id}:{kind}:{entity_id}"})
                for run_id, kind, entity_id in saves
            ))
            for (run_id, kind, entity_id), (status, body) in zip(saves, responses):
                self.assertEqual(status, 200)
                result = json.loads(body)
                self.assertEqual(set(result), {"run_id", "kind", "id", "annotation"})
                self.assertEqual((result["run_id"], result["kind"], result["id"]), (run_id, kind, entity_id))

            self.assertEqual({path.name for path in reviews.iterdir()}, {f"{RUN_A}.json", f"{RUN_B}.json"})
            app = build_app(feedback_dir=reviews)
            for run_id in (RUN_A, RUN_B, RUN_A):
                path = reviews / f"{run_id}.json"
                original = path.read_bytes()
                status, body = await self.request(app, f"/api/runs/{run_id}/feedback")
                self.assertEqual(status, 200)
                document = json.loads(body)
                self.assertEqual(document["run_id"], run_id)
                self.assertEqual(json.loads(original)["run_id"], run_id)
                self.assertNotIn("writable", json.loads(original))
                for kind in EMPTY_ANNOTATIONS:
                    self.assertEqual(set(document["annotations"][kind]), {"1", "001", "note-A"})
                    for entity_id, record in document["annotations"][kind].items():
                        self.assertEqual(record["note"], f"{run_id}:{kind}:{entity_id}")
                self.assertEqual(path.read_bytes(), original, "GET must not rewrite a document")

    async def test_invalid_uuid_and_path_inputs_cannot_select_files(self):
        invalid_ids = (
            "not-a-uuid", "..", "../outside", "..%2Foutside", "%2e%2e", "0001", RUN_A.upper(),
            RUN_A.replace("-", ""), "{" + RUN_A + "}", "urn:uuid:" + RUN_A, RUN_A + ".json",
        )
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "reviews"
            app = build_app(feedback_dir=reviews)
            for run_id in invalid_ids:
                for method in ("GET", "PUT"):
                    with self.subTest(run_id=run_id, method=method):
                        url = f"/api/runs/{run_id}/feedback" + ("/note/001" if method == "PUT" else "")
                        status, _ = await self.request(app, url, method=method, payload={"note": "Do not save"})
                        self.assertIn(status, (400, 404, 405))
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_corrupt_or_conflicting_documents_are_never_overwritten(self):
        cases = [(b"{", 500), (b"", 500), (b"\xff", 500), (b"null", 500), (b"[]", 500)]
        for identity in (None, 123, "invalid", RUN_A.upper(), RUN_B):
            cases.append((json.dumps({"run_id": identity, "annotations": {}}).encode(), 409))
        cases.append((b'{"annotations": {}}', 409))
        for annotations in (None, [], {"note": []}, {"group": None}, {"variable": {"400": None}},
                            {"note": {"001": "bad record"}}, {"note": {"001": {"note": []}}},
                            {"note": {"001": {"flags": "not-a-list"}}}, {"note": {"001": {"flags": [123]}}}):
            cases.append((json.dumps({"run_id": RUN_A, "annotations": annotations}).encode(), 500))
        for expected in (0, False, [], {}, ["001"]):
            cases.append((json.dumps({"run_id": RUN_A, "annotations": {"note": {"001": {"expected": expected}}}}).encode(), 500))
        cases.append((json.dumps({"run_id": RUN_A}).encode(), 500))
        cases.append((f'{{"run_id": "{RUN_B}", "run_id": "{RUN_A}", "annotations": {{}}}}'.encode(), 500))
        cases.append((f'{{"run_id": "{RUN_A}", "annotations": {{"note": {{"001": {{}}, "001": {{}}}}}}}}'.encode(), 500))
        for number in ("NaN", "Infinity", "-Infinity", "1e400"):
            cases.append((f'{{"run_id": "{RUN_A}", "annotations": {{}}, "metadata": [{number}]}}'.encode(), 500))
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory)
            path = reviews / f"{RUN_A}.json"
            app = build_app(feedback_dir=reviews)
            for original, expected_status in cases:
                path.write_bytes(original)
                for method, suffix in (("GET", ""), ("PUT", "/note/001"), ("PUT", "/group?entity_id=unrelated")):
                    with self.subTest(original=original, method=method, suffix=suffix):
                        url = f"/api/runs/{RUN_A}/feedback" + suffix
                        status, body = await self.request(app, url, method=method, payload={"note": "Do not overwrite"})
                        self.assertEqual(status, expected_status)
                        self.assertEqual(json.loads(body)["run_id"], RUN_A)
                        self.assertTrue(json.loads(body)["detail"])
                        self.assertEqual(path.read_bytes(), original)
                        self.assertEqual(list(reviews.iterdir()), [path])

    async def test_existing_non_file_is_an_error_not_an_empty_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{RUN_A}.json"
            path.mkdir()
            app = build_app(feedback_dir=Path(directory))
            for method in ("GET", "PUT"):
                url = f"/api/runs/{RUN_A}/feedback" + ("/note/001" if method == "PUT" else "")
                status, _ = await self.request(app, url, method=method, payload={"note": "Do not overwrite"})
                self.assertEqual(status, 500)
                self.assertTrue(path.is_dir())

    async def test_legacy_single_file_binding_and_explicit_save_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            path = Path(directory) / "legacy.json"
            legacy = {
                "schema_version": "legacy", "state_file": "original.json", "reviewer": {"id": "reviewer-1"},
                "updated_at": "yesterday", "annotations": {
                    "note": {"001": {"note": "Keep me", "custom": "retained"}},
                    "variable": {"400": {"flags": ["wrong"]}},
                },
            }
            original = json.dumps(legacy, indent=4).encode()
            app = build_app(state_path=state, feedback_path=path)
            status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["annotations"], EMPTY_ANNOTATIONS)
            self.assertIs(json.loads(body)["writable"], True)
            self.assertFalse(path.exists())
            path.write_bytes(original)
            for url in ("/api/feedback", f"/api/runs/{RUN_A}/feedback"):
                status, body = await self.request(app, url)
                self.assertEqual(status, 200)
                document = json.loads(body)
                self.assertEqual(document["run_id"], RUN_A)
                self.assertIs(document["writable"], True)
                self.assertIsNone(document["read_only_reason"])
                self.assertEqual(document["annotations"]["note"], {
                    "001": {**legacy["annotations"]["note"]["001"], "flags": [], "expected": None},
                })
                self.assertEqual(document["annotations"]["variable"], {
                    "400": {"flags": ["wrong"], "note": "", "expected": None},
                })
                self.assertEqual(document["annotations"]["group"], {})
                self.assertEqual(path.read_bytes(), original)

            status, body = await self.request(app, f"/api/runs/{RUN_B}/feedback")
            self.assertEqual(status, 200)
            document = json.loads(body)
            self.assertEqual(document["run_id"], RUN_B)
            self.assertIs(document["writable"], False)
            self.assertIn("--feedback-dir", document["read_only_reason"])
            self.assertEqual(document["annotations"], EMPTY_ANNOTATIONS)
            status, _ = await self.request(app, f"/api/runs/{RUN_B}/feedback/note/001",
                                           method="PUT", payload={"note": "Wrong run"})
            self.assertEqual(status, 409)
            self.assertEqual(path.read_bytes(), original)

            status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback/note/1",
                                              method="PUT", payload={"note": " New note "})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["run_id"], RUN_A)
            migrated = json.loads(path.read_bytes())
            self.assertEqual(migrated["run_id"], RUN_A)
            self.assertEqual(migrated["annotations"]["note"]["001"], legacy["annotations"]["note"]["001"])
            self.assertEqual(migrated["annotations"]["note"]["1"]["note"], "New note")
            self.assertEqual(migrated["annotations"]["variable"], legacy["annotations"]["variable"])
            for key in ("schema_version", "state_file", "reviewer"):
                self.assertEqual(migrated[key], legacy[key])
            self.assertNotEqual(migrated["updated_at"], legacy["updated_at"])

    async def test_single_file_rejects_recorded_mismatches_and_corruption_on_all_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            path = Path(directory) / "feedback.json"
            app = build_app(state_path=state, feedback_path=path)
            for original, expected_status in (
                (b"{", 500),
                (json.dumps({"run_id": RUN_B, "annotations": {}}).encode(), 409),
                (json.dumps({"run_id": None, "annotations": {}}).encode(), 409),
            ):
                path.write_bytes(original)
                for base in ("/api/feedback", f"/api/runs/{RUN_A}/feedback"):
                    for method in ("GET", "PUT"):
                        with self.subTest(original=original, base=base, method=method):
                            url = base + ("/note/001" if method == "PUT" else "")
                            status, _ = await self.request(app, url, method=method, payload={"note": "Do not overwrite"})
                            self.assertEqual(status, expected_status)
                            self.assertEqual(path.read_bytes(), original)

    async def test_unscoped_routes_stay_startup_bound_in_directory_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            reviews = Path(directory) / "reviews"
            app = build_app(state_path=state, feedback_dir=reviews)
            status, _ = await self.request(app, f"/api/runs/{RUN_B}/feedback/note/001",
                                           method="PUT", payload={"note": "Run B"})
            self.assertEqual(status, 200)
            state.write_text(json.dumps({"run": {"run_id": RUN_B}}))
            status, body = await self.request(app, "/api/feedback")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["run_id"], RUN_A)
            self.assertEqual(json.loads(body)["annotations"], EMPTY_ANNOTATIONS)
            status, body = await self.request(app, "/api/feedback/note/001", method="PUT", payload={"note": "Run A"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["run_id"], RUN_A)
            for run_id, expected in ((RUN_A, "Run A"), (RUN_B, "Run B")):
                document = json.loads((reviews / f"{run_id}.json").read_text())
                self.assertEqual(document["annotations"]["note"]["001"]["note"], expected)

    async def test_invalid_startup_identity_preserves_raw_serving_and_local_run_recovery(self):
        invalid_artifacts = (
            b"{bad json", b"\xff", b"[]", b"null", b"{}", b'{"run": []}',
            b'{"run": {"run_id": null}}', json.dumps({"run": {"run_id": RUN_A.upper()}}).encode(),
            b"[" * 2000 + b"]" * 2000,
        )
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            path = Path(directory) / "feedback.json"
            original_feedback = b'{"annotations": {"note": {"001": {"note": "Unbound"}}}}'
            path.write_bytes(original_feedback)
            for original in invalid_artifacts:
                state.write_bytes(original)
                for mode in ({}, {"feedback_path": path}, {"feedback_dir": Path(directory) / "reviews"}):
                    with self.subTest(original=original, mode=mode):
                        app = build_app(state_path=state, **mode)
                        status, body = await self.request(app, "/case_state.json")
                        self.assertEqual((status, body), (200, original))
                        status, body = await self.request(app, "/api/feedback")
                        self.assertEqual(status, 200)
                        document = json.loads(body)
                        self.assertIsNone(document["run_id"])
                        self.assertIs(document["writable"], False)
                        self.assertTrue(document["read_only_reason"])
                        self.assertEqual(document["annotations"], EMPTY_ANNOTATIONS)
                        status, _ = await self.request(app, "/api/feedback/note/001", method="PUT", payload={"note": "No"})
                        self.assertEqual(status, 409)
                        status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback")
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(body)["run_id"], RUN_A)
                        writable = "feedback_dir" in mode
                        self.assertIs(json.loads(body)["writable"], writable)
                        status, _ = await self.request(app, f"/api/runs/{RUN_A}/feedback/note/001",
                                                       method="PUT", payload={"note": "Local recovery"})
                        self.assertEqual(status, 200 if writable else 409)
                        self.assertEqual(path.read_bytes(), original_feedback)
                        self.assertEqual(state.read_bytes(), original)

    async def test_missing_startup_does_not_disable_directory_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(state_path=Path(directory) / "missing.json", feedback_dir=Path(directory) / "reviews")
            status, _ = await self.request(app, "/case_state.json")
            self.assertEqual(status, 404)
            status, _ = await self.request(app, "/")
            self.assertEqual(status, 200)
            status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback")
            self.assertEqual(status, 200)
            self.assertIs(json.loads(body)["writable"], True)

    async def test_read_modify_write_is_locked_across_app_instances(self):
        first_read = threading.Event()
        contender = threading.Event()
        real_lock = server._FEEDBACK_LOCK
        original_load = server._load_feedback
        original_write = server._write_json_atomic

        class ObservedLock:
            def __enter__(self):
                if real_lock.locked():
                    contender.set()
                return real_lock.__enter__()

            def __exit__(self, *args):
                return real_lock.__exit__(*args)

        def paused_load(*args, **kwargs):
            self.assertTrue(real_lock.locked(), "Read must be inside the lock")
            document = original_load(*args, **kwargs)
            if not first_read.is_set():
                first_read.set()
                self.assertTrue(contender.wait(5), "Second save must contend while the first holds its snapshot")
            return document

        def checked_write(*args, **kwargs):
            self.assertTrue(real_lock.locked(), "Write must be inside the same lock")
            return original_write(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            apps = [build_app(feedback_dir=Path(directory)), build_app(feedback_dir=Path(directory))]
            with patch.object(server, "_FEEDBACK_LOCK", ObservedLock()), \
                    patch.object(server, "_load_feedback", side_effect=paused_load), \
                    patch.object(server, "_write_json_atomic", side_effect=checked_write):
                first = asyncio.create_task(self.request(apps[0], f"/api/runs/{RUN_A}/feedback/note/001",
                                                        method="PUT", payload={"note": "First"}))
                self.assertTrue(await asyncio.to_thread(first_read.wait, 5))
                second = asyncio.create_task(self.request(apps[1], f"/api/runs/{RUN_A}/feedback/variable/400",
                                                         method="PUT", payload={"note": "Second"}))
                responses = await asyncio.wait_for(asyncio.gather(first, second), timeout=10)
                self.assertEqual([status for status, _ in responses], [200, 200])
            document = json.loads((Path(directory) / f"{RUN_A}.json").read_text())
            self.assertEqual(document["annotations"]["note"]["001"]["note"], "First")
            self.assertEqual(document["annotations"]["variable"]["400"]["note"], "Second")

    async def test_atomic_failure_keeps_prior_document_and_later_save_can_clear_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{RUN_A}.json"
            app = build_app(feedback_dir=Path(directory))
            url = f"/api/runs/{RUN_A}/feedback/note/001"
            status, _ = await self.request(app, url, method="PUT", payload={"note": "Original", "flags": ["review"]})
            self.assertEqual(status, 200)
            document = json.loads(path.read_bytes())
            document["annotations"]["group"]["other"] = {"note": "Manual edit between saves"}
            path.write_text(json.dumps(document))
            original = path.read_bytes()
            with patch.object(server.os, "replace", side_effect=OSError("Simulated failure")):
                status, _ = await self.request(app, url, method="PUT", payload={"note": "Replacement"})
            self.assertEqual(status, 500)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path], "Failed replacement must clean up its temp file")
            status, _ = await self.request(app, url, method="PUT", payload={"note": "Invalid", "expected": float("nan")})
            self.assertEqual(status, 422)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            status, body = await self.request(app, url, method="PUT", payload={"flags": [], "note": "  "})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"run_id": RUN_A, "kind": "note", "id": "001", "annotation": None})
            saved = json.loads(path.read_bytes())
            self.assertEqual(saved["annotations"]["note"], {})
            self.assertEqual(saved["annotations"]["group"], document["annotations"]["group"])

    async def test_invalid_kind_and_payload_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "reviews"
            app = build_app(feedback_dir=reviews)
            for kind, payload, expected in (
                ("unknown", {"note": "No"}, 404),
                ("note", {"note": 123}, 422), ("note", {"flags": "not-a-list"}, 422),
                ("note", {"flags": [123]}, 422), ("note", {"flags": [None]}, 422),
                ("note", {"flags": [{}]}, 422), ("note", {"flags": [False]}, 422),
                ("note", {"expected": 0}, 422), ("note", {"expected": False}, 422),
                ("note", {"expected": []}, 422), ("note", {"expected": {}}, 422),
                ("note", ["not-an-object"], 422),
            ):
                with self.subTest(kind=kind, payload=payload):
                    for suffix in ("/001", "?entity_id=001"):
                        status, _ = await self.request(app, f"/api/runs/{RUN_A}/feedback/{kind}" + suffix,
                                                       method="PUT", payload=payload)
                        self.assertEqual(status, expected)
            self.assertFalse(reviews.exists())

    async def test_expected_only_annotation_is_meaningful_and_all_empty_fields_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            reviews = Path(directory) / "reviews"
            app = build_app(state_path=state, feedback_dir=reviews)
            path = reviews / f"{RUN_A}.json"
            for url in ("/api/feedback/variable/400", f"/api/runs/{RUN_A}/feedback/variable/400",
                        f"/api/runs/{RUN_A}/feedback/variable?entity_id=400"):
                for expected in ("001", "0", " value "):
                    with self.subTest(url=url, expected=expected):
                        status, body = await self.request(app, url, method="PUT", payload={"expected": expected})
                        self.assertEqual(status, 200)
                        record = json.loads(body)["annotation"]
                        self.assertEqual(record["expected"], expected)
                        self.assertEqual(record["flags"], [])
                        self.assertEqual(record["note"], "")
                        status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback")
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(body)["annotations"]["variable"]["400"], record)
                        self.assertEqual(json.loads(path.read_bytes())["annotations"]["variable"]["400"], record)
                for expected in (None, "", " \t "):
                    with self.subTest(url=url, empty_expected=expected):
                        status, body = await self.request(app, url, method="PUT",
                                                          payload={"flags": [], "note": "  ", "expected": expected})
                        self.assertEqual(status, 200)
                        self.assertIsNone(json.loads(body)["annotation"])
                        self.assertEqual(json.loads(path.read_bytes())["annotations"]["variable"], {})

    async def test_legacy_response_defaults_preserve_bytes_and_unedited_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            legacy_records = {
                "missing": {"custom": {"source": "manual"}},
                "null": {"flags": None, "note": None, "expected": None, "custom": ["kept"]},
                "empty": {"flags": [], "note": "", "expected": ""},
            }
            original = json.dumps({"annotations": {"note": legacy_records}}, indent=4).encode()
            path.write_bytes(original)
            app = build_app(state_path=EXAMPLE_DIR / "case_state.json", feedback_path=path)
            startup_id = json.loads((EXAMPLE_DIR / "case_state.json").read_text())["run"]["run_id"]
            for url in ("/api/feedback", f"/api/runs/{startup_id}/feedback"):
                status, body = await self.request(app, url)
                self.assertEqual(status, 200)
                records = json.loads(body)["annotations"]["note"]
                self.assertEqual(records["missing"], {"flags": [], "note": "", "expected": None, "custom": {"source": "manual"}})
                self.assertEqual(records["null"], {"flags": [], "note": "", "expected": None, "custom": ["kept"]})
                self.assertEqual(records["empty"], legacy_records["empty"])
                self.assertEqual(path.read_bytes(), original)
            status, _ = await self.request(app, f"/api/runs/{startup_id}/feedback/group?entity_id=new",
                                           method="PUT", payload={"note": "New record"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(path.read_bytes())["annotations"]["note"], legacy_records)

    async def test_edit_merges_owned_fields_and_preserves_annotation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{RUN_A}.json"
            original_record = {
                "flags": ["old"], "expected": "old", "note": "Old note", "updated_at": "yesterday",
                "reviewer": {"id": "reviewer-1"}, "audit": ["imported"], "custom": 7,
            }
            original = {"run_id": RUN_A, "annotations": {"note": {
                "001": original_record, "other": {"note": "Do not normalize on save"},
            }}}
            path.write_text(json.dumps(original))
            app = build_app(feedback_dir=Path(directory))
            for suffix in ("?entity_id=001", "/001"):
                status, body = await self.request(app, f"/api/runs/{RUN_A}/feedback/note" + suffix, method="PUT",
                                                  payload={"flags": ["new"], "expected": "001", "note": " Edited ",
                                                           "reviewer": {"id": "do-not-replace"}, "updated_at": "client-time"})
                self.assertEqual(status, 200)
                record = json.loads(body)["annotation"]
                self.assertEqual(record["flags"], ["new"])
                self.assertEqual(record["expected"], "001")
                self.assertEqual(record["note"], "Edited")
                self.assertNotIn(record["updated_at"], ("yesterday", "client-time"))
                for key in ("reviewer", "audit", "custom"):
                    self.assertEqual(record[key], original_record[key])
                saved = json.loads(path.read_bytes())
                self.assertEqual(saved["annotations"]["note"]["001"], record)
                self.assertEqual(saved["annotations"]["note"]["other"], original["annotations"]["note"]["other"])

    async def test_query_transport_round_trips_arbitrary_string_ids(self):
        entity_ids = ("1", "001", "a/b", "/", ".", "..", "../../other.json", "%", "%2F", "%2e%2e",
                      "a?b=c&d#e", "plus+space ", "\u00e9/\u764c", "")
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(feedback_dir=Path(directory))
            for run_id in (RUN_A, RUN_B):
                for kind in EMPTY_ANNOTATIONS:
                    for entity_id in entity_ids:
                        with self.subTest(run_id=run_id, kind=kind, entity_id=entity_id):
                            url = f"/api/runs/{run_id}/feedback/{kind}?" + urlencode({"entity_id": entity_id})
                            status, body = await self.request(app, url, method="PUT", payload={"expected": run_id})
                            self.assertEqual(status, 200)
                            result = json.loads(body)
                            self.assertEqual((result["run_id"], result["kind"], result["id"]), (run_id, kind, entity_id))
                            self.assertEqual(result["annotation"]["expected"], run_id)
                status, body = await self.request(app, f"/api/runs/{run_id}/feedback")
                self.assertEqual(status, 200)
                annotations = json.loads(body)["annotations"]
                self.assertEqual(json.loads((Path(directory) / f"{run_id}.json").read_bytes())["annotations"], annotations)
                for kind in EMPTY_ANNOTATIONS:
                    self.assertEqual(set(annotations[kind]), set(entity_ids))
                    self.assertTrue(all(record["expected"] == run_id for record in annotations[kind].values()))
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {f"{RUN_A}.json", f"{RUN_B}.json"})

    async def test_query_transport_requires_id_and_honors_uuid_and_read_only_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            path = Path(directory) / "feedback.json"
            app = build_app(state_path=state, feedback_path=path)
            for url, expected in (
                (f"/api/runs/{RUN_A}/feedback/note", 422),
                ("/api/runs/invalid/feedback/note?entity_id=x%2Fy", 400),
                (f"/api/runs/{RUN_A.upper()}/feedback/note?entity_id=x%2Fy", 400),
                (f"/api/runs/{RUN_B}/feedback/note?entity_id=x%2Fy", 409),
            ):
                status, _ = await self.request(app, url, method="PUT", payload={"entity_id": "not-a-query-id", "note": "No"})
                self.assertEqual(status, expected)
                self.assertFalse(path.exists())
            status, _ = await self.request(build_app(), f"/api/runs/{RUN_A}/feedback/note?entity_id=001",
                                           method="PUT", payload={"note": "No destination"})
            self.assertEqual(status, 409)

    async def test_scoped_ground_truth_stays_bound_when_startup_artifact_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            truth = Path(directory) / "truth.json"
            values = {"400": "C500", "001": "Keep string keys"}
            original = json.dumps(values, indent=4).encode()
            truth.write_bytes(original)
            app = build_app(state_path=state, ground_truth_path=truth)
            status, body = await self.request(app, f"/api/runs/{RUN_A}/ground-truth")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"run_id": RUN_A, "values": values})
            replacement = json.dumps({"run": {"run_id": RUN_B}}, indent=4).encode()
            state.write_bytes(replacement)
            status, body = await self.request(app, "/case_state.json")
            self.assertEqual((status, body), (200, replacement))
            with patch.object(server, "_read_json", side_effect=AssertionError("Must reject before reading ground truth")):
                status, body = await self.request(app, f"/api/runs/{RUN_B}/ground-truth")
            self.assertEqual(status, 409)
            self.assertEqual(json.loads(body)["run_id"], RUN_B)
            self.assertNotIn("values", json.loads(body))
            status, body = await self.request(app, f"/api/runs/{RUN_A}/ground-truth")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"run_id": RUN_A, "values": values})
            status, body = await self.request(app, "/api/ground-truth")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), values, "Legacy route retains its shipped unscoped response")
            self.assertEqual(truth.read_bytes(), original)
            self.assertEqual(state.read_bytes(), replacement)

    async def test_scoped_ground_truth_validates_ids_and_returns_empty_without_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            app = build_app(state_path=state)
            status, body = await self.request(app, f"/api/runs/{RUN_A}/ground-truth")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"run_id": RUN_A, "values": {}})
            for run_id, expected in ((RUN_B, 409), ("invalid", 400), (RUN_A.upper(), 400), ("..", 400)):
                status, body = await self.request(app, f"/api/runs/{run_id}/ground-truth")
                self.assertEqual(status, expected)
                self.assertEqual(json.loads(body)["run_id"], run_id)
                self.assertNotIn("values", json.loads(body))
            status, body = await self.request(app, "/api/ground-truth")
            self.assertEqual((status, json.loads(body)), (200, {}))

    async def test_scoped_ground_truth_cannot_bind_after_unresolved_startup_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.json"
            truth.write_text('{"400": "C500"}')
            for index, original in enumerate((None, b"{broken", b"{}", b'{"run": {"run_id": "invalid"}}')):
                for configured in (False, True):
                    with self.subTest(original=original, configured=configured):
                        state = Path(directory) / f"run-{index}-{configured}.json"
                        if original is not None:
                            state.write_bytes(original)
                        app = build_app(state_path=state, ground_truth_path=truth if configured else None)
                        for replaced in (False, True):
                            if replaced:
                                state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
                            status, body = await self.request(app, f"/api/runs/{RUN_A}/ground-truth")
                            self.assertEqual(status, 409)
                            self.assertEqual(json.loads(body)["run_id"], RUN_A)
                            self.assertNotIn("values", json.loads(body))
                        status, body = await self.request(app, "/api/ground-truth")
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(body), {"400": "C500"} if configured else {})

    async def test_scoped_ground_truth_read_errors_do_not_masquerade_as_empty_values(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"run": {"run_id": RUN_A}}))
            for index, original in enumerate((None, b"{broken", b"null", b"[]", b'{"400": NaN}')):
                with self.subTest(original=original):
                    truth = Path(directory) / f"truth-{index}.json"
                    if original is not None:
                        truth.write_bytes(original)
                    app = build_app(state_path=state, ground_truth_path=truth)
                    status, body = await self.request(app, f"/api/runs/{RUN_A}/ground-truth")
                    self.assertEqual(status, 500)
                    self.assertEqual(json.loads(body)["run_id"], RUN_A)
                    self.assertNotIn("values", json.loads(body))

    def test_feedback_destinations_are_mutually_exclusive_for_direct_callers(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_app(feedback_path=Path("feedback.json"), feedback_dir=Path("reviews"))

    def test_startup_bound_options_require_explicit_state_for_direct_callers(self):
        for options in (
            {"feedback_path": Path("feedback.json")},
            {"ground_truth_path": Path("truth.json")},
            {"feedback_path": Path("feedback.json"), "ground_truth_path": Path("truth.json")},
            {"feedback_dir": Path("reviews"), "ground_truth_path": Path("truth.json")},
        ):
            with self.subTest(options=options), patch.object(server, "_read_json") as read:
                with self.assertRaisesRegex(ValueError, "require an explicit --state"):
                    build_app(**options)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
