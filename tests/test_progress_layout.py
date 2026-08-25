import unittest
from dataclasses import replace
from pathlib import Path

from cipoc.tools import load_group_hierarchy, load_variable_groups
from cipoc.utils.progress.layout import build_rows, render_lines
from cipoc.utils.progress.events import ProgressEvent
from cipoc.utils.progress.model import BranchSnapshot, ProgressModel, Stage


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_GROUPS = REPO_ROOT / "config" / "variable_groups.json"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "progress_layout.golden"
VIEWPORTS = ((100, 24), (80, 24), (60, 24), (100, 50), (80, 50), (60, 50))


def dashboard_snapshot():
    groups = load_variable_groups(VARIABLE_GROUPS)
    model = ProgressModel(
        "Orchestrator",
        100.0,
        target_groups=groups,
        group_hierarchy=load_group_hierarchy(VARIABLE_GROUPS),
        graph_input={"note_corpus": {item: {} for item in range(14)}},
        show_note_counts=True,
    )
    base = model.snapshot()
    variables = dict(base.variables)

    def update(item_id, **changes):
        variables[item_id] = replace(variables[item_id], **changes)

    def terminal(item_ids, status, **changes):
        for item_id in item_ids:
            update(item_id, stage=Stage.DONE, status=status, **changes)

    update(
        390,
        stage=Stage.DONE,
        status="extracted",
        value="20260115",
        confidence="max",
    )
    update(400, stage=Stage.DONE, status="structured_data", value="C50.9")
    update(410, stage=Stage.DONE, status="not_found", confidence="low")

    terminal(
        (690, 700, 710, 720, 740),
        "not_applicable",
        detail="Corpus gate not met: treatment.",
    )
    terminal(
        (1200, 1210, 1220, 1230, 1240, 1270, 1280),
        "not_applicable",
        detail="Corpus gate not met: treatment.",
    )

    update(1112, stage=Stage.DONE, status="extracted", value="1", confidence="max")
    update(1113, stage=Stage.DONE, status="not_found", confidence="low")
    update(1114, stage=Stage.VALIDATE, attempt=2)
    update(1115, stage=Stage.EXTRACT, attempt=1)
    update(1116, stage=Stage.DONE, status="extracted", value="1", confidence="high")
    update(
        1117,
        stage=Stage.DONE,
        status="extracted",
        value="0",
        confidence="low",
        flag="?",
    )

    terminal((674, 676, 682), "extracted", value="1", confidence="high")
    update(
        672,
        stage=Stage.DONE,
        status="error",
        detail="Invalid code returned by extractor.",
        confidence="medium",
        flag="!",
    )

    update(670, stage=Stage.RETRIEVE)
    update(671, stage=Stage.IDLE)
    update(
        3843,
        stage=Stage.DONE,
        status="error",
        detail="Invalid grade code.",
        confidence="medium",
        flag="!",
    )
    update(3844, stage=Stage.EXTRACT, attempt=1)
    update(522, stage=Stage.DONE, status="extracted", value="8500", confidence="max")
    update(523, stage=Stage.DONE, status="extracted", value="3", confidence="max")

    terminal((764, 820, 830, 1182, 3280, 756), "not_found", confidence="low")
    terminal(
        (832, 3836, 3838, 3839, 3840, 3841, 3842),
        "not_applicable",
        detail="Primary site does not apply.",
    )

    group_changes = {
        "initial_llm_extraction": {"note_count": 3},
        "first_course_treatment": {"annotation": "gate:treatment ✗"},
        "metastases": {
            "annotation": "gate:mets ✓",
            "active": True,
            "stage": Stage.VALIDATE,
            "note_count": 3,
        },
        "lymph_node_removal": {"annotation": "gate:nodes ✓", "note_count": 2},
        "site_specific_codes": {"stage": Stage.EXTRACT, "note_count": 7},
        "histologic_type_and_behavior": {"note_count": 7},
        "other": {"note_count": 4},
    }
    snapshot_groups = tuple(
        replace(group, **group_changes.get(group.group_id, {})) for group in base.groups
    )
    counts = {}
    for variable in variables.values():
        counts[variable.status] = counts.get(variable.status, 0) + 1

    return replace(
        base,
        groups=snapshot_groups,
        variables=variables,
        branches=(
            BranchSnapshot(
                key="metastases",
                label="Metastases",
                stage=Stage.VALIDATE,
                variables=2,
                note_count=3,
                started_at=160.0,
            ),
        ),
        notes_done=14,
        counts=counts,
        review_flags=3,
    )


class ProgressLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = dashboard_snapshot()

    def test_viewport_goldens(self):
        sections = []
        for width, height in VIEWPORTS:
            rows = build_rows(self.snapshot, width, height, now=172.0, tick=0)
            sections.append(
                f"== {width}x{height} ==\n"
                + "\n".join(render_lines(rows)).rstrip()
            )
        actual = "\n\n".join(sections) + "\n"
        self.assertEqual(actual, GOLDEN.read_text())

    def test_every_frame_fits_its_viewport(self):
        for width, height in VIEWPORTS:
            with self.subTest(width=width, height=height):
                rows = build_rows(self.snapshot, width, height, now=172.0)
                self.assertLessEqual(len(rows), height)
                self.assertTrue(all(len(row.text) <= width for row in rows))

    def test_unbounded_layout_expands_every_variable(self):
        rows = build_rows(self.snapshot, 100, None, now=172.0)
        variable_rows = [row for row in rows if row.kind == "variable"]
        self.assertEqual(len(variable_rows), len(self.snapshot.variables))
        self.assertEqual(
            [int(row.text[2:6]) for row in variable_rows],
            list(self.snapshot.variables),
        )

    def test_active_group_is_not_collapsed(self):
        rows = build_rows(self.snapshot, 80, 24, now=172.0)
        text = "\n".join(render_lines(rows))
        self.assertIn("▾ Metastases", text)
        self.assertIn("1114", text)
        self.assertIn("validate·2", text)

    def test_24_row_dashboard_keeps_all_top_level_groups_and_active_details(self):
        rows = build_rows(self.snapshot, 100, 24, now=172.0)
        text = "\n".join(render_lines(rows))

        for group in self.snapshot.groups:
            if group.depth == 0:
                with self.subTest(group=group.name):
                    self.assertIn(group.name, text)
        self.assertIn("▾ Metastases", text)
        self.assertIn("1114", text)
        self.assertIn("validate·2", text)

    def test_final_report_prompt_is_live_layout_only(self):
        snapshot = replace(self.snapshot, finished=True, branches=())

        live = "\n".join(
            render_lines(
                build_rows(
                    snapshot,
                    80,
                    24,
                    now=172.0,
                    report_prompt=True,
                )
            )
        )
        persistent = "\n".join(
            render_lines(build_rows(snapshot, 80, None, now=172.0))
        )

        self.assertIn("Press Enter to view report", live)
        self.assertNotIn("Press Enter to view report", persistent)

    def test_standalone_extractor_combines_variable_table_and_node_timeline(self):
        graph_input = {
            "requested_variables": {
                "group_id": "standalone",
                "name": "Standalone",
                "variables": [
                    {"item_id": 390, "name": "Date of Diagnosis"},
                    {"item_id": 400, "name": "Primary Site"},
                ],
            },
            "notes": [{"note_id": 1}],
        }
        model = ProgressModel(
            "Extractor",
            10.0,
            graph_input=graph_input,
            show_note_counts=True,
        )
        model.ingest(
            ProgressEvent(
                kind="task_start",
                namespace=(),
                node="variable_branch",
                task_id="variable-1",
                payload={
                    "task": {
                        "variable": {"item_id": 390},
                        "extraction_attempts": 1,
                    }
                },
            ),
            11.0,
        )

        snapshot = model.snapshot()
        self.assertEqual(snapshot.groups[0].note_count, 1)
        rows = build_rows(snapshot, 100, 24, now=12.0)
        text = "\n".join(render_lines(rows))
        self.assertIn("CIPOC · Extractor · 2 variables", text)
        self.assertIn("Date of Diagnosis", text)
        self.assertIn("extract", text)
        self.assertIn("variable_branch", text)
        self.assertIn("1n", text)
        for width in (60, 80, 100):
            frame = build_rows(snapshot, width, 24, now=12.0)
            self.assertLessEqual(len(frame), 24)
            self.assertTrue(all(len(row.text) <= width for row in frame))

    def test_compact_agent_uses_step_chrome_and_completion_copy(self):
        model = ProgressModel("Note Scanner", 10.0, graph_input={})
        model.ingest(
            ProgressEvent(
                kind="task_start",
                namespace=(),
                node="detect_concepts",
                task_id="scan-1",
                payload={},
            ),
            11.0,
        )
        model.ingest(
            ProgressEvent(
                kind="task_end",
                namespace=(),
                node="detect_concepts",
                task_id="scan-1",
                payload={},
            ),
            12.0,
        )
        model.finish()

        rows = build_rows(model.snapshot(), 80, 12, now=13.0)
        text = "\n".join(render_lines(rows))
        self.assertIn("steps", text)
        self.assertIn("1/1", text)
        self.assertIn("detect_concepts", text)
        self.assertIn("complete in 00:03", text)
        self.assertNotIn("0/0 variables", text)
        for width in (60, 80, 100):
            frame = build_rows(model.snapshot(), width, 12, now=13.0)
            self.assertLessEqual(len(frame), 12)
            self.assertTrue(all(len(row.text) <= width for row in frame))


if __name__ == "__main__":
    unittest.main()
