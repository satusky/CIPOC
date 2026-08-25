"""Phase 3 — the web frontend contract: assets, the ``/api/graph`` topology, and
the fine-node → overview-block map the animated Panel-1 highlighting depends on.

The frontend itself is browser JavaScript, so these tests pin the *contract* it
consumes rather than its rendering: the static assets exist and are wired up, the
graph endpoint returns a drawable overview chart, and the coarse map that decides
which map block lights up covers every node the graph can actually emit (and only
real nodes), so the map can never silently drift from the pipeline.
"""

import json
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from cipoc.demo.mapping import (  # noqa: E402
    _INITIALIZE_BY_AGENT,
    map_node_id,
    map_node_ids,
    overview_block_map,
)
from cipoc.demo.server import (  # noqa: E402
    WEB_DIR,
    build_app,
    load_replay_session,
    overview_chart,
)
from cipoc.utils.progress.model import DEFAULT_NODE_KINDS  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"
WEB_ASSETS = ("index.html", "app.js", "styles.css", "cytoscape.min.js")


def _client():
    from fastapi.testclient import TestClient

    return TestClient(build_app(load_replay_session(FIXTURE)))


def _reachable_fine_ids() -> set[str]:
    """Every agent_system node ID that :func:`map_node_id` can emit at runtime."""
    ids: set[str] = set(_INITIALIZE_BY_AGENT.values())
    for name in DEFAULT_NODE_KINDS:
        resolved = map_node_id(name)
        if resolved is not None:
            ids.add(resolved)
    return ids


def _overview_node_ids() -> set[str]:
    chart = overview_chart()
    return {node["data"]["id"] for node in chart["elements"]["nodes"]}


class WebAssetTests(unittest.TestCase):
    def test_all_assets_present(self):
        for name in WEB_ASSETS:
            self.assertTrue((WEB_DIR / name).is_file(), f"missing web asset: {name}")

    def test_vendored_cytoscape_is_the_library(self):
        text = (WEB_DIR / "cytoscape.min.js").read_text(errors="ignore")
        self.assertIn("cytoscape", text.lower())
        self.assertGreater((WEB_DIR / "cytoscape.min.js").stat().st_size, 100_000)

    def test_index_wires_up_assets(self):
        html = (WEB_DIR / "index.html").read_text()
        for ref in ("app.js", "styles.css", "cytoscape.min.js"):
            self.assertIn(ref, html)

    def test_app_js_fetches_the_backend_contract(self):
        app_js = (WEB_DIR / "app.js").read_text()
        for endpoint in ("/api/meta", "/api/graph", "/api/steps", "/api/stream"):
            self.assertIn(endpoint, app_js)

    def test_app_js_wires_phase4_component_views(self):
        """Panel-2 dispatch, inline evidence highlighting, and live push exist."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in (
            "componentHeadline",  # per-component dispatch
            "renderEvidence",     # evidence-span block
            "highlightContent",   # inline <mark> highlighting
            "/api/notes",         # raw note text for highlighting
            "viewExtractions",    # extractor coded values + repair loop
            "viewRetriever",      # kept/dropped candidate notes
            "applyLive",          # live-mode push-on-event handling
            "renderFanoutDetail", # collapsed fan-out step -> per-instance cards
            "renderInstanceDetail",
            "viewNote",           # per-note summary + concepts + mentions together
            "pendingSlot",        # live skeleton: unfilled characterization slot
        ):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")

    def test_styles_cover_phase4_views(self):
        css = (WEB_DIR / "styles.css").read_text()
        for cls in (".extraction", ".repair-badge", ".note-text mark", ".concept-chip",
                    ".headline-fact.pending"):
            self.assertIn(cls, css, f"styles.css missing {cls}")

    def test_app_js_wires_reference_map_layout(self):
        app_js = (WEB_DIR / "app.js").read_text()
        for wiring in (
            "const MAP_POS",
            "const MAP_HIDE",
            "function mapStyle",
            'edge[kind="loop"]',
            '"taxi-turn": "90%"',
            'visitedCoarse.add("relevant_notes_gate")',
        ):
            self.assertIn(wiring, app_js, f"app.js missing map wiring: {wiring}")


class OverviewChartTests(unittest.TestCase):
    def test_chart_is_drawable(self):
        chart = overview_chart()
        self.assertTrue(chart["elements"]["nodes"])
        self.assertTrue(chart["elements"]["edges"])
        self.assertTrue(chart["style"])
        self.assertIn("orchestrator", chart["agent_colors"])

    def test_graph_endpoint_serves_the_chart(self):
        graph = _client().get("/api/graph").json()
        self.assertIn("coarse_map", graph)
        self.assertEqual(
            {n["data"]["id"] for n in graph["elements"]["nodes"]},
            _overview_node_ids(),
        )

    def test_static_assets_served_by_app(self):
        client = _client()
        self.assertEqual(client.get("/").status_code, 200)
        for name in ("app.js", "styles.css", "cytoscape.min.js"):
            self.assertEqual(client.get(f"/{name}").status_code, 200, name)


class CoarseMapCoverageTests(unittest.TestCase):
    """The fine → overview-block map must fully and validly cover the graph."""

    def test_every_reachable_node_has_a_block(self):
        coarse = overview_block_map()
        missing = _reachable_fine_ids() - coarse.keys()
        self.assertEqual(missing, set(), f"unmapped map nodes: {sorted(missing)}")

    def test_every_block_is_a_real_overview_node(self):
        blocks = set(overview_block_map().values())
        unknown = blocks - _overview_node_ids()
        self.assertEqual(unknown, set(), f"blocks not in overview chart: {sorted(unknown)}")

    def test_every_key_is_a_real_agent_system_node(self):
        keys = set(overview_block_map())
        unknown = keys - map_node_ids()
        self.assertEqual(unknown, set(), f"keys not in agent_system: {sorted(unknown)}")

    def test_fixture_snapshot_nodes_all_map_to_blocks(self):
        """Nodes the actual fixture reports must all resolve to a lit block."""
        session = load_replay_session(FIXTURE)
        coarse = overview_block_map()
        snap = session.step_snapshot(len(session.steps) - 1)
        reported = set(snap["visited_map_nodes"]) | set(snap["details"].keys())
        unmapped = {n for n in reported if n not in coarse}
        self.assertEqual(unmapped, set(), f"fixture nodes with no block: {sorted(unmapped)}")

    def test_extract_wrapper_maps_to_relevant_notes_gate(self):
        self.assertEqual(map_node_id("extract"), "relevant_notes_gate")
        self.assertEqual(
            overview_block_map()["relevant_notes_gate"],
            "relevant_notes_gate",
        )


if __name__ == "__main__":
    unittest.main()
