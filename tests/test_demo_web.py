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

    def test_app_js_renders_one_container_per_variable(self):
        """An extraction pass is per-group and then per-variable."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in (
            "renderExtractDetail",   # extract_branch step -> per-group sections
            "renderGroupDetail",     # one container per group in the pass
            "renderVariableDetail",  # one container per variable
            "renderAttempts",        # repeated validation behind one dropdown
            "variable_branch",       # the fan-out instances it selects
            "stepTaskIds",           # scoped to the groups *this* pass fanned out
        ):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")
        # The merged group result just repeats the variable cards, so the node
        # that produces it is demoted out of the extraction step's cards.
        minor = app_js.split("const MINOR_NODES")[1].split("]")[0]
        self.assertIn("merge_variable_results", minor)

    def test_app_js_quiets_structural_nodes(self):
        """Initialization / fan-outs / gates are quiet rows in the step timeline."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("MINOR_NODES", "renderMinorRow", "renderTimeline", "isMinor",
                       "NODE_TITLES", "nodeHead"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")
        # Raw payload dropdowns are opt-in (pinned components), not the default.
        self.assertIn('raw ? collapsible("Task input"', app_js)

    def test_app_js_renders_model_calls_readably(self):
        """Inline icon, role-colored bubbles, and compact highlighted JSON."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("CALL_ICON", "msgBubble", "roleClass", "compactJSON",
                       "highlightJSON", "codeBlock"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")
        self.assertNotIn("🧠", app_js, "the brain emoji should be gone")
        # Each note sub-step's call belongs inside that sub-step's slot.
        self.assertIn("NOTE_SLOT_NODES", app_js)
        self.assertIn("callsFor", app_js)

    def test_app_js_summarizes_the_case(self):
        """Case facts get their own container, and the run ends with a summary."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("viewCaseFacts", "viewFinalSummary"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")

    def test_app_js_wires_the_panel_splitter(self):
        app_js = (WEB_DIR / "app.js").read_text()
        html = (WEB_DIR / "index.html").read_text()
        self.assertIn("wireSplitter", app_js)
        self.assertIn("--vars-h", app_js)
        self.assertIn('id="row-split"', html)

    def test_styles_cover_phase4_views(self):
        css = (WEB_DIR / "styles.css").read_text()
        for cls in (".extraction", ".repair-badge", ".note-text mark", ".concept-chip",
                    ".headline-fact.pending"):
            self.assertIn(cls, css, f"styles.css missing {cls}")

    def test_styles_cover_the_cleanup_views(self):
        css = (WEB_DIR / "styles.css").read_text()
        for cls in (".node-head", ".node-head-title", ".minor-row",
                    ".node-detail.variable", ".node-detail.group", ".attempt",
                    ".row-split", "--vars-h", ".msg.role-system", ".call-icon",
                    ".j-key", "pre.code.json", ".map-tip", ".map-scrub", ".cy-wrap"):
            self.assertIn(cls, css, f"styles.css missing {cls}")

    def test_app_js_derives_the_map_from_the_run(self):
        """The map is built from this run's notes/groups/variables, not a chart."""
        app_js = (WEB_DIR / "app.js").read_text()
        for wiring in (
            "function buildMapModel",   # nodes/edges from the snapshot
            "function computeLayout",   # positions, since counts are run-dependent
            "function buildMapIndex",   # per-instance timing from the event list
            "function renderMapAt",     # classes at a point in the run
            "const BLOCK_TO_MAP",       # coarse block -> drawn element
            "function stateAt",
        ):
            self.assertIn(wiring, app_js, f"app.js missing map wiring: {wiring}")
        # The hand-authored positions are gone with the static topology.
        self.assertNotIn("const MAP_POS", app_js)

    def test_the_layout_is_chosen_against_the_panel(self):
        """The arrangement is searched for, not fixed.

        A drawing whose aspect does not match the panel's is scaled to whichever
        side binds and the rest of the panel is thrown away, which is what made
        the labels unreadable. So candidate packings are scored on the zoom they
        would actually achieve in this container.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("function packLayout", "function modelParts",
                       "function viewportBox", "function refitMap", "layoutKey"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")
        # Scored against the real container, not a guessed target shape.
        chooser = app_js.split("function computeLayout")[1].split("\n}")[0]
        self.assertIn("viewportBox", chooser)
        self.assertIn("packLayout", chooser)
        # The two constants that used to guess for it are gone.
        for dead in ("bandTargetW", "varCols: 4"):
            self.assertNotIn(dead, app_js, f"{dead} should be gone")
        # A resize can change the winner, so it re-packs rather than only re-fitting.
        self.assertIn("new ResizeObserver(refitMap)", app_js)
        self.assertIn("lastRenderT", app_js)  # and repaints the frame it was on

    def test_the_agent_color_legend_is_gone(self):
        """Obsolete once the map carries agent color on the elements themselves."""
        for name in ("app.js", "index.html", "styles.css"):
            self.assertNotIn("legend", (WEB_DIR / name).read_text(),
                             f"{name} still references the legend")

    def test_the_case_is_one_box_that_names_its_own_phase(self):
        """The case is a plain box, not a container of Plan/Update slabs.

        Everything that happens *to* the case — initializing, planning against
        it, updating it, finalizing it — is the same box, told apart by a label
        that is big enough to read because nothing else is in there. The label
        also takes the colour of whatever is flowing, so it agrees with the lit
        lines rather than merely sitting between them.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("const CASE_ID", "CASE_REST", "CASE_PHASE_NODES",
                       "function casePhase", "function passStartBefore",
                       "function anyGroupActive"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")
        # Four blocks land on the one box...
        table = app_js.split("const BLOCK_TO_MAP = {")[1].split("};")[0]
        for block in ("initialize_case", "eligible_groups_gate", "update_case",
                      "finalize_case"):
            self.assertIn(f"{block}: CASE_ID", table, f"{block} should map to the case")
        # ...so the poles, the merged hub and the separate Initialize/Finalize
        # slabs are all gone, along with the edge that had to loop back.
        for dead in ("stage:hub", "stage:update", "stage:initialize",
                     "stage:finalize", "pole:", '"loop"'):
            self.assertNotIn(dead, app_js, f"{dead} should be gone")
        # Each group has exactly two lines: dispatched from the case, reported
        # back to it. The gate's own variables carry their state in place.
        self.assertIn('link(CASE_ID, gate, "fan gate-in"', app_js)
        self.assertIn('link(gate, CASE_ID, "fan to-extractor grp-out"', app_js)
        self.assertIn("tone-retr", app_js)   # the label agrees with the lines
        self.assertIn("tone-extr", app_js)

    def test_the_front_of_the_pipeline_is_one_container(self):
        """Notes and corpus characterization are a single box of discs.

        The notes were a container and characterization a slab beside it, joined
        by an arrow that only ever said "and then" — and the notes are what the
        characterization is *made of*, so they live inside it. Nothing is wired
        per note; the discs fill in place.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        self.assertIn("const CORPUS_ID", app_js)
        self.assertIn('add({ id: CORPUS_ID, label: "Scan & characterize notes"', app_js)
        self.assertIn("parent: CORPUS_ID", app_js)     # the discs live inside it
        self.assertIn('link(CORPUS_ID, CASE_ID, "spine")', app_js)
        # Both blocks resolve to the one box, so it lights across the pair.
        table = app_js.split("const BLOCK_TO_MAP = {")[1].split("};")[0]
        self.assertIn("characterize_corpus: CORPUS_ID", table)
        self.assertIn("scanner_agent_block: CORPUS_ID", table)
        # The separate notes box, the arrow between them, and every per-note and
        # per-variable line are gone — as is the slab, which nothing is now.
        for dead in ("NOTES_ID", "scan-out", "note-in", "note-out",
                     "var-in", "var-out", "node.slab"):
            self.assertNotIn(dead, app_js, f"{dead} should be gone")

    def test_map_edges_belong_to_the_phase_that_owns_them(self):
        """Lines are scoped to their step, not accumulated across the run.

        Wiring that persisted past its moment turned the map into a static
        diagram of the whole run drawn over the part of it that was moving. The
        planner's verdicts show during the check, the retriever's dispatch during
        the pass, the extractor's results during the merge, nothing after.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        state = app_js.split("function edgeState")[1].split("\n}")[0]
        for phase in ("ctx.deciding", "ctx.dispatching", "ctx.returning"):
            self.assertIn(phase, state, f"edgeState missing {phase}")
        # The case-to-gate edge is the verdict during a check and the dispatch
        # during a pass, so its colour is per-frame, not fixed when it is built.
        # The wire classes are named for the verdict, so `wire-${verdict}` in
        # edgeState resolves without a lookup table to drift out of sync.
        styles = app_js.split("function stateStyles")[1].split("\n}")[0]
        self.assertIn("wire-${verdict}", state)
        for verdict in ("pending", "open", "shut", "skipped", "ungated", "dispatch"):
            self.assertIn(f"edge.wire-{verdict}", styles, f"no style for wire-{verdict}")
        # The old dim-but-present treatment for a ruled-out group's wire is gone
        # — a failed check now says so with a pink line while the check is on.
        self.assertNotIn("edge.blocked", app_js)

    def test_a_group_with_no_gate_gets_no_verdict(self):
        """An ungated group never passed a check, so it does not show a ✓.

        Four of demo2's ten groups have no `gate:`/`site:` predicate at all;
        giving them the same green tick as a group that cleared a corpus gate
        claims a decision that was never made.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        verdict = app_js.split("function gateVerdict")[1].split("\n}")[0]
        self.assertIn('return "ungated"', verdict)
        self.assertIn("planChecked", verdict)   # gated ones wait for the planner
        self.assertIn('ungated: "↓"', app_js)
        self.assertIn("gate-ungated", app_js)
        # The verdict lands when the planner reaches it, not when corpus
        # characterization first made it computable.
        index = app_js.split("function buildMapIndex")[1].split("\n}")[0]
        self.assertIn('ev.map_node_id === "plan_extraction"', index)

    def test_the_extraction_plan_is_grouped_by_verdict(self):
        """A plan that rules three groups out must not read as ten passing.

        Panel 2's chips took their ✓/✗ from the *retriever's* stage, so at the
        planning step — before the retriever has run — every group wore a green
        tick and the plan contradicted the map directly above it. Both now read
        the same annotation, and each verdict keeps the colour it has on the map.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        css = (WEB_DIR / "styles.css").read_text()
        self.assertIn("function annotationVerdict", app_js)
        # Panel 1's gate disc and Panel 2's chip share it, so they cannot drift.
        self.assertEqual(app_js.count("annotationVerdict("), 3)  # def + gate + plan
        self.assertIn("const PLAN_ROWS", app_js)
        for verdict in ("ungated", "open", "shut"):
            self.assertIn(f".gate-chip.{verdict}", css)
            self.assertIn(f".plan-row.{verdict}", css)
        # The old flat treatment keyed off the retriever's stage is gone.
        self.assertNotIn(".gate-chip.eligible", css)
        # Pink, not brick: the two files have to agree on the failure colour.
        self.assertIn("--err: #d02670", css)
        self.assertIn('err: "#d02670"', app_js)

    def test_the_check_has_nothing_to_add_to_a_step_it_was_merged_into(self):
        """`check_state` is merged twice over, and is a repeat either way."""
        app_js = (WEB_DIR / "app.js").read_text()
        self.assertIn('check_state: ["plan_extraction", "finalize_case"]', app_js)

    def test_the_variables_panel_is_put_away_by_default(self):
        """The map is what is being presented; the variable table is a reference.

        Left open it took a third of the map's height for the whole talk, so it
        collapses to its own header at the bottom edge and is pulled up when
        wanted — and reopens at whatever size the splitter was last dragged to.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        css = (WEB_DIR / "styles.css").read_text()
        html = (WEB_DIR / "index.html").read_text()
        self.assertIn("function wireVarsPane", app_js)
        self.assertIn("wireVarsPane()", app_js)
        self.assertIn("const VARS_OPEN_KEY", app_js)      # remembered per presenter
        # Opened before it has ever been dragged it takes everything the map can
        # spare — a third of a screen shows a handful of forty-four rows — and
        # the clamp is shared with the drag so neither can starve the map.
        self.assertIn("function maxVarsHeight", app_js)
        self.assertIn("setVarsHeight(grid, maxVarsHeight(grid))", app_js)
        # One clamp, shared with the drag, and it discounts the grid's own
        # padding and row gap — those belong to neither row, so ignoring them
        # left the map about 40px short of MAP_MIN.
        self.assertIn("padding - gap - MAP_MIN", app_js)
        self.assertEqual(app_js.count("grid.style.setProperty(\"--vars-h\""), 1)
        self.assertIn('id="vars-toggle"', html)
        self.assertIn('aria-expanded="false"', html)      # closed on first load
        # Collapsed is a row of `auto`, not a height, so it cannot disagree with
        # what the header measures and the dragged size survives underneath it.
        self.assertIn(".grid.vars-collapsed { grid-template-rows: minmax(0, 1fr) auto; }", css)
        self.assertIn(".grid.vars-collapsed #vars { display: none; }", css)

    def test_panel_two_container_cards_collapse(self):
        """A step's shape first; one card's contents when asked for."""
        app_js = (WEB_DIR / "app.js").read_text()
        css = (WEB_DIR / "styles.css").read_text()
        self.assertIn("function cardSection", app_js)
        self.assertIn("const openCards", app_js)   # survives a re-render
        self.assertIn("function watchCards", app_js)
        # Notes and variable groups both become containers that open on demand,
        # and neither is a bare <section> any more.
        self.assertEqual(app_js.count("cardSection("), 3)  # def + notes + groups
        self.assertNotIn('<section class="node-detail instance', app_js)
        self.assertNotIn('<section class="node-detail group"', app_js)
        self.assertIn("details.node-detail > summary.node-head", css)

    def test_map_edges_are_drawn_only_when_they_have_something_to_say(self):
        """An edge with nothing to say is absent, not dim.

        A hundred hairlines showing the run's final wiring before any of it has
        happened is a grey web the lit edges have to fight through.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        state = app_js.split("function edgeState")[1].split("\n}")[0]
        self.assertIn("undrawn", state)
        # The backbone is the one edge that is always drawn, so it is also the
        # only place edgeState may still fall through to st-idle.
        self.assertLessEqual(state.count('"st-idle"'), 1)
        self.assertIn("function planChecked", app_js)  # gate lines wait for the check
        # display:none would drop the edge out of cy.fit()'s bounds, so the
        # viewport would lurch every time one appeared mid-animation.
        styles = app_js.split("function stateStyles")[1].split("\n}")[0]
        self.assertIn('selector: ".undrawn"', styles)
        self.assertIn("opacity: 0", styles)
        self.assertNotIn('display: "none"', styles)

    def test_block_to_map_covers_every_overview_block(self):
        """Every coarse block must resolve to something the new map draws.

        `mapping.py` still owns runtime-node -> block; this is the last hop, and
        a block missing here would silently stop lighting up.
        """
        app_js = (WEB_DIR / "app.js").read_text()
        table = app_js.split("const BLOCK_TO_MAP = {")[1].split("};")[0]
        for block in set(overview_block_map().values()):
            self.assertIn(f"{block}:", table, f"BLOCK_TO_MAP missing {block}")

    def test_app_js_animates_within_a_step(self):
        """Edge-in-on-start / edge-out-on-finish only reads if the step moves."""
        app_js = (WEB_DIR / "app.js").read_text()
        for symbol in ("function playStep", "function seekStep", "function stepSpan",
                       "function settleStep", "dashLoop", "gate-in", "grp-out"):
            self.assertIn(symbol, app_js, f"app.js missing {symbol}")


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
