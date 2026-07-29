"""Pins the opt-in async execution mode.

Async mode is a build-time decision: ``use_async`` picks which callable each
LLM-backed node registers, and everything downstream (``arun``, the orchestrator's
subagents, the mode guard) follows from that one choice. These tests cover the
choice itself, the parity of the two graphs' results, that the async path really
overlaps I/O, and that the config flag never leaks into the model client.

Stdlib ``unittest`` only — there is no pytest-asyncio here and none should be
added (airgapped DBR target), so async cases run under ``asyncio.run``.
"""

import asyncio
import time
import unittest

from langgraph.graph.state import CompiledStateGraph

from cipoc.agents import (
    ExtractorAgent,
    NoteRetrieverAgent,
    NoteScannerAgent,
    OrchestratorAgent,
)
from cipoc.agents.note_retriever import RetrieverInput
from cipoc.agents.orchestrator import CaseState
from cipoc.llm import BaseAgentModel, LLMConfig
from cipoc.models import (
    Case,
    CaseVariableResult,
    NoteDigest,
    VariableGroupInfo,
    VariableInfo,
    VariableStatus,
)
from cipoc.utils.utils import CipocConfig

from tests.fake_orchestrator import (
    Outcome,
    Script,
    build_fake_orchestrator,
    graph_input,
    load_notes,
)


# Enough for llm_config() to build an OpenAIConfig; no client is ever created
# because every agent here is constructed with an injected llm=.
BASE_LLM = {
    "model": "stub-model",
    "api_key": "stub-key",
    "base_url": "http://endpoint.invalid",
    "provider": "openai",
}

# The documented twin set: exactly the nodes that issue an LLM request, directly
# or through a child graph. Anything else must be the same callable in both modes.
EXPECTED_TWINS = {
    NoteScannerAgent: {"detect_concepts", "summarize_note", "get_cancer_mentions"},
    NoteRetrieverAgent: {"identify_relevant_notes"},
    ExtractorAgent: {
        "extract_group_values",
        "extract_individual_value",
        "repair_invalid_extraction",
    },
    OrchestratorAgent: {"note_branch", "retrieve_notes", "extract"},
}


class FakeLLM:
    """A model wrapper that is never called; agents here are only built, not run."""

    def structured(self, schema, messages, **kwargs):
        raise AssertionError("structured() called in a build-only test")

    async def astructured(self, schema, messages, **kwargs):
        raise AssertionError("astructured() called in a build-only test")


def _node_steps(graph: CompiledStateGraph) -> dict[str, object]:
    """Map node name to the ``RunnableCallable`` LangGraph compiled for it,
    descending into subgraph nodes so the extractor's variable branch and the
    orchestrator's extract branch are covered too.

    Read ``.func`` / ``.afunc`` off the result: LangGraph sets ``func`` for a node
    registered with a plain callable (and synthesizes an ``afunc`` that runs it in
    an executor), and leaves ``func`` None for one registered with a coroutine
    function. ``func is None`` is therefore the signal that a twin was bound.
    """
    steps: dict[str, object] = {}
    for name, node in graph.nodes.items():
        if name.startswith("__"):
            continue
        step = node.node.steps[0]
        if isinstance(step, CompiledStateGraph):
            steps.update(_node_steps(step))
            continue
        steps[name] = step
    return steps


def _underlying(fn):
    """The plain function behind a bound method, for cross-instance comparison."""
    return getattr(fn, "__func__", fn)


def _build(cls, **kwargs):
    return cls(llm=FakeLLM(), **kwargs)


class NodeBindingTests(unittest.TestCase):
    """``use_async`` swaps in the ``a``-prefixed twins and nothing else."""

    def test_async_mode_binds_the_twins_and_leaves_pure_nodes_alone(self):
        for cls, expected in EXPECTED_TWINS.items():
            with self.subTest(cls.__name__):
                sync_steps = _node_steps(_build(cls, use_async=False)._graph)
                async_steps = _node_steps(_build(cls, use_async=True)._graph)
                self.assertEqual(set(sync_steps), set(async_steps))

                swapped = set()
                for name, sync_step in sync_steps.items():
                    async_step = async_steps[name]
                    self.assertIsNotNone(sync_step.func, name)
                    if async_step.func is not None:
                        # Deterministic node: literally the same function in both
                        # graphs, which the async runner hands to an executor.
                        self.assertIs(
                            _underlying(async_step.func), _underlying(sync_step.func), name
                        )
                        continue
                    swapped.add(name)
                    self.assertEqual(async_step.afunc.__name__, f"a{name}")
                    self.assertTrue(
                        asyncio.iscoroutinefunction(async_step.afunc), name
                    )

                self.assertEqual(swapped, expected)

    def test_sync_mode_is_the_default(self):
        for cls in EXPECTED_TWINS:
            with self.subTest(cls.__name__):
                self.assertFalse(_build(cls)._async)

    def test_the_orchestrator_propagates_its_mode_to_its_subagents(self):
        """Its nodes await the subagents' arun, so a subagent left in sync mode
        would trip the mode guard mid-run."""
        for mode in (False, True):
            with self.subTest(use_async=mode):
                agent = _build(OrchestratorAgent, use_async=mode)
                for sub in (agent._scanner, agent._retriever, agent._extractor):
                    self.assertIs(sub._async, mode)


class ModeGuardTests(unittest.TestCase):
    """Calling the wrong verb fails here, not deep inside LangGraph."""

    REQUEST = RetrieverInput(
        requested_variables=VariableGroupInfo(
            name="test", variables=[VariableInfo(item_id=400, name="Primary Site")]
        ),
        available_digests={1: NoteDigest(note_id=1, type="pathology", summary="s")},
    )

    def test_run_on_an_async_agent_raises(self):
        agent = _build(NoteRetrieverAgent, use_async=True)
        with self.assertRaises(RuntimeError) as caught:
            agent.run(self.REQUEST, progress=False)
        self.assertIn("use_async=False", str(caught.exception))

    def test_arun_on_a_sync_agent_raises(self):
        agent = _build(NoteRetrieverAgent, use_async=False)
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(agent.arun(self.REQUEST))
        self.assertIn("use_async=True", str(caught.exception))

    def test_astream_results_on_a_sync_orchestrator_raises(self):
        agent = _build(OrchestratorAgent, use_async=False)

        async def drive():
            async for _ in agent.astream_results([], progress=False):
                pass

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(drive())
        self.assertIn("use_async=True", str(caught.exception))


class ConfigTests(unittest.TestCase):
    def _config(self, **extra) -> CipocConfig:
        return CipocConfig({"llm": {**BASE_LLM, **extra}})

    def test_async_mode_is_not_forwarded_to_the_model_client(self):
        """OpenAIConfig allows extra fields, so a stray `async_mode` key would
        reach ChatOpenAI as a model kwarg — the same hole the `retry` pop closes."""
        config = self._config(async_mode=True)
        self.assertNotIn("async_mode", config.llm_config().model_dump())
        self.assertNotIn("async_mode", config.llm_config("extractor").model_dump())

    def test_the_config_flag_selects_the_mode(self):
        agent = _build(NoteRetrieverAgent, config=self._config(async_mode=True))
        self.assertTrue(agent._async)

    def test_a_per_agent_override_selects_the_mode(self):
        config = CipocConfig(
            {"llm": BASE_LLM, "agents": {"note_retriever": {"async_mode": True}}}
        )
        self.assertFalse(_build(NoteScannerAgent, config=config)._async)
        self.assertTrue(_build(NoteRetrieverAgent, config=config)._async)

    def test_an_explicit_kwarg_beats_the_config(self):
        config = self._config(async_mode=True)
        self.assertFalse(_build(NoteRetrieverAgent, config=config, use_async=False)._async)


class ParityTests(unittest.TestCase):
    """The two graphs differ in call verb only, so they must agree on results."""

    @staticmethod
    def _case(use_async: bool, script: Script | None = None):
        agent = build_fake_orchestrator(script, use_async=use_async)
        notes = load_notes()
        if use_async:
            return asyncio.run(
                agent.arun([note.model_dump() for note in notes], progress=False)
            )
        # invoke rather than run(): identical graph, without painting a progress
        # dashboard over the test output.
        return CaseState(**agent._graph.invoke(graph_input(notes))).to_case()

    def test_the_async_run_produces_the_same_case(self):
        self.assertEqual(
            self._case(False).model_dump(), self._case(True).model_dump()
        )

    def test_parity_holds_through_the_repair_loop_and_a_missing_value(self):
        outcomes = {
            400: Outcome(repairs=1),      # one validation failure, then accepted
            410: Outcome(value=None),     # coded as not found
            522: Outcome(exhausted=True),  # repair budget spent, still invalid
        }
        self.assertEqual(
            self._case(False, Script(outcomes=outcomes)).model_dump(),
            self._case(True, Script(outcomes=outcomes)).model_dump(),
        )


class IncrementalDeliveryTests(unittest.TestCase):
    """``astream_results`` hands back a group's results when its branch lands,
    not when the whole case finishes."""

    @staticmethod
    def _drain(agent, structured_data=None, progress_marker=None):
        """Every streamed result, the ``Case``, and — when ``progress_marker`` is a
        ``Script`` — how much simulated work had been done as each result landed."""
        raw = [note.model_dump() for note in load_notes()]

        async def drive():
            results: list[CaseVariableResult] = []
            marks: list[int] = []
            case: Case | None = None
            async for item in agent.astream_results(
                raw, structured_data, progress=False
            ):
                if isinstance(item, Case):
                    case = item
                    continue
                results.append(item)
                marks.append(progress_marker.paused if progress_marker else 0)
            return results, marks, case

        return asyncio.run(drive())

    def _assert_union_is_the_case(self, results, case):
        item_ids = [result.item_id for result in results]
        self.assertEqual(len(item_ids), len(set(item_ids)), "an item was yielded twice")
        # Nothing streamed that the Case does not carry, nothing withheld until
        # the end. This is the property that makes the stream a real substitute
        # for waiting on `arun`.
        self.assertEqual(
            {result.item_id: result.model_dump() for result in results},
            {
                item_id: result.model_dump()
                for item_id, result in case.variable_results.items()
            },
        )

    def test_every_result_is_yielded_exactly_once_and_the_case_comes_last(self):
        results, _, case = self._drain(build_fake_orchestrator(use_async=True))
        self.assertIsNotNone(case)
        self._assert_union_is_the_case(results, case)

    def test_results_arrive_while_the_run_still_has_work_to_do(self):
        """The incremental claim itself: the first result reaches the caller with
        simulated endpoint calls still outstanding, so it waited neither for the
        fan-in barrier nor for the dependent wave."""
        script = Script()
        agent = build_fake_orchestrator(script, use_async=True)

        results, marks, case = self._drain(agent, progress_marker=script)

        self.assertTrue(results)
        self.assertLess(marks[0], script.paused)
        self.assertTrue(case.variable_results)

    def test_structured_data_seeds_are_streamed_too(self):
        """They go terminal in ``initialize`` and are never extracted, so they
        reach the caller only because every node's ``variable_results`` write is
        streamed — not just ``extract``'s. That is also what keeps the union above
        equal to the final Case."""
        agent = build_fake_orchestrator(use_async=True)

        # 390 is Date of Diagnosis, an initial-stage variable.
        results, _, case = self._drain(agent, {390: "20250224"})

        seeded = {result.item_id: result for result in results}[390]
        self.assertEqual(seeded.status, VariableStatus.STRUCTURED_DATA)
        self.assertEqual(seeded.value, "20250224")
        self._assert_union_is_the_case(results, case)


class ConcurrencyTests(unittest.TestCase):
    """Two halves: the fan-out really overlaps, and the endpoint cap really binds."""

    def test_the_async_fan_out_overlaps(self):
        script = Script(delay=0.02)
        agent = build_fake_orchestrator(script, use_async=True)
        raw = [note.model_dump() for note in load_notes()]

        started = time.monotonic()
        case = asyncio.run(agent.arun(raw, progress=False))
        elapsed = time.monotonic() - started

        self.assertTrue(case.variable_results)
        # Direct evidence rather than a timing heuristic: simulated endpoint calls
        # were genuinely in flight together.
        self.assertGreater(script.peak_in_flight, 1)
        # And the wall clock agrees — `paused * delay` is what the same run would
        # have cost with no overlap at all.
        self.assertLess(elapsed, script.paused * script.delay)

    def test_the_sync_and_async_runs_do_the_same_amount_of_work(self):
        """Overlap must come from concurrency, not from skipping calls."""
        sync_script, async_script = Script(), Script()
        build_fake_orchestrator(sync_script)._graph.invoke(graph_input(load_notes()))
        asyncio.run(
            build_fake_orchestrator(async_script, use_async=True).arun(
                [note.model_dump() for note in load_notes()], progress=False
            )
        )
        self.assertEqual(sync_script.paused, async_script.paused)
        self.assertEqual(async_script.in_flight, 0)

    def test_max_concurrency_bounds_the_async_path(self):
        """The layer-1 guard: one wrapper, one semaphore, N awaited calls."""

        class _Tracker:
            def __init__(self):
                self.in_flight = 0
                self.peak = 0

        class _Runnable:
            def __init__(self, tracker, delay):
                self._tracker = tracker
                self._delay = delay

            async def ainvoke(self, messages, **kwargs):
                self._tracker.in_flight += 1
                self._tracker.peak = max(self._tracker.peak, self._tracker.in_flight)
                await asyncio.sleep(self._delay)
                self._tracker.in_flight -= 1
                return "ok"

        class _Chat:
            def __init__(self, runnable):
                self._runnable = runnable

            def with_structured_output(self, schema):
                return self._runnable

        class _StubModel(BaseAgentModel):
            def __init__(self, config, runnable):
                self._runnable = runnable
                super().__init__(config)

            def _initialize_model(self, **kwargs):
                return _Chat(self._runnable)

        for cap, expected_peak in ((2, 2), (None, 6)):
            with self.subTest(max_concurrency=cap):
                tracker = _Tracker()
                model = _StubModel(
                    LLMConfig(**{**BASE_LLM, "max_concurrency": cap}),
                    _Runnable(tracker, 0.01),
                )

                async def drive():
                    await asyncio.gather(
                        *(model.astructured(str, "hi") for _ in range(6))
                    )

                asyncio.run(drive())
                self.assertEqual(tracker.peak, expected_peak)
                self.assertEqual(tracker.in_flight, 0)


if __name__ == "__main__":
    unittest.main()
