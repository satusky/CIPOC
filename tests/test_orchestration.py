import unittest
from pathlib import Path

from cipoc.models import (
    CaseFacts,
    ConfidenceLevel,
    CorpusGate,
    NoteCorpusDescriptors,
    TargetGroup,
    ValidatedVariableOutput,
    VariableInfo,
)
from cipoc.models.case import CaseVariableResult, VariableStatus
from cipoc.tools import (
    eligible_groups,
    load_variable_groups,
    resolve_leftovers,
    site_applies,
    unmet_dependencies,
    validate_dependencies,
)
from cipoc.tools.coding_context import load_rule_store, scope_coding_context
from cipoc.tools.orchestration import stage_is_ready
from tests.fake_orchestrator import Outcome, Script, build_fake_orchestrator, graph_input


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_GROUPS = REPO_ROOT / "config" / "variable_groups.json"
RULES_DIR = REPO_ROOT / "documents" / "rules"

EMPTY_CORPUS = NoteCorpusDescriptors(unique_flags=set())


def group(group_id, item_ids, **kwargs) -> TargetGroup:
    return TargetGroup(
        group_id=group_id,
        name=group_id,
        variables=[VariableInfo(item_id=item_id) for item_id in item_ids],
        **kwargs,
    )


def extraction(item_id: int, value: str | None) -> ValidatedVariableOutput:
    return ValidatedVariableOutput(
        item_id=item_id,
        value=value,
        explanation="fixture",
        most_important_note=None,
        spans=[],
        presence_confidence=ConfidenceLevel.HIGH,
        is_valid=True,
    )


def _fields(item_id: int, status: VariableStatus) -> dict:
    """The minimum ``CaseVariableResult`` demands for a status, per its validators."""
    match status:
        case VariableStatus.EXTRACTED:
            return {"value": "1", "extraction": extraction(item_id, "1")}
        case VariableStatus.STRUCTURED_DATA:
            return {"value": "1"}
        case VariableStatus.NOT_FOUND:
            return {"extraction": extraction(item_id, None)}
        case VariableStatus.ERROR | VariableStatus.BLOCKED | VariableStatus.NOT_APPLICABLE:
            return {"reason": "fixture"}
        case _:
            return {}


def results(**by_status) -> dict[int, CaseVariableResult]:
    """Build a results dict from ``status_name=[item_id, ...]`` keywords."""
    return {
        item_id: CaseVariableResult(
            item_id=item_id,
            status=VariableStatus(status),
            **_fields(item_id, VariableStatus(status)),
        )
        for status, item_ids in by_status.items()
        for item_id in item_ids
    }


class SiteApplicabilityTests(unittest.TestCase):
    def test_item_832_accepts_coded_breast_primary_site(self):
        target = next(
            target
            for target in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in target.variables)
        )

        self.assertTrue(site_applies(target.applies_to, CaseFacts(primary_site="C50.4")))
        self.assertFalse(site_applies(target.applies_to, CaseFacts(primary_site="C34.9")))


class DependencyOrderingTests(unittest.TestCase):
    """``depends_on`` sequences groups within a stage."""

    def setUp(self):
        self.histology = group("histology", [522], stage="dependent")
        self.grade = group("grade", [3844], stage="dependent", depends_on=["histology"])
        self.groups = [self.grade, self.histology]

    def test_waits_while_the_dependency_is_pending(self):
        state = results(pending=[522, 3844])

        self.assertFalse(stage_is_ready(self.grade, self.groups, state))
        self.assertTrue(stage_is_ready(self.histology, self.groups, state))
        self.assertEqual(
            [other.group_id for other in unmet_dependencies(self.grade, self.groups, state)],
            ["histology"],
        )

    def test_releases_once_the_dependency_is_terminal(self):
        state = results(extracted=[522], pending=[3844])

        self.assertTrue(stage_is_ready(self.grade, self.groups, state))
        self.assertEqual(unmet_dependencies(self.grade, self.groups, state), [])

    def test_a_dependency_that_finds_nothing_still_releases_its_dependents(self):
        """The gate is terminal, not coded: blocking on a fact the notes do not
        carry would be worse than the widened scope this ordering replaces."""
        for status in ("not_found", "not_applicable", "blocked", "error"):
            with self.subTest(status=status):
                state = results(**{status: [522]}, pending=[3844])
                self.assertTrue(stage_is_ready(self.grade, self.groups, state))

    def test_only_the_dependent_is_held_back_in_the_first_wave(self):
        ready = eligible_groups(self.groups, results(pending=[522, 3844]), EMPTY_CORPUS, None)

        self.assertEqual([target.group_id for target in ready], ["histology"])


class LeftoverResolutionTests(unittest.TestCase):
    def test_blocked_cites_the_unmet_dependency_not_the_initial_stage(self):
        groups = [
            group("initial", [400], stage="initial"),
            group("histology", [522], stage="dependent"),
            group("grade", [3844], stage="dependent", depends_on=["histology"]),
        ]

        updates = resolve_leftovers(groups, results(pending=[400, 522, 3844]), EMPTY_CORPUS, None)

        self.assertEqual(updates[3844].status, VariableStatus.BLOCKED)
        self.assertEqual(updates[3844].blocking_item_ids, [522])
        self.assertIn("histology", updates[3844].reason)
        # The stage barrier is still the reason for a dependent with no edges.
        self.assertEqual(updates[522].blocking_item_ids, [400])

    def test_exclusions_resolve_first_so_they_can_release_a_dependent(self):
        """A gate-excluded dependency must not stamp its dependent BLOCKED in the
        same pass — excluding it is exactly what makes the dependent runnable."""
        groups = [
            group("gated", [522], stage="dependent", gate=[CorpusGate.METASTASIS_PRESENT]),
            group("grade", [3844], stage="dependent", depends_on=["gated"]),
        ]
        state = results(pending=[522, 3844])

        first = resolve_leftovers(groups, state, EMPTY_CORPUS, None)
        self.assertEqual(set(first), {522})
        self.assertEqual(first[522].status, VariableStatus.NOT_APPLICABLE)

        # With the gated group terminal, the dependent is eligible rather than blocked.
        state.update(first)
        ready = eligible_groups(groups, state, EMPTY_CORPUS, None)
        self.assertEqual([target.group_id for target in ready], ["grade"])

    def test_blocked_is_stamped_once_a_pass_excludes_nothing_new(self):
        groups = [
            group("initial", [400], stage="initial"),
            group("grade", [3844], stage="dependent"),
        ]

        updates = resolve_leftovers(groups, results(pending=[400, 3844]), EMPTY_CORPUS, None)

        self.assertEqual(updates[3844].status, VariableStatus.BLOCKED)
        self.assertEqual(updates[400].status, VariableStatus.BLOCKED)


class DependencyValidationTests(unittest.TestCase):
    def test_unknown_dependency_raises(self):
        groups = [group("grade", [3844], depends_on=["typo"])]

        with self.assertRaisesRegex(ValueError, "unknown group"):
            validate_dependencies(groups)

    def test_cycle_raises_and_names_the_cycle(self):
        groups = [
            group("a", [1], depends_on=["b"]),
            group("b", [2], depends_on=["a"]),
        ]

        with self.assertRaisesRegex(ValueError, "Cyclic depends_on"):
            validate_dependencies(groups)

    def test_self_dependency_raises(self):
        with self.assertRaisesRegex(ValueError, "Cyclic depends_on"):
            validate_dependencies([group("a", [1], depends_on=["a"])])

    def test_diamond_is_acyclic(self):
        validate_dependencies([
            group("a", [1]),
            group("b", [2], depends_on=["a"]),
            group("c", [3], depends_on=["a"]),
            group("d", [4], depends_on=["b", "c"]),
        ])


class ConfigDependencyTests(unittest.TestCase):
    def setUp(self):
        self.groups = {
            target.group_id: target for target in load_variable_groups(VARIABLE_GROUPS)
        }

    def test_histology_scoped_groups_wait_on_histology(self):
        for group_id in ("site_specific_codes", "other"):
            with self.subTest(group_id=group_id):
                self.assertEqual(
                    self.groups[group_id].depends_on, ["histologic_type_and_behavior"]
                )

    def test_subgroups_do_not_inherit_depends_on(self):
        """``histologic_type_and_behavior`` is a *subgroup* of ``site_specific_codes``
        in the config and flattens into its peer, so inheriting the parent's
        ``depends_on`` would hand it a self-dependency."""
        self.assertIsNone(self.groups["histologic_type_and_behavior"].depends_on)

    def test_histology_is_not_itself_held_back(self):
        state = results(
            extracted=[390, 400, 410],
            pending=[522, 523, 670, 671, 3843, 3844],
        )
        target = self.groups["histologic_type_and_behavior"]

        self.assertTrue(stage_is_ready(target, list(self.groups.values()), state))
        self.assertFalse(
            stage_is_ready(self.groups["site_specific_codes"], list(self.groups.values()), state)
        )


class WaveOrderingTests(unittest.TestCase):
    """End-to-end through the deterministic fake graph: no LLM, real topology."""

    def scoped(self, script: Script | None = None) -> list[tuple[str, CaseFacts | None]]:
        """Run a fake case, recording ``(group_id, case_facts)`` per scoped group.

        ``build_fake_orchestrator`` stubs ``_scope_group`` to a pass-through so the
        fixture never reads the 2 MB data dictionary; this re-stubs it to record
        instead, which is where the contract lives — the facts a group is scoped
        against are exactly what ``build_variable_group`` would reduce codes with.
        """
        agent = build_fake_orchestrator(script)
        calls: list[tuple[str, CaseFacts | None]] = []

        def record(group, case_facts):
            calls.append((group.group_id, case_facts))
            return group

        agent._scope_group = record
        agent._graph.invoke(graph_input())
        return calls

    def test_dependents_are_scoped_after_histology_is_known(self):
        script = Script(outcomes={
            390: Outcome(value="2022-01-01"),
            400: Outcome(value="C509"),
            522: Outcome(value="8500"),
            523: Outcome(value="3"),
        })
        calls = self.scoped(script)
        by_group = {group_id: facts for group_id, facts in calls}

        self.assertIsNone(by_group["histologic_type_and_behavior"].histology)
        for group_id in ("site_specific_codes", "other"):
            with self.subTest(group_id=group_id):
                self.assertEqual(by_group[group_id].histology, "8500")
                self.assertEqual(by_group[group_id].primary_site, "C509")

    def test_histology_is_scoped_in_an_earlier_wave_than_its_dependents(self):
        order = [group_id for group_id, _ in self.scoped()]

        for group_id in ("site_specific_codes", "other"):
            with self.subTest(group_id=group_id):
                self.assertLess(
                    order.index("histologic_type_and_behavior"), order.index(group_id)
                )

    def test_undependent_groups_still_share_one_wave(self):
        """The edge sequences two groups, not the whole dependent stage."""
        order = [group_id for group_id, _ in self.scoped()]
        late = {"site_specific_codes", "other"}

        self.assertTrue(all(group_id in late for group_id in order[-len(late):]))
        self.assertIn("tnm_staging", order[: -len(late)])


class GradeScopingTests(unittest.TestCase):
    """Why the ordering exists: histology collapses the applicable code tables.

    Grade Pathological (3844) has no code descriptions in the data dictionary, so
    its valid codes come entirely from SEER*RSA ``code_table`` units. Site alone
    matches every schema whose topography range covers it — ~20 tables whose codes
    ``1``/``2``/``3`` mean different things (Nottingham/SBR vs FNCLCC vs Gleason).
    Adding histology selects exactly one schema.
    """

    @classmethod
    def setUpClass(cls):
        cls.store = load_rule_store(RULES_DIR)

    def code_tables(self, item_id: int, **facts) -> int:
        context = scope_coding_context(
            [item_id], CaseFacts(date_of_diagnosis="2022-01-01", **facts), self.store
        )[item_id]
        return sum(1 for unit in context.units if unit.kind == "code_table")

    def test_histology_narrows_grade_to_a_single_schema(self):
        for site, histology in [
            ("C509", "8500"),  # breast ductal
            ("C349", "8140"),  # lung adenocarcinoma
            ("C619", "8140"),  # prostate adenocarcinoma
            ("C189", "8140"),  # colon adenocarcinoma
        ]:
            with self.subTest(site=site, histology=histology):
                site_only = self.code_tables(3844, primary_site=site)
                with_histology = self.code_tables(
                    3844, primary_site=site, histology=histology
                )
                self.assertGreater(site_only, 1)
                self.assertEqual(with_histology, 1)


if __name__ == "__main__":
    unittest.main()
