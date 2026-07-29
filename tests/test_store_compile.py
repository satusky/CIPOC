"""End-to-end tests for the STORE 2024 compile path.

The tagging step is the pipeline's only LLM call, so it is faked here with a
tagger that echoes verbatim lines back out of the section it was handed. That
exercises everything deterministic around it: exact heading resolution, item
forcing, the merge of per-item applicability, provenance/fidelity validation,
the files written, and the two ways this manual can silently compile to nothing
at runtime — the 2024 manifest date versus pre-2024 diagnoses, and CoC-family
units versus SEER precedence.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from cipoc.models import CaseFacts
from cipoc.tools.coding_context import load_rule_store, scope_coding_context

from scripts.rule_compilation.compile_manual import compile_sections, write_outputs
from scripts.rule_compilation.store_index import (
    ABSENT,
    MANIFEST_ENTRY,
    MANUAL,
    SOURCE,
    TARGETS,
    build_index,
)
from scripts.rule_compilation.tag import SectionTagging, TaggedUnit

RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")


class _FakeStructuredModel:
    """Stands in for ``with_structured_output(SectionTagging)``.

    Emits units whose text is copied verbatim out of the section it was given, so
    the downstream fidelity check sees what a well-behaved model would produce.
    Also claims a wrong ``item_ids`` on purpose, to prove the compiler overrides it.
    """

    def __init__(self, recorder):
        self._recorder = recorder

    def invoke(self, messages):
        prompt = messages[-1].content
        self._recorder.append(prompt)
        body = prompt.split("Section text:\n", 1)[-1]
        excerpts = [
            line.strip()
            for line in body.splitlines()
            if len(line.strip()) > 60 and "*" not in line and "|" not in line
        ][:2]
        return SectionTagging(
            units=[
                TaggedUnit(kind="instruction", item_ids=[999], text=text)
                for text in excerpts
            ]
        )


class _FakeModel:
    def __init__(self, recorder):
        self._structured = _FakeStructuredModel(recorder)

    def with_structured_output(self, schema):
        return self._structured


class _FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []
        self.model = _FakeModel(self.prompts)


class StoreIndexTests(unittest.TestCase):
    """The pure half: target resolution and the effective dates it compiles with."""

    @classmethod
    def setUpClass(cls):
        for path in (SOURCE, DATA_DICTIONARY):
            if not path.exists():
                raise unittest.SkipTest(f"{path} not present")
        cls.index = build_index(SOURCE)

    def test_every_target_resolves_to_its_own_section_subtree(self):
        self.assertEqual(len(self.index), len(TARGETS))
        for item, sections in self.index:
            self.assertTrue(sections, f"item {item.item_id} resolved to no sections")
            self.assertEqual(sections[0].heading, item.heading)
            self.assertEqual(sections[0].level, 1)
            self.assertTrue(all(s.level > 1 for s in sections[1:]))

    def test_stems_and_item_ids_are_unique(self):
        self.assertEqual(len({t.stem for t in TARGETS}), len(TARGETS))
        self.assertEqual(len({t.item_id for t in TARGETS}), len(TARGETS))

    def test_absent_items_are_not_also_targets(self):
        self.assertFalse(set(ABSENT) & {t.item_id for t in TARGETS})

    def test_requesting_an_absent_item_fails_with_the_reason(self):
        with self.assertRaises(ValueError) as caught:
            build_index(SOURCE, items=[690])
        self.assertIn("690", str(caught.exception))
        self.assertIn("retired", str(caught.exception))

    def test_a_moved_heading_fails_rather_than_compiling_another_item(self):
        """The cross-check that makes exact-heading resolution safe to trust.

        'AJCC TNM Clin T' is a prefix of 'AJCC TNM Clin T Suffix', so a substring
        search resolves the wrong section. Pointing a target at a real heading for
        a different item must raise, not compile.
        """
        from dataclasses import replace

        from scripts.rule_compilation.segment import segment_markdown
        from scripts.rule_compilation.store_index import sections_for

        sections = segment_markdown(SOURCE.read_text(), max_heading_level=2)
        clin_t = next(t for t in TARGETS if t.item_id == 1001)
        with self.assertRaises(ValueError):
            sections_for(replace(clin_t, heading="AJCC TNM Clin T Suffix"), sections)

    def test_effective_dates_match_the_data_dictionary(self):
        """dx_date_min must be the item's implementation year, not the edition's.

        These dates are what keep a 2018-implemented item in scope for a 2019 case
        compiled from the 2024 edition, so they are hand-entered in TARGETS and
        pinned here against the dictionary rather than trusted to stay right.
        """
        dictionary = json.loads(DATA_DICTIONARY.read_text())
        for item in TARGETS:
            implemented = dictionary[str(item.item_id)]["Year Implemented"]
            if implemented is None:
                continue  # STORE marks these "All Years"; the project floor stands
            self.assertEqual(
                item.dx_date_min,
                f"{int(implemented)}-01-01",
                f"item {item.item_id} dx_date_min disagrees with the data dictionary",
            )

    def test_targets_are_items_no_compiled_manual_already_covers(self):
        """The premise of this compile: STORE is filling gaps, not competing.

        resolve_precedence prefers SEER over every other family for any kind SEER
        also covers, so a CoC-family target that a SEER manual already carries
        would compile and then be dropped at runtime.
        """
        if not RULES_DIR.exists():
            self.skipTest("compiled rule store not present")
        covered: set[int] = set()
        for path in RULES_DIR.rglob("*.json"):
            if path.name == "manifest.json" or path.suffixes != [".json"]:
                continue
            if path.parent.name == MANUAL:
                continue
            for unit in json.loads(path.read_text()):
                covered.update(unit.get("item_ids") or [])
        overlap = sorted(covered & {t.item_id for t in TARGETS})
        self.assertEqual(overlap, [], f"targets already covered elsewhere: {overlap}")


class StoreDriverPlanTests(unittest.TestCase):
    """The driver's filter/plan path, which runs before any LLM call is made."""

    @classmethod
    def setUpClass(cls):
        if not SOURCE.exists():
            raise unittest.SkipTest(f"{SOURCE} not present")

    def _plan(self, argv):
        """Run the driver in plan mode (no --run) and return its stdout."""
        import contextlib
        import io

        from scripts.rule_compilation.compile_store import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(main(argv), 0)
        return buffer.getvalue()

    def test_group_filter_selects_only_that_variable_group(self):
        output = self._plan(["--group", "tnm_staging"])
        listed = {int(line.split()[0]) for line in output.splitlines()
                  if line.startswith("  ") and line.split() and line.split()[0].isdigit()}
        self.assertEqual(
            listed, {t.item_id for t in TARGETS if t.group_id == "tnm_staging"}
        )

    def test_item_filter_selects_only_those_items(self):
        output = self._plan(["--item", "3843", "3844"])
        self.assertIn("grade_clinical", output)
        self.assertIn("grade_pathological", output)
        self.assertNotIn("ajcc_tnm_clin_t", output)

    def test_unknown_group_fails_rather_than_compiling_everything(self):
        from scripts.rule_compilation.compile_store import main

        with self.assertRaises(SystemExit):
            main(["--group", "no_such_group"])


class StoreCompileTests(unittest.TestCase):
    """The compile half, driven through the fake tagger."""

    ITEM_ID = 1001  # AJCC TNM Clin T: uncovered, 2018-implemented, CoC-only

    @classmethod
    def setUpClass(cls):
        for path in (SOURCE, RULES_DIR, DATA_DICTIONARY):
            if not path.exists():
                raise unittest.SkipTest(f"{path} not present")
        cls.item, cls.sections = build_index(SOURCE, items=[cls.ITEM_ID])[0]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rules_dir = Path(self._tmp.name) / "rules"
        shutil.copytree(RULES_DIR, self.rules_dir)
        self.addCleanup(self._tmp.cleanup)

    def _compile(self):
        llm = _FakeLLM()
        units, validations, usage = compile_sections(
            manual=MANUAL,
            site_group=self.item.stem,
            sections=self.sections,
            source_path=SOURCE,
            data_dictionary_path=DATA_DICTIONARY,
            default_applicability=self.item.applicability,
            llm=llm,
            item_ids=[self.item.item_id],
        )
        return llm, units, validations, usage

    def _write(self, units, validations, usage):
        return write_outputs(
            units, validations, usage,
            rules_dir=self.rules_dir, manual=MANUAL, site_group=self.item.stem,
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )

    def test_units_are_forced_to_the_target_item(self):
        _, units, _, _ = self._compile()
        self.assertTrue(units)
        for unit in units:
            self.assertEqual(unit.item_ids, [self.ITEM_ID])

    def test_prompt_tells_the_model_not_to_assign_items(self):
        llm, _, _, _ = self._compile()
        self.assertTrue(all(str(self.ITEM_ID) in prompt for prompt in llm.prompts))

    def test_item_applicability_is_merged_into_every_unit(self):
        _, units, _, _ = self._compile()
        for unit in units:
            self.assertEqual(unit.applies_to.dx_date_min, "2018-01-01")
            # Site-agnostic item: nothing should narrow it to a primary site.
            self.assertIsNone(unit.applies_to.sites)

    def test_verbatim_units_pass_validation(self):
        _, units, validations, _ = self._compile()
        failures = [(v.rule_id, v.errors) for v in validations if not v.ok]
        self.assertEqual(failures, [])
        self.assertEqual(len(validations), len(units))

    def test_write_outputs_promotes_units_and_seeds_the_manifest(self):
        _, units, validations, usage = self._compile()
        out_path, report_path, usage_path, accepted = self._write(units, validations, usage)
        self.assertEqual(len(accepted), len(units))
        self.assertTrue(report_path.exists() and usage_path.exists())
        self.assertEqual(len(json.loads(out_path.read_text())), len(accepted))

        entry = json.loads((self.rules_dir / "manifest.json").read_text())[MANUAL]
        self.assertEqual(entry["family"], "CoC")
        self.assertEqual(entry["publication_date"], "2024-01-01")
        self.assertEqual(entry["source_markdown"], str(SOURCE))

    def test_units_apply_to_cases_diagnosed_before_the_edition_was_published(self):
        """The regression this manual's per-item dx_date_min exists to prevent.

        The 2024 manifest date would exclude every unit for a 2019 case under the
        manual-level temporal filter; each unit's own dx_date_min is what keeps
        the 2018-implemented AJCC items in scope.
        """
        _, units, validations, usage = self._compile()
        self._write(units, validations, usage)
        store = load_rule_store(self.rules_dir, use_cache=False)

        in_scope = scope_coding_context(
            [self.ITEM_ID], CaseFacts(date_of_diagnosis="2019-03-04"), store
        )[self.ITEM_ID]
        self.assertTrue([u for u in in_scope.units if u.source_doc == MANUAL])

    def test_units_do_not_apply_before_the_item_was_implemented(self):
        _, units, validations, usage = self._compile()
        self._write(units, validations, usage)
        store = load_rule_store(self.rules_dir, use_cache=False)

        pre_2018 = scope_coding_context(
            [self.ITEM_ID], CaseFacts(date_of_diagnosis="2016-03-04"), store
        )[self.ITEM_ID]
        self.assertFalse([u for u in pre_2018.units if u.source_doc == MANUAL])

    def test_coc_units_survive_scoping_alongside_the_seer_manuals(self):
        """CoC family must not be wiped by SEER precedence for a gap item.

        resolve_precedence drops every non-SEER unit of any kind SEER also
        covers. That is correct where the two overlap, and harmless here only
        because no SEER manual carries item 1001 — which is the whole reason this
        item is being compiled. Pinned so the premise fails loudly if it changes.
        """
        _, units, validations, usage = self._compile()
        self._write(units, validations, usage)
        store = load_rule_store(self.rules_dir, use_cache=False)

        context = scope_coding_context(
            [self.ITEM_ID], CaseFacts(date_of_diagnosis="2021-06-01"), store
        )[self.ITEM_ID]
        scoped = {u.rule_id for u in context.units}
        self.assertEqual({u.rule_id for u in units}, scoped)

    def test_compiled_units_are_site_agnostic(self):
        """AJCC T is coded for every primary; scoping must not filter it by site."""
        _, units, validations, usage = self._compile()
        self._write(units, validations, usage)
        store = load_rule_store(self.rules_dir, use_cache=False)

        for facts in (
            CaseFacts(primary_site="C509", date_of_diagnosis="2021-06-01"),
            CaseFacts(primary_site="C349", date_of_diagnosis="2021-06-01"),
        ):
            context = scope_coding_context([self.ITEM_ID], facts, store)[self.ITEM_ID]
            self.assertTrue(
                [u for u in context.units if u.source_doc == MANUAL],
                f"units vanished for {facts.primary_site}",
            )


if __name__ == "__main__":
    unittest.main()
