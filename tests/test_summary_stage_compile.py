"""End-to-end tests for the Summary Stage 2018 compile path.

The tagging step is the pipeline's only LLM call, so it is faked here with a
tagger that echoes verbatim lines back out of the section it was handed. That is
enough to exercise everything deterministic around it: item forcing, the merge of
chapter applicability into each unit, provenance/fidelity validation, the files
written, and — the failure this manual is most exposed to — whether the compiled
units actually survive runtime scoping alongside the SPCSM units for item 764.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from cipoc.models import CaseFacts
from cipoc.tools.coding_context import load_rule_store, scope_coding_context

from scripts.rule_compilation.compile_manual import compile_sections, write_outputs
from scripts.rule_compilation.segment import segment_markdown
from scripts.rule_compilation.summary_stage_index import (
    MANIFEST_ENTRY,
    SUMMARY_STAGE_ITEM_ID,
    build_index,
    sections_for,
)
from scripts.rule_compilation.tag import SectionTagging, TaggedUnit

SOURCE = Path("documents/markdown/Summary-Stage_v3.3.md")
RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")

MANUAL = "summary_stage_2018"


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


class SummaryStageCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (SOURCE, RULES_DIR, DATA_DICTIONARY):
            if not path.exists():
                raise unittest.SkipTest(f"{path} not present")
        chapters = build_index(SOURCE)
        cls.breast = next(c for c in chapters if c.site_group == "breast")
        cls.sections = segment_markdown(SOURCE.read_text(), max_heading_level=2)

    def setUp(self):
        """Build a store holding SPCSM plus whatever this test compiles — nothing else.

        Copying the whole committed store would load the ~90 already-compiled
        Summary Stage chapters alongside the breast units under test. Every
        scoping assertion below filters by ``source_doc == MANUAL``, so those
        chapters would answer for the breast compile: the site-agnostic `general`
        chapter scopes to any primary site, and the full store's 368 code_table
        units for item 764 displace the SPCSM code_table this test uses to
        demonstrate cross-manual precedence. SPCSM is the only other manual these
        tests reason about, so it is the only one copied in.
        """
        self._tmp = tempfile.TemporaryDirectory()
        self.rules_dir = Path(self._tmp.name) / "rules"
        self.rules_dir.mkdir(parents=True)
        shutil.copy(RULES_DIR / "manifest.json", self.rules_dir / "manifest.json")
        shutil.copytree(RULES_DIR / "spcsm_2024", self.rules_dir / "spcsm_2024")
        self.addCleanup(self._tmp.cleanup)

    def _compile_breast(self):
        llm = _FakeLLM()
        units, validations, usage = compile_sections(
            manual=MANUAL,
            site_group="breast",
            sections=sections_for(self.breast, self.sections),
            source_path=SOURCE,
            data_dictionary_path=DATA_DICTIONARY,
            default_applicability=self.breast.applicability,
            llm=llm,
            item_ids=[SUMMARY_STAGE_ITEM_ID],
        )
        return llm, units, validations, usage

    def test_units_are_forced_to_the_summary_stage_item(self):
        _, units, _, _ = self._compile_breast()
        self.assertTrue(units)
        for unit in units:
            self.assertEqual(unit.item_ids, [SUMMARY_STAGE_ITEM_ID])

    def test_chapter_applicability_is_merged_into_every_unit(self):
        _, units, _, _ = self._compile_breast()
        for unit in units:
            self.assertEqual(unit.applies_to.sites, ["C500-C506", "C508-C509"])
            self.assertEqual(unit.applies_to.dx_date_min, "2018-01-01")
            self.assertIn("8720-8790", unit.applies_to.histologies)

    def test_prompt_tells_the_model_not_to_assign_items(self):
        llm, _, _, _ = self._compile_breast()
        self.assertTrue(all("764" in prompt for prompt in llm.prompts))

    def test_verbatim_units_pass_validation(self):
        _, units, validations, _ = self._compile_breast()
        failures = [(v.rule_id, v.errors) for v in validations if not v.ok]
        self.assertEqual(failures, [])
        self.assertEqual(len(validations), len(units))

    def test_write_outputs_promotes_units_and_seeds_the_manifest(self):
        _, units, validations, usage = self._compile_breast()
        out_path, report_path, usage_path, accepted = write_outputs(
            units, validations, usage,
            rules_dir=self.rules_dir, manual=MANUAL, site_group="breast",
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )
        self.assertEqual(len(accepted), len(units))
        self.assertTrue(report_path.exists() and usage_path.exists())
        self.assertEqual(len(json.loads(out_path.read_text())), len(accepted))

        entry = json.loads((self.rules_dir / "manifest.json").read_text())[MANUAL]
        self.assertEqual(entry["publication_date"], MANIFEST_ENTRY["publication_date"])
        self.assertEqual(entry["family"], "SEER")
        self.assertEqual(entry["source_markdown"], str(SOURCE))

    def test_manifest_upsert_preserves_hand_edits(self):
        _, units, validations, usage = self._compile_breast()
        kwargs = dict(
            rules_dir=self.rules_dir, manual=MANUAL, site_group="breast",
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )
        write_outputs(units, validations, usage, **kwargs)

        manifest_path = self.rules_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[MANUAL]["title"] = "Hand edited title"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        write_outputs(units, validations, usage, **kwargs)
        reloaded = json.loads(manifest_path.read_text())[MANUAL]
        self.assertEqual(reloaded["title"], "Hand edited title")

    def test_compiled_units_survive_scoping_against_the_spcsm_units(self):
        """The regression this manual's manifest date exists to prevent.

        SPCSM 2024 already carries instruction units for item 764. Precedence
        keeps only the newest publication_date per (kind, family), so a Summary
        Stage entry dated to its 2018 effective date would lose to SPCSM and every
        unit compiled here would vanish at runtime.
        """
        _, units, validations, usage = self._compile_breast()
        write_outputs(
            units, validations, usage,
            rules_dir=self.rules_dir, manual=MANUAL, site_group="breast",
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )

        store = load_rule_store(self.rules_dir, use_cache=False)
        context = scope_coding_context(
            [SUMMARY_STAGE_ITEM_ID],
            CaseFacts(gross_primary_site="breast", date_of_diagnosis="2021-06-01"),
            store,
        )[SUMMARY_STAGE_ITEM_ID]

        scoped = {unit.rule_id for unit in context.units}
        self.assertEqual({u.rule_id for u in units}, scoped & {u.rule_id for u in units})

        # Same kind, same family, newer edition: the Summary Stage instructions
        # displace the SPCSM ones. SPCSM's code_table unit has no Summary Stage
        # counterpart in this compile and rightly stays.
        surviving_spcsm = {u.kind for u in context.units if u.source_doc == "spcsm_2024"}
        self.assertNotIn("instruction", surviving_spcsm)
        self.assertIn("code_table", surviving_spcsm)

    def test_units_apply_to_cases_diagnosed_before_the_edition_was_published(self):
        """The 2025 edition governs every case diagnosed 2018+, not just 2025+.

        The manual-level temporal filter would exclude a 2019 case; each unit's own
        dx_date_min is what keeps it in scope.
        """
        _, units, validations, usage = self._compile_breast()
        write_outputs(
            units, validations, usage,
            rules_dir=self.rules_dir, manual=MANUAL, site_group="breast",
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )
        store = load_rule_store(self.rules_dir, use_cache=False)

        in_scope = scope_coding_context(
            [SUMMARY_STAGE_ITEM_ID],
            CaseFacts(gross_primary_site="breast", date_of_diagnosis="2019-03-04"),
            store,
        )[SUMMARY_STAGE_ITEM_ID]
        self.assertTrue([u for u in in_scope.units if u.source_doc == MANUAL])

        pre_2018 = scope_coding_context(
            [SUMMARY_STAGE_ITEM_ID],
            CaseFacts(gross_primary_site="breast", date_of_diagnosis="2016-03-04"),
            store,
        )[SUMMARY_STAGE_ITEM_ID]
        self.assertFalse([u for u in pre_2018.units if u.source_doc == MANUAL])

    def test_breast_units_do_not_scope_to_an_unrelated_primary_site(self):
        _, units, validations, usage = self._compile_breast()
        write_outputs(
            units, validations, usage,
            rules_dir=self.rules_dir, manual=MANUAL, site_group="breast",
            source_path=SOURCE, manifest_defaults=MANIFEST_ENTRY,
        )
        store = load_rule_store(self.rules_dir, use_cache=False)

        lung = scope_coding_context(
            [SUMMARY_STAGE_ITEM_ID],
            CaseFacts(primary_site="C349", date_of_diagnosis="2021-06-01"),
            store,
        )[SUMMARY_STAGE_ITEM_ID]
        self.assertFalse([u for u in lung.units if u.source_doc == MANUAL])


if __name__ == "__main__":
    unittest.main()
