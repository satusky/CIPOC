"""Tests for the SEER*RSA staging ingest.

No network and no LLM: the unit builder is pure, and the driver is exercised
against an in-memory ZIP assembled from ``tests/fixtures/staging/`` — real
schema, selection table, and code tables trimmed out of ``eod_public-3.3.zip``.

Three things are pinned that this ingest can silently get wrong at runtime:
the ``applies_to`` built from the schema-selection table (it is the only reason
a site-specific code set narrows anything), the per-unit ``dx_date_min`` (without
it the manual-level temporal filter drops the whole manual for a pre-2024 case),
and the ``SEER-RSA`` manifest family (registering as ``SEER`` would delete
``summary_stage_2018``'s item-764 tables by precedence).
"""

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from cipoc.models import CaseFacts
from cipoc.tools.coding_context import load_rule_store, scope_coding_context

from scripts.rule_compilation.compile_staging import compile_schema
from scripts.rule_compilation.staging_fetch import StagingRelease, verify_release
from scripts.rule_compilation.staging_index import (
    ALGORITHMS,
    EXCLUDE_ITEMS,
    VARIABLE_GROUPS,
    anchor_for,
    rule_id,
    schema_stem,
    target_items,
)
from scripts.rule_compilation.staging_units import (
    build_schema_units,
    extract_code_table,
    schema_applicabilities,
)
from scripts.rule_compilation.staging_validate import validate_staging_units

FIXTURES = Path("tests/fixtures/staging")
RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")

EOD = ALGORITHMS["eod"]

# Every table the fixture schema's inputs name, plus its selection table.
_FIXTURE_TABLES = (
    "schema_selection_breast",
    "year_dx_validation",
    "primary_site",
    "histology",
    "ss2018_breast_69079",
    "grade_pathological_47031",
)


def _load_fixture_schema() -> dict:
    return json.loads((FIXTURES / "breast_schema.json").read_text())


def _load_fixture_tables() -> dict[str, dict]:
    return {name: json.loads((FIXTURES / f"{name}.json").read_text()) for name in _FIXTURE_TABLES}


def _fixture_release() -> StagingRelease:
    """Pack the fixtures into an in-memory ZIP shaped like the real release.

    Goes through ``StagingRelease`` rather than around it so the member layout
    the fetcher assumes (``schemas/<id>.json``, ``tables/<id>.json``) is part of
    what these tests pin.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("schemas/breast.json", (FIXTURES / "breast_schema.json").read_text())
        for name in _FIXTURE_TABLES:
            archive.writestr(f"tables/{name}.json", (FIXTURES / f"{name}.json").read_text())
    return StagingRelease(EOD, zipfile.ZipFile(buffer))


class StagingIndexTests(unittest.TestCase):
    """The pure index: what is in scope, what is excluded, and the id scheme."""

    def test_target_items_derive_from_the_variable_group_config(self):
        from cipoc.tools.orchestration import load_variable_groups

        configured = {v.item_id for g in load_variable_groups(VARIABLE_GROUPS) for v in g.variables}
        self.assertEqual(target_items(), frozenset(configured) - set(EXCLUDE_ITEMS))

    def test_excluded_items_are_never_in_the_default_scope(self):
        self.assertFalse(target_items() & set(EXCLUDE_ITEMS))

    def test_requesting_an_excluded_item_fails_with_the_reason(self):
        with self.assertRaises(ValueError) as caught:
            target_items(items=[3844, 522])
        self.assertIn("522", str(caught.exception))
        self.assertIn("selection criteria", str(caught.exception))

    def test_item_override_widens_beyond_the_configured_groups(self):
        """--item is how the ingest is deliberately widened; it must not be
        silently intersected back down to variable_groups.json."""
        self.assertEqual(target_items(items=[940, 950]), frozenset({940, 950}))

    def test_manifest_family_is_seer_rsa_not_seer(self):
        """resolve_precedence drops every non-SEER unit of any kind SEER covers.

        Registering these as 'SEER' with a recent publication_date would not
        merely compete with summary_stage_2018 and spcsm_2024 — it would delete
        their units for every kind this ingest also emits.
        """
        for algorithm in ALGORITHMS.values():
            self.assertEqual(algorithm.manifest_entry["family"], "SEER-RSA")
            self.assertTrue(algorithm.dx_date_min)

    def test_stems_and_rule_ids_are_stable_and_distinct_per_schema(self):
        self.assertEqual(schema_stem("Bone Pelvis"), "bone_pelvis")
        # TNM tables such as clin_n_daj are shared across ~128 schemas, and
        # load_rule_store raises on a duplicate rule_id anywhere in the store.
        left = rule_id("seer_rsa_tnm", "breast", "clin_n_daj", "code_table", 3)
        right = rule_id("seer_rsa_tnm", "lung", "clin_n_daj", "code_table", 3)
        self.assertNotEqual(left, right)
        self.assertEqual(left, "seer_rsa_tnm:breast:clin_n_daj:c3")

    def test_anchor_names_the_release_member_it_was_copied_from(self):
        self.assertEqual(
            anchor_for(EOD, "tables/grade_pathological_47031.json"),
            "eod_public-3.3:tables/grade_pathological_47031.json",
        )


class StagingUnitTests(unittest.TestCase):
    """``build_schema_units`` against the trimmed breast fixture."""

    @classmethod
    def setUpClass(cls):
        cls.schema = _load_fixture_schema()
        cls.tables = _load_fixture_tables()
        cls.units = build_schema_units(
            cls.schema, cls.tables, EOD, items=frozenset({764, 3844})
        )

    def _code_table(self, item_id: int):
        return next(
            u for u in self.units if u.kind == "code_table" and u.item_ids == [item_id]
        )

    def test_the_grade_table_becomes_a_code_table_on_item_3844(self):
        """The gap this ingest exists to fill: the dictionary's Code Descriptions
        for 3843/3844 is ``{}``, so nothing is enforceable without these."""
        unit = self._code_table(3844)
        self.assertTrue(unit.codes["1"].startswith("G1: Low combined histologic grade"))
        self.assertIn("L", unit.codes)
        self.assertEqual(unit.anchor, "eod_public-3.3:tables/grade_pathological_47031.json")

    def test_applicability_comes_from_the_schema_selection_table(self):
        predicates = schema_applicabilities(self.schema, self.tables, EOD)
        self.assertEqual(len(predicates), 1)  # fixture selection table has one row
        applies_to = predicates[0]
        self.assertEqual(applies_to.sites, ["C500-C506", "C508-C509"])
        self.assertEqual(applies_to.histologies, ["8000-8700", "8982-8983"])
        self.assertIsNone(applies_to.dx_date_max)

    def test_every_unit_carries_its_own_dx_date_min(self):
        """scope_coding_context skips the manual-level temporal filter only for
        units with their own dates; a unit without one dies on publication_date."""
        self.assertTrue(self.units)
        for unit in self.units:
            self.assertIsNotNone(unit.applies_to)
            self.assertEqual(unit.applies_to.dx_date_min, "2018-01-01")

    def test_schema_notes_become_one_general_instruction(self):
        general = [u for u in self.units if u.kind == "instruction" and not u.item_ids]
        self.assertEqual(len(general), 1)
        self.assertEqual(general[0].anchor, "eod_public-3.3:schemas/breast.json")
        self.assertIn("Nipple", general[0].text)

    def test_table_notes_become_an_instruction_on_the_same_item(self):
        notes = [u for u in self.units if u.kind == "instruction" and u.item_ids == [3844]]
        self.assertEqual(len(notes), 1)
        self.assertIn("must not be blank", notes[0].text)

    def test_rule_ids_are_unique_within_a_schema(self):
        ids = [u.rule_id for u in self.units]
        self.assertEqual(len(ids), len(set(ids)))

    def test_selection_and_lookup_tables_produce_nothing(self):
        """Item 400/522/390 are excluded by index, but the table shapes must be
        rejected on their own merits too — the two defences are independent."""
        widened = build_schema_units(
            self.schema, self.tables, EOD, items=frozenset({390, 400, 522, 764, 3844})
        )
        self.assertFalse([u for u in widened if u.kind == "code_table" and u.item_ids == [522]])
        self.assertFalse([u for u in widened if u.kind == "code_table" and u.item_ids == [390]])
        # A range list with no DESCRIPTION column, and a match table with an
        # ENDPOINT column, are not value sets.
        self.assertIsNone(extract_code_table(self.tables["histology"], "hist"))
        self.assertIsNone(extract_code_table(self.tables["year_dx_validation"], "year_dx"))
        # ... and a value set read under the wrong input key is not one either.
        self.assertIsNone(
            extract_code_table(self.tables["grade_pathological_47031"], "grade_clin")
        )

    def test_year_dx_column_narrows_the_date_window(self):
        """Selection cells look like '2018-2024', '2023-9998, 9999,' or '*'.
        9998/9999 are the unknown/blank sentinels, i.e. no upper bound."""
        cases = {
            "2018-2024": ("2018-01-01", "2024-12-31"),
            "2023-9998, 9999,": ("2023-01-01", None),
            "2018-2020, 9999,": ("2018-01-01", "2020-12-31"),
            "*": (EOD.dx_date_min, None),
        }
        for cell, (expected_min, expected_max) in cases.items():
            with self.subTest(cell=cell):
                selection = {
                    "definition": [
                        {"key": "site", "type": "INPUT"},
                        {"key": "year_dx", "type": "INPUT"},
                        {"key": "result", "type": "ENDPOINT"},
                    ],
                    "rows": [["C340-C349", cell, "MATCH"]],
                }
                applies_to, = schema_applicabilities(
                    {"id": "lung", "schema_selection_table": "sel"}, {"sel": selection}, EOD
                )
                self.assertEqual(applies_to.dx_date_min, expected_min)
                self.assertEqual(applies_to.dx_date_max, expected_max)

    def test_sex_is_not_mapped_onto_applicability(self):
        """The selection tables encode sex as a NAACCR code while CaseFacts.sex
        holds free text; matches_case compares them as strings, so a mapping
        would turn every sexed schema into a false exclusion."""
        selection = {
            "definition": [
                {"key": "site", "type": "INPUT"},
                {"key": "sex_at_birth", "type": "INPUT"},
                {"key": "result", "type": "ENDPOINT"},
            ],
            "rows": [["C481, C482", "2", "MATCH"]],
        }
        applies_to, = schema_applicabilities(
            {"id": "peritoneum", "schema_selection_table": "sel"}, {"sel": selection}, EOD
        )
        self.assertIsNone(applies_to.sex)

    def _multi_row_schema(self):
        """A two-row selection table, shaped like the soft-tissue schemas.

        Row 1 pairs C470-C529 with 8992; row 2 pairs C473/C475 with 8000-8803.
        Breast C509 + 8500 satisfies row 1's site and row 2's histology, and no
        row at all.
        """
        selection = {
            "definition": [
                {"key": "site", "type": "INPUT"},
                {"key": "hist", "type": "INPUT"},
                {"key": "result", "type": "ENDPOINT"},
            ],
            "rows": [
                ["C470-C529", "8992", "MATCH"],
                ["C473, C475", "8000-8803", "MATCH"],
            ],
        }
        schema = {
            "id": "soft_tissue_other",
            "title": "Soft Tissue Other",
            "schema_selection_table": "sel",
            "notes": "chapter notes",
            "inputs": [{"key": "grade_path", "naaccr_item": 3844, "table": "grade"}],
        }
        return schema, {"sel": selection, "grade": self.tables["grade_pathological_47031"]}

    def test_a_multi_row_schema_yields_one_predicate_per_row(self):
        schema, tables = self._multi_row_schema()
        predicates = schema_applicabilities(schema, tables, EOD)
        self.assertEqual(
            [(p.sites, p.histologies) for p in predicates],
            [(["C470-C529"], ["8992"]), (["C473", "C475"], ["8000-8803"])],
        )

    def test_unioning_the_rows_would_invent_a_site_histology_pair(self):
        """The regression this split exists to prevent.

        A single unioned predicate claims sites C470-C529 *and* histologies
        8000-8803, so a breast ductal carcinoma matches a soft-tissue schema and
        picks up its grading system. Split per row, it matches neither.
        """
        from cipoc.models import RuleApplicability
        from cipoc.tools.coding_context import matches_case

        schema, tables = self._multi_row_schema()
        breast_ductal = CaseFacts(
            primary_site="C509", histology="8500", date_of_diagnosis="2021-06-14"
        )
        predicates = schema_applicabilities(schema, tables, EOD)
        self.assertFalse([p for p in predicates if matches_case(p, breast_ductal)])

        unioned = RuleApplicability(
            sites=[s for p in predicates for s in p.sites],
            histologies=[h for p in predicates for h in p.histologies],
            dx_date_min=EOD.dx_date_min,
        )
        self.assertTrue(matches_case(unioned, breast_ductal), "premise of this test")

        # ... and a case each row genuinely selects still matches.
        for site, histology in (("C509", "8992"), ("C473", "8140")):
            with self.subTest(site=site, histology=histology):
                facts = CaseFacts(
                    primary_site=site, histology=histology, date_of_diagnosis="2021-06-14"
                )
                self.assertTrue([p for p in predicates if matches_case(p, facts)])

    def test_every_unit_is_repeated_once_per_predicate(self):
        schema, tables = self._multi_row_schema()
        units = build_schema_units(schema, tables, EOD, items=frozenset({3844}))
        by_kind = {"code_table": [], "instruction": []}
        for unit in units:
            by_kind[unit.kind].append(unit)
        self.assertEqual(len(by_kind["code_table"]), 2)
        # schema notes + the grade table's own notes, each twice
        self.assertEqual(len(by_kind["instruction"]), 4)
        self.assertEqual(len({u.rule_id for u in units}), len(units))
        self.assertEqual(len({u.text for u in by_kind["code_table"]}), 1)


class StagingValidationTests(unittest.TestCase):
    """Validation keeps every dictionary check and drops only the markdown one."""

    @classmethod
    def setUpClass(cls):
        if not DATA_DICTIONARY.exists():
            raise unittest.SkipTest(f"{DATA_DICTIONARY} not present")
        cls.units = build_schema_units(
            _load_fixture_schema(), _load_fixture_tables(), EOD, items=frozenset({764, 3844})
        )

    def test_ingested_units_pass_without_a_markdown_anchor(self):
        results = validate_staging_units(self.units, data_dictionary_path=DATA_DICTIONARY)
        self.assertEqual([(r.rule_id, r.errors) for r in results if not r.ok], [])
        self.assertEqual(len(results), len(self.units))

    def test_a_code_outside_the_dictionary_set_is_quarantined(self):
        """The check that catches a table whose vocabulary drifted from NAACCR's
        — the three FIGO tables with codes like '1C2' land here by design."""
        bad = self.units[0].model_copy(
            update={
                "rule_id": "seer_rsa_eod:breast:bogus:c99",
                "kind": "code_table",
                "item_ids": [764],
                "codes": {"not-a-summary-stage-code": "nonsense"},
            }
        )
        results = validate_staging_units([bad], data_dictionary_path=DATA_DICTIONARY)
        self.assertFalse(results[0].ok)
        self.assertTrue(any("not in item 764 set" in e for e in results[0].errors))

    def test_provenance_is_still_enforced_for_markdown_compiles(self):
        """require_provenance defaults to True, so every existing caller is
        unaffected by the flag this ingest added."""
        from scripts.rule_compilation.validate import validate_unit

        result = validate_unit(
            self.units[0], source_lines=["a line"], data_dictionary={"3844": {}}
        )
        self.assertTrue(any("Anchor" in e for e in result.errors))


class StagingCompileTests(unittest.TestCase):
    """The driver end to end: fixture ZIP → written store → runtime scoping."""

    FACTS = CaseFacts(gross_primary_site="breast", date_of_diagnosis="2021-06-01")

    @classmethod
    def setUpClass(cls):
        for path in (RULES_DIR, DATA_DICTIONARY):
            if not path.exists():
                raise unittest.SkipTest(f"{path} not present")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rules_dir = Path(self._tmp.name) / "rules"
        shutil.copytree(RULES_DIR, self.rules_dir)
        self.addCleanup(self._tmp.cleanup)

        release = _fixture_release()
        verify_release(release)
        self.out_path, self.report_path, self.units, self.accepted = compile_schema(
            release, "breast",
            items=target_items(),
            rules_dir=self.rules_dir,
            data_dictionary_path=DATA_DICTIONARY,
        )
        # Cache-busting is mandatory: load_rule_store memoizes by resolved path,
        # and a temp directory can reuse one.
        self.store = load_rule_store(self.rules_dir, use_cache=False)
        self.fixture_ids = {u.rule_id for u in self.accepted}

    def _scope(self, item_id: int, facts: CaseFacts | None = None):
        return scope_coding_context([item_id], facts or self.FACTS, self.store)[item_id]

    def _fixture_units(self, context) -> list:
        """The units this fixture compile produced, isolated from the rest of the store.

        ``source_doc == 'seer_rsa_eod'`` is not a usable filter: the committed
        store already holds every EOD schema, and a breast case legitimately
        matches several of them (melanoma skin, the soft-tissue schemas and GIST
        all include C50 sites — with histology unknown, scoping widens to all of
        them by design). Only the fixture's own rule_ids identify what this
        compile put there.
        """
        return [u for u in context.units if u.rule_id in self.fixture_ids]

    def test_outputs_written_and_manifest_seeded(self):
        self.assertEqual(len(self.accepted), len(self.units))
        self.assertTrue(self.report_path.exists())
        self.assertFalse((self.out_path.parent / "breast.usage.json").exists())
        self.assertEqual(len(json.loads(self.out_path.read_text())), len(self.accepted))

        entry = json.loads((self.rules_dir / "manifest.json").read_text())[EOD.manual]
        self.assertEqual(entry["family"], "SEER-RSA")
        self.assertEqual(entry["publication_date"], EOD.manifest_entry["publication_date"])

    def test_breast_grade_codes_reach_the_scoped_context(self):
        """The whole point: item 3844's dictionary code set is empty, so before
        this ingest the extractor had nothing to code against.

        Only the key set is asserted against ``reduced_codes``. Descriptions are
        not: ``reduce_valid_codes`` folds the applicable code tables with a plain
        ``dict.update`` in specificity order, so the *least* specific table wins
        every key it shares — here ``store_2024``'s degenerate ``{'1': '1', …}``
        table overwrites the breast text. That is pre-existing runtime behaviour,
        not something this ingest introduces; the unit's own codes are checked
        instead.
        """
        context = self._scope(3844)
        ours = [u for u in self._fixture_units(context) if u.kind == "code_table"]
        self.assertEqual(len(ours), 1)
        self.assertTrue(ours[0].codes["1"].startswith("G1: Low combined"))

        self.assertIsNotNone(context.reduced_codes)
        # 'L'/'M'/'H' are the in-situ nuclear grades, which only the breast
        # schema's table carries — proof the site-specific set is what landed.
        self.assertTrue({"1", "L", "M", "H"} <= set(context.reduced_codes))

    def test_a_different_primary_site_gets_none_of_them(self):
        context = self._scope(3844, CaseFacts(primary_site="C349", date_of_diagnosis="2021-06-01"))
        self.assertFalse(self._fixture_units(context))
        # Scoping selects as well as excludes: a C349 case still gets EOD grade
        # guidance, from the lung schema rather than the breast one.
        self.assertTrue([
            u for u in context.units
            if u.source_doc == EOD.manual and ":lung:" in u.rule_id
        ])

    def test_a_pre_2018_diagnosis_gets_none_of_them(self):
        context = self._scope(
            3844, CaseFacts(gross_primary_site="breast", date_of_diagnosis="2015-01-01")
        )
        self.assertFalse(self._fixture_units(context))

    def test_seer_rsa_family_yields_to_seer_but_survives_where_seer_is_silent(self):
        """Decision 3, both directions, in one assertion pair.

        Item 764 is carried by summary_stage_2018 (family SEER), so precedence
        keeps that manual's code tables and drops ours — the known, accepted
        consequence of registering as SEER-RSA. Item 3844 is carried by no SEER
        manual, so ours survive. Flip the family to 'SEER' and the first of these
        inverts, silently deleting summary_stage_2018's item-764 tables.
        """
        emitted = {u.item_ids[0] for u in self.accepted if u.kind == "code_table" and u.item_ids}
        self.assertTrue({764, 3844} <= emitted, "fixture must emit both items to test this")

        summary_stage = self._scope(764)
        self.assertFalse(self._fixture_units(summary_stage))
        self.assertTrue([u for u in summary_stage.units if u.source_doc == "summary_stage_2018"])

        grade = self._scope(3844)
        self.assertTrue(self._fixture_units(grade))


if __name__ == "__main__":
    unittest.main()
