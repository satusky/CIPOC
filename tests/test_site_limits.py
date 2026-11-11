"""The site-limited groups cover every Primary Site where their items are collected.

FIGO Stage (3836), the Gleason items (3838-3842) and Date of Sentinel Lymph Node
Biopsy (832) are the only items with a site restriction in the variable groups
files. Each restriction is site-level only: it may let in cases the item doesn't
apply to (a male peritoneum case, a lymphoma of the cervix), but it must never
shut out one it does apply to.
"""

import json
import unittest
from pathlib import Path

from cipoc.models import CaseFacts, TargetGroup
from cipoc.tools import load_scoping_dictionary, load_variable_groups, site_applies, tissue_site_codes


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    REPO_ROOT / "config" / "variable_groups.json",
    REPO_ROOT / "config" / "variable_groups_+AJCC.json",
)
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scoping_dictionary.json"
# canonical-naaccr, checked out next to this repo, generates the site dictionary.
CANONICAL = REPO_ROOT.parent / "canonical-naaccr"
CANONICAL_BUILD = CANONICAL / "build" / "cipoc_data_dictionary.json"
CANONICAL_EOD_SCHEMAS = CANONICAL / "staging" / "eod" / "schemas"

FIGO_STAGE = 3836
GLEASON = (3838, 3839, 3840, 3841, 3842)
SENTINEL_NODE_DATE = 832

# The selection sites of the schemas that collect FIGO Stage: EOD's 13 schemas
# with a 3836 input (cervix x3, corpus x3, fallopian tube, ovary, placenta,
# primary peritoneal carcinoma, vagina, vulva x2), and the CS and TNM schemas
# that hold it in an SSF (cervix, corpus x3, fallopian tube, ovary,
# peritoneum_female_gen, placenta, vagina, vulva). Both give the same 23 codes.
FIGO_SITES = (
    "C481", "C482", "C488",
    "C510", "C511", "C512", "C518", "C519",
    "C529",
    "C530", "C531", "C538", "C539",
    "C540", "C541", "C542", "C543", "C548", "C549", "C559",
    "C569",
    "C570",
    "C589",
)
# eod/prostate, cs/prostate and tnm/prostate.
GLEASON_SITES = ("C619",)
# eod/breast: C500-C506, C508-C509, the whole breast tissue (there's no C507).
BREAST_SITES = ("C500", "C501", "C502", "C503", "C504", "C505", "C506", "C508", "C509")
# eod/melanoma_skin's selection sites, for histologies 8720-8790. STORE 2024
# collects 832 "for breast and cutaneous melanoma cases only".
MELANOMA_SKIN_SITES = (
    "C000", "C001", "C002", "C006",
    "C440", "C441", "C442", "C443", "C444", "C445", "C446", "C447", "C448", "C449",
    "C500",
    "C510", "C511", "C512", "C518", "C519",
    "C600", "C601", "C602", "C608", "C609",
    "C632",
)


def group_holding(groups: list[TargetGroup], item_id: int) -> TargetGroup:
    return next(group for group in groups if any(variable.item_id == item_id for variable in group.variables))


class SiteLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tissue_codes = tissue_site_codes(load_scoping_dictionary(FIXTURE))
        cls.configs = {path.name: load_variable_groups(path) for path in CONFIGS}

    def applies(self, group, primary_site=None, histology=None, candidates=()):
        facts = CaseFacts(
            primary_site=primary_site, histology=histology, candidate_gross_sites=list(candidates)
        )
        return site_applies(group.applies_to, facts, tissue_codes=self.tissue_codes)

    def groups_holding(self, item_id):
        """``(config file name, group)`` for the group holding the item in each config."""
        return [(name, group_holding(groups, item_id)) for name, groups in self.configs.items()]

    def test_figo_stage_covers_every_gynecologic_schema_site(self):
        for name, group in self.groups_holding(FIGO_STAGE):
            with self.subTest(config=name):
                self.assertEqual([site for site in FIGO_SITES if not self.applies(group, site)], [])
                # Retroperitoneum, the other female genital organs, prostate, breast, lung.
                for site in ("C480", "C571", "C579", "C619", "C509", "C341"):
                    self.assertFalse(self.applies(group, site), site)

    def test_figo_stage_reaches_the_partly_covered_tissues_from_the_notes(self):
        for name, group in self.groups_holding(FIGO_STAGE):
            with self.subTest(config=name):
                for tissue in ("retroperitoneum_and_peritoneum", "female_genital_organ", "uterus"):
                    self.assertTrue(self.applies(group, candidates=[tissue]), tissue)
                self.assertFalse(self.applies(group, candidates=["lung"]))

    def test_gleason_is_prostate_only(self):
        for item_id in GLEASON:
            for name, group in self.groups_holding(item_id):
                with self.subTest(config=name, item=item_id):
                    self.assertTrue(self.applies(group, "C619"))
                    for site in ("C569", "C341", "C679"):
                        self.assertFalse(self.applies(group, site), site)

    def test_sentinel_node_date_covers_breast_and_cutaneous_melanoma(self):
        for name, group in self.groups_holding(SENTINEL_NODE_DATE):
            with self.subTest(config=name):
                self.assertEqual(
                    [site for site in BREAST_SITES if not self.applies(group, site, "8500")], []
                )
                self.assertEqual(
                    [site for site in MELANOMA_SKIN_SITES if not self.applies(group, site, "8720")], []
                )
                # Squamous carcinoma of the skin, and melanoma outside the skin schema.
                self.assertFalse(self.applies(group, "C440", "8070"))
                self.assertFalse(self.applies(group, "C341", "8720"))

def _site_codes(value: str, all_codes: list[str]) -> set[str]:
    codes: set[str] = set()
    for token in value.split(","):
        low, _, high = token.strip().partition("-")
        codes.update(code for code in all_codes if low <= code <= (high or low))
    return codes


def _first_histology(row: dict) -> str | None:
    hist = row.get("hist")
    return hist.split(",")[0].strip().partition("-")[0] if hist else None


class ReferenceCoverageTests(unittest.TestCase):
    """Checks the restrictions against canonical-naaccr's latest build.

    canonical-naaccr's build, not ``documents/``, because the copy in
    ``documents/`` is made by hand and lags the generator. The build is what
    the next copy will be. Both are absent on a fresh clone or the target
    cluster, so these skip.
    """

    @classmethod
    def setUpClass(cls):
        if not CANONICAL_BUILD.is_file():
            raise unittest.SkipTest(f"{CANONICAL_BUILD} isn't present; canonical-naaccr isn't checked out next door.")
        with open(CANONICAL_BUILD) as f:
            cls.build = json.load(f)
        if not any("ssf" in schema for schema in cls.build["schemas"].values()):
            raise unittest.SkipTest(f"{CANONICAL_BUILD} predates G7 and has no SSF mappings.")
        cls.all_codes = sorted(
            code for tissue in cls.build["tissues"].values() for code in tissue["codes"]["400"]
        )
        cls.tissue_codes = {
            tissue: frozenset(entry["codes"]["400"]) for tissue, entry in cls.build["tissues"].items()
        }
        cls.configs = {path.name: load_variable_groups(path) for path in CONFIGS}

    def uncovered(self, item_id: int, schema_keys) -> list[str]:
        """Each config's misses, as "config: schema site/hist"."""
        misses = []
        for name, groups in self.configs.items():
            group = group_holding(groups, item_id)
            for key in schema_keys:
                for row in self.build["schemas"][key]["select"]:
                    histology = _first_histology(row)
                    for site in sorted(_site_codes(row.get("site", ",".join(self.all_codes)), self.all_codes)):
                        facts = CaseFacts(primary_site=site, histology=histology)
                        if not site_applies(group.applies_to, facts, tissue_codes=self.tissue_codes):
                            misses.append(f"{name}: {key} {site}/{histology}")
        return misses

    def test_every_cs_and_tnm_schema_holding_the_item_in_an_ssf(self):
        for item_id in (FIGO_STAGE, *GLEASON):
            with self.subTest(item=item_id):
                keys = [key for key, schema in self.build["schemas"].items() if str(item_id) in schema.get("ssf", {})]
                self.assertTrue(keys)
                self.assertEqual(self.uncovered(item_id, keys), [])

    def test_every_eod_schema_collecting_the_item(self):
        if not CANONICAL_EOD_SCHEMAS.is_dir():
            self.skipTest(f"{CANONICAL_EOD_SCHEMAS} isn't present.")
        collecting: dict[int, list[str]] = {}
        for path in sorted(CANONICAL_EOD_SCHEMAS.glob("*.json")):
            with open(path) as f:
                schema = json.load(f)
            if not isinstance(schema, dict):
                continue
            for schema_input in schema.get("inputs", []):
                collecting.setdefault(schema_input.get("naaccr_item"), []).append(f"eod/{schema['id']}")
        for item_id in (FIGO_STAGE, *GLEASON):
            with self.subTest(item=item_id):
                self.assertTrue(collecting.get(item_id))
                self.assertEqual(self.uncovered(item_id, collecting[item_id]), [])

    def test_sentinel_node_date_covers_eod_breast_and_melanoma_skin(self):
        # 832 isn't a staging input, so the schemas are named here (STORE 2024).
        self.assertEqual(self.uncovered(SENTINEL_NODE_DATE, ["eod/breast", "eod/melanoma_skin"]), [])


if __name__ == "__main__":
    unittest.main()
