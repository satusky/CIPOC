"""Algorithm index for the SEER*RSA staging ingest.

Pure code, no network and no LLM. SEER*RSA (``staging.seer.cancer.gov``)
publishes, per staging *schema* — i.e. per site/histology group — the code
tables and registrar notes for the NAACCR items that schema collects. That is
exactly the dimension the NAACCR data dictionary lacks: its ``Code Descriptions``
are site-agnostic where they exist at all, and for Grade Clinical/Pathological
(3843/3844) they are empty, which leaves ``VariableValueValidator`` with nothing
to enforce.

The data comes from the ``imsweb/staging-client-java`` release ZIPs rather than
the website, because the ZIPs *are* the website's data: ``schemas/<id>.json``
carries the schema's notes and its inputs (each with its NAACCR item and the id
of its code table), ``tables/<id>.json`` carries the code rows and the registrar
notes, and ``tables/schema_selection_<id>.json`` carries the ICD-O-3 site and
histology ranges that select the schema. stdlib ``urllib`` + ``zipfile`` + ``json``
read all of it, so nothing new is added to a DBR-18.2-pinned project.

This module holds the parts that must not drift: the pinned release tag, the two
algorithms compiled from it, which NAACCR items are in scope, and the rule_id
scheme. ``staging_units`` turns parsed schemas into units; ``compile_staging``
drives the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

VARIABLE_GROUPS = Path("config/variable_groups.json")

# Pinned so a re-run is reproducible. The staging-client releases version the
# algorithm data as well as the client, so a newer tag can ship a newer EOD or
# TNM edition; bumping this is a deliberate act with a new manifest entry.
RELEASE_TAG = "v11.9.4"
RELEASE_URL = "https://github.com/imsweb/staging-client-java/releases/download/{tag}/{asset}"


@dataclass(frozen=True)
class StagingAlgorithm:
    """One SEER*RSA algorithm release, and how it lands in the rule store.

    ``dx_date_min`` is the algorithm's effective date, not the release date. It
    is the floor every emitted unit carries when its schema-selection table does
    not state a narrower ``year_dx`` range — see ``MANIFEST_ENTRY`` for why the
    unit-level date is what actually keeps these rules in scope.
    """

    name: str            # --algorithm value: 'eod' or 'tnm'
    manual: str          # manifest key / source_doc
    algorithm_id: str    # the 'algorithm' field inside every schema/table JSON
    version: str         # the 'version' field inside every schema/table JSON
    asset: str           # release asset file name
    dx_date_min: str
    manifest_entry: dict = field(repr=False)

    @property
    def url(self) -> str:
        return RELEASE_URL.format(tag=RELEASE_TAG, asset=self.asset)

    @property
    def source_ref(self) -> str:
        """Stable provenance prefix, e.g. 'eod_public-3.3'. Used in anchors."""
        return self.asset.removesuffix(".zip")


# publication_date in both entries is *edition recency for precedence only*, not
# an effective date, and family is 'SEER-RSA' rather than 'SEER'. Both are traps:
#
#   * resolve_precedence() groups by kind, keeps the newest publication_date
#     within each family, then — when 'SEER' is among the survivors and more than
#     one family is present — drops every non-SEER unit of that kind. Registering
#     these as 'SEER' with a recent date would silently delete
#     summary_stage_2018's item-764 code tables and displace spcsm_2024's
#     instructions wholesale. As 'SEER-RSA' the ingest fills the genuine gaps
#     (3843/3844, 3836, 3838-3842 — no SEER manual carries them) and yields where
#     SEER already speaks. Same contract store_2024 uses with 'CoC'.
#   * scope_coding_context() skips the manual-level temporal filter for any unit
#     whose applies_to carries its own dates, so every emitted unit sets
#     dx_date_min. Without it, a 2019 case would drop the whole manual on the
#     publication_date alone.
_MANIFEST_NOTE = (
    "Edition recency for precedence only, not an effective date. Every unit carries its "
    "own applies_to.dx_date_min (the algorithm's effective year, or the schema-selection "
    "table's year_dx range when it states one)."
)

ALGORITHMS: dict[str, StagingAlgorithm] = {
    "eod": StagingAlgorithm(
        name="eod",
        manual="seer_rsa_eod",
        algorithm_id="eod_public",
        version="3.3",
        asset="eod_public-3.3.zip",
        dx_date_min="2018-01-01",
        manifest_entry={
            "title": "SEER*RSA EOD Public 3.3",
            "family": "SEER-RSA",
            "publication_date": "2024-01-01",
            "effective_note": _MANIFEST_NOTE,
        },
    ),
    "tnm": StagingAlgorithm(
        name="tnm",
        manual="seer_rsa_tnm",
        algorithm_id="tnm",
        version="2.1",
        asset="tnm-2.1.zip",
        dx_date_min="2016-01-01",
        manifest_entry={
            "title": "SEER*RSA TNM 2.1",
            "family": "SEER-RSA",
            "publication_date": "2016-01-01",
            "effective_note": _MANIFEST_NOTE,
        },
    ),
}

# Items whose SEER*RSA "table" is not a coding value set, and why. Kept explicit
# in the spirit of store_index.ABSENT: both are real inputs on every schema, so
# without this they would compile ~140 times each and be wrong in both cases.
EXCLUDE_ITEMS: dict[int, str] = {
    400: "Primary Site: the schema's 'primary_site' table is the global 330-code ICD-O-3 "
         "topography list, identical for every schema. It narrows nothing, and the site "
         "scope that does belongs in applies_to, built from the schema-selection table.",
    522: "Histology: the schema's 'histology' table is a list of ICD-O-3 morphology "
         "*ranges* ('8000-8005') with no descriptions, i.e. selection criteria rather "
         "than codes. Those ranges belong in applies_to, not in a code_table.",
    390: "Date of Diagnosis: the 'year_dx_validation' table is a match predicate over the "
         "diagnosis year, not a code set, and validate_unit rejects a code_table on a "
         "date item anyway. The year range belongs on applies_to.dx_date_min/max.",
}


@lru_cache(maxsize=None)
def _configured_items(path: str) -> frozenset[int]:
    from cipoc.tools.orchestration import load_variable_groups

    groups = load_variable_groups(path)
    return frozenset(v.item_id for group in groups for v in group.variables)


def target_items(
    path: str | Path = VARIABLE_GROUPS, *, items: list[int] | None = None
) -> frozenset[int]:
    """NAACCR items this ingest emits code tables for.

    Derived from ``config/variable_groups.json`` rather than hand-copied, so an
    item added to a variable group is picked up by the next compile. ``items``
    narrows to an explicit subset (``--item``) and may name items outside the
    configured groups — that is how the ingest is widened deliberately — but
    never an ``EXCLUDE_ITEMS`` entry.
    """
    if items is not None:
        excluded = sorted(set(items) & set(EXCLUDE_ITEMS))
        if excluded:
            detail = "".join(f"\n  {i}: {EXCLUDE_ITEMS[i]}" for i in excluded)
            raise ValueError(f"Item(s) {excluded} are excluded from the staging ingest.{detail}")
        return frozenset(items)
    return _configured_items(str(path)) - frozenset(EXCLUDE_ITEMS)


_UNSAFE_STEM = re.compile(r"[^a-z0-9_]+")


def schema_stem(schema_id: str) -> str:
    """Output file stem for one schema id. Ids are already file-safe; this pins it."""
    stem = _UNSAFE_STEM.sub("_", schema_id.strip().casefold()).strip("_")
    if not stem:
        raise ValueError(f"Schema id {schema_id!r} does not yield a usable file stem.")
    return stem


_KIND_LETTERS = {
    "code_table": "c",
    "instruction": "i",
    "definition": "d",
    "priority_rule": "p",
    "example": "e",
}


def rule_id(manual: str, schema_id: str, table_id: str, kind: str, ordinal: int) -> str:
    """``{source_doc}:{schema_id}:{table_id}:{kind_letter}{n}``.

    The schema id is load-bearing, not decoration: TNM tables such as
    ``clin_n_daj`` are shared verbatim across ~128 schemas, and
    ``load_rule_store`` raises on a duplicate rule_id anywhere in the store. The
    ordinal counts units within a schema, so two inputs pointing at one table
    still get distinct ids.
    """
    return f"{manual}:{schema_id}:{table_id}:{_KIND_LETTERS[kind]}{ordinal}"


def anchor_for(algorithm: StagingAlgorithm, member: str) -> str:
    """Structural provenance for a deterministic copy: the exact ZIP member read.

    Markdown compiles anchor to an ``L<start>-L<end>`` region because a model
    could have paraphrased and the region is what proves it did not. Here the
    text is copied by code out of one named JSON member, so the member *is* the
    anchor — which is why ``staging_validate`` runs with ``require_provenance=False``.
    """
    return f"{algorithm.source_ref}:{member}"
