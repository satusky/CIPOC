"""Generate a synthetic evaluation store for developing the dashboard.

    python make_dummy_data.py            # writes data/ground_truth.csv, data/eval.sqlite, web/data.js

Everything is invented: patients, values, runs and token counts. The variable list
is the real one from ``config/variable_groups.json``. The generator plants a few
patterns for the dashboard to find:

- a slow overall improvement, with a jump at the model change (run 6)
- a regression in first-course treatment for runs 8-9 (a gate closing wrongly)
- metastases recall fixed at run 3, site-specific codes improved at run 11
- melanoma and ovary below the other tissues; 2023+ surgery codes weak until run 11
- each patient-variable cell is solid, systematically wrong, or flaky, so some
  patients repeat their values every run and others do not
- runs 11-14 share one configuration, so their differences are pure run-to-run noise
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import evalstore

HERE = Path(__file__).resolve().parent
VARIABLE_GROUPS = HERE.parents[1] / "config" / "variable_groups.json"
SEED = 20261113
N_PATIENTS = 150

TISSUES = {  # share of test set, skill offset
    "breast": (0.26, 0.03),
    "lung": (0.20, -0.04),
    "colorectal": (0.18, 0.0),
    "prostate": (0.16, 0.02),
    "melanoma": (0.11, -0.10),
    "ovary": (0.09, -0.06),
}
SITES = {
    "breast": ["C504", "C504", "C502", "C508", "C509", "C501"],
    "lung": ["C341", "C341", "C343", "C349", "C342"],
    "colorectal": ["C187", "C180", "C182", "C199", "C209", "C209"],
    "prostate": ["C619"],
    "melanoma": ["C445", "C446", "C447", "C443", "C444"],
    "ovary": ["C569"],
}
HISTOLOGY = {
    "breast": ["8500", "8500", "8500", "8520", "8522", "8480"],
    "lung": ["8140", "8140", "8070", "8041", "8046"],
    "colorectal": ["8140", "8140", "8140", "8480", "8210"],
    "prostate": ["8140", "8140", "8140", "8550"],
    "melanoma": ["8720", "8743", "8721", "8742"],
    "ovary": ["8441", "8441", "8380", "8310"],
}
DATE_ITEMS = {390, 682, 832, 1200, 1210, 1220, 1230, 1240, 1270, 1280}

# Chance a cell is coded correctly at baseline, before run and tissue effects.
BASE_SKILL = {
    390: 0.80, 400: 0.90, 410: 0.90, 522: 0.70, 523: 0.90,
    690: 0.80, 700: 0.80, 710: 0.78, 720: 0.74, 740: 0.66,
    1200: 0.72, 1210: 0.70, 1220: 0.70, 1230: 0.66, 1240: 0.66, 1270: 0.62, 1280: 0.60,
    1112: 0.74, 1113: 0.76, 1114: 0.66, 1115: 0.74, 1116: 0.72, 1117: 0.62,
    672: 0.68, 674: 0.72, 676: 0.64, 682: 0.66,
    670: 0.62, 671: 0.40, 3843: 0.64, 3844: 0.68,
    764: 0.58, 820: 0.70, 830: 0.64, 1182: 0.60, 3280: 0.84, 756: 0.60,
    832: 0.70, 3836: 0.62, 3838: 0.66, 3839: 0.72, 3840: 0.70, 3841: 0.76, 3842: 0.60,
}
# Share of a variable's errors that are a missing value rather than a wrong one.
MISS_SHARE = {1112: 0.7, 1113: 0.7, 1114: 0.7, 1115: 0.7, 1116: 0.7, 1117: 0.7, 1182: 0.1}

RUNS = [
    # date, label, model, version, digest, note
    ("2026-08-31", "baseline", "llm-medium", "0.3.0", "a41c07", "First full sweep of the test set."),
    ("2026-09-04", "baseline rerun", "llm-medium", "0.3.0", "a41c07", "Same configuration as run 1."),
    ("2026-09-09", "mets keywords", "llm-medium", "0.3.1", "5be290", "Retriever keyword list widened for metastases."),
    ("2026-09-14", "mets keywords rerun", "llm-medium", "0.3.1", "5be290", "Same configuration as run 3."),
    ("2026-09-18", "date prompt", "llm-medium", "0.3.2", "c9d1e4", "Extractor prompt: explicit date-selection rules."),
    ("2026-09-24", "llm-large", "llm-large", "0.3.2", "0f7a3b", "Extractor moved to the larger model."),
    ("2026-09-29", "llm-large rerun", "llm-large", "0.3.2", "0f7a3b", "Same configuration as run 6."),
    ("2026-10-05", "gate refactor", "llm-large", "0.4.0", "71e5cc", "Treatment gate rewritten."),
    ("2026-10-08", "gate refactor rerun", "llm-large", "0.4.0", "71e5cc", "Same configuration as run 8."),
    ("2026-10-13", "gate fix", "llm-large", "0.4.1", "d2308f", "Treatment gate no longer closes on negated mentions."),
    ("2026-10-20", "site dictionary", "llm-large", "0.4.2", "8ac614", "Site data dictionary refreshed; 2023 surgery codes added."),
    ("2026-10-27", "replicate 1", "llm-large", "0.4.2", "8ac614", "Same configuration as run 11."),
    ("2026-11-03", "replicate 2", "llm-large", "0.4.2", "8ac614", "Same configuration as run 11."),
    ("2026-11-10", "replicate 3", "llm-large", "0.4.2", "8ac614", "Same configuration as run 11."),
]
GLOBAL_GAIN = [0, 0, 0.02, 0.02, 0.04, 0.10, 0.10, 0.11, 0.11, 0.12, 0.13, 0.13, 0.13, 0.13]
FAILED = {5: 3, 7: 2}  # run index -> patients whose case failed


def ymd(value: date) -> str:
    return value.strftime("%Y%m%d")


def pick(rng: random.Random, weighted: list[tuple[str, float]]) -> str:
    return rng.choices([v for v, _ in weighted], [w for _, w in weighted])[0]


def make_patient(rng: random.Random, index: int) -> dict:
    tissue = rng.choices(list(TISSUES), [share for share, _ in TISSUES.values()])[0]
    return {
        "patient_id": f"P{index + 1:04d}",
        "tissue": tissue,
        "year": rng.choice(range(2018, 2025)),
        "n_notes": max(3, min(140, round(rng.lognormvariate(3.0, 0.7)))),
    }


def make_truth(rng: random.Random, patient: dict) -> dict[int, str | None]:
    tissue, year = patient["tissue"], patient["year"]
    dx = date(year, 1, 1) + timedelta(days=rng.randrange(330))
    after = lambda low, high: ymd(dx + timedelta(days=rng.randrange(low, high)))  # noqa: E731
    truth: dict[int, str | None] = {390: ymd(dx), 400: rng.choice(SITES[tissue])}
    truth[410] = "0" if tissue in ("prostate", "colorectal") else pick(rng, [("1", 5), ("2", 5), ("9", 0.4)])
    truth[522] = rng.choice(HISTOLOGY[tissue])
    truth[523] = pick(rng, [("3", 9), ("2", 1)])

    metastatic = rng.random() < (0.38 if tissue == "lung" else 0.16)
    surgery = rng.random() < (0.35 if metastatic else 0.85)
    radiation = rng.random() < {"breast": 0.6, "lung": 0.5, "prostate": 0.45}.get(tissue, 0.2)
    chemo = rng.random() < {"lung": 0.6, "colorectal": 0.5, "ovary": 0.8, "breast": 0.4}.get(tissue, 0.1)
    hormone = rng.random() < {"breast": 0.7, "prostate": 0.5}.get(tissue, 0.02)
    brm = rng.random() < {"melanoma": 0.5, "lung": 0.3}.get(tissue, 0.05)

    truth[690] = pick(rng, [("1", 8), ("2", 1), ("4", 1)]) if radiation else "0"
    truth[700] = pick(rng, [("03", 6), ("02", 3), ("01", 1)]) if chemo else pick(rng, [("00", 9), ("87", 1)])
    truth[710] = "01" if hormone else "00"
    truth[720] = "01" if brm else "00"
    truth[740] = pick(rng, [("02", 6), ("00", 2), ("05", 1), ("01", 1)])
    truth[1200] = after(14, 90) if surgery else None
    truth[1210] = after(40, 160) if radiation else None
    truth[1220] = after(20, 120) if chemo else None
    truth[1230] = after(30, 200) if hormone else None
    truth[1240] = after(20, 140) if brm else None
    given = [truth[i] for i in (1200, 1210, 1220, 1230, 1240) if truth[i]]
    truth[1270] = min(given) if given else None
    truth[1280] = after(-20, 10) if truth[740] != "00" else None

    for item, share in ((1112, 0.4), (1113, 0.2), (1114, 0.25), (1115, 0.3), (1116, 0.3), (1117, 0.15)):
        truth[item] = "1" if metastatic and rng.random() < share else "0"

    nodes = surgery and tissue != "prostate" and rng.random() < 0.75
    truth[672] = pick(rng, [("2", 4), ("5", 3), ("6", 2), ("1", 1)]) if nodes else "0"
    truth[674] = pick(rng, [("0", 9), ("1", 0.5), ("4", 0.5)])
    examined = rng.randrange(1, 28) if nodes else 0
    truth[676] = f"{examined:02d}"
    truth[682] = truth[1200] if nodes and truth[672] in ("5", "6") else None
    truth[830] = f"{examined:02d}"
    truth[820] = f"{min(examined, int(rng.expovariate(0.5))):02d}" if nodes else "98"

    surgery_code = pick(rng, [("20", 3), ("30", 3), ("40", 2), ("50", 1), ("22", 1)]) if surgery else "00"
    truth[670] = surgery_code if year <= 2022 else None
    truth[671] = (f"A{surgery_code}0" if surgery else "A000") if year >= 2023 else None
    truth[3843] = pick(rng, [("1", 2), ("2", 4), ("3", 3), ("9", 2)])
    truth[3844] = pick(rng, [("1", 2), ("2", 4), ("3", 3)]) if surgery else "9"
    truth[764] = "7" if metastatic else pick(rng, [("1", 5), ("2", 2), ("3", 2), ("4", 1), ("0", 0.5)])
    truth[1182] = pick(rng, [("0", 6), ("1", 3), ("9", 1)]) if surgery else "9"
    truth[3280] = pick(rng, [("0", 3), ("2", 1), ("4", 1)]) if metastatic else "0"
    truth[756] = f"{min(989, max(1, round(rng.lognormvariate(3.1, 0.6)))):03d}" if rng.random() < 0.9 else "999"

    truth[832] = after(14, 60) if tissue in ("breast", "melanoma") and nodes and truth[672] == "2" else None
    truth[3836] = pick(rng, [("IA", 2), ("IC", 2), ("IIB", 1), ("IIIC", 4), ("IVB", 1)]) if tissue == "ovary" else None
    if tissue == "prostate":
        primary, secondary = pick(rng, [("3", 6), ("4", 4)]), pick(rng, [("3", 4), ("4", 5), ("5", 1)])
        truth[3838] = primary + secondary
        truth[3840] = f"{int(primary) + int(secondary):02d}"
        truth[3839] = truth[3838] if surgery else "X7"
        truth[3841] = truth[3840] if surgery else "X7"
        truth[3842] = pick(rng, [("X9", 7), ("50", 2), ("40", 1)]) if surgery else "X7"
    else:
        truth.update({3838: None, 3839: None, 3840: None, 3841: None, 3842: None})
    return truth


def shift_date(rng: random.Random, value: str) -> str:
    moved = date(int(value[:4]), int(value[4:6]), int(value[6:])) + timedelta(
        days=rng.choice([-1, 1]) * rng.choice([1, 1, 2, 7, 14, 30])
    )
    return ymd(moved)


def wrong_value(rng: random.Random, item: int, truth: str | None, pool: list[str], other_dates: list[str]) -> str:
    """A plausible wrong answer: a neighbouring date or another code for the item."""
    if item in DATE_ITEMS:
        candidates = [d for d in other_dates if d != truth]
        if candidates and (truth is None or rng.random() < 0.6):
            return rng.choice(candidates)
        return shift_date(rng, truth or rng.choice(pool))
    return rng.choice([v for v in pool if v != truth] or ["9"])


def skill(run: int, item: int, group: str, patient: dict) -> float:
    value = BASE_SKILL[item] + GLOBAL_GAIN[run] + TISSUES[patient["tissue"]][1]
    if patient["year"] <= 2019:
        value -= 0.04
    if group == "metastases" and run >= 2:
        value += 0.12
    if item in DATE_ITEMS and run >= 4:
        value += 0.07
    if item in (522, 523) and run >= 5:
        value += 0.05
    if item in (670, 671, 3843, 3844) and run >= 10:
        value += 0.20 if item == 671 else 0.08
    return max(0.05, min(0.985, value))


def main() -> None:
    rng = random.Random(SEED)
    names = evalstore.variable_names(VARIABLE_GROUPS)
    items = list(names)
    patients = [make_patient(rng, i) for i in range(N_PATIENTS)]
    truths = [make_truth(rng, p) for p in patients]
    pools = {item: sorted({t[item] for t in truths if t[item] is not None}) for item in items}

    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    truth_csv = data_dir / "ground_truth.csv"
    with open(truth_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*evalstore.PATIENT_COLUMNS, *items])
        for patient, truth in zip(patients, truths):
            writer.writerow([*patient.values(), *("" if truth[i] is None else truth[i] for i in items)])

    db_path = data_dir / "eval.sqlite"
    db_path.unlink(missing_ok=True)
    conn = evalstore.connect(db_path)
    evalstore.load_ground_truth(conn, truth_csv, names)

    # Latent, run-independent state for every patient and patient-variable cell.
    cells = []
    for patient, truth in zip(patients, truths):
        chaotic = rng.random() < 0.08
        noise = min(1.0, rng.betavariate(1.2, 6) + patient["n_notes"] / 400 + (0.6 if chaotic else 0))
        patient_dates = sorted({truth[i] for i in DATE_ITEMS if truth[i]})
        row = {}
        for item in items:
            missed = truth[item] is not None and rng.random() < MISS_SHARE.get(item, 0.25)
            row[item] = {
                "u": rng.random(),
                "flaky_width": (0.01 + 0.05 * noise) * (3.0 if chaotic else 1.0) * (1.4 if item in DATE_ITEMS or item == 756 else 1.0),
                "flaky_p": rng.uniform(0.35, 0.85),
                "wrong": None if missed else wrong_value(rng, item, truth[item], pools[item], patient_dates),
            }
        cells.append({"cells": row, "gate_victim": rng.random() < 0.45, "dates": patient_dates})

    for run_index, (day, label, model, version, digest, note) in enumerate(RUNS):
        run_id = f"sweep-{run_index + 1:02d}"
        evalstore.record_run(
            conn, run_id, f"{day}T09:00:00+00:00",
            label=label, model=model, cipoc_version=version, config_digest=digest, notes=note,
        )
        failed = set(rng.sample(range(N_PATIENTS), FAILED.get(run_index, 0)))
        for p, (patient, truth, latent) in enumerate(zip(patients, truths, cells)):
            notes = patient["n_notes"]
            large = model == "llm-large"
            tokens_in = round(
                (notes * rng.uniform(1500, 2100) + rng.uniform(60_000, 90_000))
                * (1.12 if large else 1.0) * (0.8 if run_index >= 10 else 1.0)
            )
            tokens_out = round((notes * 140 + rng.uniform(5000, 8000)) * (1.8 if large else 1.0))
            calls = notes + rng.randrange(14, 22)
            seconds = round(calls * rng.uniform(0.5, 0.9) * (1.5 if large else 1.0), 1)
            if p in failed:
                evalstore.record_patient(
                    conn, run_id, patient["patient_id"], {}, status="failed",
                    model_invocations=calls // 3,
                )
                continue

            gate_closed = run_index in (7, 8) and latent["gate_victim"]
            results = {}
            for item in items:
                group = names[item]["group_id"]
                cell, expected = latent["cells"][item], truth[item]
                if gate_closed and group == "first_course_treatment":
                    results[item] = (None, "not_applicable")
                    continue
                s = skill(run_index, item, group, patient)
                if expected is None:  # blank truth: mostly left alone, sometimes hallucinated
                    s = min(0.99, s + 0.22)
                if cell["u"] < s - cell["flaky_width"]:
                    correct = rng.random() < 0.997
                elif cell["u"] < s + cell["flaky_width"]:
                    correct = rng.random() < cell["flaky_p"]
                else:
                    correct = rng.random() < 0.02
                if correct:
                    value = expected
                elif rng.random() < 0.95:
                    value = cell["wrong"]
                else:
                    value = wrong_value(rng, item, expected, pools[item], latent["dates"])
                if value is not None:
                    status = "extracted"
                elif expected is None and group not in ("first_course_treatment", "lymph_node_removal"):
                    status = "not_applicable"
                else:
                    status = "error" if rng.random() < 0.03 else "not_found"
                results[item] = (value, status)
            evalstore.record_patient(
                conn, run_id, patient["patient_id"], results,
                case_run_id=f"{rng.getrandbits(128):032x}",
                # A few cases come back without provider usage.
                input_tokens=None if rng.random() < 0.01 else tokens_in,
                output_tokens=tokens_out,
                model_invocations=calls,
                duration_seconds=seconds,
            )
        conn.commit()

    evalstore.export_bundle(conn, HERE / "web" / "data.js")
    conn.close()
    print(f"{N_PATIENTS} patients, {len(items)} variables, {len(RUNS)} runs")
    print(f"  {truth_csv}\n  {db_path}\n  {HERE / 'web' / 'data.js'}")


if __name__ == "__main__":
    main()
