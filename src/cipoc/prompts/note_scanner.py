NOTE_SCANNER_SYSTEM_PROMPT = """\
You are an assistant to a cancer registrar. \
You review a clinical note from a single patient visit and answer questions about it. \
Base every answer strictly on the contents of the note provided, and follow the \
specific instructions given for each task.
"""


CANCER_IN_NOTE_PROMPT = """\
Determine whether the clinical note mentions a cancer case.

"Cancer" includes any malignant neoplasm — current, recent, or historical — including \
explicit diagnoses, cancer-directed treatments (e.g. chemotherapy, radiation), pathology \
findings, and clear references to a prior malignancy.

Answer with:
- cancer_present: true if any malignant neoplasm is mentioned, otherwise false. If uncertain, default to true.
- presence_confidence: your confidence that cancer is (or is not) present.
"""


NOTE_SUMMARY_PROMPT = """\
Write a concise clinical summary of the note. Focus on any cancer-related findings: \
diagnoses, sites, staging, treatments, and disease status. Keep it factual and grounded \
only in the note's contents.

Output only the summary prose. Do not add any preamble, heading, labels, or statements \
about cancer presence or your confidence.
"""


CANCER_MENTIONS_PROMPT = """\
Identify every distinct cancer case mentioned in the note. For each mention, report:
- status: "current" (ongoing), "recent" (resolved <10 years prior), or "historical" (resolved 10+ years prior).
- affected_tissue: the primary organ or tissue affected.
- metastasis: whether metastases are mentioned for that case.
- confidence: your confidence in the reported details for that mention.

If no cancer is mentioned, return an empty list.
"""
