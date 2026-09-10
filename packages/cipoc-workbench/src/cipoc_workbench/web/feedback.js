"use strict";
/* Reviewer feedback — canned failure-mode flags plus free text, per entity.
 *
 * The flags name *where in the pipeline* something went wrong rather than how
 * bad it was, so a body of annotations aggregates into "what should I fix
 * next" instead of a pile of prose. Each entity kind gets its own vocabulary,
 * because the ways a gate can be wrong and the ways a coded value can be wrong
 * have nothing in common.
 *
 * Persistence needs the workbench server (`cipoc-workbench serve`).
 * Under a plain static server the form still renders, disabled, with the
 * command that enables it — failing at save time, after someone has typed a
 * paragraph, would be the worse trade.
 */

const FEEDBACK_FLAGS = {
  variable: [
    ["wrong_value", "wrong value"],
    ["missed_value", "missed a value that is there"],
    ["spurious_value", "coded a value that should not exist"],
    ["wrong_evidence", "right value, wrong evidence"],
    ["wrong_rule", "wrong coding rule applied"],
    ["bad_validation", "validation rejected a good value"],
    /* Without this, a mistake in a hand-built reference file gets recorded as
     * an extraction defect and pollutes every count derived from it. */
    ["truth_wrong", "the ground truth is wrong"],
  ],
  group: [
    ["gate_wrongly_excluded", "gate wrongly excluded this group"],
    ["gate_wrongly_admitted", "gate wrongly admitted this group"],
    ["wrong_notes", "wrong notes selected"],
    ["wrong_order", "dependency order wrong"],
    ["wrong_scoping", "wrong site or histology scoping"],
  ],
  note: [
    ["bad_concepts", "concepts mis-scanned"],
    ["bad_temporality", "cancer temporality wrong"],
    ["bad_summary", "summary is misleading"],
  ],
};

const EMPTY_ANNOTATION = { flags: [], expected: null, note: "" };

const draftKey = (kind, id) => kind + ":" + id;

/* The saved record for an entity, or null. */
function annotationFor(kind, id) {
  const bucket = App.feedback[kind] || {};
  return Object.hasOwn(bucket, String(id)) ? bucket[String(id)] : null;
}

const isAnnotated = (kind, id) => annotationFor(kind, id) !== null;

const annotationCount = () =>
  Object.values(App.feedback).reduce((n, bucket) => n + Object.keys(bucket).length, 0);

/* What the form should currently show: an unsaved draft if one exists, else the
 * saved record, else empty. */
function currentAnnotation(kind, id) {
  const draft = App.feedbackDraft.get(draftKey(kind, id));
  if (draft) return draft;
  const saved = annotationFor(kind, id);
  return saved ? { ...saved } : { ...EMPTY_ANNOTATION, flags: [] };
}

function setDraft(kind, id, patch) {
  const key = draftKey(kind, id);
  if (!App.feedbackWritable || App.feedbackPending.has(key)) return;
  App.feedbackDraft.set(key, { ...currentAnnotation(kind, id), ...patch });
  App.feedbackErrors.delete(key);
}

/* ------------------------------------------------------------------ load */

function indexFeedback(document) {
  const annotations = document.annotations;
  if (!isRecord(annotations)) throw new Error("Feedback annotations must be an object.");
  const feedback = {};
  for (const kind of Object.keys(FEEDBACK_FLAGS)) {
    if (!isRecord(annotations[kind])) throw new Error("Invalid feedback bucket: " + kind);
    feedback[kind] = Object.fromEntries(Object.entries(annotations[kind]).map(([id, annotation]) => {
      if (!isRecord(annotation)) throw new Error("Malformed feedback annotation.");
      // Persisted single-file reviews can omit these fields. Normalize a copy,
      // retaining metadata and rejecting explicit malformed values such as flags:null.
      const copy = { flags: [], expected: null, note: "", ...annotation };
      validateAnnotation(copy);
      return [id, copy];
    }));
  }
  App.feedback = feedback;
}

function validateAnnotation(annotation) {
  if (!isRecord(annotation) || !Array.isArray(annotation.flags) ||
      !annotation.flags.every((flag) => typeof flag === "string") ||
      (annotation.note != null && typeof annotation.note !== "string") ||
      (annotation.expected != null && typeof annotation.expected !== "string")) {
    throw new Error("Malformed feedback annotation; saved feedback was not replaced.");
  }
}

/* Probing at boot rather than at save time: the form has to know whether it can
 * promise anything before someone starts typing into it. */
async function loadFeedback(activation = App.activation) {
  if (activation !== App.activation || !canonicalRunId(App.run.run_id)) return false;
  const runId = App.run.run_id;
  const generation = ++App.feedbackGeneration;
  const current = () => activation === App.activation && runId === App.run.run_id && generation === App.feedbackGeneration;
  App.feedbackWritable = false;
  App.feedbackStatus = "loading";
  App.feedbackReason = "Loading this run's feedback...";
  try {
    const response = await fetch("/api/runs/" + runId + "/feedback", { cache: "no-store" });
    const document = await response.json();
    if (!current()) return false;
    if (!response.ok) throw new Error(document.detail || "HTTP " + response.status);
    if (document.run_id !== runId) throw new Error("Feedback response run identity does not match this run.");
    if (typeof document.writable !== "boolean") throw new Error("Feedback service did not report write capability.");
    indexFeedback(document);
    App.feedbackWritable = document.writable;
    App.feedbackStatus = document.writable ? "writable" : "read-only";
    App.feedbackReason = document.writable ? "Feedback is saved separately for this run."
      : document.read_only_reason || "No feedback destination configured. Use cipoc-workbench serve --feedback-dir DIRECTORY to save across runs.";
  } catch (err) {
    if (!current()) return false;
    App.feedbackWritable = false;
    App.feedbackStatus = "unavailable";
    App.feedbackReason = "Feedback unavailable: " + err.message + ". The run remains viewable; annotations cannot be saved. " +
      "Check the server's feedback destination, or serve with cipoc-workbench serve --feedback-dir DIRECTORY.";
  }
  renderChrome();
  render();
  if (App.selection) renderDetail();
  return App.feedbackWritable;
}

async function saveAnnotation(kind, id, annotation, status) {
  const key = draftKey(kind, id);
  const runId = App.run.run_id;
  const activation = App.activation;
  const current = () => activation === App.activation && runId === App.run.run_id;
  if (!App.feedbackWritable || !canonicalRunId(runId) || !Object.hasOwn(FEEDBACK_FLAGS, kind) || App.feedbackPending.has(key)) return false;
  validateAnnotation(annotation);
  const submitted = { flags: [...annotation.flags], expected: annotation.expected ?? null, note: annotation.note || "" };
  App.feedbackDraft.set(key, submitted);
  App.feedbackPending.set(key, submitted);
  App.feedbackErrors.delete(key);
  if (status) { status.className = "chip"; status.textContent = "saving..."; }
  if (isSelected(kind, id)) renderDetail();
  let success = false;
  try {
    const response = await fetch("api/runs/" + runId + "/feedback/" + kind + "?entity_id=" + encodeURIComponent(String(id)), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitted),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || "HTTP " + response.status);
    }
    const document = await response.json();
    if (!current()) return false;
    if (document.run_id !== runId || document.kind !== kind || String(document.id) !== String(id)) {
      throw new Error("Feedback save response identity does not match this annotation.");
    }
    if (!Object.hasOwn(document, "annotation")) throw new Error("Feedback save response is missing its annotation.");
    const saved = document.annotation;
    if (saved !== null) validateAnnotation(saved);
    if (saved) Object.defineProperty(App.feedback[kind], String(id), { value: saved, writable: true, enumerable: true, configurable: true });
    else delete App.feedback[kind][String(id)];

    if (App.feedbackDraft.get(key) === submitted) App.feedbackDraft.delete(key);
    success = true;
  } catch (err) {
    if (!current()) return false;
    App.feedbackErrors.set(key, "Not saved: " + err.message);
    if (status) { status.className = "chip bad"; status.textContent = App.feedbackErrors.get(key); }
  } finally {
    if (current()) {
      App.feedbackPending.delete(key);
      renderChrome();
      render();
      if (isSelected(kind, id)) renderDetail();
    }
  }
  return success;
}

/* ------------------------------------------------------------------ form */

function feedbackSection(kind, id) {
  const flags = FEEDBACK_FLAGS[kind];
  if (!flags) return null;

  const current = currentAnnotation(kind, id);
  const saved = annotationFor(kind, id);
  const key = draftKey(kind, id);
  const activation = App.activation;
  const pending = App.feedbackPending.has(key);
  const disabled = !App.feedbackWritable || pending;
  const editable = () => activation === App.activation && App.feedbackWritable && !App.feedbackPending.has(key);

  const boxes = h("div", { class: "flag-grid" }, flags.map(([flagId, label]) => {
    const input = h("input", {
      type: "checkbox",
      disabled: disabled || undefined,
      onchange: (e) => {
        if (!editable()) return;
        const next = new Set(currentAnnotation(kind, id).flags);
        if (e.target.checked) next.add(flagId);
        else next.delete(flagId);
        setDraft(kind, id, { flags: [...next], expected: expected ? expected.value : null });
        markDirty();
      },
    });
    input.checked = current.flags.includes(flagId);
    return h("label", { class: "check" }, input, label);
  }));

  const text = h("textarea", {
    class: "fb-text",
    rows: "3",
    placeholder: "What went wrong, and what should have happened?",
    disabled: disabled || undefined,
    oninput: (e) => {
      if (editable()) {
        setDraft(kind, id, { note: e.target.value, expected: expected ? expected.value : null });
        markDirty();
      }
    },
  });
  text.value = current.note || "";

  /* Prefilled from the reference file when there is a disagreement, so the
     common case — confirming what the answer should have been — is no typing. */
  let expected = null;
  if (kind === "variable") {
    const verdict = hasTruth() ? verdictFor(App.variables.find((v) => v.item_id === Number(id))) : null;
    expected = h("input", {
      type: "text",
      class: "fb-input",
      placeholder: "the value it should have had",
      disabled: disabled || undefined,
      oninput: (e) => { if (editable()) { setDraft(kind, id, { expected: e.target.value }); markDirty(); } },
    });
    expected.value = App.feedbackDraft.has(key) || saved || current.expected != null
      ? current.expected ?? ""
      : (verdict && verdict.verdict !== "match" && verdict.verdict !== "untested" ? verdict.expected : "") || "";
  }

  const status = h("span", {
    class: App.feedbackErrors.has(key) ? "chip bad" : "chip",
    text: pending ? "saving..." : App.feedbackErrors.get(key) || "",
  });
  /* Seeded from the draft store, not from this render: navigating away and back
     rebuilds the form, and the edits survive that (currentAnnotation reads the
     draft) — so the "unsaved" marker has to survive it too, or the form would
     come back looking saved while holding unsaved edits. */
  const dirty = h("span", {
    class: "chip warn", text: "unsaved",
    hidden: !App.feedbackDraft.has(draftKey(kind, id)),
  });
  function markDirty() { dirty.hidden = false; status.textContent = ""; status.className = "chip"; }

  const save = h("button", {
    type: "button",
    class: "fb-save",
    text: "Save",
    disabled: disabled || undefined,
    /* Read the live controls rather than the draft. The draft only holds what
       someone has *edited*, and `expected` is prefilled from the reference file
       — so a reviewer who accepts the prefill (the common case, and the reason
       it is prefilled at all) never fires an input event, and a draft-sourced
       save would silently drop the value that is plainly on screen. */
    onclick: () => editable() && saveAnnotation(kind, id, {
      flags: currentAnnotation(kind, id).flags,
      expected: expected ? expected.value.trim() || null : null,
      note: text.value,
    }, status),
  });

  return section("Feedback",
    !App.feedbackWritable
      ? h("p", { class: "faint", style: "margin:0 0 10px" },
          App.feedbackReason)
      : null,
    boxes,
    expected ? h("div", { class: "fb-row" }, h("span", { class: "fb-label", text: "expected" }), expected) : null,
    text,
    h("div", { class: "fb-actions" },
      save,
      dirty,
      status,
      saved ? h("span", { class: "faint", style: "font-size:11px", text: "saved " + saved.updated_at }) : null));
}
