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
  return (App.feedback[kind] || {})[String(id)] || null;
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
  App.feedbackDraft.set(key, { ...currentAnnotation(kind, id), ...patch });
}

/* ------------------------------------------------------------------ load */

function indexFeedback(document) {
  const annotations = (document || {}).annotations || {};
  App.feedback = {
    variable: annotations.variable || {},
    group: annotations.group || {},
    note: annotations.note || {},
  };
}

/* Probing at boot rather than at save time: the form has to know whether it can
 * promise anything before someone starts typing into it. */
async function loadFeedback() {
  try {
    const response = await fetch("api/feedback", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    indexFeedback(await response.json());
    App.feedbackWritable = true;
  } catch (err) {
    indexFeedback(null);
    App.feedbackWritable = false;
  }
}

async function saveAnnotation(kind, id, annotation, status) {
  status.className = "chip";
  status.textContent = "saving…";
  try {
    const response = await fetch("api/feedback/" + kind + "/" + encodeURIComponent(id), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotation),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || "HTTP " + response.status);
    }
    const saved = (await response.json()).annotation;
    if (saved) App.feedback[kind][String(id)] = saved;
    else delete App.feedback[kind][String(id)];

    App.feedbackDraft.delete(draftKey(kind, id));
    /* The full re-render replaces this node, so the status chip below is only
       ever seen on failure — which is the case that must not be silent. */
    renderChrome();
    render();
    renderDetail();
  } catch (err) {
    status.className = "chip bad";
    status.textContent = "not saved — " + err.message;
  }
}

/* ------------------------------------------------------------------ form */

function feedbackSection(kind, id) {
  const flags = FEEDBACK_FLAGS[kind];
  if (!flags) return null;

  const current = currentAnnotation(kind, id);
  const saved = annotationFor(kind, id);
  const disabled = !App.feedbackWritable;

  const boxes = h("div", { class: "flag-grid" }, flags.map(([flagId, label]) => {
    const input = h("input", {
      type: "checkbox",
      disabled: disabled || undefined,
      onchange: (e) => {
        const next = new Set(currentAnnotation(kind, id).flags);
        if (e.target.checked) next.add(flagId);
        else next.delete(flagId);
        setDraft(kind, id, { flags: [...next] });
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
    oninput: (e) => { setDraft(kind, id, { note: e.target.value }); markDirty(); },
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
      oninput: (e) => { setDraft(kind, id, { expected: e.target.value }); markDirty(); },
    });
    expected.value = current.expected != null
      ? current.expected
      : (verdict && verdict.verdict !== "match" && verdict.verdict !== "untested" ? verdict.expected : "") || "";
  }

  const status = h("span", { class: "chip" });
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
    onclick: () => saveAnnotation(kind, id, {
      flags: currentAnnotation(kind, id).flags,
      expected: expected ? expected.value.trim() || null : null,
      note: text.value,
    }, status),
  });

  return section("Feedback",
    disabled
      ? h("p", { class: "faint", style: "margin:0 0 10px" },
          "Read-only: no workbench server. Start one with ",
          h("code", { class: "mono", text: "cipoc-workbench serve --feedback FILE" }),
          " to save annotations.")
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
