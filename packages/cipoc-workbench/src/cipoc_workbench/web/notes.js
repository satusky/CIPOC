"use strict";
/* Notes view — one row per scanned note, organised by date and type, showing
 * the scan results the note carries. */

function noteMatches(note) {
  const q = App.noteFilter;
  if (!q) return true;
  return (
    String(note.note_id).includes(q) ||
    (note.note_type || "").toLowerCase().includes(q) ||
    (note.date || "").includes(q) ||
    (note.summary || "").toLowerCase().includes(q) ||
    (note.content || "").toLowerCase().includes(q) ||
    (note.flags || []).some((f) => f.toLowerCase().includes(q))
  );
}

function presentConcepts(note) {
  return Object.entries(note.concepts || {}).filter(([, c]) => c.presence);
}

const noteTemporality = (note) => {
  const statuses = note.cancer_status || [];
  if (statuses.includes("current")) return "current";
  if (statuses.includes("recent")) return "recent";
  if (statuses.includes("historical")) return "historical";
  return "";
};

function sortedNotes() {
  return [...App.notes.values()].sort((a, b) =>
    String(a.date || "").localeCompare(String(b.date || "")) || (a.note_id - b.note_id));
}

function noteRow(note) {
  const concepts = presentConcepts(note);
  const mentions = note.cancer_mentions || [];
  const temporality = noteTemporality(note);
  const citing = variablesCitingNote(note.note_id).length;

  return h("li", {},
    h("button", {
      type: "button",
      class: "note-row " + temporality,
      dataset: { entity: "note:" + note.note_id,
                 annotated: isAnnotated("note", note.note_id) ? "1" : null },
      onclick: () => select("note", note.note_id),
    },
      h("div", {},
        h("div", { class: "note-date", text: note.date || "undated" }),
        h("div", { class: "faint", style: "font-size:11px;font-family:var(--mono)", text: "#" + note.note_id })),
      h("div", { class: "note-main" },
        h("h3", { text: note.note_type || "Note" }),
        h("p", { class: "note-summary", text: note.summary || "No summary recorded." })),
      h("div", { class: "note-meta" },
        temporality ? h("span", { class: "chip on", text: temporality }) : h("span", { class: "chip", text: "no cancer" }),
        h("span", { class: "chip", text: concepts.length + " concept" + (concepts.length === 1 ? "" : "s") }),
        h("span", { class: "chip", text: mentions.length + " mention" + (mentions.length === 1 ? "" : "s") }),
        citing ? h("span", { class: "chip good", text: citing + " cited by" }) : null)
    ));
}

function renderNotes() {
  const list = clear($("#notes-list"));
  const notes = sortedNotes().filter(noteMatches);
  if (!notes.length) {
    list.append(h("li", {}, h("p", { class: "empty", text: "No notes match this filter." })));
    return;
  }
  for (const note of notes) list.append(noteRow(note));
  markSelection();
}
