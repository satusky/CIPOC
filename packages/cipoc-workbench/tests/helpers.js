"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function descendants(node) {
  return [node, ...node.children.flatMap(descendants)];
}

// Only the selector vocabulary used by the Workbench, not a browser emulator.
function matches(node, selector) {
  if (node.tag.startsWith("#")) return false;
  const attrs = [...selector.matchAll(/\[([^\]=]+)(?:=["']?([^\]"']*)["']?)?\]/g)];
  let simple = selector.replace(/\[[^\]]+\]/g, "");
  for (const [, key, value] of attrs) {
    if (!node.hasAttribute(key) || (value !== undefined && node.getAttribute(key) !== value)) return false;
  }
  for (const state of ["checked", "disabled", "hidden"]) {
    if (simple.includes(":" + state)) {
      if (!node[state]) return false;
      simple = simple.replace(":" + state, "");
    }
  }
  const tag = simple.match(/^[\w-]+/);
  if (tag && node.tag !== tag[0].toLowerCase()) return false;
  const id = simple.match(/#([\w-]+)/);
  if (id && node.id !== id[1]) return false;
  for (const [, cls] of simple.matchAll(/\.([\w-]+)/g)) if (!node.classList.contains(cls)) return false;
  if (/[^\w#.*-]/.test(simple)) throw new Error("Unsupported harness selector: " + selector);
  return true;
}

function selectAll(root, selector) {
  const selectors = selector.match(/(?:\[[^\]]*\]|[^,])+/g)
    .map((s) => s.trim().match(/(?:\[[^\]]*\]|[^\s])+/g));
  return descendants(root).slice(1).filter((node) => selectors.some((parts) => {
    if (!matches(node, parts.at(-1))) return false;
    let parent = node.parentElement;
    for (let i = parts.length - 2; i >= 0; i--) {
      while (parent && !matches(parent, parts[i])) parent = parent.parentElement;
      if (!parent) return false;
      parent = parent.parentElement;
    }
    return true;
  }));
}

class FakeNode {
  constructor(tag = "div") {
    this.tag = tag.toLowerCase();
    this.tagName = this.tag.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.attributes = {};
    this.style = {};
    this.listeners = {};
    this.value = "";
    this.checked = false;
    this.files = [];
    const dataName = (key) => "data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
    this.dataset = new Proxy({}, {
      get: (_, key) => this.attributes[dataName(key)],
      set: (_, key, value) => { this.attributes[dataName(key)] = String(value); return true; },
      deleteProperty: (_, key) => { delete this.attributes[dataName(key)]; return true; },
    });
  }

  get parentNode() { return this.parentElement; }
  get firstChild() { return this.children[0] || null; }
  get childNodes() { return this.children; }
  get id() { return this.getAttribute("id") || ""; }
  set id(value) { this.setAttribute("id", value); }
  get className() { return this.getAttribute("class") || ""; }
  set className(value) { this.setAttribute("class", value); }
  get hidden() { return this.hasAttribute("hidden"); }
  set hidden(value) { value ? this.setAttribute("hidden", "") : this.removeAttribute("hidden"); }
  get disabled() { return this.hasAttribute("disabled"); }
  set disabled(value) { value ? this.setAttribute("disabled", "") : this.removeAttribute("disabled"); }
  get open() { return this.hasAttribute("open"); }
  set open(value) { value ? this.setAttribute("open", "") : this.removeAttribute("open"); }
  get isConnected() { return this.tag === "#document" || !!this.parentElement?.isConnected; }
  get classList() {
    const values = () => new Set(this.className.split(/\s+/).filter(Boolean));
    return {
      contains: (name) => values().has(name),
      add: (...names) => { this.className = [...new Set([...values(), ...names])].join(" "); },
      remove: (...names) => { this.className = [...values()].filter((v) => !names.includes(v)).join(" "); },
      toggle: (name, force) => {
        const set = values();
        const enabled = force ?? !set.has(name);
        enabled ? set.add(name) : set.delete(name);
        this.className = [...set].join(" ");
        return enabled;
      },
    };
  }
  append(...children) {
    for (const value of children) {
      const child = value instanceof FakeNode ? value : new FakeText(String(value));
      if (child.parentElement) child.parentElement.removeChild(child);
      child.parentElement = this;
      this.children.push(child);
    }
  }
  appendChild(child) { this.append(child); return child; }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index < 0) throw new Error("Not a child");
    this.children.splice(index, 1);
    child.parentElement = null;
    return child;
  }
  replaceChildren(...children) { this.textContent = ""; this.append(...children); }
  remove() { this.parentElement?.removeChild(this); }
  contains(node) { return descendants(this).includes(node); }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "checked") this.checked = true;
    if (name === "value") this.value = String(value);
    if (name === "style") for (const declaration of String(value).split(";")) {
      const [key, val] = declaration.split(":");
      if (key?.trim()) this.style[key.trim()] = val?.trim();
    }
  }
  getAttribute(name) { return Object.hasOwn(this.attributes, name) ? this.attributes[name] : null; }
  getAttributeNames() { return Object.keys(this.attributes); }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  removeEventListener(name, listener) { this.listeners[name] = (this.listeners[name] || []).filter((l) => l !== listener); }
  dispatchEvent(event) {
    event.target ||= this;
    event.currentTarget = this;
    event.preventDefault ||= () => { event.defaultPrevented = true; };
    event.stopPropagation ||= () => { event.cancelBubble = true; };
    const calls = [...(this.listeners[event.type] || [])];
    if (this["on" + event.type]) calls.push(this["on" + event.type]);
    this.eventResults = calls.map((listener) => listener(event));
    if (event.bubbles && !event.cancelBubble) this.parentElement?.dispatchEvent(event);
    return !event.defaultPrevented;
  }
  click() {
    if (this.disabled) return;
    if (this.tag === "input" && this.getAttribute("type") === "checkbox") this.checked = !this.checked;
    this.dispatchEvent({ type: "click", bubbles: true });
    if (this.tag === "input" && this.getAttribute("type") === "checkbox") this.dispatchEvent({ type: "change", bubbles: true });
  }
  focus() { this.dispatchEvent({ type: "focus" }); }
  showModal() { this.open = true; }
  close() { this.open = false; }
  getBoundingClientRect() { return { left: 0, top: 0, bottom: 20, width: 100, height: 20 }; }
  querySelectorAll(selector) { return selectAll(this, selector); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  matches(selector) { return matches(this, selector); }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
  get textContent() { return this.children.map((child) => child.textContent).join(""); }
  set textContent(value) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    if (String(value)) this.append(new FakeText(String(value)));
  }
}

class FakeText extends FakeNode {
  constructor(text) { super("#text"); this._text = text; }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value); }
}

const decode = (text) => text.replace(/&(#\d+|amp|lt|gt|quot|middot|hellip);/g, (_, entity) =>
  entity.startsWith("#") ? String.fromCodePoint(Number(entity.slice(1)))
    : ({ amp: "&", lt: "<", gt: ">", quot: '"', middot: "\u00b7", hellip: "\u2026" })[entity]);

function loadWorkbench(options = {}) {
  const web = path.join(__dirname, "..", "src", "cipoc_workbench", "web");
  const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
  const document = new FakeNode("#document");
  const stack = [document];
  const markup = html.replace(/<!--[\s\S]*?-->/g, "").replace(/(<script\b[^>]*>)[\s\S]*?<\/script>/g, "$1</script>");
  for (const [token] of markup.matchAll(/<![^>]*>|<[^>]+>|[^<]+/g)) {
    if (token.startsWith("<!")) continue;
    if (token.startsWith("</")) { stack.pop(); continue; }
    if (!token.startsWith("<")) { stack.at(-1).append(new FakeText(decode(token))); continue; }
    const tag = token.match(/^<([\w-]+)/)[1];
    const node = new FakeNode(tag);
    const attributes = token.slice(tag.length + 1).replace(/\/?\s*>$/, "");
    for (const [, key, double, single, bare] of attributes.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
      node.setAttribute(key, decode(double ?? single ?? bare ?? ""));
    }
    stack.at(-1).append(node);
    if (!["meta", "link", "input", "br", "hr", "img"].includes(tag) && !token.endsWith("/>")) stack.push(node);
  }
  document.documentElement = document.querySelector("html");
  document.body = document.querySelector("body");
  document.createElement = (tag) => new FakeNode(tag);
  document.createTextNode = (text) => new FakeText(String(text));
  document.getElementById = (id) => document.querySelector("#" + id);
  const requests = [], reads = [], alerts = [];
  const storage = options.storage || new Map();
  const fetch = options.fetch || ((url, init = {}) => new Promise((resolve, reject) => {
    const request = { url, options: init, reject, resolve: (body, status = 200) => resolve({
      ok: status >= 200 && status < 300, status, json: async () => body,
    }) };
    requests.push(request);
    if (!options.manualFetch) request.resolve({}, 404);
  }));
  class FileReader {
    readAsText(file) {
      const read = { file,
        resolve: (text = file.content) => { this.result = text; this.onload?.({ target: this }); },
        reject: () => this.onerror?.({ target: this }),
        abort: () => this.onabort?.({ target: this }),
      };
      reads.push(read);
      if (!options.manualFiles) queueMicrotask(() => read.resolve());
    }
  }
  const window = new FakeNode("window");
  Object.assign(window, { innerWidth: 1200, innerHeight: 800,
    matchMedia: () => ({ matches: false, addEventListener() {} }),
  });
  const context = vm.createContext({ console, document, window, Node: FakeNode, FileReader, fetch,
    setTimeout, clearTimeout, URL, alert: (message) => alerts.push(message),
    localStorage: { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, String(value)) },
  });
  const scripts = [];
  for (const [, attrs, body] of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
    const src = attrs.match(/\bsrc="([^"]+)"/);
    if (src) scripts.push(src[1]);
    vm.runInContext(src ? fs.readFileSync(path.join(web, src[1]), "utf8") : body,
      context, { filename: src ? src[1] : "index.html inline script" });
  }
  return { context, document, scripts, requests, reads, alerts, storage,
    evaluate: (code) => vm.runInContext(code, context),
  };
}

function canonicalResult(schemaVersion = "1.0") {
  const note = (noteId, content) => ({
    note_id: noteId, date: "2026-09-04", note_type: "Pathology", content,
    summary: "Breast cancer pathology.",
    concepts: { cancer: { presence: true, confidence: "high", evidence: [{ note_id: noteId, text: content }] } },
    cancer_status: ["current"], cancer_mentions: [], flags: ["breast"],
  });
  const exchange = (entityKey, agent, node) => ({
    entity_key: entityKey, agent, node, attempt: 1, retry_ordinal: null, model: "test-model",
    prompt_messages: null, response: null,
    usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 }, error: null,
  });
  return {
    schema_version: schemaVersion,
    run: {
      run_id: "123e4567-e89b-42d3-a456-426614174000",
      started_at: "2026-09-04T12:00:00Z", finished_at: "2026-09-04T12:00:02Z",
      duration_seconds: 2, status: "completed", config_fingerprint: {}, contains_phi: true,
    },
    case: {
      case_facts: { gross_primary_site: "breast" },
      variable_results: { 400: {
        item_id: 400, status: "extracted", value: "C504",
        extraction: {
          item_id: 400, value: "C504", explanation: "Upper-outer quadrant.", most_important_note: "note-2",
          spans: [{ note_id: "note-2", text: "Upper-outer quadrant" }], presence_confidence: "high",
          validation_errors: [], is_valid: true, extraction_attempts: 1,
        }, reason: null, blocking_item_ids: [],
      } },
      note_selection: { "group:diagnosis": {
        group_id: "diagnosis", requested_item_ids: [400], candidate_note_ids: ["note-10", "note-2"],
        rejected_note_ids: { "note-A": ["note_type_mismatch", "cancer_status_mismatch"] },
        selected_note_ids: ["note-2"], discarded_note_ids: ["invented-X"],
        unevaluated_checks: ["keyword_filter_disabled", "temporal_anchor_unavailable"],
      } },
      fatal_blocker: null, report: { flags: [] },
    },
    inputs: {
      target_variables: [{
        group_id: "diagnosis", name: "Diagnosis", extract_as_group: true, stage: "initial",
        gate: ["treatment_present"], applies_to: null,
        note_filter: { note_types: ["Pathology"], keywords: ["breast"], cancer_status: [], within_days: 30 },
        variables: [{ item_id: 400, name: "Primary Site" }],
      }], structured_data: {},
    },
    corpus: {
      note_corpus: {
        "note-2": note("note-2", "Upper-outer quadrant"), "note-A": note("note-A", "Other note"),
        "note-10": note("note-10", "Earlier lexical identifier"),
      },
      note_digests: { "note-2": { note_id: "note-2", note_type: "Pathology", summary: "Digest" } },
      note_corpus_descriptors: { note_count: 3, concepts: { surgery: { presence: true } } },
    },
    observability: {
      ...(schemaVersion !== "1.0" ? { collection_status: "complete", collection_issues: [], unattributed_exchanges: [] } : {}),
      llm_content_captured: false, max_content_chars: null, content_truncated: false,
      variable_attempts: { "group:diagnosis/variable:400": [{
        attempt: 1, mode: "group", candidate: { value: "C504", presence_confidence: "high", explanation: "Candidate" },
        validation_errors: [], is_valid: true,
      }] },
      llm_exchanges: {
        "note:note-2": [exchange("note:note-2", "note_scanner", "detect_concepts")],
        "group:diagnosis": [exchange("group:diagnosis", "note_retriever", "identify_relevant_notes"),
          exchange("group:diagnosis", "extractor", "extract_group_values")],
        "group:diagnosis/variable:400": [exchange("group:diagnosis/variable:400", "extractor", "extract_variable")],
      },
      llm_usage_summary: {
        input_tokens: 40, output_tokens: 8, total_tokens: 48, logical_calls: 4, model_invocations: 4,
        successful_invocations: 4, failed_invocations: 0, retry_invocations: 0,
        usage_reported_invocations: 4, missing_usage_invocations: 0, by_agent: {}, by_node: {}, by_model: {},
      },
    },
  };
}

module.exports = { loadWorkbench, canonicalResult, descendants };
