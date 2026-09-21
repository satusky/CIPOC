const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const web = path.join(__dirname, '../src/cipoc/demo/web');
const app = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const entitySelector = '[data-entity-kind][data-entity-id]';
const groupId = 'g:one/"[x]';
const noteId = 'note:51/"[x]';
const itemId = '400:part/"[x]';
const scope = 'extract_branch:pass:two';
const escape = (value) => String(value).replace(/[&<>"']/g, (c) =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const unescape = (value) => value.replace(/&(amp|lt|gt|quot|#39);/g, (_, key) =>
  ({amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'"}[key]));

// A small DOM harness for delegated events and focus after innerHTML replacement.
// Layout/geometry is tested with the real vendored Cytoscape, not this DOM stub.
class Element {
  constructor(document, tag = 'div', attributes = {}) {
    this.document = document;
    this.tagName = tag.toUpperCase();
    this.attributes = {};
    this.dataset = {};
    this.children = [];
    this.listeners = new Map();
    this._html = '';
    this.textContent = '';
    this.disabled = false;
    this.value = '';
    this.classList = {toggle() {}};
    this.style = {setProperty() { throw new Error('card navigation changed layout'); }};
    for (const [key, value] of Object.entries(attributes)) this.setAttribute(key, value);
  }
  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key === 'id') this.id = String(value);
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
  }
  hasAttribute(key) { return Object.hasOwn(this.attributes, key); }
  removeAttribute(key) { delete this.attributes[key]; }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(value) { if (value) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }
  get isConnected() { return this === this.document.body || !!this.parentElement?.isConnected; }
  get isContentEditable() { return this.attributes.contenteditable === 'true' || !!this.parentElement?.isContentEditable; }
  append(child) { child.parentElement = this; this.children.push(child); return child; }
  contains(child) { return !!child && (child === this || this.children.some((c) => c.contains(child))); }
  matches(selector) {
    return selector.split(',').some((part) => {
      part = part.trim();
      if (part.includes(':not(')) {
        if (this.attributes.contenteditable === 'false') return false;
        part = part.replace(/:not\(.*\)/, '');
      }
      if (part.startsWith('.')) return (this.attributes.class || '').split(' ').includes(part.slice(1));
      const tag = part.match(/^[a-z]+/i)?.[0];
      if (tag && this.tagName !== tag.toUpperCase()) return false;
      const attrs = [...part.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)];
      return (!!tag || attrs.length > 0) && attrs.every(([, key, value]) =>
        this.hasAttribute(key) && (value === undefined || this.attributes[key] === value));
    });
  }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
  querySelectorAll(selector) {
    return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(callback);
  }
  focus(options) {
    this.document.activeElement = this;
    this.document.focuses.push({element: this, options});
  }
  get innerHTML() { return this._html; }
  set innerHTML(html) {
    if (this.contains(this.document.activeElement)) this.document.activeElement = this.document.body;
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this._html = html;
    // Only native controls need nodes for these tests; preserve their attributes
    // exactly as a browser would after decoding escaped HTML attribute values.
    for (const [, tag, attrs] of html.matchAll(/<(button|a|input)\b([^>]*)>/g)) {
      const attributes = {};
      for (const [, key, value] of attrs.matchAll(/([\w-]+)="([^"]*)"/g)) attributes[key] = unescape(value);
      this.append(new Element(this.document, tag, attributes));
    }
  }
}

function browser({viewer = false, cards = true} = {}) {
  const document = {
    listeners: new Map(), focuses: [],
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(callback);
    },
    getElementById(id) { return this.body.querySelectorAll('[id]').find((el) => el.id === id) || null; },
  };
  document.body = new Element(document, 'body');
  document.activeElement = document.body;
  const ids = ['run-title', 'run-sub', 'mode-badge', 'btn-prev', 'btn-next', 'btn-play',
    'step-select', 'step-counter', 'btn-replay-step', 'map-scrub', 'map-tip', 'detail',
    'detail-node', 'detail-heading', 'detail-window', 'detail-back', 'detail-close', 'vars', 'vars-summary',
    'omop-modal', 'omop-title', 'omop-sub', 'omop-body', 'omop-close', 'omop-back', 'run-issues'];
  for (const id of ids) {
    const tag = /^(btn-|detail-back|detail-close|omop-close)/.test(id) ? 'button' : 'div';
    const el = document.body.append(new Element(document, tag, {id}));
    if (['omop-modal', 'detail-window', 'detail-back', 'detail-close'].includes(id)) el.hidden = true;
  }
  const calls = [];
  const requests = [];
  const context = vm.createContext({
    document, console, URLSearchParams,
    window: {location: {search: viewer ? '?viewer=1' : ''}},
    setTimeout: () => 1, clearTimeout() {},
    requestAnimationFrame: () => 1, cancelAnimationFrame() {}, performance: {now: () => 0},
    fetch: async (url, options) => { requests.push({url, options}); return new Response('{}'); },
  });
  if (cards) context.DemoCards = {
    render(selection, snapshot, notes, catalog) {
      calls.push({selection, snapshot, notes, catalog});
      if (!selection) return '';
      const link = selection.kind === 'group'
        ? {kind: 'variable', id: itemId, groupId, instanceKey: scope}
        : selection.kind === 'variable' ? {kind: 'note', id: noteId} : null;
      return `<article><p>${escape(snapshot.phase || 'pending')} · ${escape(selection.id)}</p>${link ?
        `<button type="button" data-entity-kind="${link.kind}" data-entity-id="${escape(link.id)}"${link.groupId ?
          ` data-group-id="${escape(link.groupId)}" data-instance-key="${escape(link.instanceKey)}"` : ''}>Open</button>` : ''}</article>`;
    },
  };
  vm.runInContext(app, context);
  const run = (source) => vm.runInContext(source, context);
  run('cacheEls(); wireControls(); wireDetailCards(); wireOmopModal();');
  const get = (id) => document.getElementById(id);
  const dispatch = (type, target, extra = {}) => {
    const event = {type, target, defaultPrevented: false, ...extra,
      preventDefault() { this.defaultPrevented = true; }};
    for (let el = target; el; el = el.parentElement) {
      for (const callback of el.listeners.get(type) || []) callback(event);
    }
    for (const callback of document.listeners.get(type) || []) callback(event);
    return event;
  };
  const click = (target) => dispatch('click', target);
  const key = (value, target = document.activeElement) => dispatch('keydown', target, {key: value});
  const selection = () => JSON.parse(run('JSON.stringify(detailSelection)'));
  const controls = (id) => get(id).querySelectorAll(entitySelector);
  const view = (snapshot, cursor = 2) => ({cursor, playing: true, follow_live: false,
    step: {index: cursor, title: cursor ? 'Extract groups' : 'Initialize case', node: 'extract_branch',
      start_seq: cursor * 10, end_seq: snapshot.seq}, snapshot});
  const setView = (value) => { context.nextView = value; run('applyView(nextView)'); };
  // Layout and timing functions are tripwires during local navigation. applyView
  // still follows its normal render path; separate map tests install real nodes.
  run(`syncMap = () => {}; playStep = () => {}; settleStep = () => {};
    fitMap = refitMap = computeLayout = setVarsHeight = () => { throw new Error('unexpected geometry work'); };`);
  return {context, document, calls, requests, run, get, click, key, selection, controls, view, setView};
}

function snapshot(seq = 29, phase = 'done') {
  return {seq, phase, instances: {[scope]: {key: scope, node: 'extract_branch', input: {
    requested_variables: {group_id: groupId}}, result: {}}}, details: {}, progress: {
    totals: {terminal: 0, variables: 1, done_groups: 0, groups: 1}, notes_done: 0, notes_total: 1,
    groups: [{group_id: groupId, name: '<Group>', stage: 'extracting', annotation: 'gate: future?'}],
    variables: [{item_id: itemId, group_id: groupId, name: '<Variable>', stage: 'pending', value: null}],
  }};
}

function mapHarness(t) {
  const h = browser();
  h.setView(h.view(snapshot()));
  vm.runInContext(fs.readFileSync(path.join(web, 'cytoscape.min.js'), 'utf8'), h.context);
  h.context.noteId = noteId;
  h.context.groupId = groupId;
  h.context.itemId = itemId;
  h.run(`const testTheme = Object.fromEntries([
      'ok','warn','err','line','muted','navy','ink','inkMuted','panel','sunk','errSoft','lineSoft',
      'scanner','extractor','retriever','orchestrator','scannerInk'].map(key => [key, '#334455']));
    testTheme.font = 'sans-serif';
    mapIndex.notes = [{id: 'note:' + noteId, noteId, type: 'Pathology'}];
    cy = cytoscape({headless: true, styleEnabled: true,
      style: [...mapStyle(testTheme), ...stateStyles(testTheme)]});
    const model = buildMapModel(lastView.snapshot);
    cy.add(model.nodes.map((node, i) => ({...node, position: {x: i * 100, y: i % 2 * 100}})));
    cy.add(model.edges);
    wireMapCards();
    cy.fit = cy.resize = cy.layout = () => { throw new Error('card navigation moved map'); };
    stepAnim = 77; stepFallback = 88; layoutKey = 'unchanged';`);
  h.tap = (id) => { h.context.tapId = id; h.run('cy.getElementById(tapId).emit("tap")'); };
  t.after(() => h.run('cy.destroy()'));
  return h;
}

test('map discs select exact entities, including compound bubbling and special IDs', (t) => {
  const h = mapHarness(t);
  for (const [node, expected] of [
    [`note:${noteId}`, {kind: 'note', id: noteId}],
    [`grp:${groupId}`, {kind: 'group', id: groupId}],
    [`gate:${groupId}`, {kind: 'group', id: groupId}],
    [`var:${itemId}`, {kind: 'variable', id: itemId, groupId}],
  ]) {
    const before = h.calls.length;
    h.tap(node);
    assert.deepEqual(h.selection(), expected);
    assert.equal(h.calls.length, before + 1, 'compound parent must not open a second card');
    assert.equal(h.calls.at(-1).snapshot, h.run('lastView.snapshot'));
  }
  for (const node of ['stage:case', 'stage:corpus', 'stage:corpus:mark']) {
    h.tap(`var:${itemId}`);
    h.tap(node);
    assert.equal(h.selection(), null);
    assert.equal(h.get('detail').innerHTML, '');
    assert.equal(h.get('detail-window').hidden, true);
    assert.equal(h.get('detail-node').textContent, 'Step 3 · Extract groups');
  }
  h.tap(`note:${noteId}`);
  h.run('cy.emit("tap")');
  assert.equal(h.selection(), null);
  assert.deepEqual(h.requests, []);
  assert.equal(h.run('lastView.cursor'), 2);
  assert.equal(h.run('lastView.playing'), true);
  assert.equal(h.run('stepAnim'), 77);
  assert.equal(h.run('stepFallback'), 88);
});

test('selection highlight survives animation and close without resizing, fitting, or repacking', (t) => {
  const h = mapHarness(t);
  const geometry = () => h.run(`JSON.stringify({nodes: cy.nodes().map(n =>
    [n.id(), n.position(), n.width(), n.height()]), pan: cy.pan(), zoom: cy.zoom(), layoutKey})`);
  h.run('renderMapAt(0, lastView.snapshot, lastView.step)');
  const before = geometry();
  h.tap(`var:${itemId}`);
  assert.equal(h.run('cy.getElementById("var:" + itemId).hasClass("entity-selected")'), true);
  assert.equal(geometry(), before);
  h.run('renderMapAt(0, lastView.snapshot, lastView.step)');
  assert.equal(h.run('cy.getElementById("var:" + itemId).hasClass("entity-selected")'), true);
  assert.equal(h.run('cy.getElementById("var:" + itemId).hasClass("st-idle")'), true);
  h.click(h.get('detail-close'));
  assert.equal(h.run('cy.nodes(".entity-selected").length'), 0);
  assert.equal(geometry(), before);
  h.tap(`gate:${groupId}`);
  assert.equal(h.run('cy.nodes(".entity-selected").length'), 2);
  assert.equal(geometry(), before);
});

test('within-step scrubbing animates the map without changing the step-end card', (t) => {
  const h = mapHarness(t);
  h.tap(`var:${itemId}`);
  const html = h.get('detail').innerHTML;
  const renders = h.calls.length;
  h.run(`events = [{seq: 20, t: 10, namespace: []}, {seq: 29, t: 19, namespace: []}];
    seekStep(0);`);
  assert.equal(h.run('lastRenderT'), 10);
  assert.equal(h.get('detail').innerHTML, html);
  assert.equal(h.calls.length, renders);
  assert.equal(h.calls.at(-1).snapshot.seq, 29);
  assert.equal(h.run('lastView.cursor'), 2);
  assert.deepEqual(h.requests, []);
});

test('table names and group headings are escaped native buttons; OMOP remains independent', () => {
  const h = browser();
  h.setView(h.view(snapshot()));
  const [group, variable] = h.controls('vars');
  assert.equal(group.tagName, 'BUTTON');
  assert.equal(variable.tagName, 'BUTTON');
  assert.match(h.get('vars').innerHTML, /&lt;Group&gt;/);
  assert.match(h.get('vars').innerHTML, /&lt;Variable&gt;/);
  h.click(group);
  assert.deepEqual(h.selection(), {kind: 'group', id: groupId});
  const nested = variable.append(new Element(h.document, 'span'));
  h.click(nested);
  assert.deepEqual(h.selection(), {kind: 'variable', id: itemId, groupId});
  h.run('var previews = []; openOmopModal = (...args) => previews.push(args); wireDetailCards();');
  const renders = h.calls.length;
  h.click(h.get('vars').querySelectorAll('.omop-btn')[0]);
  assert.equal(h.calls.length, renders);
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(previews)')), [[itemId, '<Variable>']]);
  assert.equal(h.get('detail').listeners.get('click').length, 1);
  assert.equal(h.get('detail').listeners.has('toggle'), false);
  assert.deepEqual(h.requests, []);
});

test('group → scoped variable → evidence note supports Back and Close restores the replaced origin', () => {
  const h = browser();
  h.setView(h.view(snapshot()));
  const origin = h.controls('vars')[0];
  assert.equal(h.get('detail-window').hidden, true);
  origin.focus();
  h.click(origin);
  assert.equal(h.get('detail-window').hidden, false);
  assert.equal(h.get('detail-close').hidden, false);
  assert.equal(h.get('detail-back').hidden, true);
  h.click(h.controls('detail')[0]);
  assert.deepEqual(h.selection(), {kind: 'variable', id: itemId, groupId, instanceKey: scope});
  h.click(h.controls('detail')[0]);
  assert.deepEqual(h.selection(), {kind: 'note', id: noteId});
  assert.equal(h.get('detail-back').hidden, false);
  h.click(h.get('detail-back'));
  assert.equal(h.selection().instanceKey, scope);
  h.click(h.get('detail-back'));
  assert.deepEqual(h.selection(), {kind: 'group', id: groupId});
  assert.equal(h.get('detail-back').hidden, true);
  const next = snapshot(30);
  next.progress.variables[0].value = 'updated';
  h.setView(h.view(next));
  assert.equal(origin.isConnected, false);
  h.click(h.get('detail-close'));
  assert.equal(h.selection(), null);
  assert.equal(h.get('detail-window').hidden, true);
  assert.equal(h.get('detail-close').hidden, true);
  assert.equal(h.document.activeElement, h.controls('vars')[0]);
  assert.equal(h.document.focuses.at(-1).options.preventScroll, true);
  assert.deepEqual(h.requests, []);
});

test('replay retains entity identity but clears pass/history, supplying only current snapshot and identity catalog', () => {
  const h = browser();
  const later = snapshot();
  h.setView(h.view(later));
  h.click(h.controls('vars')[0]);
  h.click(h.controls('detail')[0]);
  h.run(`mapIndex.notes = [{id: 'note:future', noteId: 'future', type: 'Pathology', result: 'FUTURE'}];
    mapIndex.groups.set('future', {id: 'future', name: 'Future group', annotation: 'FUTURE',
      variables: [{itemId: 400, name: 'Site', description: 'FUTURE', value: 'FUTURE'}]});
    notesById = {future: {content: 'source note', summary: 'FUTURE'}};`);
  const earlier = {seq: 1, phase: 'pending', instances: {}, progress: null,
    details: {initialize_case: {llm_calls: [{prompt_messages: [{content: 'RAW PROMPT'}]}]}}};
  const focusCount = h.document.focuses.length;
  h.setView(h.view(earlier, 0));
  assert.deepEqual(h.selection(), {kind: 'variable', id: itemId, groupId});
  assert.equal(h.get('detail-back').hidden, true);
  assert.equal(h.calls.at(-1).snapshot, earlier);
  assert.match(h.get('detail').innerHTML, /pending/);
  assert.doesNotMatch(h.get('detail').innerHTML, /FUTURE|RAW PROMPT|done/);
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls.at(-1).catalog)), {
    notes: [{id: 'note:future', noteId: 'future', type: 'Pathology'}],
    groups: [{id: 'future', name: 'Future group', variables: [{itemId: '400', name: 'Site'}]}],
  });
  assert.equal(h.document.focuses.length, focusCount, 'snapshot updates do not steal focus');
  h.click(h.get('detail-back'));
  assert.equal(h.selection().instanceKey, undefined);
  h.setView(h.view(snapshot(2, 'active'), 0));
  assert.match(h.get('detail').innerHTML, /active/);
  assert.deepEqual(h.requests, []);
});

test('same-step live updates preserve scope and focused links; a rewind drops stale pass bindings', () => {
  const h = browser();
  h.setView(h.view(snapshot()));
  h.click(h.controls('vars')[0]);
  h.click(h.controls('detail')[0]);
  const link = h.controls('detail')[0];
  link.focus();
  h.setView(h.view(snapshot(30, 'active')));
  assert.equal(h.selection().instanceKey, scope);
  assert.equal(h.document.activeElement, h.controls('detail')[0]);
  assert.notEqual(h.document.activeElement, link);
  const count = h.document.focuses.length;
  h.run('renderDetail(lastView)');
  assert.equal(h.document.focuses.length, count, 'identical refresh leaves DOM/focus intact');
  h.setView(h.view(snapshot(28, 'pending')));
  assert.equal(h.selection().instanceKey, undefined);
  assert.equal(h.get('detail-back').hidden, true);
});

test('a pending notes fetch rerenders the current selection/snapshot, including a closed card', async () => {
  const h = browser();
  h.setView(h.view(snapshot()));
  h.click(h.controls('vars')[0]);
  let finish;
  h.context.fetch = (url) => {
    h.requests.push({url});
    return new Promise((resolve) => { finish = (notes) => resolve({ok: true, json: async () => notes}); });
  };
  const fetching = h.run('ensureNotes()');
  h.click(h.controls('detail')[0]);
  h.setView(h.view(snapshot(1, 'pending'), 0));
  const focusCount = h.document.focuses.length;
  finish({[noteId]: {content: 'Source text'}});
  await fetching;
  assert.equal(h.calls.at(-1).selection.kind, 'variable');
  assert.equal(h.calls.at(-1).snapshot.seq, 1);
  assert.equal(h.calls.at(-1).notes[noteId].content, 'Source text');
  assert.equal(h.document.focuses.length, focusCount);
  const again = h.run('ensureNotes()');
  h.click(h.get('detail-close'));
  finish({[noteId]: {content: 'Source text'}});
  await again;
  assert.equal(h.calls.at(-1).selection, null);
  assert.equal(h.get('detail').innerHTML, '');
  assert.equal(h.get('detail-window').hidden, true);
  assert.deepEqual(h.requests, [{url: '/api/notes'}, {url: '/api/notes'}]);
});

test('interactive controls keep native keys; global replay keys work only outside them', () => {
  const h = browser();
  h.setView(h.view(snapshot()));
  const targets = [h.controls('vars')[0], h.controls('vars')[1], h.get('detail-close')];
  for (const tag of ['input', 'select', 'textarea', 'button', 'summary']) {
    targets.push(h.document.body.append(new Element(h.document, tag)));
  }
  for (const attrs of [{contenteditable: 'true'}, {contenteditable: ''}, {role: 'button'}]) {
    targets.push(h.document.body.append(new Element(h.document, 'div', attrs)).append(new Element(h.document, 'span')));
  }
  targets.push(h.document.body.append(new Element(h.document, 'a', {href: '#'})));
  for (const target of targets) {
    for (const value of ['Enter', ' ', 'ArrowLeft', 'ArrowRight']) {
      assert.equal(h.key(value, target).defaultPrevented, false, `${target.tagName} ${value}`);
    }
  }
  assert.deepEqual(h.requests, []);
  for (const value of ['ArrowLeft', 'ArrowRight', ' ']) assert.equal(h.key(value, h.document.body).defaultPrevented, true);
  assert.deepEqual(h.requests.map(({url}) => url), ['/api/prev', '/api/next', '/api/pause']);
});

test('Escape closes OMOP before the card, and viewers can navigate cards without control requests', () => {
  const h = browser({viewer: true});
  h.setView(h.view(snapshot()));
  const origin = h.controls('vars')[1];
  h.click(origin);
  h.get('omop-modal').hidden = false;
  assert.equal(h.key('Escape', h.get('omop-close')).defaultPrevented, true);
  assert.equal(h.get('omop-modal').hidden, true);
  assert.equal(h.selection().kind, 'variable');
  assert.equal(h.key('Escape', h.get('detail-close')).defaultPrevented, true);
  assert.equal(h.selection(), null);
  assert.equal(h.document.activeElement, origin);
  for (const value of ['ArrowLeft', 'ArrowRight', ' ']) h.key(value, h.document.body);
  assert.deepEqual(h.requests, []);
});

test('missing cards.js keeps the empty stage blank and reports unavailable selected details', () => {
  const h = browser({cards: false});
  h.run(`renderExtractDetail = renderFanoutDetail = renderNodeDetail = () => {
    throw new Error('old detail renderer is active');
  };`);
  const snap = snapshot();
  snap.details.extractor_extract_group_values = {llm_calls: [{prompt_messages: [{content: 'RAW PROMPT'}]}]};
  h.setView(h.view(snap));
  assert.equal(h.get('detail').innerHTML, '');
  assert.equal(h.get('detail-window').hidden, true);
  assert.equal(h.get('detail-heading').textContent, 'Details');
  h.click(h.controls('vars')[1]);
  assert.match(h.get('detail').innerHTML, /Entity details are unavailable/);
  assert.doesNotMatch(h.get('detail').innerHTML, /RAW PROMPT|Model call|pinned component/);
});

test('real DemoCards links bind a variable pass and evidence note, then replay clears future output', () => {
  const h = browser({cards: false});
  vm.runInContext(fs.readFileSync(path.join(web, 'cards.js'), 'utf8'), h.context);
  const later = snapshot();
  const variable = {item_id: itemId, name: 'Site', description: 'FUTURE DEFINITION'};
  const variableKey = `${scope}/extract:e/variable_branch:v1`;
  Object.assign(later.instances[scope], {active: 0, status: 'done', input: {
    requested_variables: {group_id: groupId, name: 'Site group', variables: [variable]},
  }});
  later.instances[variableKey] = {key: variableKey, node: 'variable_branch', active: 0, status: 'done',
    input: {task: {variable}}, result: {variable_results: [{item_id: itemId, value: 'C509',
      is_valid: true, explanation: 'FUTURE EXPLANATION', presence_confidence: 'high',
      spans: [{note_id: noteId, text: 'cancer'}]}]}};
  later.instances['note_branch:n1'] = {key: 'note_branch:n1', node: 'note_branch', active: 0, status: 'done',
    input: {note_id: noteId, note_type: 'Pathology', date: '2020-01-01', content: 'A cancer finding.'},
    result: {summary: 'FUTURE SUMMARY', concepts: {}, cancer_mentions: []}};
  later.details.extractor_extract_group_values = {llm_calls: [{prompt_messages: [{content: 'RAW PROMPT'}]}]};
  Object.assign(later.progress.variables[0], {terminal: true, status: 'extracted', value: 'C509'});
  h.context.sourceNotes = {[noteId]: {note_id: noteId, content: 'A cancer finding.', summary: 'RAW FUTURE SUMMARY'}};
  h.run('notesById = sourceNotes;');
  h.setView(h.view(later));
  h.click(h.controls('vars')[0]);
  const tile = h.controls('detail').find((el) => el.dataset.entityKind === 'variable');
  assert.equal(tile.dataset.instanceKey, variableKey);
  h.click(tile);
  assert.deepEqual(h.selection(), {kind: 'variable', id: itemId, groupId, instanceKey: variableKey});
  assert.match(h.get('detail').innerHTML, /C509/);
  assert.match(h.get('detail').innerHTML, /FUTURE DEFINITION/);
  assert.match(h.get('detail').innerHTML, /FUTURE EXPLANATION/);
  const evidence = h.controls('detail').find((el) => el.dataset.entityKind === 'note');
  h.click(evidence);
  assert.deepEqual(h.selection(), {kind: 'note', id: noteId});
  assert.match(h.get('detail').innerHTML, /FUTURE SUMMARY/);
  assert.doesNotMatch(h.get('detail').innerHTML, /RAW FUTURE SUMMARY|RAW PROMPT/);
  h.click(h.get('detail-back'));
  const early = snapshot(1, 'pending');
  early.instances = {};
  early.progress.variables[0].status = 'pending';
  h.setView(h.view(early, 0));
  assert.deepEqual(h.selection(), {kind: 'variable', id: itemId, groupId});
  assert.match(h.get('detail').innerHTML, /Pending/);
  assert.match(h.get('detail').innerHTML, /Definition not recorded/);
  assert.doesNotMatch(h.get('detail').innerHTML, /C509|FUTURE|RAW PROMPT/);
  h.context.noteSelection = {kind: 'note', id: noteId};
  h.run('openEntityCard(noteSelection)');
  assert.match(h.get('detail').innerHTML, /Pending/);
  assert.doesNotMatch(h.get('detail').innerHTML, /FUTURE|RAW PROMPT/);
  h.setView(h.view(later));
  assert.match(h.get('detail').innerHTML, /FUTURE SUMMARY/);
  assert.equal(h.selection().id, noteId);
  assert.deepEqual(h.requests, []);
});
