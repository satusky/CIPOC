const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function browser(search = '') {
  const timers = [];
  const elements = new Map();
  const context = vm.createContext({
    document: { addEventListener() {}, getElementById(id) {
      if (!elements.has(id)) elements.set(id, {textContent: ''});
      return elements.get(id);
    } },
    window: { location: { search } }, URLSearchParams,
    setTimeout(callback) { timers.push(callback); return timers.length; },
    clearTimeout() {}, console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/cipoc/demo/web/app.js'), 'utf8'), context);
  return { context, timers, run: (source) => vm.runInContext(source, context) };
}

test('burst refreshes serialize requests and fetch only missing events', async () => {
  const { context, timers, run } = browser();
  const urls = [];
  let resolve;
  context.fetch = (url) => {
    urls.push(url);
    return new Promise((ready) => { resolve = (data) => ready({ ok: true, json: async () => data }); });
  };
  run(`applyMeta = () => {}; buildStepSelect = () => {};
       applyView = (view) => { lastView = view; }; updateControls = () => {};`);
  const first = run('refreshLive()');
  await run('refreshLive()');
  await run('refreshLive()');
  assert.deepEqual(urls, ['/api/update?after=-1']);
  resolve({ events: [{seq: 0}], steps: [], meta: {}, view: {cursor: 0, snapshot: {seq: 0}} });
  await first;
  assert.equal(timers.length, 1);
  const second = timers.shift()();
  assert.deepEqual(urls, ['/api/update?after=-1', '/api/update?after=0']);
  resolve({ events: [{seq: 1}], steps: [], meta: {}, view: {cursor: 0, snapshot: {seq: 1}} });
  await second;
  assert.equal(run('events.length'), 2);
  assert.equal(run('lastView.snapshot.seq'), 1);
});

test('pause while following live pauses presentation through the server', () => {
  const { context, run } = browser();
  const requests = [];
  context.capture = (url) => requests.push(url);
  run('post = capture; lastView = { follow_live: true, playing: false }; togglePlay();');
  assert.deepEqual(requests, ['/api/pause']);
});

test('viewer controls never post execution or playback requests', async () => {
  const { context, timers, run } = browser('?viewer=1');
  const response = new Response('unused');
  const requests = [];
  context.fetch = async (url) => { requests.push(url); return response; };
  await run('post("/api/start")');
  await run('post("/api/next")');
  await run('post("/api/goto/3")');
  assert.deepEqual(requests, []);
  assert.equal(response.bodyUsed, false);
  assert.equal(timers.length, 0);
  assert.equal(run('document.getElementById("run-issues").textContent'), '');
});

for (const status of [200, 409]) {
  test(`control responses drain large HTTP ${status} bodies before refresh or error reporting`, async (t) => {
    const { context, timers, run } = browser();
    let body;
    const response = new Response(new ReadableStream({start(controller) { body = controller; }}), {status});
    t.after(() => { try { body.close(); } catch { /* already closed */ } });
    response.json = () => { throw new Error('control responses must not be parsed'); };
    context.fetch = async (url, options) => {
      assert.equal(url, '/api/goto/3');
      assert.equal(options.method, 'POST');
      return response;
    };
    run('lastView = {cursor: 1, snapshot: {seq: 10}};');
    const original = run('lastView');
    const pending = run('post("/api/goto/3")');
    // Let fetch deliver its headers; the body deliberately remains in flight.
    await new Promise(setImmediate);
    assert.equal(response.bodyUsed, true, 'the body reader must start on error responses too');
    assert.equal(timers.length, 0, 'do not queue a refresh while this connection is still occupied');
    assert.equal(run('document.getElementById("run-issues").textContent'), '');
    // A ~2.2 MB body reproduces the response size that exhausted browser slots.
    // It need not be valid JSON: these bytes must be discarded, not interpreted.
    body.enqueue(new Uint8Array(2_200_000));
    body.close();
    await pending;
    assert.equal(run('lastView'), original, 'control responses never replace the selected snapshot');
    assert.equal(timers.length, status === 200 ? 1 : 0);
    assert.equal(run('document.getElementById("run-issues").textContent'),
      status === 200 ? '' : 'Control request failed (409).');
  });
}

test('repeated controls release every body and coalesce successful refreshes', async () => {
  const { context, timers, run } = browser();
  const responses = [];
  context.fetch = async () => {
    const response = new Response('{"cursor":999,"snapshot":{"seq":999}}');
    response.json = () => { throw new Error('control responses must not be parsed'); };
    responses.push(response);
    return response;
  };
  run('lastView = {cursor: 0, snapshot: {seq: 0}};');
  const original = run('lastView');
  // More than the five connections left after the browser opens the SSE stream.
  for (let step = 1; step <= 8; step++) await run(`post("/api/goto/${step}")`);
  assert.equal(responses.length, 8);
  assert.ok(responses.every((response) => response.bodyUsed));
  assert.equal(timers.length, 1);
  assert.equal(run('lastView'), original);
  assert.equal(run('document.getElementById("run-issues").textContent'), '');
});

test('bodyless successful controls still schedule refresh', async () => {
  const { context, timers, run } = browser();
  context.fetch = async () => new Response(null, {status: 204});
  await run('post("/api/pause")');
  assert.equal(timers.length, 1);
  assert.equal(run('document.getElementById("run-issues").textContent'), '');
});

test('control fetch rejections are reported without scheduling refresh', async () => {
  const { context, timers, run } = browser();
  context.fetch = async () => { throw new Error('connection lost'); };
  await run('post("/api/next")');
  assert.equal(timers.length, 0);
  assert.equal(run('document.getElementById("run-issues").textContent'), 'connection lost');
});

test('control body-read rejections are reported without scheduling refresh', async () => {
  const { context, timers, run } = browser();
  const response = new Response(new ReadableStream({
    pull(controller) { controller.error(new Error('body interrupted')); },
  }));
  context.fetch = async () => response;
  await run('post("/api/next")');
  assert.equal(response.bodyUsed, true);
  assert.equal(timers.length, 0);
  assert.equal(run('document.getElementById("run-issues").textContent'), 'body interrupted');
});

test('retriever proposals do not invent a candidate pool from unrelated notes', () => {
  const { run } = browser();
  const html = run('notesById = {unrelated: {}}; viewRetriever({relevant_note_ids: ["n1"]}, {})');
  assert.match(html, /Retriever-proposed notes/);
  assert.doesNotMatch(html, /unrelated|extraction skipped/);
});

test('validated selection explains rejected and discarded notes with escaping', () => {
  const { run } = browser();
  const html = run(`viewSelection({ candidate_note_ids: ['a'], selected_note_ids: ['a'],
    rejected_note_ids: {'<b>': ['note_type_mismatch']}, discarded_note_ids: ['unoffered'],
    unevaluated_checks: ['temporal_anchor_unavailable'] })`);
  assert.match(html, /Validated note selection/);
  assert.match(html, /&lt;b&gt;/);
  assert.match(html, /Discarded unoffered proposals/);
  assert.match(html, /temporal_anchor_unavailable/);
});

test('model cards distinguish active invocation, transport retry and timing', () => {
  const { run } = browser();
  const html = run(`renderLLMCall({complete: false, model: '<model>', transport_retry_ordinal: 1,
    queue_seconds: 0, service_seconds: 1.25, prompt_messages: [], response: null})`);
  assert.match(html, /active/);
  assert.match(html, /transport retry 1/);
  assert.match(html, /0.00s local wait/);
  assert.match(html, /&lt;model&gt;/);
});

function liveMap(t) {
  const harness = browser();
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/cipoc/demo/web/cytoscape.min.js'), 'utf8'), harness.context);
  harness.context.requestAnimationFrame = () => 1;
  harness.context.cancelAnimationFrame = () => {};
  harness.context.performance = { now: () => 0 };
  harness.run(`cy = cytoscape({headless: true});
    fitMap = () => {}; startDashLoop = () => {};
    renderDetail = () => {}; renderVars = () => {}; updateControls = () => {};
    closeOmopModal = () => {}; applyMeta = () => {}; buildStepSelect = () => {};`);
  t.after(() => harness.run('cy.destroy()'));
  harness.update = async (newEvents, snapshot, options = {}) => {
    const view = {
      cursor: 0, follow_live: true, at_end: true,
      step: {index: 0, node: 'extract_branch', start_seq: 0, end_seq: snapshot.seq},
      snapshot: JSON.parse(JSON.stringify(snapshot)), ...options,
    };
    harness.context.fetch = async () => ({ok: true, json: async () => ({
      events: newEvents, steps: [view.step], view, meta: {mode: 'live'},
    })});
    await harness.run('refreshLive()');
  };
  return harness;
}

function task(seq, node, id, type, payload = {}, namespace = []) {
  return {seq, t: seq, type, node, task_id: id, namespace, payload};
}

test('live map grows from one loaded note to the entire bundle and closes their windows', async (t) => {
  const {run, context, update} = liveMap(t);
  const notes = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/note_bundle.json'), 'utf8'));
  const first = notes.find(note => note.note_id === 51);
  const start = task(0, 'note_branch', 'n51', 'task_start', first);
  context.initial = [start];
  run('events = initial; buildMapIndex();');
  await update([], {seq: 0});
  const later = notes.filter(note => note !== first).map((note, i) =>
    task(i + 1, 'note_branch', `n${note.note_id}`, 'task_start', note));
  const finishes = notes.map((note, i) =>
    task(later.length + i + 1, 'note_branch', `n${note.note_id}`, 'task_end'));
  await update([...later, ...finishes], {seq: finishes.at(-1).seq});
  const ids = JSON.parse(run('JSON.stringify(cy.nodes(".note").map(node => node.id()).sort())'));
  assert.deepEqual(ids, notes.map(note => `note:${note.note_id}`).sort());
  assert.equal(run('cy.nodes(".note.st-done").length'), notes.length);
});

test('live bubbles finish before the group merge, retaining null and invalid outcomes', async (t) => {
  const {run, context, update} = liveMap(t);
  const variables = [400, 410, 390, 522].map(item_id => ({item_id, name: `Item ${item_id}`}));
  const requested = {group_id: 'g', name: 'Group', variables};
  const starts = [task(0, 'extract_branch', 'g1', 'task_start', {requested_variables: requested}),
    ...variables.map((variable, i) => task(i + 1, 'variable_branch', `v${variable.item_id}`,
      'task_start', {task: {variable}}, ['extract_branch:g1', 'extract:e1']))];
  const snapshot = {seq: 4, instances: {}, progress: {
    groups: [{group_id: 'g', name: 'Group', annotation: ''}],
    variables: variables.map(variable => ({...variable, group_id: 'g', value: null, terminal: false})),
  }};
  context.initial = starts;
  run('events = initial; buildMapIndex();');
  await update([], snapshot);
  const finishes = variables.slice(0, 3).map((variable, i) => {
    const extraction = {item_id: variable.item_id, value: i === 1 ? null : 'C509',
      is_valid: i !== 2, presence_confidence: 'high', validation_errors: i === 2 ? ['Invalid code'] : []};
    snapshot.instances[`extract_branch:g1/extract:e1/variable_branch:v${variable.item_id}`] = {
      node: 'variable_branch', active: 0, status: i === 2 ? 'invalid' : 'done',
      result: {variable_results: [extraction]},
    };
    return task(i + 5, 'variable_branch', `v${variable.item_id}`, 'task_end',
      {variable_results: [extraction]}, ['extract_branch:g1', 'extract:e1']);
  });
  snapshot.seq = 7;
  await update(finishes, snapshot);
  assert.equal(run('cy.getElementById("var:400").hasClass("st-done")'), true);
  assert.equal(run('cy.getElementById("var:400").hasClass("st-empty")'), false);
  assert.equal(run('cy.getElementById("var:410").hasClass("st-empty")'), true);
  assert.equal(run('cy.getElementById("var:390").hasClass("st-flagged")'), true);
  assert.equal(run('cy.getElementById("var:522").hasClass("st-active")'), true);
  assert.equal(run('stateAt("grp:g", 7)'), 'active');
  const graph = run('cy');
  await update([], snapshot);
  assert.equal(run('cy'), graph);
  // Historical scrubbing must not reveal the latest completion prematurely.
  run('renderMapAt(4, lastView.snapshot, lastView.step)');
  assert.equal(run('cy.getElementById("var:400").hasClass("st-active")'), true);
  assert.equal(run('cy.getElementById("var:390").hasClass("st-flagged")'), false);
});

test('follow live paints the newest frame immediately, while review retains animation', async (t) => {
  const {run, context, update} = liveMap(t);
  const painted = [];
  context.paint = (time) => painted.push(time);
  run('renderMapAt = paint;');
  const timeline = [task(0, 'note_branch', 'n', 'task_start', {note_id: 51}),
    task(5, 'note_branch', 'n', 'task_end')];
  await update(timeline, {seq: 5});
  assert.equal(painted.at(-1), 5);
  await update([], {seq: 5}, {cursor: 1, follow_live: false});
  assert.equal(painted.at(-1), 0);
  await update([], {seq: 5}, {cursor: 1, follow_live: true});
  assert.equal(painted.at(-1), 5);
});
