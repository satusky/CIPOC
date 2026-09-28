const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '../src/cipoc/demo/web/cards.js'), 'utf8');
// No app.js, DOM, fetch, browser APIs, module loader or timers are available.
const context = vm.createContext({});
vm.runInContext(script, context);
const {render} = context.DemoCards;
const catalog = {notes: [{id: 'note:01', noteId: '01', type: 'Pathology'}], groups: [
  {id: 'g', name: 'Primary cancer', variables: [{itemId: 400, name: 'Primary Site'}]},
  {id: 'h', name: 'Other group', variables: [{itemId: 400, name: 'Other site'}]},
]};
const note = (id, summary, options = {}) => ({
  key: `note_branch:${id}`, node: 'note_branch', index: 1, started_t: 1, status: 'done',
  input: {note_id: id, date: '2025-02-24', note_type: 'Pathology'},
  result: {summary}, ...options,
});
const group = (id = 'g', pass = 'a', variables = [{item_id: 400, name: 'Primary Site', description: 'Recorded site definition'}], options = {}) => ({
  key: `extract_branch:${id}-${pass}`, node: 'extract_branch', index: 2, started_t: 2, status: 'done',
  input: {requested_variables: {group_id: id, name: `${id} group`, variables}}, ...options,
});
const output = (value = 'C509', extra = {}) => ({item_id: 400, value, is_valid: true,
  explanation: 'Recorded extraction explanation.', spans: [], presence_confidence: 'high', ...extra});
const variable = (parent, value = 'C509', options = {}) => ({
  key: `${parent.key}/extract:e/variable_branch:400`, node: 'variable_branch', index: 3, started_t: 3, status: 'done',
  input: {task: {variable: {item_id: 400, name: 'Primary Site'}}},
  result: {variable_results: [output(value)]}, attempts: [], ...options,
});
const snapshot = (...items) => ({instances: Object.fromEntries(items.map((i) => [i.key, i]))});
const variableSelection = {kind: 'variable', id: '400', groupId: 'g'};
const cardStatus = (html) => /class="entity-status status-([^"]+)"/.exec(html)?.[1];
const occurrenceCount = (html, text) => html.split(text).length - 1;

test('standalone global/CommonJS export, blank empty selection, deterministic and immutable', () => {
  assert.equal(typeof render, 'function');
  assert.equal(typeof require('../src/cipoc/demo/web/cards.js').render, 'function');
  assert.equal(render(null), '');
  const g = group();
  const state = snapshot(g, variable(g));
  const before = JSON.stringify(state);
  assert.equal(render(variableSelection, state), render(variableSelection, state));
  assert.equal(JSON.stringify(state), before);
  assert.doesNotMatch(render(variableSelection, state), /<details|<summary|role="tab|overflow|onclick|<script/);
});

test('concurrent notes keep their own summary, keywords, concepts, mentions and metadata', () => {
  const a = note('01', 'Summary A', {result: {summary: 'Summary A',
    flags: ['biopsy', 'ER+/PR+ & HER2−'],
    concepts: {surgery: {presence: false, confidence: 'max'}, cancer: {presence: true, confidence: 'high'}},
    cancer_mentions: [{affected_tissue: 'Breast A', histology: 'Ductal A', status: 'current', confidence: 'high', metastasis: false}],
  }});
  const b = note('02', 'Summary B', {started_t: 10, result: {summary: 'Summary B',
    flags: ['lung resection'],
    concepts: {immunotherapy: {presence: true}}, cancer_mentions: [{affected_tissue: 'Lung B'}]}});
  const state = {...snapshot(a, b), details: {scanner_summarize_note: {result: b.result}}};
  const html = render({kind: 'note', id: 'note:01'}, state, {}, catalog);
  for (const expected of ['Summary A', '2025-02-24', 'Pathology', 'Absent', 'Breast A', 'Ductal A', 'current', 'Confidence: high']) assert.ok(html.includes(expected), expected);
  assert.doesNotMatch(html, /Summary B|Lung B|immunotherapy|lung resection/);
  assert.match(html, /class="entity-keyword">biopsy<\/li>/);
  assert.match(html, /class="entity-keyword">ER\+\/PR\+ &amp; HER2−<\/li>/);
  assert.equal(occurrenceCount(html, 'class="entity-section-title"'), 5);
  const missing = render({kind: 'note', id: '02'}, state);
  assert.match(missing, /Histology: Not recorded/);
});

test('latest note is selected by start/order, not object insertion or last completion', () => {
  const earlier = note('01', 'OLD', {key: 'note_branch:old', index: 50, started_t: 1, finished_t: 100});
  const later = note('01', 'NEW', {key: 'note_branch:new', index: 1, started_t: 2, finished_t: 3});
  assert.match(render({kind: 'note', id: '01'}, snapshot(later, earlier)), /NEW/);
  assert.doesNotMatch(render({kind: 'note', id: '01'}, snapshot(later, earlier)), /OLD/);
  earlier.started_t = 2;
  assert.match(render({kind: 'note', id: '01'}, snapshot(later, earlier)), /OLD/);
});

test('explicit stale or mismatched instance keys never fall back to another entity', () => {
  const g = group();
  const v = variable(g);
  const state = snapshot(note('01', 'PRIVATE SUMMARY'), note('02', 'WRONG SUMMARY'), g, v);
  for (const selection of [
    {kind: 'note', id: '01', instanceKey: 'missing'},
    {kind: 'note', id: '01', instanceKey: 'note_branch:02'},
    {...variableSelection, instanceKey: 'missing'},
    {...variableSelection, id: '410', instanceKey: v.key},
    {...variableSelection, groupId: 'h', instanceKey: v.key},
    {kind: 'group', id: 'g', instanceKey: 'missing'},
  ]) {
    const html = render(selection, state, {}, catalog);
    assert.equal(cardStatus(html), 'unavailable');
    assert.doesNotMatch(html, /PRIVATE SUMMARY|WRONG SUMMARY|C509|Recorded extraction explanation/);
  }
});

test('two groups containing the same item never mix outcomes, definitions or checks', () => {
  const g = group();
  const h = group('h', 'a', [{item_id: 400, name: 'Other site', description: 'OTHER DEFINITION'}], {started_t: 4});
  const a = variable(g, 'A', {attempts: [{node: 'validate_extraction', is_valid: true, value: 'A', attempt: 1}]});
  const b = variable(h, 'B', {started_t: 5, attempts: [{node: 'validate_extraction', is_valid: false, value: 'B', validation_errors: ['OTHER ERROR']}]});
  const state = {...snapshot(g, h, a, b), details: {extractor_validate_extraction: {input: b.input, result: b.result}}};
  const html = render(variableSelection, state, {}, catalog);
  assert.match(html, /Recorded site definition/);
  assert.match(html, /Candidate: A/);
  assert.doesNotMatch(html, /OTHER DEFINITION|OTHER ERROR|Candidate: B/);
  const tiles = render({kind: 'group', id: 'g'}, state, {}, catalog);
  assert.match(tiles, /data-entity-kind="variable" data-entity-id="400" data-group-id="g"/);
  assert.ok(tiles.includes(`data-instance-key="${a.key}"`));
  assert.doesNotMatch(tiles, /OTHER|>B</);
});

test('latest group pass excludes old variable branches, explicit historical pins stay historical', () => {
  const old = group();
  const oldV = variable(old, 'OLD VALUE');
  const fresh = group('g', 'b', [{item_id: 400, name: 'Primary Site', description: 'NEW DEFINITION'}], {started_t: 10, status: 'active'});
  const state = {...snapshot(old, oldV, fresh), progress: {groups: [{group_id: 'g', item_ids: [400]}], variables: [
    {item_id: 400, group_id: 'g', status: 'pending', terminal: false, value: null},
  ]}};
  let html = render(variableSelection, state);
  assert.doesNotMatch(html, /OLD VALUE|Recorded site definition|Passed recorded checks/);
  assert.match(html, /NEW DEFINITION/);
  html = render({...variableSelection, instanceKey: oldV.key}, state);
  assert.match(html, /OLD VALUE|Recorded site definition/);
  assert.doesNotMatch(html, /NEW DEFINITION/);
  const newV = variable(fresh, 'NEW VALUE', {started_t: 11});
  state.instances[newV.key] = newV;
  html = render({kind: 'group', id: 'g', instanceKey: old.key}, state);
  assert.match(html, /OLD VALUE/);
  assert.doesNotMatch(html, /NEW VALUE/);
});

test('no-instance progress roster preserves every terminal status, values and meaningful reasons', () => {
  const statuses = ['structured_data', 'not_found', 'invalid', 'error', 'not_applicable', 'skipped', 'blocked', 'extracted', 'pending'];
  const state = {progress: {groups: [{group_id: 'g', name: 'Planned group', item_ids: statuses.map((_, i) => i + 1)}],
    variables: statuses.map((status, i) => ({item_id: i + 1, name: `Variable ${i + 1}`, group_id: 'g', status,
      terminal: status !== 'pending', value: ['structured_data', 'extracted'].includes(status) ? `00${i}` : null,
      detail: status === 'blocked' ? 'Depends on unresolved site 400' : status === 'not_applicable' ? 'Outside site scope' : null}))}};
  const html = render({kind: 'group', id: 'g'}, state);
  assert.equal(occurrenceCount(html, 'class="entity-tile"'), statuses.length);
  for (const status of statuses) assert.ok(html.includes(`status-${status.replaceAll('_', '-')}`), status);
  assert.match(html, /Depends on unresolved site 400|Outside site scope/);
  assert.match(html, />000<|>007</);
  const v = render({kind: 'variable', id: '1', groupId: 'g'}, state);
  assert.equal(cardStatus(v), 'structured-data');
  assert.match(v, /000/);
  assert.match(v, /Not recorded · validation history unavailable/);
  assert.doesNotMatch(v, /Passed recorded checks|Validation checks · 0/);
});

test('group roster includes planned skipped/structured items omitted from its extraction request', () => {
  const g = group();
  const state = {...snapshot(g, variable(g)), progress: {groups: [{group_id: 'g', item_ids: [400, 410, 390]}],
    variables: [{item_id: 410, name: 'Laterality', group_id: 'g', status: 'structured_data', terminal: true, value: '2'},
      {item_id: 390, name: 'Date', group_id: 'g', status: 'blocked', terminal: true, detail: 'Missing date anchor'}]}};
  const html = render({kind: 'group', id: 'g'}, state);
  assert.equal(occurrenceCount(html, 'class="entity-tile"'), 3);
  assert.match(html, /Laterality|Missing date anchor/);
  const structured = render({kind: 'variable', id: '410', groupId: 'g', instanceKey: g.key}, state);
  assert.equal(cardStatus(structured), 'structured-data');
  assert.match(structured, /<strong>2<\/strong>/);
});

test('instance-free branch results preserve not-found explanations and invalid candidate errors', () => {
  const g = group();
  g.result = {variable_results: {'400': {item_id: 400, status: 'not_found', value: null,
    extraction: output(null, {explanation: 'No notes survived the recorded selection.'})}}};
  let html = render(variableSelection, snapshot(g));
  assert.equal(cardStatus(html), 'not-found');
  assert.match(html, /No notes survived the recorded selection/);
  assert.match(html, /validation history unavailable/);
  g.result.variable_results['400'] = {item_id: 400, status: 'error', value: null,
    extraction: output('BAD', {is_valid: false, validation_errors: ['Final bad code'], explanation: 'Candidate failed.'})};
  html = render(variableSelection, snapshot(g));
  assert.equal(cardStatus(html), 'invalid');
  assert.match(html, /Final bad code/);
  assert.match(html, /<strong>BAD<\/strong>/);
  assert.doesNotMatch(html, /class="entity-attempt"/);
});

test('root details enrich compressed blockers but stale root results cannot replace progress', () => {
  const state = {progress: {variables: [{item_id: 400, name: 'Site', group_id: 'g', status: 'blocked', value: null,
    terminal: true, detail: '←1,2,3'}]}, details: {finalize_case: {finished_t: 50, result: {
    variable_results: {'400': {item_id: 400, status: 'blocked', value: null, reason: 'All prerequisite facts missing', blocking_item_ids: [1, 2, 3, 4]}},
  }}}};
  let html = render(variableSelection, state);
  assert.match(html, /All prerequisite facts missing/);
  assert.match(html, /Blocked by items: 1, 2, 3, 4/);
  state.progress.variables[0] = {...state.progress.variables[0], status: 'structured_data', value: 'C509', detail: null};
  html = render(variableSelection, state);
  assert.equal(cardStatus(html), 'structured-data');
  assert.match(html, /C509/);
  assert.doesNotMatch(html, /All prerequisite facts missing|Blocked by items/);
});

test('new active pass cannot borrow an old terminal progress outcome before its branch starts', () => {
  const old = group();
  const fresh = group('g', 'b', undefined, {started_t: 10, status: 'active'});
  const state = {...snapshot(old, variable(old, 'OLD VALUE'), fresh), progress: {variables: [
    {item_id: 400, name: 'Site', group_id: 'g', status: 'extracted', terminal: true, value: 'OLD VALUE'},
  ]}};
  const html = render(variableSelection, state);
  assert.equal(cardStatus(html), 'pending');
  assert.doesNotMatch(html, /OLD VALUE|status-extracted/);
});

test('future catalog metadata cannot become observed definitions, outcomes or roster counts', () => {
  const future = {notes: [{id: 'note:01', noteId: '01', type: 'Pathology', summary: 'FUTURE SUMMARY'}], groups: [
    {id: 'g', name: 'Identity name', status: 'extracted', variables: [
      {itemId: 400, name: 'Identity variable', description: 'FUTURE DEFINITION', value: 'FUTURE VALUE', status: 'extracted'}]},
  ]};
  const raw = {'01': {date: '2025-01-01', note_type: 'Pathology', content: 'RAW TEXT', summary: 'FUTURE SUMMARY', concepts: {future: {presence: true}}}};
  for (const selection of [{kind: 'note', id: '01'}, {kind: 'group', id: 'g'}, variableSelection]) {
    const html = render(selection, {}, raw, future);
    assert.doesNotMatch(html, /FUTURE|RAW TEXT|status-extracted|class="entity-tile"/);
  }
  assert.match(render(variableSelection, {}, raw, future), /Identity variable|Definition not recorded/);
});

test('null, invalid, active, pending, error and branch completion remain different', () => {
  const g = group();
  for (const [expected, v] of [
    ['not-found', variable(g, null)],
    ['invalid', variable(g, null, {result: {variable_results: [output(null, {is_valid: false, validation_errors: ['No usable code']})]}})],
    ['active', variable(g, null, {status: 'active', active: 1})],
    ['error', variable(g, null, {status: 'error', error: 'Endpoint failed'})],
    ['done', variable(g, null, {result: null, attempts: [{node: 'extract_individual_value', value: null, is_valid: false}]})],
  ]) {
    const html = render(variableSelection, snapshot(g, v));
    assert.equal(cardStatus(html), expected);
    if (expected === 'done') assert.match(html, /Branch completed; extraction outcome not recorded/);
  }
  assert.equal(cardStatus(render(variableSelection, snapshot(g))), 'pending');
  const html = render({kind: 'group', id: 'g'}, snapshot(g));
  assert.equal(cardStatus(html), 'done');
  assert.doesNotMatch(html, /status-extracted/);
});

test('only validation nodes count, including one clean check and every failure reason', () => {
  const g = group();
  const attempts = [
    {node: 'extract_individual_value', attempt: 1, is_valid: false, value: 'BAD'},
    {node: 'validate_extraction', attempt: 1, is_valid: false, value: 'BAD', validation_errors: ['Invalid code', 'Wrong length']},
    {node: 'repair_invalid_extraction', attempt: 2, is_valid: false, value: 'C509'},
    {node: 'validate_extraction', attempt: 2, is_valid: true, value: 'C509', validation_errors: []},
  ];
  const html = render(variableSelection, snapshot(g, variable(g, 'C509', {attempts})));
  assert.match(html, /Validation checks · 2/);
  assert.equal(occurrenceCount(html, 'class="entity-attempt"'), 2);
  assert.match(html, /Candidate: BAD/);
  assert.match(html, /Invalid code/);
  assert.match(html, /Wrong length/);
  assert.match(html, /Passed recorded checks/);
  const clean = render(variableSelection, snapshot(g, variable(g, 'C509', {attempts: [attempts[3]]})));
  assert.match(clean, /Validation checks · 1/);
  assert.match(clean, /Passed recorded checks/);
  assert.doesNotMatch(clean, /Invalid code/);
});

test('repair resets never establish invalidity or reuse the old explanation/evidence', () => {
  const g = group();
  const v = variable(g, 'OLD', {status: 'active', result: null,
    input: {task: {variable: {item_id: 400}, candidate: output('OLD', {explanation: 'STALE EXPLANATION'}), extraction_attempts: 1}},
    attempts: [{node: 'validate_extraction', attempt: 1, is_valid: false, value: 'OLD', validation_errors: ['Old code invalid']},
      {node: 'repair_invalid_extraction', attempt: 2, is_valid: false, value: 'NEW'}]});
  const html = render(variableSelection, snapshot(g, v));
  assert.equal(cardStatus(html), 'active');
  assert.match(html, /Candidate value<\/span>\s*<strong>NEW/);
  assert.match(html, /Old code invalid/);
  assert.doesNotMatch(html, /STALE EXPLANATION/);
  v.status = 'done';
  assert.equal(cardStatus(render(variableSelection, snapshot(g, v))), 'done');
});

test('definitions come from variable task first and only its matching scoped group second', () => {
  const g = group('g', 'a', [{item_id: 410, description: 'WRONG ITEM'}, {item_id: 400, description: 'RIGHT GROUP DEFINITION'}]);
  const v = variable(g);
  assert.match(render(variableSelection, snapshot(g, v)), /RIGHT GROUP DEFINITION/);
  v.input.task.variable.description = 'TASK DEFINITION';
  const html = render(variableSelection, snapshot(g, v));
  assert.match(html, /TASK DEFINITION/);
  assert.doesNotMatch(html, /WRONG ITEM|RIGHT GROUP DEFINITION/);
});

test('evidence deduplicates identical note spans and merges overlaps while retaining labels', () => {
  const n = note('01', 'Summary', {result: {
    concepts: {cancer: {presence: true, confidence: 'high', evidence: [{note_id: '01', text: 'invasive carcinoma'}]},
      surgery: {presence: false, evidence: [{note_id: '01', text: 'carcinoma in breast'}]}},
    cancer_mentions: [{affected_tissue: 'breast', status: 'current', evidence: [{note_id: '01', text: 'invasive carcinoma'}]}],
  }});
  const html = render({kind: 'note', id: '01'}, snapshot(n), {'01': {content: 'The invasive carcinoma in breast was recorded.', note_type: 'Path', date: '2025-01-02'}});
  assert.equal(occurrenceCount(html, 'class="entity-evidence-item"'), 1);
  assert.match(html, /<mark>invasive carcinoma in breast<\/mark>/);
  const labels = /class="entity-evidence-labels">([^<]*)/.exec(html)[1];
  for (const label of ['cancer · Present', 'surgery · Absent', 'Cancer mention 1 · breast · current']) assert.ok(labels.includes(label));
  assert.match(html, /The <mark>/);
  assert.match(html, /Source-note excerpt/);
  assert.match(html, /data-entity-kind="note" data-entity-id="01"/);
  assert.doesNotMatch(html, /<mark>[^<]*<mark>/);
});

test('multiple notes, repeated text, missing source and unmatched punctuation remain honest', () => {
  const g = group();
  const spans = [
    {note_id: '01', text: 'same phrase'}, {note_id: '01', text: 'same phrase'},
    {note_id: '02', text: 'same phrase'}, {note_id: 'missing', text: 'absent source quote'},
    {note_id: '03', text: 'curly “quote”'}, {note_id: '03', text: 'source text'},
  ];
  const v = variable(g, 'C509', {result: {variable_results: [output('C509', {spans})]}});
  const notes = {'01': {content: 'same phrase -- same phrase', date: 'DATE ONE', note_type: 'TYPE ONE'},
    '02': {content: 'other same phrase here', date: 'DATE TWO', note_type: 'TYPE TWO'},
    '03': {content: 'curly "quote" SOURCE TEXT'}};
  const html = render(variableSelection, snapshot(g, v), notes);
  assert.equal(occurrenceCount(html, '<mark>same phrase</mark>'), 2);
  assert.match(html, /2 exact occurrences; first shown/);
  assert.match(html, /DATE ONE · TYPE ONE/);
  assert.match(html, /DATE TWO · TYPE TWO/);
  assert.match(html, /Source note unavailable · cited quote, not highlighted/);
  assert.match(html, /Exact span not found in source note · cited quote, not highlighted/);
  assert.doesNotMatch(html, /<mark>absent source quote|<mark>curly|<mark>source text/);
});

test('HTML-unsafe IDs, values, source text, labels, errors and navigation attributes are escaped', () => {
  const id = 'n"><img src=x onerror=alert(1)>';
  const g = group('g"<&');
  const v = variable(g, '<script>bad()</script>', {result: {variable_results: [output('<script>bad()</script>', {
    explanation: '<img src=x>', spans: [{note_id: id, text: '<unsafe>&"'}], most_important_note: id,
  })]}, attempts: [{node: 'validate_extraction', value: '<candidate>', is_valid: false, validation_errors: ['<error> & reason']}]});
  const html = render({kind: 'variable', id: '400', groupId: 'g"<&'}, snapshot(g, v), {[id]: {content: 'Before <unsafe>&" after', note_type: '<type>', date: '<date>'}});
  assert.doesNotMatch(html, /<script>|<img|<unsafe>|<error>|<type>|<candidate>/);
  assert.match(html, /&lt;script&gt;bad\(\)&lt;\/script&gt;/);
  assert.match(html, /<mark>&lt;unsafe&gt;&amp;&quot;<\/mark>/);
  assert.match(html, /data-entity-id="n&quot;&gt;&lt;img/);
  const tiles = render({kind: 'group', id: 'g"<&'}, snapshot(g, v));
  assert.match(tiles, /data-group-id="g&quot;&lt;&amp;"/);
  assert.match(tiles, /data-instance-key="extract_branch:g&quot;&lt;&amp;/);
});

test('captured prompts, model response strings and unrelated source notes never leak', () => {
  const secret = 'DO_NOT_DISPLAY_CAPTURED_MESSAGES';
  const g = group();
  const v = variable(g, null, {result: {messages: [{content: secret}], response: secret, group_context: {explanation: secret}},
    llm_calls: [{prompt_messages: [{content: `{"description":"${secret}"}`}], response: `{"value":"${secret}"}`}],
  });
  const state = {...snapshot(g, v), details: {extractor_extract_individual_value: {llm_calls: [{response: secret}]}}};
  const html = render(variableSelection, state, {unrelated: {content: secret}});
  assert.doesNotMatch(html, /DO_NOT_DISPLAY_CAPTURED_MESSAGES|prompt_messages|group_context/);
  assert.match(html, /Evidence not recorded/);
});

test('long prose and cited text use labeled deterministic excerpts without dropping checks', () => {
  const g = group();
  const text = 'START ' + 'x'.repeat(5000) + ' END';
  const attempts = Array.from({length: 12}, (_, i) => ({node: 'validate_extraction', value: String(i), is_valid: false,
    validation_errors: [`Meaningful error ${i}`]}));
  const v = variable(g, 'C509', {input: {task: {variable: {item_id: 400, description: text}}}, attempts,
    result: {variable_results: [output('C509', {explanation: text, spans: [{note_id: '01', text}]})]}});
  const html = render(variableSelection, snapshot(g, v), {'01': {content: 'discardable context '.repeat(100) + text + ' tail '.repeat(100)}});
  assert.match(html, /Deterministic excerpt · first 650/);
  assert.match(html, /Deterministic cited-text excerpt/);
  assert.match(html, /cited characters omitted/);
  assert.match(html, / END<\/mark>/);
  assert.doesNotMatch(html, /discardable context/);
  assert.equal(occurrenceCount(html, 'class="entity-attempt"'), 12);
  for (let i = 0; i < 12; i++) assert.ok(html.includes(`Meaningful error ${i}`));
});

test('actual committed trace produces isolated note, group and repaired-variable cards', () => {
  // Fold just the InstanceDetail channels exactly as state.py does. This pins
  // tests to real serialized shapes without Python/runtime/endpoint dependencies.
  const events = fs.readFileSync(path.join(__dirname, 'fixtures/demo_trace.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const state = {instances: {}};
  const notes = {};
  for (const event of events) {
    const root = ['note_branch', 'extract_branch', 'variable_branch'].includes(event.node) && ['task_start', 'task_end'].includes(event.type);
    const ns = root ? [...event.namespace, `${event.node}:${event.task_id}`] : event.namespace;
    const depth = ns.findLastIndex((s) => /^(note_branch|extract_branch|variable_branch):/.test(s));
    if (depth === -1) continue;
    const key = ns.slice(0, depth + 1).join('/');
    const inst = state.instances[key] ||= {key, node: ns[depth].split(':')[0], started_t: event.t, index: Object.keys(state.instances).length + 1, status: 'active', attempts: []};
    if (root && event.type === 'task_start') {
      inst.input = event.payload;
      if (event.node === 'note_branch') notes[event.payload.note_id] = event.payload;
    } else if (root && event.type === 'task_end') inst.status = 'done';
    else if (event.type === 'values') inst.result = {...inst.result, ...event.payload};
    else if (event.type === 'task_end' && ['extract_individual_value', 'validate_extraction', 'repair_invalid_extraction'].includes(event.node)) {
      const t = event.payload.task;
      inst.attempts.push({node: event.node, attempt: t.extraction_attempts, is_valid: t.is_valid, validation_errors: t.validation_errors, value: t.candidate?.value});
    }
  }
  const noteHtml = render({kind: 'note', id: '51'}, state, notes);
  assert.match(noteHtml, /Pathology Report recorded 2025-02-24/);
  const groupHtml = render({kind: 'group', id: 'lymph_node_removal'}, state, notes);
  assert.equal(occurrenceCount(groupHtml, 'class="entity-tile"'), 4);
  assert.doesNotMatch(groupHtml, /data-entity-id="390"/);
  const variableHtml = render({kind: 'variable', id: '674', groupId: 'lymph_node_removal'}, state, notes);
  assert.match(variableHtml, /Validation checks · 2/);
  assert.match(variableHtml, /Passed recorded checks/);
  assert.match(variableHtml, /<mark>Specimen: Left breast/);
  assert.doesNotMatch(variableHtml, /Synthetic prompt|fake-model|You are the/);
});
