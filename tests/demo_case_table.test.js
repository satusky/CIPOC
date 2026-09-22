const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/cipoc/demo/web/app.js'), 'utf8');
function renderer() {
  const context = vm.createContext({document: {addEventListener() {}}, console});
  vm.runInContext(source, context);
  // Future identities and results must never enter the case overview.
  vm.runInContext(`mapIndex.groups.set('FUTURE', {id: 'FUTURE', name: 'FUTURE',
    variables: [{itemId: 9999, name: 'FUTURE', value: 'FUTURE'}]});
    latestMeta = {summary: {case_facts: {primary_site: 'FUTURE'}}};`, context);
  return (snapshot) => {
    context.snapshot = snapshot;
    return vm.runInContext('renderCaseTable(snapshot)', context);
  };
}

function progress() {
  return {
    totals: {terminal: 2, variables: 8, done_groups: 1, groups: 3},
    notes_done: 4, notes_total: 6, review_flags: 2,
    groups: [{group_id: 'site', name: 'Site', stage: 'extracting'},
      {group_id: 'other', name: 'Other', stage: 'waiting'}],
    variables: [
      {item_id: 400, group_id: 'site', name: 'Primary site', stage: 'done', status: 'extracted',
        value: 'C509', confidence: 'high', flag: 'Review site'},
      {item_id: 410, group_id: 'site', name: 'Laterality', stage: 'done', status: 'not_found',
        value: null, confidence: null},
      {item_id: 522, group_id: 'other', name: 'Histology', stage: 'pending', status: null,
        value: null, confidence: null},
    ],
  };
}

test('case facts stay above one accessible scroll region containing every group and row', () => {
  const render = renderer();
  const snapshot = {case_facts: {primary_site: 'C509', gross_primary_site: 'breast', histology: '8500',
    behavior: '3', sex: '2', date_of_diagnosis: '2024-01-02'}, progress: progress()};
  const before = JSON.stringify(snapshot);
  const html = render(snapshot);
  assert.match(html, /^<article class="entity-card entity-case">/);
  assert.match(html, /<header class="entity-header"><h2 class="entity-title">Case<\/h2>/);
  assert.match(html, /<section class="case-facts"><h3 class="entity-section-title">Case facts<\/h3>/);
  for (const [label, value] of [['Primary site', 'C509'], ['Gross primary site', 'breast'],
    ['Histology', '8500'], ['Behavior', '3'], ['Sex', '2'], ['Date of diagnosis', '2024-01-02']]) {
    assert.ok(html.includes(`<div class="case-fact"><dt>${label}</dt><dd>${value}</dd></div>`));
  }
  const [above, scroll] = html.split('<div class="case-table-scroll" tabindex="0" role="region" aria-label="Case variables">');
  assert.ok(scroll);
  assert.match(above, /case-fact-grid/);
  assert.doesNotMatch(above, /class="vargroup"|class="vartable"/);
  assert.equal((scroll.match(/class="vargroup"/g) || []).length, 2);
  assert.equal((scroll.match(/data-entity-kind="variable"/g) || []).length, 3);
  assert.equal((scroll.match(/class="omop-btn"/g) || []).length, 3);
  assert.match(scroll, /<th scope="col" class="vt-id">Item ID<\/th>/);
  assert.match(scroll, /<td class="vt-id">400<\/td>/);
  assert.match(scroll, /<td class="vt-id">522<\/td>/);
  assert.equal(JSON.stringify(snapshot), before, 'rendering is pure');
});

test('counts, stages, terminal null values, confidence and flags use recorded progress', () => {
  const html = renderer()({progress: progress()});
  for (const expected of ['Variables <b>2/8</b>', 'Groups <b>1/3</b>', 'Notes <b>4/6</b>',
    '⚑ 2 flag(s)', 'title="Review site"', 'stage-extracting', 'stage-waiting',
    '<span class="conf"> · high</span>', 'st-extracted">extracted', 'st-not_found">not_found',
    'st-pending">pending']) assert.ok(html.includes(expected), expected);
  const nullRow = html.match(/<tr>\s*<td class="vt-id">410<\/td>[\s\S]*?<\/tr>/)[0];
  assert.match(nullRow, /class="vt-value">—/);
  assert.doesNotMatch(nullRow, /class="conf"/);
  assert.doesNotMatch(html, /FUTURE|undefined|st-null/);
});

test('absent facts differ from recorded unknown fields, and known zero/false are retained', () => {
  const render = renderer();
  for (const snapshot of [{}, {case_facts: null}]) {
    const html = render(snapshot);
    assert.match(html, /Case facts not recorded at this step/);
    assert.doesNotMatch(html, /case-fact-grid|<dt>Primary site/);
  }
  const unknown = render({case_facts: {}});
  assert.equal((unknown.match(/<dd>—<\/dd>/g) || []).length, 6);
  assert.doesNotMatch(unknown, /Case facts not recorded at this step/);
  const html = render({case_facts: {primary_site: null, behavior: 0, sex: false,
    additional_fact: {code: 0, recorded: false}}, progress: {
      variables: [{item_id: 1, group_id: 'g', value: 0, confidence: 0},
        {item_id: 2, group_id: 'g', value: false, confidence: false}],
    }});
  assert.match(html, /<dt>Primary site<\/dt><dd>—<\/dd>/);
  assert.match(html, /<dt>Behavior<\/dt><dd>0<\/dd>/);
  assert.match(html, /<dt>Sex<\/dt><dd>false<\/dd>/);
  assert.ok(html.indexOf('<dt>Additional fact') > html.indexOf('<dt>Date of diagnosis'));
  assert.match(html, /&quot;code&quot;: 0/);
  assert.match(html, /class="vt-value">0/);
  assert.match(html, /class="vt-value">false/);
  assert.match(html, /class="conf"> · 0/);
  assert.match(html, /class="conf"> · false/);
  assert.match(html, /Variables <b>—\/—<\/b>/);
  assert.doesNotMatch(html, /stage-pending/);
});

test('an early or empty snapshot cannot borrow a future plan, case facts, or prompts', () => {
  const render = renderer();
  for (const progress of [undefined, null, {}, {groups: [], variables: []}]) {
    const html = render({progress, case: {case_facts: {primary_site: 'FUTURE'}},
      details: {finalize_case: {result: {case_facts: {primary_site: 'FUTURE'}}},
        initialize_case: {llm_calls: [{prompt_messages: [{content: 'RAW PROMPT'}]}]}},
      instances: {final: {result: {variable_results: {400: {value: 'FUTURE'}}}}}});
    assert.match(html, /Case facts not recorded at this step/);
    assert.match(html, /The extraction plan has not been recorded at this step/);
    assert.doesNotMatch(html, /FUTURE|9999|RAW PROMPT|class="vartable"|data-entity-kind/);
  }
  const html = render({progress: {groups: [{group_id: 'g', name: 'Recorded group', stage: 'waiting'}], variables: []}});
  assert.match(html, /Recorded group/);
  assert.match(html, /No variables recorded for this group yet/);
  assert.doesNotMatch(html, /class="vartable"|FUTURE/);
});

test('all recorded fields and identifiers are escaped in text and attributes', () => {
  const unsafe = '<img src=x onerror="evil()">&\'';
  const snapshot = {case_facts: {[unsafe]: unsafe, primary_site: unsafe}, progress: {
    totals: {terminal: unsafe, variables: unsafe, done_groups: unsafe, groups: unsafe},
    notes_done: unsafe, notes_total: unsafe, review_flags: unsafe,
    groups: [{group_id: unsafe, name: unsafe, stage: unsafe}],
    variables: [{item_id: unsafe, group_id: unsafe, name: unsafe, value: {code: unsafe},
      confidence: unsafe, status: unsafe, flag: unsafe}],
  }};
  const html = renderer()(snapshot);
  assert.doesNotMatch(html, /<img|"evil\(\)"/);
  assert.match(html, /&lt;img src=x onerror=&quot;evil\(\)&quot;&gt;&amp;&#39;/);
  assert.match(html, /data-entity-id="&lt;img/);
  assert.match(html, /data-item-id="&lt;img/);
});

test('recorded variables survive missing or partial group metadata without invented stages', () => {
  const prog = progress();
  prog.groups.pop();
  const html = renderer()({progress: prog});
  assert.match(html, /data-entity-id="other">other<\/button>/);
  assert.match(html, /<td class="vt-id">522<\/td>/);
  assert.equal((html.match(/class="stage-badge /g) || []).length, 1);
  assert.doesNotMatch(html, /FUTURE/);
});
