const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const web = path.join(__dirname, '../src/cipoc/demo/web');
const app = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const engine = fs.readFileSync(path.join(web, 'cytoscape.min.js'), 'utf8');
const boundsOptions = '{includeLabels: true, includeOverlays: false, includeUnderlays: false}';

function harness(t) {
  // Page coordinates intentionally differ from rendered Cytoscape coordinates.
  const rects = {
    canvas: {left: 26, top: 140, width: 1894, height: 912},
    drawing: {left: 26, top: 188, width: 966, height: 864},
  };
  const reads = [];
  const canvas = {getBoundingClientRect: () => rects.canvas, childNodes: []};
  const probe = {getBoundingClientRect: () => rects.drawing};
  const timers = new Map();
  let timer = 0;
  const context = vm.createContext({
    document: {
      addEventListener() {},
      getElementById(id) {
        reads.push(id);
        assert.equal(id, 'map-layout', 'geometry must never consult a detail card');
        return rects.drawing ? probe : null;
      },
    },
    window: {}, console, canvas, rects,
    setTimeout(callback, delay) { timers.set(++timer, {callback, delay}); return timer; },
    clearTimeout(id) { timers.delete(id); },
    requestAnimationFrame: () => 1, cancelAnimationFrame() {},
  });
  vm.runInContext(engine, context);
  vm.runInContext(app, context);
  const run = (source) => vm.runInContext(source, context);
  const json = (source) => JSON.parse(run(`JSON.stringify(${source})`));
  run(`const testTheme = Object.fromEntries([
      'ok','warn','err','line','muted','navy','ink','inkMuted','panel','sunk','errSoft','lineSoft',
      'scanner','extractor','retriever','orchestrator','scannerInk'].map(key => [key, '#334455']));
    testTheme.font = 'sans-serif';
    cy = cytoscape({headless: true, styleEnabled: true, minZoom: 0.2, maxZoom: 2.5,
      style: [...mapStyle(testTheme), ...stateStyles(testTheme)]});
    cy.container = () => canvas;
    cy.width = () => rects.canvas.width;
    cy.height = () => rects.canvas.height;

    // The null renderer has no canvas font metrics. Supply deterministic text
    // measurements to the vendored base renderer's actual label projector, and
    // enable label-aware bounds without creating a browser renderer. Edges use
    // the same endpoint approximation as headless Cytoscape; compound sizing,
    // label bounds, packing and viewport transforms all stay real.
    cy.headless = () => false;
    const labelRenderer = Object.create(cytoscape('renderer', 'base').prototype);
    labelRenderer.calculateLabelDimensions = (node, text) => {
      const fontSize = node.pstyle('font-size').pfValue;
      const lines = text.split('\\n');
      return {width: Math.max(...lines.map(line => line.length)) * fontSize * 0.6,
        height: lines.length * fontSize};
    };
    cy.renderer().recalculateRenderedStyle = (elements) => {
      elements.nodes().forEach(node => labelRenderer.recalculateNodeLabelProjection(node));
      elements.edges().forEach(edge => {
        const source = edge.source().position(), target = edge.target().position();
        Object.assign(edge._private.rstyle, {
          srcX: source.x, srcY: source.y, tgtX: target.x, tgtY: target.y,
          midX: (source.x + target.x) / 2, midY: (source.y + target.y) / 2, arrowWidth: 0,
        });
      });
    };
    const counts = [9, 14, 4, 8, 6, 3, 12, 7, 5, 10];
    const snapshot = {progress: {
      groups: counts.map((_, i) => ({group_id: 'g' + i, name: 'Variable group ' + i})),
      variables: counts.flatMap((count, i) => Array.from({length: count}, (_, j) => ({
        item_id: i * 100 + j, group_id: 'g' + i, name: 'Variable ' + j}))),
    }};
    mapIndex.notes = Array.from({length: 18}, (_, i) => ({id: 'note:' + i, noteId: i}));
    syncMap(snapshot);`);
  t.after(() => run('if (cy) { cy.container = () => undefined; cy.destroy(); }'));
  const resize = () => {
    run('refitMap()');
    for (const [id, entry] of timers) {
      if (entry.delay !== 120) continue;
      timers.delete(id);
      entry.callback();
    }
  };
  return {run, json, rects, reads, resize};
}

function close(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-6, `${message}: ${actual} ≈ ${expected}`);
}

function fittedBounds(h) {
  const view = h.json('viewportBox()');
  const bounds = h.json(`cy.elements().renderedBoundingBox(${boundsOptions})`);
  const pad = h.run('FIT_PAD');
  assert.ok(bounds.x1 >= view.x + pad - 1e-6, 'left label/body bound fits');
  assert.ok(bounds.y1 >= view.y + pad - 1e-6, 'top label/body bound fits');
  assert.ok(bounds.x2 <= view.x + view.w - pad + 1e-6, 'right label/body bound fits');
  assert.ok(bounds.y2 <= view.y + view.h - pad + 1e-6, 'bottom label/body bound fits');
  close((bounds.x1 + bounds.x2) / 2, view.x + view.w / 2, 'horizontal center');
  close((bounds.y1 + bounds.y2) / 2, view.y + view.h / 2, 'vertical center');
  return bounds;
}

test('full-width canvas fits the labeled map beside the overlay and below its title', (t) => {
  const h = harness(t);
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 48, w: 966, h: 864});
  const bounds = fittedBounds(h);
  assert.ok(bounds.x2 < 982, 'graph stays left of the 896px window at right:16px');
  assert.ok(h.run(`cy.elements().boundingBox(${boundsOptions}).h >
    cy.elements().boundingBox({includeLabels: false, includeOverlays: false, includeUnderlays: false}).h`),
  'real label projection contributes to the fitted bounds');

  const zoom = h.run('cy.zoom()');
  h.rects.drawing.width = h.rects.canvas.width * 0.45;
  h.resize();
  fittedBounds(h);
  assert.ok(zoom > h.run('cy.zoom()'), 'representative graph renders larger than in the former 45% region');
});

test('probe offsets and actual width/height changes drive packing and fitting', (t) => {
  const h = harness(t);
  const originalKey = h.run('layoutKey');
  h.run(`lastRenderT = 7.25; lastView = {snapshot, step: {node: 'plan_extraction'}};
    stepAnim = 77; stepFallback = 88;
    const repaints = [];
    renderMapAt = (...args) => repaints.push(args);`);
  h.rects.canvas = {left: 50, top: 200, width: 1500, height: 1100};
  h.rects.drawing = {left: 74, top: 256, width: 620, height: 1044};
  h.resize();
  assert.deepEqual(h.json('viewportBox()'), {x: 24, y: 56, w: 620, h: 1044});
  assert.notEqual(h.run('layoutKey'), originalKey, 'a tall rectangle chooses a new packing');
  fittedBounds(h);
  assert.equal(h.run('repaints.length'), 1);
  assert.equal(h.run('repaints[0][0]'), 7.25, 'resize preserves the current animation frame');
  assert.equal(h.run('repaints[0][1] === lastView.snapshot'), true);
  assert.equal(h.run('repaints[0][2] === lastView.step'), true);
  assert.equal(h.run('stepAnim'), 77);
  assert.equal(h.run('stepFallback'), 88);

  h.rects.canvas = {left: 50, top: 200, width: 2200, height: 640};
  h.rects.drawing = {left: 50, top: 248, width: 1272, height: 592};
  h.resize();
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 48, w: 1272, h: 592});
  fittedBounds(h);
});

test('fitting honors custom minimum and maximum zoom while keeping the probe center', (t) => {
  const h = harness(t);
  h.run('cy.minZoom(0.1); cy.maxZoom(0.35); fitMap()');
  close(h.run('cy.zoom()'), 0.35, 'maximum zoom');
  fittedBounds(h);
  h.run('cy.maxZoom(4); cy.minZoom(3); fitMap()');
  close(h.run('cy.zoom()'), 3, 'minimum zoom');
  const bounds = h.json(`cy.elements().renderedBoundingBox(${boundsOptions})`);
  close((bounds.x1 + bounds.x2) / 2, 483, 'minimum-clamped horizontal center');
  close((bounds.y1 + bounds.y2) / 2, 480, 'minimum-clamped vertical center');
});

test('missing probe, missing DOM geometry and headless initialization use full-canvas fallback', (t) => {
  const h = harness(t);
  const drawing = h.rects.drawing;
  h.rects.drawing = null;
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1894, h: 912});
  h.run('fitMap()');
  fittedBounds(h);
  h.rects.drawing = drawing;
  h.run('cy.container = () => undefined');
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1894, h: 912});
  h.run('cy.container = () => ({}); document.getElementById = undefined');
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1894, h: 912});
  h.rects.canvas.width = 1;
  h.rects.canvas.height = 1;
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1100, h: 500});
  h.run('fitMap()');
  fittedBounds(h);
  h.run('cy.container = () => undefined; cy.destroy(); cy = null');
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1100, h: 500});
  assert.doesNotThrow(() => h.run('fitMap()'));
});

test('empty elements, zero-sized containers and invalid bounds remain safe', (t) => {
  const h = harness(t);
  h.rects.drawing.width = 0;
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1894, h: 912});
  h.rects.canvas.width = 0;
  h.rects.canvas.height = 0;
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 0, w: 1100, h: 500});
  h.run('fitMap()');
  fittedBounds(h);
  const viewport = h.json('({zoom: cy.zoom(), pan: cy.pan()})');
  h.run('cy.elements().remove(); fitMap()');
  assert.deepEqual(h.json('({zoom: cy.zoom(), pan: cy.pan()})'), viewport);

  h.run(`const originalElements = cy.elements;
    cy.elements = () => ({empty: () => false, boundingBox: () =>
      ({x1: NaN, y1: 0, x2: Infinity, y2: 10, w: Infinity, h: 10})});
    fitMap(); cy.elements = originalElements;`);
  assert.deepEqual(h.json('({zoom: cy.zoom(), pan: cy.pan()})'), viewport);
});

test('nonzero probe is authoritative even when small; unusable padded area does not move the map', (t) => {
  const h = harness(t);
  const viewport = h.json('({zoom: cy.zoom(), pan: cy.pan()})');
  h.rects.drawing.width = 20;
  assert.deepEqual(h.json('viewportBox()'), {x: 0, y: 48, w: 20, h: 864});
  h.run('fitMap()');
  assert.deepEqual(h.json('({zoom: cy.zoom(), pan: cy.pan()})'), viewport);
});

test('selection and activity decoration do not affect layout or fitted geometry', (t) => {
  const h = harness(t);
  const geometry = () => h.json(`({positions: cy.nodes().map(node => [node.id(), node.position()]),
    pan: cy.pan(), zoom: cy.zoom(), layoutKey})`);
  const before = geometry();
  const packed = h.json('computeLayout(lastMapModel)');
  h.run(`detailSelection = new Proxy({}, {get() { throw new Error('geometry consulted selection'); }});
    cy.nodes().addClass('entity-selected');
    cy.nodes().style({'underlay-opacity': 1, 'underlay-padding': 200});`);
  assert.ok(h.run(`cy.elements().boundingBox().w > cy.elements().boundingBox(${boundsOptions}).w`),
    'decorative bounds are larger than the drawing');
  h.run('fitMap()');
  assert.deepEqual(h.json('computeLayout(lastMapModel)'), packed);
  assert.deepEqual(geometry(), before);
  h.run(`detailSelection = null; cy.nodes().removeClass('entity-selected');
    cy.nodes().removeStyle('underlay-opacity underlay-padding'); fitMap();`);
  assert.deepEqual(geometry(), before);
  assert.ok(h.reads.every(id => id === 'map-layout'));
});
