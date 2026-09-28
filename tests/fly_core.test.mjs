import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');

async function core() {
  try {
    return await import(pathToFileURL(path.join(ROOT, 'fly_core.js')).href + `?t=${Date.now()}`);
  } catch (err) {
    assert.fail(`fly_core.js must exist and load as an ES module: ${err.message}`);
  }
}

test('matching resonator accumulates more activity than a mismatched resonator', async () => {
  const { makeFly, driveFly } = await core();
  const matching = makeFly({ omega: 0.94, r: 0.92, energy: 100 });
  const mismatch = makeFly({ omega: 0.31, r: 0.92, energy: 100 });
  let heardMatch = 0;
  let heardMismatch = 0;
  for (let t = 0; t < 180; t++) {
    const sample = 128 + 110 * Math.sin(0.94 * t);
    heardMatch += driveFly(matching, sample).heard;
    heardMismatch += driveFly(mismatch, sample).heard;
  }
  assert.ok(heardMatch > heardMismatch * 1.5, `${heardMatch} should dominate ${heardMismatch}`);
});

test('a click rewards only flies carrying eligibility', async () => {
  const { makeFly, Soma } = await core();
  const active = makeFly({ omega: 0.94, energy: 30 });
  const quiet = makeFly({ omega: 0.31, energy: 30 });
  active.settled = quiet.settled = true;
  active.heard = 20;
  quiet.heard = 0;
  const soma = new Soma();
  for (let i = 0; i < 8; i++) soma.step([active, quiet], { click: false, learn: true });
  const beforeA = active.energy;
  const beforeQ = quiet.energy;
  const result = soma.step([active, quiet], { click: true, learn: true });
  assert.ok(active.energy > beforeA);
  assert.equal(quiet.energy, beforeQ);
  assert.ok(result.fed.some(({ fly }) => fly === active));
  assert.ok(!result.fed.some(({ fly }) => fly === quiet));
});

test('teaching opposite rhythms reverses the population tuning shift', async () => {
  const { runTeachingExperiment } = await core();
  const taughtA = runTeachingExperiment({ targetOmega: 0.94, seed: 11, frames: 1600 });
  const taughtB = runTeachingExperiment({ targetOmega: 0.31, seed: 11, frames: 1600 });
  assert.ok(taughtA.medianOmega > taughtB.medianOmega + 0.20, `${taughtA.medianOmega} vs ${taughtB.medianOmega}`);
  assert.ok(taughtA.nearTargetShare > 0.35, `A share ${taughtA.nearTargetShare}`);
  assert.ok(taughtB.nearTargetShare > 0.35, `B share ${taughtB.nearTargetShare}`);
});

test('arbor grows only when rewarded positions deposit resource', async () => {
  const { Arbor, seededRandom } = await core();
  const rng = seededRandom(3);
  const empty = new Arbor({ x: 20, y: 50, rng });
  for (let i = 0; i < 30; i++) empty.step();
  assert.equal(empty.nodes.length, 1);

  const grown = new Arbor({ x: 20, y: 50, rng: seededRandom(3) });
  grown.feed([{ x: 90, y: 50, color: '#ffcc55' }], { perPoint: 8, jitter: 2 });
  for (let i = 0; i < 50; i++) grown.step();
  assert.ok(grown.nodes.length > 4, `nodes=${grown.nodes.length}`);
  assert.ok(grown.nodes.at(-1).x > 35, `tip x=${grown.nodes.at(-1).x}`);
});

test('public page exposes synthetic start, camera, teaching, reversal and reset controls', async () => {
  let html = '';
  try {
    html = await fs.readFile(path.join(ROOT, 'index.html'), 'utf8');
  } catch (err) {
    assert.fail(`index.html must exist: ${err.message}`);
  }
  for (const id of ['startBtn', 'cameraBtn', 'teachBtn', 'reverseBtn', 'resetBtn', 'worldCanvas', 'arborCanvas']) {
    assert.match(html, new RegExp(`id=["']${id}["']`));
  }
  assert.match(html, /type=["']module["']/);
  assert.match(html, /fly_core\.js/);
});
