// FlyBench browser core: dependency-free causal rules shared by the live page and tests.
// The constants mirror the Python clicker neuron where practical, but the browser
// uses one local luminance sample per fly rather than the Python 9-point eye.

export const PARAMS = Object.freeze({
  RATE: 0.10,
  KAPPA: 0.05,
  THETA_MIN: 1.0,
  LAM: 0.85,
  BETA: 2.0,
  G: 2.0,
  BUDGET: 120.0,
  CAP: 40.0,
  FA_WINDOW: 30,
  FA_COST: 3.0,
  BUD_E: 130.0,
  MAX_POP: 60,
  ANCHOR: 400,
  OMEGA_MUT_SD: 0.12,
  MEAN_ALPHA: 0.02,
  NOISE_FLOOR: 4.0,
});

export function seededRandom(seed = 1) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makeFly({
  x = 0,
  y = 0,
  omega = 0.5,
  r = 0.9,
  energy = 100,
  angle = 0,
} = {}) {
  return {
    x, y, angle,
    omega: Math.max(0.03, Math.min(2.8, omega)),
    r: Math.max(0.5, Math.min(0.985, r)),
    energy,
    mean: null,
    zRe: 0,
    zIm: 0,
    heard: 0,
    activity: 0,
    elig: 0,
    contrib: 0,
    settled: true,
    anchor: 0,
    vx: 0,
    vy: 0,
  };
}

export function driveFly(fly, sample, { noiseFloor = PARAMS.NOISE_FLOOR } = {}) {
  if (fly.mean == null) fly.mean = sample;
  const d = sample - fly.mean;
  fly.mean += PARAMS.MEAN_ALPHA * d;

  const c = Math.cos(fly.omega);
  const s = Math.sin(fly.omega);
  const oldRe = fly.zRe;
  const oldIm = fly.zIm;
  const rotatedRe = c * oldRe - s * oldIm;
  const rotatedIm = s * oldRe + c * oldIm;
  fly.zRe = fly.r * rotatedRe + (1 - fly.r) * d;
  fly.zIm = fly.r * rotatedIm;
  fly.activity = Math.hypot(fly.zRe, fly.zIm);
  fly.heard = Math.max(fly.activity - noiseFloor, 0);
  return { heard: fly.heard, activity: fly.activity, delta: d };
}

function weightedMedian(values) {
  if (!values.length) return NaN;
  const a = [...values].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export class Soma {
  constructor({ theta = 5, rng = Math.random, maxPop = PARAMS.MAX_POP } = {}) {
    this.theta = theta;
    this.t = 0;
    this.pending = [];
    this.V = 0;
    this.y = false;
    this.born = 0;
    this.rng = rng;
    this.maxPop = maxPop;
  }

  step(flies, { click = false, learn = true } = {}) {
    this.t += 1;
    let V = 0;
    for (const fly of flies) {
      const w = Math.max(0.2, Math.min(1.5, fly.energy / 100));
      fly.contrib = fly.settled ? w * fly.heard : 0;
      V += fly.contrib;
    }
    const y = V > this.theta;
    this.theta = Math.max(
      PARAMS.THETA_MIN,
      this.theta * Math.exp(PARAMS.KAPPA * ((y ? 1 : 0) - PARAMS.RATE)),
    );
    this.V = V;
    this.y = y;

    for (const fly of flies) {
      fly.elig = PARAMS.LAM * (fly.elig || 0) + fly.contrib * (1 + PARAMS.BETA * (y ? 1 : 0));
    }

    if (!learn) return { y, V, fed: [], born: [] };

    for (const fly of flies) if (fly.anchor > 0) fly.anchor -= 1;
    if (y) {
      const contributors = flies
        .filter((fly) => fly.contrib > 0)
        .map((fly) => ({ fly, contrib: fly.contrib }));
      this.pending.push({ t: this.t, contributors });
    }

    const fed = [];
    if (click) {
      const wants = flies
        .filter((fly) => fly.elig > 0)
        .map((fly) => ({ fly, want: Math.min(PARAMS.G * fly.elig, PARAMS.CAP) }));
      const total = wants.reduce((sum, item) => sum + item.want, 0);
      const scale = total > 0 ? Math.min(1, PARAMS.BUDGET / total) : 0;
      for (const item of wants) {
        const food = item.want * scale;
        item.fly.energy = Math.min(item.fly.energy + food, 150);
        if (food > 1) {
          item.fly.anchor = PARAMS.ANCHOR;
          fed.push({ fly: item.fly, food });
        }
      }
      this.pending = [];
    }

    const keep = [];
    for (const spike of this.pending) {
      if (this.t - spike.t >= PARAMS.FA_WINDOW) {
        const total = spike.contributors.reduce((sum, item) => sum + item.contrib, 0);
        if (total > 0) {
          for (const item of spike.contributors) {
            item.fly.energy -= PARAMS.FA_COST * item.contrib / total;
          }
        }
      } else {
        keep.push(spike);
      }
    }
    this.pending = keep;

    const born = this.bud(flies);
    return { y, V, fed, born };
  }

  bud(flies) {
    const born = [];
    const original = [...flies];
    for (const fly of original) {
      if (fly.energy <= PARAMS.BUD_E || flies.length + born.length >= this.maxPop) continue;
      const child = makeFly({
        x: fly.x + (this.rng() * 24 - 12),
        y: fly.y + (this.rng() * 24 - 12),
        angle: fly.angle,
        omega: fly.omega * Math.exp(gaussian(this.rng) * PARAMS.OMEGA_MUT_SD),
        r: fly.r + gaussian(this.rng) * 0.01,
        energy: fly.energy / 2,
      });
      child.anchor = fly.anchor;
      fly.energy /= 2;
      born.push(child);
    }
    flies.push(...born);
    this.born += born.length;
    return born;
  }
}

export class Arbor {
  constructor({ x, y, rng = Math.random, attract = 110, kill = 9, seg = 4, noise = 0.15, life = 900, maxNodes = 6000 }) {
    this.nodes = [{ x, y, parent: -1, color: '#ffffff', count: 1 }];
    this.resources = [];
    this.rng = rng;
    this.attract = attract;
    this.kill = kill;
    this.seg = seg;
    this.noise = noise;
    this.life = life;
    this.maxNodes = maxNodes;
  }

  feed(points, { perPoint = 3, jitter = 22 } = {}) {
    for (const p of points) {
      for (let k = 0; k < perPoint; k++) {
        this.resources.push({
          x: p.x + gaussian(this.rng) * jitter,
          y: p.y + gaussian(this.rng) * jitter,
          color: p.color || '#ffffff',
          age: 0,
        });
      }
    }
  }

  step() {
    if (!this.resources.length || this.nodes.length >= this.maxNodes) return 0;
    const assignments = new Map();
    const keep = [];

    for (const resource of this.resources) {
      resource.age += 1;
      if (resource.age >= this.life) continue;
      let best = -1;
      let bestD = Infinity;
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const d = Math.hypot(resource.x - n.x, resource.y - n.y);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (bestD <= this.kill) continue;
      keep.push(resource);
      if (bestD < this.attract) {
        if (!assignments.has(best)) assignments.set(best, []);
        assignments.get(best).push(resource);
      }
    }

    let grew = 0;
    for (const [parentIndex, rs] of assignments) {
      if (this.nodes.length >= this.maxNodes) break;
      const p = this.nodes[parentIndex];
      let dx = 0, dy = 0;
      for (const r of rs) {
        const len = Math.hypot(r.x - p.x, r.y - p.y) || 1;
        dx += (r.x - p.x) / len;
        dy += (r.y - p.y) / len;
      }
      dx /= rs.length;
      dy /= rs.length;
      dx += gaussian(this.rng) * this.noise;
      dy += gaussian(this.rng) * this.noise;
      const len = Math.hypot(dx, dy) || 1;
      dx /= len; dy /= len;
      const node = {
        x: p.x + this.seg * dx,
        y: p.y + this.seg * dy,
        parent: parentIndex,
        color: rs[0].color || p.color,
        count: 1,
      };
      this.nodes.push(node);
      let j = parentIndex;
      while (j >= 0) {
        this.nodes[j].count += 1;
        j = this.nodes[j].parent;
      }
      grew += 1;
    }
    this.resources = keep;
    return grew;
  }
}

export function runTeachingExperiment({ targetOmega, seed = 1, frames = 1600 } = {}) {
  const rng = seededRandom(seed);
  const flies = [];
  for (let i = 0; i < 28; i++) {
    flies.push(makeFly({
      omega: 0.12 + rng() * 1.08,
      r: 0.88 + rng() * 0.07,
      energy: 72 + rng() * 8,
    }));
  }
  const soma = new Soma({ rng, theta: 12 });
  let clicks = 0;
  for (let t = 0; t < frames; t++) {
    const sample = 128 + 110 * Math.sin(targetOmega * t);
    for (const fly of flies) {
      driveFly(fly, sample, { noiseFloor: 20 });
      fly.settled = true;
      fly.energy -= 0.07;
    }
    const click = t > 100 && t % 22 === 0;
    if (click) clicks += 1;
    soma.step(flies, { click, learn: true });
    for (let i = flies.length - 1; i >= 0; i--) if (flies[i].energy <= 4) flies.splice(i, 1);
    while (flies.length < 10) {
      flies.push(makeFly({ omega: 0.12 + rng() * 1.08, r: 0.88 + rng() * 0.07, energy: 80 }));
    }
  }
  const omegas = flies.map((f) => f.omega);
  const band = 0.16;
  return {
    medianOmega: weightedMedian(omegas),
    nearTargetShare: omegas.filter((w) => Math.abs(w - targetOmega) < band).length / omegas.length,
    population: flies.length,
    clicks,
  };
}

export function hueForOmega(omega, min = 0.18, max = 1.12) {
  const u = Math.max(0, Math.min(1, (omega - min) / (max - min)));
  return `hsl(${Math.round(210 - 190 * u)} 88% 64%)`;
}
