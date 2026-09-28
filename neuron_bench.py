"""
neuron_bench.py - can a fly colony be clicker-trained into a neuron?

World: the room, one lamp in the middle that always flickers, switching between
  A = 0.15 cycles/frame (target)   and   B = 0.05 cycles/frame (distractor),
in segments of 120-240 frames, plus the room light switching (x0.6 / x1.0)
every 400-700 frames.

Teacher: clicks 8 frames after an A segment starts, then every 30 frames while
it lasts (someone watching the lamp and clicking when it does the thing).

Conditions (same world, same seeds):
  clicker     teacher clicks on A                      the neuron
  clicker_B   teacher clicks on B                      same world, other history
  shuffled    same number of clicks at random times    is it the teacher's timing?
  no_credit   no teacher; flies eat what they hear     the plain resonator colony
              (as in resonator_flies) with the same soma
  + bank      40 fixed random resonators at random places, logistic readout
              trained on every frame's true label (much more supervision)

Test (identical for all): 900 frames, A/B alternating in 150-frame segments,
light switches at 300 and 750; learning, energy decay and births off.
Score: soma firing rate during A, during B, and in the 60 frames after a
light switch; plus the colony's tuning.
"""
import sys, os, json, time, random
import numpy as np
import cv2
import flybench as fb
import neuron as nr

W, H = fb.W, fb.H
LAMP = (640, 360)
FA, FB = 0.15, 0.05
TRAIN, TEST = 4000, 900


class LampScene:
    def __init__(self, seed, schedule, toggles):
        self.room = fb.RoomScene(seed, person=False)
        yy, xx = np.mgrid[0:H, 0:W]
        dd = np.hypot(xx - LAMP[0], yy - LAMP[1])
        self.mask = np.clip((110 - dd) / 20 + 0.5, 0, 1).astype(np.float32)
        self.schedule, self.toggles = schedule, sorted(toggles)

    def label(self, t):
        for s, e, lab in self.schedule:
            if s <= t < e:
                return lab
        return None

    def light(self, t):
        return 0.6 if sum(1 for x in self.toggles if x <= t) % 2 else 1.0

    def frame(self, t):
        f = self.room.frame(t).astype(np.float32)[:, :, 0] * self.light(t)
        lab = self.label(t)
        if lab:
            f += self.mask * 45.0 * np.sin(2 * np.pi * (FA if lab == "A" else FB) * t)
        return cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def train_world(seed):
    rng = np.random.default_rng(100 + seed)
    sched, t, lab = [], 0, rng.choice(["A", "B"])
    while t < TRAIN:
        L = int(rng.integers(120, 241))
        sched.append((t, t + L, lab))
        t += L
        lab = "B" if lab == "A" else "A"
    tog, t = [], int(rng.integers(400, 701))
    while t < TRAIN:
        tog.append(t)
        t += int(rng.integers(400, 701))
    return LampScene(seed, sched, tog)


def test_world(seed):
    sched = [(TRAIN + i * 150, TRAIN + (i + 1) * 150, "AB"[i % 2]) for i in range(6)]
    return LampScene(seed, sched, [TRAIN + 300, TRAIN + 750])


def clicks_for(scene, teach, rng, n_frames):
    on = set()
    if teach in ("A", "B"):
        for s, e, lab in scene.schedule:
            if lab == teach:
                on.update(range(s + 8, min(e, n_frames), 30))
    return on


def run(cond, seed, verbose=False):
    random.seed(seed); np.random.seed(seed)
    cfg = fb.BugConfig()
    cfg.max_population = 60
    Bug = nr.make_neuron_bug(fb.EnhancedBug, self_feed=1.0 if cond == "no_credit" else 0.0)
    g = fb.EnhancedBug.try_mate.__globals__
    saved = g["EnhancedBug"]; g["EnhancedBug"] = Bug
    try:
        return _run(cond, seed, cfg, Bug, verbose)
    finally:
        g["EnhancedBug"] = saved


def _run(cond, seed, cfg, Bug, verbose):
    rng = np.random.default_rng(seed)
    tr, te = train_world(seed), test_world(seed)
    teach = {"clicker": "A", "clicker_B": "B"}.get(cond)
    clicks = clicks_for(tr, teach, rng, TRAIN)
    if cond == "shuffled":
        n = len(clicks_for(tr, "A", rng, TRAIN))
        clicks = set(rng.choice(np.arange(TRAIN), n, replace=False).tolist())

    def new_bug(x, y):
        b = Bug(x=x, y=y, config=cfg, seed=random.randint(0, 1_000_000))
        b.screen_width, b.screen_height = W, H
        return b

    bugs = [new_bug(random.uniform(50, W - 50), random.uniform(50, H - 50))
            for _ in range(cfg.initial_population)]
    soma = nr.Soma()
    log = dict(t=[], y=[], V=[], label=[], light_since=[], pop=[], clicks=int(len(clicks)))
    omega_snap = {}
    t0 = time.time()
    for t in range(TRAIN + TEST):
        testing = t >= TRAIN
        if t == TRAIN:
            cfg.energy_decay = 0.0
            cfg.max_population = 0
            omega_snap["end_train"] = [b.traits["omega"] for b in bugs]
        scene = te if testing else tr
        frame = scene.frame(t)
        grid = {}
        for b in bugs:
            grid.setdefault((int(b.x // 100), int(b.y // 100)), []).append(b)
        dead, newb = [], []
        for b in bugs:
            gx, gy = int(b.x // 100), int(b.y // 100)
            near = [o for dx in (-1, 0, 1) for dy in (-1, 0, 1) for o in grid.get((gx + dx, gy + dy), [])]
            b.nearby_bugs = [o for o in near if o is not b and np.hypot(b.x - o.x, b.y - o.y) < cfg.bug_detection_range]
            v, r = b.process_frame(frame)
            b.update_position(v, r)
        y = soma.step(bugs, click=(t in clicks) and not testing, learn=not testing)
        for b in bugs:
            if b.energy <= 0:
                dead.append(b)
            elif b.is_mating and len(bugs) + len(newb) < cfg.max_population:
                for o in b.nearby_bugs:
                    if o.is_mating and o not in newb and o.energy > 0:
                        c = b.try_mate(o)
                        if c:
                            c.screen_width, c.screen_height = W, H
                            newb.append(c)
                            break
        for b in dead:
            bugs.remove(b)
        bugs.extend(newb)
        if len(bugs) < cfg.min_population and not testing:
            while len(bugs) < cfg.initial_population:
                bugs.append(new_bug(random.uniform(0, W), random.uniform(0, H)))
        last_tog = max([x for x in scene.toggles if x <= t], default=-10 ** 9)
        log["t"].append(t); log["y"].append(int(y)); log["V"].append(soma.V)
        log["label"].append(scene.label(t) or "-"); log["light_since"].append(t - last_tog)
        log["pop"].append(len(bugs))
        if verbose and t % 1000 == 0:
            print(f"  {cond} s{seed} t={t} pop={len(bugs)} theta={soma.theta:.1f} {time.time()-t0:.0f}s", flush=True)
    omega_snap["end_test"] = [b.traits["omega"] for b in bugs]
    log["omega"] = omega_snap
    return log


def score(log, lo=TRAIN):
    t = np.array(log["t"]); y = np.array(log["y"]); lab = np.array(log["label"])
    ls = np.array(log["light_since"])
    m = t >= lo
    r = lambda mask: float(y[m & mask].mean()) if (m & mask).any() else float("nan")
    wA, wB = 2 * np.pi * FA, 2 * np.pi * FB
    om = np.array(log["omega"]["end_train"])
    return dict(rate_A=r(lab == "A"), rate_B=r(lab == "B"),
                rate_after_light_switch=r(ls < 60), rate_overall=r(np.ones_like(m)),
                pop_end_train=int(np.array(log["pop"])[t == lo - 1][0]),
                frac_tuned_A=float(np.mean(np.abs(om - wA) < 0.15)) if len(om) else float("nan"),
                frac_tuned_B=float(np.mean(np.abs(om - wB) < 0.15)) if len(om) else float("nan"),
                median_omega=float(np.median(om)) if len(om) else float("nan"),
                clicks=log["clicks"])


if __name__ == "__main__":
    conds = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    os.makedirs("neuron_runs", exist_ok=True)
    for c in conds:
        for s in seeds:
            lg = run(c, s, verbose=True)
            sc = score(lg)
            json.dump(dict(score=sc, log=lg), open(f"neuron_runs/{c}_s{s}.json", "w"))
            print(c, s, json.dumps(sc), flush=True)
