"""
flybench.py - dumbflies.py as a research bench.

Runs the ORIGINAL BugConfig / ThinkingField / EnhancedBug classes from
dumbflies.py unchanged (exec'd straight from the source file), with the
webcam replaced by a synthetic room scene so runs are reproducible, and the
Tk GUI loop replaced by a headless copy of BugGUI.process_frame's logic.

Logs two kinds of data per fly per frame:
  observable : x, y, heading  (what an outside observer can see)
  hidden     : energy, decision-field centroid, pattern count, is_mating,
               returned velocity / rotation (ground truth for checking)
"""
import sys, os, random, time, types
import numpy as np
import cv2
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def find_dumbflies():
    """Look for dumbflies.py: $DUMBFLIES, then this folder, its parent, the cwd."""
    cands = [os.environ.get("DUMBFLIES", ""),
             os.path.join(HERE, "dumbflies.py"),
             os.path.join(HERE, "..", "dumbflies.py"),
             os.path.join(os.getcwd(), "dumbflies.py")]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit("Can't find dumbflies.py. Put it in this folder or the folder above,\n"
             "or point to it:  set DUMBFLIES=C:\\path\\to\\dumbflies.py\n"
             "Looked in:\n  " + "\n  ".join(os.path.abspath(c) for c in cands if c))


SRC = find_dumbflies()


def load_fly_classes():
    """Exec the model part of dumbflies.py (BugConfig .. EnhancedBug), untouched."""
    src = open(SRC, encoding="utf-8", errors="replace", newline="").read()
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    try:
        start = src.rindex("@dataclass", 0, src.index("class BugConfig"))
        end = src.index("# BugGUI Class")
    except ValueError:
        sys.exit(f"{SRC} doesn't look like the dumbflies.py this bench was built on\n"
                 "(expected '@dataclass class BugConfig' ... '# BugGUI Class').")
    ns = {}
    prelude = (
        "import numpy as np, cv2, random, time, logging\n"
        "from dataclasses import dataclass, field\n"
        "from collections import deque\n"
        "from typing import Dict, Tuple, List, Optional\n"
        "from scipy.ndimage import gaussian_filter\n"
    )
    exec(prelude + src[start:end], ns)
    return ns["BugConfig"], ns["ThinkingField"], ns["EnhancedBug"]


BugConfig, ThinkingField, EnhancedBug = load_fly_classes()

# --- speed patch -----------------------------------------------------------
# The original detect_patterns / associate_patterns / generate_decision are
# per-pixel Python loops (~100x too slow for batch runs). These vectorized
# versions compute the same quantities; check_equivalence() verifies it.
from scipy.ndimage import convolve as _conv

_orig = {k: getattr(ThinkingField, k) for k in
         ("detect_patterns", "associate_patterns", "generate_decision")}


def _detect(self):
    self.patterns = np.argwhere(self.field > self.pattern_threshold)   # row-major (i,j)
    return self.patterns


def _associate(self, src):
    if len(src):
        m = np.zeros_like(self.field)
        np.add.at(m, (np.asarray(src)[:, 0], np.asarray(src)[:, 1]), 1.0)
        self.field += 0.02 * _conv(m, np.ones((3, 3)), mode="constant")
    self.field = np.clip(self.field, 0, 1)


def _decide(self):
    if len(self.patterns) == 0:
        return (0.0, 0.0, False, False)
    p = np.asarray(self.patterns)
    n = self.field_size
    dx = (p[:, 1].mean() - n / 2) / (n / 2)
    dy = (p[:, 0].mean() - n / 2) / (n / 2)
    dens = len(p) / (n * n)
    return (dx, dy, dens > 0.1, dens > 0.2)


def check_equivalence(trials=200):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        f0 = rng.random((32, 32)) * rng.uniform(0.3, 1.2)
        a, b = ThinkingField(), ThinkingField()
        a.field, b.field = f0.copy(), f0.copy()
        pa = _orig["detect_patterns"](a)
        pb = _detect(b)
        assert [tuple(x) for x in pb] == pa
        da, db = _orig["generate_decision"](a), _decide(b)
        assert da[2:] == db[2:]
        worst = max(worst, abs(da[0] - db[0]), abs(da[1] - db[1]))
        g0 = rng.random((32, 32)) * 0.5
        a.field, b.field = g0.copy(), g0.copy()
        _orig["associate_patterns"](a, pa)
        _associate(b, pb)
        worst = max(worst, np.abs(a.field - b.field).max())
    return worst


ThinkingField.detect_patterns = _detect
ThinkingField.associate_patterns = _associate
ThinkingField.generate_decision = _decide

# get_vision_cone: cv2.bitwise_and with a mask is ~40 ms/call on this machine.
# Grayscale is per-pixel, so gray(mask(img)) == mask(gray(img)); drawing the
# other bugs directly in gray with their gray value is identical (LINE_8, no AA).
_orig_cone = EnhancedBug.get_vision_cone
_gray_cache = {"id": None, "g": None}


def _gray_of(bgr):
    return int(cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2GRAY)[0, 0])


_G_MATE, _G_PLAIN = _gray_of((255, 0, 255)), _gray_of((255, 255, 255))


def _cone(self, frame):
    if _gray_cache["id"] is not frame:
        _gray_cache["id"], _gray_cache["g"] = frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = _gray_cache["g"].copy()
    h, w = g.shape
    ang, L = self.traits["vision_cone_angle"], self.traits["vision_cone_length"]
    la, ra = self.angle - ang / 2, self.angle + ang / 2
    pts = np.array([[int(self.x), int(self.y)],
                    [int(self.x + L * np.cos(la)), int(self.y + L * np.sin(la))],
                    [int(self.x + L * np.cos(ra)), int(self.y + L * np.sin(ra))]], np.int32)
    for o in self.nearby_bugs:
        if o != self:
            cv2.circle(g, (int(o.x), int(o.y)), self.config.bug_visibility_size,
                       _G_MATE if o.is_mating else _G_PLAIN, -1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    g[mask == 0] = 0
    return cv2.resize(g, (self.config.vision_width, self.config.vision_height))


def check_cone_equivalence(frame, trials=100):
    rng = np.random.default_rng(2)
    cfg = BugConfig()
    for k in range(trials):
        bugs = [EnhancedBug(rng.uniform(0, 1280), rng.uniform(0, 720), cfg) for _ in range(6)]
        for b in bugs:
            b.is_mating = bool(rng.random() < 0.5)
        me = bugs[0]
        me.x, me.y = bugs[1].x + 20, bugs[1].y + 10
        me.nearby_bugs = bugs[1:]
        _gray_cache["id"] = None
        a = _orig_cone(me, frame)
        b = _cone(me, frame)
        if not np.array_equal(a, b):
            return False
    return True


EnhancedBug.get_vision_cone = _cone

W, H = 1280, 720


# ---------------------------------------------------------------- scenes
class RoomScene:
    """Static textured room + bright window on the left + a dark 'person'
    who walks in, sits, walks out, on a loop. Deterministic."""

    def __init__(self, seed=0, person=True, noise=4.0):
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, (H // 8, W // 8))
        base = cv2.GaussianBlur(base, (0, 0), 3)
        base = (base - base.min()) / (np.ptp(base) + 1e-9)
        bg = cv2.resize(base, (W, H)) * 90 + 60           # 60..150 room texture
        bg[:, :200] += 90                                  # bright window strip
        bg[430:560, 700:1150] -= 35                        # dark table
        self.bg = np.clip(bg, 0, 255).astype(np.float32)
        self.person = person
        self.noise = noise
        self.rng = np.random.default_rng(seed + 1)

    def person_pos(self, t):
        # 900-frame cycle: absent, walk in from right, sit, walk out
        c = t % 900
        if c < 150:
            return None
        if c < 350:                       # walk in
            a = (c - 150) / 200
            return (1250 - a * 800, 420)
        if c < 700:                       # sit, small sway
            return (450 + 15 * np.sin(c / 20), 430 + 5 * np.sin(c / 7))
        a = (c - 700) / 200               # walk out
        return (450 + a * 800, 420)

    def frame(self, t):
        f = self.bg.copy()
        if self.person:
            p = self.person_pos(t)
            if p is not None:
                cv2.ellipse(f, (int(p[0]), int(p[1])), (110, 230), 0, 0, 360, 35, -1)
                cv2.circle(f, (int(p[0]), int(p[1]) - 250), 70, 120, -1)
        if self.noise:
            f += self.rng.normal(0, self.noise, f.shape).astype(np.float32)
        g = np.clip(f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class ConstScene:
    def __init__(self, value):
        self.f = np.full((H, W, 3), value, np.uint8)

    def person_pos(self, t):
        return None

    def frame(self, t):
        return self.f


class RhythmScene(RoomScene):
    """The room + two flickering patches at different rates (a lamp, a screen).
    Patch A: 0.05 cycles/frame (1.5 Hz at 30 fps), Patch B: 0.15 cycles/frame."""
    PATCHES = [((330, 260), 0.05), ((960, 470), 0.15)]
    RADIUS, AMP = 100, 45.0

    def __init__(self, seed=0, person=True, noise=4.0):
        super().__init__(seed, person, noise)
        yy, xx = np.mgrid[0:H, 0:W]
        self.masks = []
        for (cx, cy), _ in self.PATCHES:
            dd = np.hypot(xx - cx, yy - cy)
            self.masks.append(np.clip((self.RADIUS - dd) / 20 + 0.5, 0, 1).astype(np.float32))

    def frame(self, t):
        f = super().frame(t).astype(np.float32)[:, :, 0]
        for m, (_, fr) in zip(self.masks, self.PATCHES):
            f += m * self.AMP * np.sin(2 * np.pi * fr * t)
        g = np.clip(f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


SCENES = {
    "room": lambda s: RoomScene(s, person=True),
    "room_static": lambda s: RoomScene(s, person=False),
    "gray": lambda s: ConstScene(128),
    "black": lambda s: ConstScene(0),
    "rhythm": lambda s: RhythmScene(s, person=True),
}


def brain_class(name):
    if name == "original":
        return EnhancedBug
    import resonator_brain as rb
    return rb.make_bug_class(EnhancedBug, name)


# ---------------------------------------------------------------- run loop
def run(scene_name="room", n_frames=3000, seed=0, verbose=False, brain="original",
        pop=None, breed=True):
    """Headless copy of BugGUI.initialize_population + process_frame.
    pop: override initial population; breed=False disables births (no lineages,
    starved flies are replaced by fresh random ones via the original repopulate)."""
    random.seed(seed)
    np.random.seed(seed)
    cfg = BugConfig()
    if pop:
        cfg.initial_population = pop
    if not breed:
        cfg.max_population = 0
    scene = SCENES[scene_name](seed)
    Bug = brain_class(brain)

    uid = [0]

    def new_bug(x, y, **kw):
        b = Bug(x=x, y=y, config=cfg, seed=random.randint(0, 1_000_000), **kw)
        b.screen_width, b.screen_height = W, H
        b.uid = uid[0]
        uid[0] += 1
        return b

    # initialize_population (same grid layout as the GUI)
    bugs = []
    margin = 50
    gs = int(np.ceil(np.sqrt(cfg.initial_population)))
    sx, sy = (W - 2 * margin) / gs, (H - 2 * margin) / gs
    for i in range(cfg.initial_population):
        gx, gy = i % gs, i // gs
        bugs.append(new_bug(margin + gx * sx + random.uniform(-sx / 4, sx / 4),
                            margin + gy * sy + random.uniform(-sy / 4, sy / 4)))

    rows = []
    births = deaths = 0
    t0 = time.time()
    for t in range(n_frames):
        frame = scene.frame(t)
        grid = {}
        for b in bugs:
            grid.setdefault((int(b.x // 100), int(b.y // 100)), []).append(b)
        dead, newb = [], []
        for b in bugs:
            gx, gy = int(b.x // 100), int(b.y // 100)
            near = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    near.extend(grid.get((gx + dx, gy + dy), []))
            b.nearby_bugs = [o for o in near if o is not b and
                             np.hypot(b.x - o.x, b.y - o.y) < cfg.bug_detection_range]
            x0, y0, a0 = b.x, b.y, b.angle
            vel, rot = b.process_frame(frame)
            # hidden ground truth read BEFORE the move
            pats = np.asarray(b.decision_field.patterns)
            if len(pats):
                ci = float(pats[:, 0].mean())
                cj = float(pats[:, 1].mean())
            else:
                ci = cj = np.nan
            b.update_position(vel, rot)
            rows.append((t, b.uid, x0, y0, a0, b.x, b.y, b.angle,
                         vel, rot, b.energy, len(pats), ci, cj, int(b.is_mating),
                         b.traits.get("omega", np.nan), b.traits.get("r", np.nan)))
            if b.energy <= 0:
                dead.append(b)
                deaths += 1
            if b.is_mating and len(bugs) < cfg.max_population:
                for o in b.nearby_bugs:
                    if o.is_mating and o not in newb:
                        c = b.try_mate(o)
                        if c:
                            c.screen_width, c.screen_height = W, H
                            c.uid = uid[0]; uid[0] += 1
                            newb.append(c)
                            births += 1
                            break
        for b in dead:
            bugs.remove(b)
        bugs.extend(newb)
        if len(bugs) < cfg.min_population:
            while len(bugs) < cfg.initial_population:
                bugs.append(new_bug(random.randint(0, W), random.randint(0, H)))
        if verbose and t % 500 == 0:
            print(f"  t={t} pop={len(bugs)} births={births} deaths={deaths} "
                  f"{time.time()-t0:.0f}s", flush=True)

    cols = ["t", "uid", "x0", "y0", "a0", "x1", "y1", "a1", "vel", "rot",
            "energy", "npat", "ci", "cj", "mating", "omega", "r"]
    data = {c: np.array([r[i] for r in rows], dtype=float) for i, c in enumerate(cols)}
    data["births"], data["deaths"] = births, deaths
    return data


if __name__ == "__main__":
    import pickle
    scenes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["room"]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
    brain = sys.argv[4] if len(sys.argv) > 4 else "original"     # original|resonator|window
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    for sc in scenes:
        for sd in seeds:
            print(f"== {sc} seed {sd} brain {brain}", flush=True)
            d = run(sc, n, sd, verbose=True, brain=brain)
            tag = "" if brain == "original" else f"{brain}_"
            with open(os.path.join(HERE, "runs", f"{tag}{sc}_s{sd}.pkl"), "wb") as f:
                pickle.dump(d, f)
            print(f"   births={d['births']} deaths={d['deaths']}", flush=True)
