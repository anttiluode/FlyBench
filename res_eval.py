"""
res_eval.py - score resonator flies vs same-window control vs original flies.

1. Sorting    do flies tuned to a patch's rate end up at THAT patch?
2. Evolution  does the population's omega move toward the rates in the room?
              (controls: same brain in a room with no rhythm; window brain,
              for which omega is an unused gene)
3. Camera     how much of the flies' behaviour can an observer explain from
              the video alone (instant features vs 16-frame spectral features)?
"""
import os, json, pickle
from collections import deque
import numpy as np
import cv2
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import flybench as fb
import analyze as A

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = fb.RhythmScene.PATCHES
TARGETS = [2 * np.pi * f for _, f in PATCHES]      # rad/frame: 0.314, 0.942
TOL = 0.15
R = {}


def load(name):
    p = os.path.join(HERE, "runs", name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def sorting(d):
    out = {}
    w = d["omega"]
    near = [np.hypot(d["x0"] - cx, d["y0"] - cy) < 200 for (cx, cy), _ in PATCHES]
    for k, tgt in enumerate(TARGETS):
        cls = np.abs(w - tgt) < TOL
        out[f"tuned_to_{'AB'[k]}"] = dict(
            fly_frames=int(cls.sum()),
            frac_at_A=float(near[0][cls].mean()) if cls.any() else None,
            frac_at_B=float(near[1][cls].mean()) if cls.any() else None)
    other = ~((np.abs(w - TARGETS[0]) < TOL) | (np.abs(w - TARGETS[1]) < TOL))
    out["untuned"] = dict(fly_frames=int(other.sum()),
                          frac_at_A=float(near[0][other].mean()),
                          frac_at_B=float(near[1][other].mean()))
    out["chance_area_share_each"] = float(np.pi * 200 ** 2 / (fb.W * fb.H))
    return out


def evolution(d):
    t, w = d["t"], d["omega"]
    def stats(m):
        ww = w[m]
        tuned = (np.abs(ww - TARGETS[0]) < TOL) | (np.abs(ww - TARGETS[1]) < TOL)
        dist = np.min(np.abs(ww[:, None] - np.array(TARGETS)[None]), axis=1)
        return dict(frac_tuned=float(tuned.mean()), median_dist_to_nearest_rate=float(np.median(dist)),
                    n=int(m.sum()))
    return dict(first_300_frames=stats(t < 300), last_1000_frames=stats(t >= t.max() - 1000),
                births=int(d["births"]), deaths=int(d["deaths"]),
                pop_end=int((t == t.max()).sum()))


def spectral_features(d, scene_name, seed, n_hist=16):
    """Observer-legal: 16-frame spectra at 6 points ahead of the fly."""
    t = d["t"].astype(int)
    x, y, a = d["x0"], d["y0"], d["a0"]
    n = len(t)
    F = np.zeros((n, 6 * (n_hist // 2)))
    order = np.argsort(t, kind="stable")
    bounds = np.searchsorted(t[order], np.arange(t.max() + 2))
    sc = fb.SCENES[scene_name](seed)
    buf = deque(maxlen=n_hist)
    for tt in range(t.max() + 1):
        g = cv2.blur(cv2.cvtColor(sc.frame(tt), cv2.COLOR_BGR2GRAY), (7, 7)).astype(np.float32)
        buf.append(g)
        rows = order[bounds[tt]:bounds[tt + 1]]
        if len(rows) == 0 or len(buf) < n_hist:
            continue
        stack = np.stack(buf)                                  # (16,H,W)
        k = 0
        for dist in (75, 140):
            for off in (-0.35, 0.0, 0.35):
                px = np.clip((x[rows] + dist * np.cos(a[rows] + off)).astype(int), 0, fb.W - 1)
                py = np.clip((y[rows] + dist * np.sin(a[rows] + off)).astype(int), 0, fb.H - 1)
                s = stack[:, py, px]                           # (16, nrows)
                spec = np.abs(np.fft.rfft(s - s.mean(0), axis=0))[1:n_hist // 2 + 1]
                F[rows, k * (n_hist // 2):(k + 1) * (n_hist // 2)] = spec.T / 100
                k += 1
    return F


def camera_scores(name_tr, name_te, scene="rhythm"):
    dtr, dte = load(name_tr), load(name_te)
    s_tr, t_tr = A.steps(dtr)
    s_te, t_te = A.steps(dte)
    Itr = A.img_features(dtr, scene, 0); Ite = A.img_features(dte, scene, 1)
    Str = spectral_features(dtr, scene, 0); Ste = spectral_features(dte, scene, 1)
    out = {}
    for tgt, (ytr, yte) in {"turn": (t_tr, t_te), "speed": (s_tr, s_te)}.items():
        out[tgt] = {}
        for k, (a, b) in {"camera_instant": (Itr, Ite),
                          "camera_16_frame_spectra": (np.c_[Itr, Str], np.c_[Ite, Ste])}.items():
            out[tgt][k] = A.fit_score(a, ytr, b, yte)
    return out


if __name__ == "__main__":
    for brain in ("resonator", "window"):
        for s in (0, 1):
            d = load(f"{brain}_rhythm_s{s}")
            if d is not None:
                R[f"sorting_{brain}_s{s}"] = sorting(d)
                R[f"evolution_{brain}_rhythm_s{s}"] = evolution(d)
    d = load("resonator_room_s0")
    if d is not None:
        R["evolution_resonator_room_NO_RHYTHM_s0"] = evolution(d)
    for brain, tag in (("original", ""), ("resonator", "resonator_"), ("window", "window_")):
        if load(f"{tag}rhythm_s0") is not None and load(f"{tag}rhythm_s1") is not None:
            print("camera observer:", brain, flush=True)
            R[f"camera_observer_{brain}"] = camera_scores(f"{tag}rhythm_s0", f"{tag}rhythm_s1")
    json.dump(R, open(os.path.join(HERE, "res_results.json"), "w"), indent=2)
    print(json.dumps(R, indent=2))
