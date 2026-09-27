"""
analyze.py - what can an outside observer learn about the flies' rules?

Reads runs/*.pkl from flybench.py. Uses only observables (position, heading,
the video) for the observer models; hidden variables are used only to check.
"""
import os, json, pickle, glob
import numpy as np
import cv2
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import flybench as fb

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = fb.W, fb.H
R = {}


def load(name):
    return pickle.load(open(os.path.join(HERE, "runs", name + ".pkl"), "rb"))


def steps(d):
    dx = (d["x1"] - d["x0"] + W / 2) % W - W / 2
    dy = (d["y1"] - d["y0"] + H / 2) % H - H / 2
    return np.hypot(dx, dy), d["a1"] - d["a0"]


def by_fly(d):
    """Indices per fly, time-ordered."""
    out = {}
    order = np.lexsort((d["t"], d["uid"]))
    uids = d["uid"][order]
    for u in np.unique(uids):
        out[u] = order[uids == u]
    return out


# ------------------------------------------------------------ P1: camera?
runs = {n: load(n) for n in
        ["room_s0", "room_s1", "gray_s0", "gray_s1", "room_static_s0",
         "room_static_s1", "black_s0", "black_s1"]
        if os.path.exists(os.path.join(HERE, "runs", n + ".pkl"))}

p1 = {}
for n, d in runs.items():
    sp, tu = steps(d)
    p1[n] = dict(speed_median=float(np.median(sp)), speed_p90=float(np.percentile(sp, 90)),
                 turn_abs_median=float(np.median(abs(tu))),
                 births=int(d["births"]), deaths=int(d["deaths"]),
                 rows=int(len(sp)))
R["P1_movement_by_scene"] = p1


def ks(a, b, key):
    sa, ta = steps(runs[a]); sb, tb = steps(runs[b])
    x, y = (sa, sb) if key == "speed" else (abs(ta), abs(tb))
    return float(ks_2samp(x, y).statistic)


pairs = [("room_s0", "room_s1"), ("room_s0", "gray_s0"), ("room_s0", "room_static_s0"),
         ("gray_s0", "gray_s1"), ("room_s0", "black_s0")]
R["P1_KS_distance"] = {f"{a} vs {b}": dict(speed=ks(a, b, "speed"), turn=ks(a, b, "turn"))
                       for a, b in pairs if a in runs and b in runs}


# ------------------------------------------------------------ P2: what feeds them
def energy_gain_vs_speed(d):
    idx = by_fly(d)
    sp, _ = steps(d)
    g, s = [], []
    for u, ii in idx.items():
        e = d["energy"][ii]
        t = d["t"][ii]
        for k in range(2, len(ii)):
            if t[k] - t[k - 1] != 1:
                continue
            gain = e[k] - e[k - 1] + 0.05          # add back the fixed decay
            if gain < -1:                          # reproduction cost, skip
                continue
            if d["energy"][ii[k]] >= 149.99:       # capped, gain not visible
                continue
            g.append(gain)
            s.append(sp[ii[k - 1]])                # how far it moved last frame
    return np.array(s), np.array(g)


p2 = {}
for n in ["room_s0", "gray_s0", "room_static_s0"]:
    if n in runs:
        s, g = energy_gain_vs_speed(runs[n])
        still = g[s < 1.5].mean() if (s < 1.5).any() else float("nan")
        moving = g[s > 6].mean()
        p2[n] = dict(corr_gain_vs_own_speed=float(np.corrcoef(s, g)[0, 1]),
                     mean_gain_when_nearly_still=float(still),
                     mean_gain_when_moving_fast=float(moving),
                     decay_per_frame=0.05)
R["P2_energy_source"] = p2


# ------------------------------------------------------------ P3: person attraction
def person_attraction(d_person, d_static, radius=250):
    sc = fb.RoomScene(0)
    out = []
    for d in (d_person, d_static):
        near = tot = 0
        for t in range(0, 3000, 5):
            p = sc.person_pos(t)
            if p is None:
                continue
            m = d["t"] == t
            dist = np.hypot(d["x0"][m] - p[0], d["y0"][m] - p[1])
            near += (dist < radius).sum()
            tot += m.sum()
        out.append(near / tot)
    return out


if "room_s0" in runs and "room_static_s0" in runs:
    a, b = person_attraction(runs["room_s0"], runs["room_static_s0"])
    R["P3_person"] = dict(frac_flies_within_250px_of_person=float(a),
                          same_spot_same_frames_no_person=float(b))

# ------------------------------------------------------------ P4: darkness
if "black_s0" in runs:
    d = runs["black_s0"]
    m = (d["t"] > 1500) & (d["t"] < 1990)
    sp, _ = steps(d)
    h = np.angle(np.exp(1j * d["a0"][m]))
    R["P4_dark"] = dict(median_step_px=float(np.median(sp[m])),
                        mean_resultant_heading_length=float(abs(np.exp(1j * h).mean())),
                        mean_heading_deg=float(np.degrees(np.angle(np.exp(1j * h).mean()))),
                        all_die_at_frame=int(d["t"][d["energy"] <= 0].min()))


# ------------------------------------------------------------ observer models
def img_features(d, scene_name, seed):
    """What is in front of the fly on the video, and what changed there.
    Streams frames (3000 full frames won't fit in memory)."""
    n = len(d["t"])
    t = d["t"].astype(int)
    x, y, a = d["x0"], d["y0"], d["a0"]
    img = np.zeros((n, 12))
    order = np.argsort(t, kind="stable")
    bounds = np.searchsorted(t[order], np.arange(t.max() + 2))
    sc = fb.SCENES[scene_name](seed)
    prev = None
    for tt in range(t.max() + 1):
        g = cv2.cvtColor(sc.frame(tt), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gp = g if prev is None else prev
        rows = order[bounds[tt]:bounds[tt + 1]]
        k = 0
        for dist in (75, 140):
            for off in (-0.35, 0.0, 0.35):
                px = np.clip((x[rows] + dist * np.cos(a[rows] + off)).astype(int), 0, W - 1)
                py = np.clip((y[rows] + dist * np.sin(a[rows] + off)).astype(int), 0, H - 1)
                img[rows, k] = g[py, px] / 255
                img[rows, k + 6] = abs(g[py, px] - gp[py, px]) / 255
                k += 1
        prev = g
    return img


def features(d, scene_name, seed):
    n = len(d["t"])
    idx = by_fly(d)
    sp, tu = steps(d)
    x, y, a = d["x0"], d["y0"], d["a0"]
    img = img_features(d, scene_name, seed)
    # --- HIST: where the fly is, and where it has been (EMAs of screen position)
    hist = np.zeros((n, 14))
    hist[:, 0] = x / W - 0.5
    hist[:, 1] = y / H - 0.5
    hist[:, 2] = np.cos(a)
    hist[:, 3] = np.sin(a)
    for u, ii in idx.items():
        col = 4
        for tau in (15, 45, 150):
            ex, ey = np.empty(len(ii)), np.empty(len(ii))
            cx, cy = x[ii[0]], y[ii[0]]
            al = 1 / tau
            for j, i in enumerate(ii):
                cx += al * (x[i] - cx); cy += al * (y[i] - cy)
                ex[j], ey[j] = cx, cy
            hist[ii, col] = ex / W - 0.5
            hist[ii, col + 1] = ey / H - 0.5
            col += 2
        prev_t = np.r_[0, tu[ii][:-1]]
        prev_s = np.r_[0, sp[ii][:-1]]
        hist[ii, 10] = prev_t
        hist[ii, 11] = prev_s
        hist[ii, 12] = np.r_[0, 0, tu[ii][:-2]]
        hist[ii, 13] = np.r_[0, 0, sp[ii][:-2]]
    # --- ORACLE (hidden): the true decision-field centroid, raw unwrapped angle
    ci, cj = np.nan_to_num(d["ci"], nan=16), np.nan_to_num(d["cj"], nan=16)
    tgt = np.arctan2((ci - 16) / 16, (cj - 16) / 16)
    orc = np.c_[tgt - a, np.hypot((ci - 16) / 16, (cj - 16) / 16), a,
                hist[:, 10:14]]
    return img, hist, orc, sp, tu


def fit_score(Xtr, ytr, Xte, yte):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    m.fit(Xtr, ytr)
    return float(r2_score(yte, m.predict(Xte)))


def observer(train, test):
    out = {}
    Itr, Htr, Otr, str_, ttr = features(runs[train], train.split("_s")[0], int(train[-1]))
    Ite, Hte, Ote, ste, tte = features(runs[test], test.split("_s")[0], int(test[-1]))
    sets = {"camera_only": (Itr, Ite),
            "position_history_only": (Htr, Hte),
            "camera_plus_history": (np.c_[Itr, Htr], np.c_[Ite, Hte]),
            "ORACLE_hidden_centroid": (Otr, Ote)}
    for target, (ya, yb) in {"turn": (ttr, tte), "speed": (str_, ste)}.items():
        out[target] = {k: fit_score(a, ya, b, yb) for k, (a, b) in sets.items()}
    return out


if __name__ == "__main__":
    if "room_s0" in runs and "room_s1" in runs:
        R["observer_train_room_s0_test_room_s1"] = observer("room_s0", "room_s1")
    if "gray_s0" in runs and "room_s1" in runs:
        R["observer_train_gray_s0_test_room_s1"] = observer("gray_s0", "room_s1")
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(json.dumps(R, indent=2))
