"""
bank_control.py - the control the clicker neuron has to be compared with.

40 fixed resonators (random tuning and window, same eye and same unit as the
flies), never moving, never dying. A logistic readout is trained on the TRUE
label of every training frame (A or not) - far more supervision than ~65
clicks. Tested on the same test world; its threshold is set so it fires
exactly as often as the clicker neuron did, so rates are comparable.

  bank_random   sensors at random places (like the colony's founders)
  bank_lamp     sensors all around the lamp (upper bound on placement)
"""
import sys, json, random
import numpy as np
from sklearn.linear_model import LogisticRegression
import flybench as fb
import neuron_bench as nb
import neuron as nr


def features(seed, placement):
    rng = np.random.default_rng(500 + seed)
    random.seed(seed); np.random.seed(seed)
    cfg = fb.BugConfig()
    Bug = nr.make_neuron_bug(fb.EnhancedBug)
    sens = []
    for i in range(40):
        if placement == "lamp":
            a = rng.uniform(0, 2 * np.pi); d = rng.uniform(60, 200)
            x, y = nb.LAMP[0] + d * np.cos(a), nb.LAMP[1] + d * np.sin(a)
            ang = np.arctan2(nb.LAMP[1] - y, nb.LAMP[0] - x)          # facing the lamp
        else:
            x, y, ang = rng.uniform(0, fb.W), rng.uniform(0, fb.H), rng.uniform(0, 2 * np.pi)
        b = Bug(x=x, y=y, config=cfg, seed=i)
        b.angle = ang
        sens.append(b)
    tr, te = nb.train_world(seed), nb.test_world(seed)
    X, lab = [], []
    for t in range(nb.TRAIN + nb.TEST):
        sc = te if t >= nb.TRAIN else tr
        f = sc.frame(t)
        row = []
        for b in sens:
            b.phase, b.phase_t = "listen", 10 ** 6
            b.process_frame(f)
            row.append(b.activity.mean())
        X.append(row); lab.append(sc.label(t) or "-")
    return np.array(X), np.array(lab), tr, te


def evaluate(seed, placement, target_rate):
    X, lab, tr, te = features(seed, placement)
    n = nb.TRAIN
    yA = (lab == "A").astype(int)
    clf = LogisticRegression(max_iter=2000).fit(np.log1p(X[:n]), yA[:n])
    s = clf.decision_function(np.log1p(X[n:]))
    thr = np.quantile(s, 1 - target_rate)
    fire = s > thr
    t = np.arange(n, n + len(s))
    since = np.array([t_ - max([x for x in te.toggles if x <= t_], default=-10 ** 9) for t_ in t])
    L = lab[n:]
    return dict(rate_A=float(fire[L == "A"].mean()), rate_B=float(fire[L == "B"].mean()),
                rate_after_light_switch=float(fire[since < 60].mean()), rate_overall=float(fire.mean()))


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    out = {}
    for s in seeds:
        sc = json.load(open(f"neuron_runs/clicker_s{s}.json"))["score"]
        for pl in ("random", "lamp"):
            out[f"bank_{pl}_s{s}"] = evaluate(s, pl, max(sc["rate_overall"], 0.02))
            print(s, pl, out[f"bank_{pl}_s{s}"], flush=True)
    json.dump(out, open("neuron_runs/bank.json", "w"), indent=1)
