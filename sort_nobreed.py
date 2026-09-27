"""
sort_nobreed.py - the clean sorting test.

No births (no lineages, so no family clustering), 40 fresh flies with random
omega, 2000 frames (a non-eating fly lasts exactly 2000 frames, so nobody is
removed by starvation during the test). Any link between a fly's omega and
where it spends its time is then caused by the brain, not by ancestry.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr
import flybench as fb

TARGETS = [2 * np.pi * f for _, f in fb.RhythmScene.PATCHES]
out = {}
for brain in ("resonator", "window"):
    per_fly = []
    for seed in (0, 1, 2):
        d = fb.run("rhythm", 2000, seed, brain=brain, pop=40, breed=False)
        for u in np.unique(d["uid"]):
            m = d["uid"] == u
            (ax, ay), _ = fb.RhythmScene.PATCHES[0]
            (bx, by), _ = fb.RhythmScene.PATCHES[1]
            atA = (np.hypot(d["x0"][m] - ax, d["y0"][m] - ay) < 200).mean()
            atB = (np.hypot(d["x0"][m] - bx, d["y0"][m] - by) < 200).mean()
            per_fly.append((d["omega"][m][0], atA, atB))
    pf = np.array(per_fly)
    w, atA, atB = pf.T
    tA, tB = np.abs(w - TARGETS[0]) < 0.15, np.abs(w - TARGETS[1]) < 0.15
    rho, p = spearmanr(-np.abs(w - TARGETS[1]) + np.abs(w - TARGETS[0]), atB - atA)
    out[brain] = dict(
        n_flies=len(pf),
        tuned_A=dict(n=int(tA.sum()), mean_time_at_A=float(atA[tA].mean()), mean_time_at_B=float(atB[tA].mean())),
        tuned_B=dict(n=int(tB.sum()), mean_time_at_A=float(atA[tB].mean()), mean_time_at_B=float(atB[tB].mean())),
        untuned=dict(n=int((~tA & ~tB).sum()), mean_time_at_A=float(atA[~tA & ~tB].mean()),
                     mean_time_at_B=float(atB[~tA & ~tB].mean())),
        spearman_tuning_vs_place_preference=float(rho), p_value=float(p),
        per_fly=pf.tolist())
    print(brain, {k: v for k, v in out[brain].items() if k != "per_fly"}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sort_nobreed.json"), "w"), indent=1)
