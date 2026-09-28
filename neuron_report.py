"""neuron_report.py - scores and figure for the clicker-neuron bench."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuron_bench as nb

WA, WB = 2 * np.pi * nb.FA, 2 * np.pi * nb.FB
CONDS = [("clicker", "taught A"), ("clicker_B", "taught B"), ("shuffled", "random\nclicks"),
         ("no_credit", "no teacher\n(eats what\nit hears)")]


def test_scores(log):
    t = np.array(log["t"]); y = np.array(log["y"]); lab = np.array(log["label"])
    ls = np.array(log["light_since"])
    m = t >= nb.TRAIN
    sw = ls < 60
    out = dict(A=float(y[m & (lab == "A") & ~sw].mean()), B=float(y[m & (lab == "B") & ~sw].mean()))
    # switch effect: firing in the 60 frames after a switch minus the same rhythm without a switch
    eff = []
    for L in ("A", "B"):
        a = m & (lab == L) & sw
        if a.any():
            eff.append(y[a].mean() - y[m & (lab == L) & ~sw].mean())
    out["switch_effect"] = float(np.mean(eff))
    out["omega"] = log["omega"]["end_train"]
    # learning curve: A-rate minus B-rate in 500-frame training blocks
    curve = []
    for a0 in range(0, nb.TRAIN, 500):
        k = (t >= a0) & (t < a0 + 500)
        ra = y[k & (lab == "A")].mean() if (k & (lab == "A")).any() else np.nan
        rb = y[k & (lab == "B")].mean() if (k & (lab == "B")).any() else np.nan
        curve.append(float(ra - rb))
    out["curve"] = curve
    out["pop_end_train"] = int(np.array(log["pop"])[t == nb.TRAIN - 1][0])
    return out


R = {}
for c, _ in CONDS:
    for f in sorted(glob.glob(f"neuron_runs/{c}_s*.json")):
        R.setdefault(c, []).append(test_scores(json.load(open(f))["log"]))
bank = json.load(open("neuron_runs/bank.json")) if os.path.exists("neuron_runs/bank.json") else {}
for pl in ("random", "lamp"):
    rows = [v for k, v in bank.items() if k.startswith(f"bank_{pl}_")]
    if rows:
        R[f"bank_{pl}"] = [dict(A=r["rate_A"], B=r["rate_B"]) for r in rows]

summary = {}
for c, rows in R.items():
    A = np.array([r["A"] for r in rows]); B = np.array([r["B"] for r in rows])
    s = dict(n=len(rows), rate_A=A.mean(), rate_B=B.mean(), A_minus_B=(A - B).mean(),
             A_minus_B_per_seed=(A - B).round(3).tolist())
    if "switch_effect" in rows[0]:
        s["switch_effect"] = float(np.mean([r["switch_effect"] for r in rows]))
        om = np.concatenate([r["omega"] for r in rows])
        s["median_omega"] = float(np.median(om))
        s["frac_near_A"] = float(np.mean(np.abs(om - WA) < 0.15))
        s["frac_near_B"] = float(np.mean(np.abs(om - WB) < 0.15))
        s["pop_end_train"] = [r["pop_end_train"] for r in rows]
    summary[c] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
json.dump(summary, open("neuron_results.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8983"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=130, gridspec_kw=dict(width_ratios=[1.7, 1.1, 1.1], wspace=0.3))

names = [c for c, _ in CONDS] + [k for k in ("bank_random", "bank_lamp") if k in R]
labels = dict(CONDS); labels.update({"bank_random": "fixed bank,\nreadout\ntrained on\nevery frame\n(random places)",
                                     "bank_lamp": "same,\nsensors\nat the lamp"})
x = np.arange(len(names))
for i, c in enumerate(names):
    A = [r["A"] for r in R[c]]; B = [r["B"] for r in R[c]]
    ax[0].bar(i - 0.18, np.mean(A), 0.34, color=BLUE, label="fires on A (target)" if i == 0 else None)
    ax[0].bar(i + 0.18, np.mean(B), 0.34, color=ORANGE, label="fires on B (distractor)" if i == 0 else None)
    ax[0].scatter(np.full(len(A), i - 0.18), A, s=10, color=INK, zorder=3)
    ax[0].scatter(np.full(len(B), i + 0.18), B, s=10, color=INK, zorder=3)
ax[0].set_xticks(x); ax[0].set_xticklabels([labels[c] for c in names], fontsize=7.5)
ax[0].set_ylabel("share of test frames the neuron fires")
ax[0].grid(axis="y", color=GRID, lw=0.6)
ax[0].set_ylim(0, 0.27)
ax[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
ax[0].set_title("A. Same test for every neuron (learning off)\ndots = seeds", loc="left", fontsize=9, color=INK)

bins = np.linspace(0.03, 1.4, 28)
for c, col, lab in (("clicker", BLUE, "taught A"), ("clicker_B", ORANGE, "taught B"), ("no_credit", GRAY, "no teacher")):
    if c in R:
        om = np.concatenate([r["omega"] for r in R[c]])
        ax[1].hist(om, bins=bins, histtype="step", lw=2, color=col, label=lab, density=True)
ax[1].axvline(WA, color=BLUE, ls=":", lw=1.2); ax[1].axvline(WB, color=ORANGE, ls=":", lw=1.2)
yl = ax[1].get_ylim()[1] * 1.12
ax[1].set_ylim(0, yl)
ax[1].text(WA, yl * 0.98, " target A", color=BLUE, va="top", fontsize=8)
ax[1].text(WB, yl * 0.98, " distractor B", color=ORANGE, va="top", fontsize=8)
ax[1].set_xlabel("tuning ω of the colony's flies (rad/frame)")
ax[1].set_ylabel("density")
ax[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
ax[1].set_title("B. What the colony became\n(end of training, all seeds)", loc="left", fontsize=9, color=INK)

for c, col, lab in (("clicker", BLUE, "taught A"), ("shuffled", GRAY, "random clicks"), ("no_credit", "#b5b4ad", "no teacher")):
    if c in R:
        cv = np.array([r["curve"] for r in R[c]])
        tt = np.arange(cv.shape[1]) * 500 + 250
        ax[2].plot(tt, np.nanmean(cv, 0), color=col, lw=2, label=lab)
        ax[2].fill_between(tt, np.nanmin(cv, 0), np.nanmax(cv, 0), color=col, alpha=0.12, lw=0)
ax[2].axhline(0, color=MUTED, lw=0.8)
ax[2].set_xlabel("training frame")
ax[2].set_ylabel("fires on A minus fires on B")
ax[2].grid(axis="y", color=GRID, lw=0.6)
ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title("C. Learning while it trains\n(band = range over seeds)", loc="left", fontsize=9, color=INK)
fig.suptitle("Clicker neuron: a fly colony taught with clicks tunes itself to what was clicked",
             x=0.01, ha="left", fontsize=11, weight="bold", color=INK, y=1.05)
fig.savefig("neuron_findings.png", bbox_inches="tight", facecolor="white")
print("figure ok")
