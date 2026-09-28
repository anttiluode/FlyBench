"""res_fig.py - figure for the resonator-fly results."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8983"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

S = json.load(open("sort_nobreed.json"))
RR = json.load(open("res_results.json"))
cams = {b: RR[f"camera_observer_{b}"] for b in ("original", "window", "resonator")}
TA, TB = 2 * np.pi * 0.05, 2 * np.pi * 0.15

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=130,
                         gridspec_kw=dict(width_ratios=[1.25, 1.25, 1], wspace=0.35))

# A/B: per-fly preference vs tuning, no breeding
for ax, brain, ttl in ((axes[0], "resonator", "A. Resonator brain"),
                       (axes[1], "window", "B. Same window, no tuning (control)")):
    pf = np.array(S[brain]["per_fly"])
    w, atA, atB = pf.T
    ax.axvspan(TA - 0.15, TA + 0.15, color=BLUE, alpha=0.08, lw=0)
    ax.axvspan(TB - 0.15, TB + 0.15, color=ORANGE, alpha=0.08, lw=0)
    ax.scatter(w, atA, s=18, color=BLUE, label="time at patch A (slow, 1.5 Hz)",
               edgecolor="white", linewidth=0.6)
    ax.scatter(w, atB, s=18, color=ORANGE, label="time at patch B (fast, 4.5 Hz)",
               edgecolor="white", linewidth=0.6, marker="D")
    ax.set_xlabel("fly's tuning gene ω (rad/frame)")
    ax.set_ylabel("share of its life near the patch")
    ax.set_ylim(-0.03, 1.12)
    ax.grid(axis="y", color=GRID, lw=0.6)
    rho, p = S[brain]["spearman_tuning_vs_place_preference"], S[brain]["p_value"]
    ax.set_title(f"{ttl}\n120 fresh flies, no breeding · ρ={rho:.2f}, p={p:.1g}",
                 loc="left", fontsize=9, color=INK)
    ax.text(TA, 1.1, "tuned A", ha="center", va="top", fontsize=8, color=BLUE)
    ax.text(TB, 1.1, "tuned B", ha="center", va="top", fontsize=8, color=ORANGE)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.33, -0.1))

# C: camera-explained behaviour
ax = axes[2]
names = ["original", "window", "resonator"]
labels = ["original flies", "window (control)", "resonator"]
vals = [max(cams[b]["speed"]["camera_16_frame_spectra"], 0) for b in names]
y = np.arange(3)[::-1]
ax.barh(y, vals, color=[GRAY, GRAY, BLUE], height=0.55)
for yy, v in zip(y, vals):
    ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, max(vals) * 1.35 + 0.02)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_xlabel("R² (held-out run)")
ax.set_title("C. Stop/go explained by the camera alone\n(held-out run, corrected Sep 28)", loc="left", fontsize=9, color=INK)

fig.suptitle("Resonator flies: each goes to the rhythm it is tuned to — and the camera finally matters",
             x=0.01, ha="left", fontsize=11, weight="bold", color=INK, y=1.04)
fig.savefig("resonator_findings.png", bbox_inches="tight", facecolor="white")
print("ok")
