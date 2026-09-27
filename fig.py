"""fig.py - one figure summarising the bench."""
import json, pickle, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import flybench as fb
import analyze as A

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

R = json.load(open("results.json"))
AB = json.load(open("ablation.json"))

fig = plt.figure(figsize=(12, 7.2), dpi=130)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.62)

# --- A: inside one fly's head
random.seed(0); np.random.seed(0)
cfg = fb.BugConfig(); sc = fb.RoomScene(0)
b = fb.EnhancedBug(640, 360, cfg, seed=5); b.screen_width, b.screen_height = fb.W, fb.H
trail = []
for t in range(400):
    f = sc.frame(t); b.nearby_bugs = []
    v, r = b.process_frame(f); b.update_position(v, r); trail.append((b.x, b.y))
room = cv2.cvtColor(sc.frame(400), cv2.COLOR_BGR2GRAY)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(room, cmap="gray", vmin=0, vmax=255)
tr = np.array(trail[-250:])
jumps = np.where(np.hypot(*np.diff(tr, axis=0).T) > 200)[0] + 1
for seg in np.split(tr, jumps):
    ax.plot(seg[:, 0], seg[:, 1], color=ORANGE, lw=2)
ax.plot(*tr[-1], "o", color=ORANGE, ms=8, mec="white", mew=1.5)
ax.set_title("A. What the camera shows + the fly's last 250 frames", loc="left", fontsize=9, color=INK)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[0, 1])
ax.imshow(b.decision_field.field, cmap="Blues", vmin=0, vmax=1)
ax.contour(b.decision_field.field > 0.55, levels=[0.5], colors=[ORANGE], linewidths=1.5)
ax.set_title("B. Its 'decision field' (32x32): a map of the\nwhole screen = its own smeared trail", loc="left", fontsize=9, color=INK)
ax.set_xticks([]); ax.set_yticks([])

# --- C: movement does not depend on the camera
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 25, 51)
for name, col, lab in (("room_s0", BLUE, "room + walking person"),
                       ("gray_s0", ORANGE, "flat gray frame"),
                       ("black_s0", AQUA, "black frame")):
    sp, _ = A.steps(A.runs[name])
    h, e = np.histogram(sp, bins=bins, density=True)
    ax.step(e[:-1], h, where="post", color=col, lw=2, label=lab)
ax.set_xlabel("step length per frame (px)")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_title("C. Room vs flat gray: same flies.\nBlack differs (they only see each other)", loc="left", fontsize=9, color=INK)

# --- D/E: what an observer can predict
rows = [("camera only", R["observer_train_room_s0_test_room_s1"]["turn"]["camera_only"],
         R["observer_train_room_s0_test_room_s1"]["speed"]["camera_only"]),
        ("where it is now", AB["turn"]["where_it_is_now_only"], AB["speed"]["where_it_is_now_only"]),
        ("where it has been", AB["turn"]["where_it_is_and_has_been (no momentum)"],
         AB["speed"]["where_it_is_and_has_been (no momentum)"]),
        ("its own last 2 moves", AB["turn"]["momentum_only (prev 2 turns/speeds)"],
         AB["speed"]["momentum_only (prev 2 turns/speeds)"]),
        ("all observables", AB["turn"]["full_history"], AB["speed"]["full_history"]),
        ("hidden field (oracle)", R["observer_train_room_s0_test_room_s1"]["turn"]["ORACLE_hidden_centroid"],
         R["observer_train_room_s0_test_room_s1"]["speed"]["ORACLE_hidden_centroid"])]
for k, (ttl, idx) in enumerate((("D. Predicting turning (held-out run, R²)", 1),
                                ("E. Predicting speed (held-out run, R²)", 2))):
    ax = fig.add_subplot(gs[1, k])
    labels = [r[0] for r in rows]; vals = [max(r[idx], 0) for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=[BLUE] * 5 + [MUTED], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.15); ax.set_xticks([0, 0.5, 1])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(ttl, loc="left", fontsize=9, color=INK)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
p3 = R["P3_person"]; p2 = R["P2_energy_source"]["room_s0"]
txt = (
    "F. Other rules an observer can read off\n\n"
    f"• A person does not attract them:\n   {p3['frac_flies_within_250px_of_person']*100:.1f}% of flies near the person\n"
    f"   vs {p3['same_spot_same_frames_no_person']*100:.1f}% at the same spot with nobody there\n\n"
    f"• Everyone starves on a timer: intake ≈{p2['mean_gain_when_moving_fast']:.3f}/frame\n"
    "   vs burn 0.05/frame, whatever the camera shows\n\n"
    f"• In the dark all die at frame {R['P4_dark']['all_die_at_frame']} (100 / 0.05)\n\n"
    "• Faster in the dark: a 2–3-cell field (other flies)\n   has its centroid far from the screen centre"
)
ax.text(0, 1, txt, va="top", fontsize=8.6, color=INK, linespacing=1.35)

fig.suptitle("dumbflies.py as a research bench: the camera is decoration, the fly follows its own trail",
             x=0.01, ha="left", fontsize=11.5, color=INK, weight="bold")
fig.savefig("flybench_findings.png", bbox_inches="tight", facecolor="white")
print("ok")
