"""
plot_log.py - turn flies_log.csv (written by resonator_flies.py) into a picture.

    python plot_log.py                 # reads flies_log.csv, writes flies_log.png

Top:    the population's tuning over time (each column = one 10 s row,
        colour = how many flies sit in that omega bin; slow at the bottom).
Middle: population, and births per row.
Bottom: room brightness, so light switches line up with what the flies did.
"""
import csv, os, sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "flies_log.csv")
rows = list(csv.DictReader(open(path)))
if len(rows) < 2:
    sys.exit("Need at least two rows in the log; let it run a bit longer.")

t = [datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") for r in rows]
wcols = [k for k in rows[0] if k.startswith("w")]
wmid = np.array([float(k[1:]) for k in wcols])
H = np.array([[float(r[k]) for k in wcols] for r in rows]).T          # bins x time
pop = np.array([float(r["population"]) for r in rows])
births = np.diff(np.array([float(r["births"]) for r in rows]), prepend=float(rows[0]["births"]))
bright = np.array([float(r["brightness"]) for r in rows])
fps = np.array([float(r["fps"]) for r in rows])

INK, MUTED, GRID, BLUE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), dpi=120, sharex=True,
                       gridspec_kw=dict(height_ratios=[2.2, 1, 0.8], hspace=0.12))
tn = mdates.date2num(t)
dt = np.median(np.diff(tn)) if len(tn) > 1 else 1 / 8640
edges = np.r_[tn - dt / 2, tn[-1] + dt / 2]
ax[0].pcolormesh(edges, np.r_[wmid - (wmid[1] - wmid[0]) / 2, wmid[-1] + (wmid[1] - wmid[0]) / 2],
                 H, cmap="Blues", shading="flat")
ax[0].set_ylabel("tuning ω (rad/frame)\nslow ↓   fast ↑")
ax[0].set_ylim(0, 1.6)
hz = np.median(fps) / (2 * np.pi)
sec = ax[0].secondary_yaxis("right", functions=(lambda w: w * hz, lambda f: f / hz))
sec.set_ylabel(f"Hz at median {np.median(fps):.1f} fps")
ax[0].set_title("Population tuning over time", loc="left", color=INK)

ax[1].plot(t, pop, color=BLUE, lw=1.5, label="population")
ax[1].bar(t, births, width=dt, color="#eb6834", alpha=0.7, label="births per row")
ax[1].set_ylabel("flies")
ax[1].legend(frameon=False, loc="upper left", fontsize=8)
ax[1].grid(axis="y", color=GRID, lw=0.6)

ax[2].plot(t, bright, color=INK, lw=1.2)
ax[2].set_ylabel("room\nbrightness")
ax[2].grid(axis="y", color=GRID, lw=0.6)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

out = os.path.splitext(path)[0] + ".png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
