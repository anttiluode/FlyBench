"""
clicker_neuron.py - clicker-train a neuron made of resonator flies, live on the webcam.

    python clicker_neuron.py
    python clicker_neuron.py --self-feed 1     # control: flies also eat what they hear

SPACE = the click (the modulator). Press it whenever the thing you want the
neuron to detect is happening: a hand wave, a phone blinking at a steady rate,
a nod... The flies eat ONLY what your clicks give them, shared by how active
each was just before the click (more if it helped the neuron fire). When the
neuron fires and no click follows within ~30 frames, the flies that made it
fire lose a little energy. A well-fed fly buds (splits in place, the copy's
tuning slightly mutated); unfed flies starve. There is no mating.

Screen:
  big circle bottom-left   the soma; flashes yellow when the neuron fires
  lines to the soma        the dendrite: which flies are feeding the soma now
  strip bottom-right       soma voltage (white) and threshold (red), last 200 frames
  white flash              your click
  top-left                 tuning histogram (as in resonator_flies)

If nobody clicks, the colony starves in a few minutes and is replaced by fresh
random flies: an untrained neuron. Save Population keeps a trained one.
Writes neuron_log.csv every 10 s (clicks, spikes, fps, population, tuning).
"""
import argparse, os, sys, time
from collections import deque
import numpy as np
import cv2

import resonator_flies as RF          # also imports dumbflies as df
df = RF.df
import neuron as nr

HERE = os.path.dirname(os.path.abspath(__file__))


def neuron_log(self, now):
    if now - getattr(self, "_nlog_t", 0.0) < 10.0:
        return
    self._nlog_t = now
    path = os.path.join(HERE, "neuron_log.csv")
    ws = [b.traits.get("omega", np.nan) for b in self.bugs]
    hist, _ = np.histogram([w for w in ws if np.isfinite(w)], bins=RF.OMEGA_BINS)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("time,fps,population,clicks,spikes,theta,median_omega,"
                    + ",".join(f"w{(a + b) / 2:.2f}" for a, b in zip(RF.OMEGA_BINS[:-1], RF.OMEGA_BINS[1:])) + "\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self._fps:.2f},{len(self.bugs)},"
                f"{self._n_clicks},{self._n_spikes},{self.soma.theta:.2f},"
                f"{np.nanmedian(ws) if len(ws) else float('nan'):.3f},"
                + ",".join(str(int(c)) for c in hist) + "\n")


def neuron_update_display(self, frame):
    now = RF.tick_fps(self)
    if not hasattr(self, "soma"):
        self.soma = nr.Soma()
        self._n_clicks = self._n_spikes = 0
        self._trace = deque(maxlen=200)
        self._flash = 0
    click = getattr(self, "_clicked", False)
    self._clicked = False
    born0 = self.soma.born
    y = self.soma.step(self.bugs, click=click, learn=True)
    self.total_births += self.soma.born - born0          # buds count as births
    self._n_clicks += int(click)
    self._n_spikes += int(y)
    self._trace.append((self.soma.V, self.soma.theta))
    if click:
        self._flash = 4
    neuron_log(self, now)

    img = RF.draw_rings_and_hist(self, frame.copy())
    h, w = img.shape[:2]
    sx, sy = 70, h - 85
    # dendrite: lines from contributing flies to the soma
    cmax = max([b.contrib for b in self.bugs] + [1e-6])
    for b in self.bugs:
        if b.contrib > 0:
            th = 1 + int(4 * b.contrib / cmax)
            cv2.line(img, (int(b.x), int(b.y)), (sx, sy), RF.hue_bgr(b.traits.get("omega", 0.5)), th)
    # soma
    fill = min(self.soma.V / max(self.soma.theta, 1e-6), 1.5)
    cv2.circle(img, (sx, sy), 46, (0, 0, 0), -1)
    cv2.circle(img, (sx, sy), int(10 + 30 * min(fill, 1.0)), (0, 230, 255) if y else (90, 90, 90), -1)
    cv2.circle(img, (sx, sy), 46, (255, 255, 255), 2)
    cv2.putText(img, "SOMA", (sx - 22, sy + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # voltage / threshold trace
    tw, th_, x0, y0 = 200, 60, w - 210, h - 95
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + tw + 4, y0 + th_ + 20), (0, 0, 0), -1)
    if len(self._trace) > 1:
        tr = np.array(self._trace)
        top = max(tr.max(), 1e-6)
        pts = lambda col: np.array([[x0 + i * tw // 200, y0 + th_ - int(th_ * v / top)]
                                    for i, v in enumerate(tr[:, col])], np.int32)
        cv2.polylines(img, [pts(0)], False, (255, 255, 255), 1)
        cv2.polylines(img, [pts(1)], False, (60, 60, 255), 1)
    cv2.putText(img, f"clicks {self._n_clicks}  spikes {self._n_spikes}", (x0, y0 + th_ + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    if self._flash:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
        self._flash -= 1
    RF.push_to_canvas(self, img)


_orig_setup_gui = df.BugGUI.setup_gui


def neuron_setup_gui(self):
    _orig_setup_gui(self)
    self.root.bind("<space>", lambda e: setattr(self, "_clicked", True))
    self.root.focus_force()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-feed", type=float, default=0.0,
                    help="0 = flies eat only from clicks (default); 1 = they also eat what they hear")
    ap.add_argument("--floor", type=float, default=RF.rb.NOISE_FLOOR)
    args = ap.parse_args()
    RF.rb.NOISE_FLOOR = args.floor
    df.EnhancedBug = nr.make_neuron_bug(df.EnhancedBug, self_feed=args.self_feed)   # try_mate sees it too
    df.BugGUI.setup_gui = neuron_setup_gui
    df.BugGUI.update_display = neuron_update_display
    df.BugGUI.update_graph = RF.patched_update_graph
    df.json = RF._SafeJSON()
    df.main()


if __name__ == "__main__":
    main()
