"""
resonator_flies.py - run the ORIGINAL dumbflies.py GUI + webcam with the
windowed-resonator brain swapped in.

    python resonator_flies.py                  # resonator brain
    python resonator_flies.py --brain window   # same eye/window, no tuning (control)
    python resonator_flies.py --floor 8        # raise if a noisy/dark camera feeds everyone

Each fly has a gene omega (its preferred flicker rate) and r (its time window).
It hops (eyes shut), lands, listens; stays and eats if what it hears matches
its rate. Try: wave a hand, or put a blinking light / a screen playing a
pulsing video in view, at a steady rate, and watch which flies come and stay.

Ring colour = tuning (blue slow ... red fast); filled ring = listening and
hearing something. The strip top-left is the population's tuning histogram,
with the rate axis in Hz using the measured frame rate.
"""
import argparse, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):
    if os.path.isfile(os.path.join(p, "dumbflies.py")) and p not in sys.path:
        sys.path.insert(0, p)
import dumbflies as df
import resonator_brain as rb


def hue_bgr(omega, lo=0.1, hi=1.3):
    h = int(np.clip((omega - lo) / (hi - lo), 0, 1) * 120)       # 0 blue .. 120 red (reversed below)
    c = cv2.cvtColor(np.uint8([[[120 - h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in c)


def patched_update_display(self, frame):
    now = time.time()
    self._fps = 0.9 * getattr(self, "_fps", 15.0) + 0.1 / max(now - getattr(self, "_last", now - 1 / 15), 1e-3)
    self._last = now
    img = frame.copy()
    for bug in self.bugs:
        w = bug.traits.get("omega", 0.5)
        col = hue_bgr(w)
        listening = getattr(bug, "phase", "") == "listen"
        heard = getattr(bug, "heard", 0.0)
        cv2.circle(img, (int(bug.x), int(bug.y)), 11, col, -1 if (listening and heard > 1) else 2)
    # population tuning histogram, top-left
    ws = np.array([b.traits.get("omega", np.nan) for b in self.bugs], float)
    ws = ws[np.isfinite(ws)]
    x0, y0, bw, hh = 10, 10, 12, 60
    cv2.rectangle(img, (x0 - 5, y0 - 5), (x0 + 24 * bw + 5, y0 + hh + 30), (0, 0, 0), -1)
    if len(ws):
        hist, edges = np.histogram(ws, bins=24, range=(0.03, 2.8))
        for i, c in enumerate(hist):
            h = int(hh * c / max(hist.max(), 1))
            cv2.rectangle(img, (x0 + i * bw, y0 + hh - h), (x0 + i * bw + bw - 2, y0 + hh),
                          hue_bgr((edges[i] + edges[i + 1]) / 2), -1)
        fps = self._fps
        for wv in (0.5, 1.0, 2.0):
            px = x0 + int((wv - 0.03) / (2.8 - 0.03) * 24 * bw)
            cv2.putText(img, f"{wv * fps / (2 * np.pi):.1f}Hz", (px - 12, y0 + hh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for bug in self.bugs:
        self.fly_visuals.draw_fly(frame_rgb, bug.x, bug.y, bug.angle, bug.is_mating,
                                  bug.traits["size"], bug.traits["shade"])
    keep = []
    for e in self.visual_effects:
        e.draw(frame_rgb)
        if e.update():
            keep.append(e)
    self.visual_effects = keep
    photo = df.ImageTk.PhotoImage(image=df.Image.fromarray(frame_rgb))
    self.canvas.create_image(0, 0, image=photo, anchor=df.tk.NW)
    self.canvas.photo = photo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="resonator", choices=["resonator", "window"])
    ap.add_argument("--floor", type=float, default=rb.NOISE_FLOOR)
    args = ap.parse_args()
    rb.NOISE_FLOOR = args.floor
    df.EnhancedBug = rb.make_bug_class(df.EnhancedBug, args.brain)   # swap the brain
    df.BugGUI.update_display = patched_update_display
    df.main()


if __name__ == "__main__":
    main()
