"""
resonator_brain.py - windowed-resonator brains for the dumbflies flies.

make_bug_class(EnhancedBug, mode) returns a subclass of the ORIGINAL fly with
a new brain. Body, movement, repulsion, energy decay, mating and inheritance
are the original code; only process_frame (the brain) is replaced.

The brain
---------
Egocentric eye: 9 points ahead of the fly (3 distances x 3 angles), sampled
from a lightly blurred grayscale camera frame. Each point feeds its own unit:

  mode="resonator"  z <- r e^{i w} z + (1 - r) d        a = |z|
  mode="window"     e <- r e + (1 - r) |d|              a = e      (control)

where d = pixel - slow mean at that point. Both have the same time window
(~1/(1-r) frames); only the resonator is frequency-tuned. w and r are genes
(traits 'omega', 'r') inherited and mutated by the original inherit_traits.

Behaviour: hop-and-listen (the same for both brains)
  hop     10 frames at 0.6 x max_speed, eyes shut (saccadic suppression)
  listen  stand still for >= 1.5/(1-r) frames; the brain cancels the body's own
          random wander turn (efference copy) so the eye doesn't sweep
  stay    keep listening and eating while it hears >= STAY_LEVEL; otherwise hop,
          turned slightly toward the louder eye side
  eat     EAT_GAIN * max(mean activity - NOISE_FLOOR, 0), only while listening

Why: moving over plain room texture registers activity 9-12, more than a
mistuned flicker (4-7), so a fly that listened while flying could live on its
own motion and rhythm would not matter.
"""
import numpy as np
import cv2

EYE = [(dist, ang) for dist in (40, 90, 150) for ang in (-0.5, 0.0, 0.5)]
EYE = np.array(EYE, dtype=float)
LEFT = EYE[:, 1] < 0
RIGHT = EYE[:, 1] > 0

NOISE_FLOOR = 6.0      # above self-motion residue and mistuned patches (calibrated)
EAT_GAIN = 0.02        # energy per unit of activity above the floor
MEAN_ALPHA = 0.02      # slow mean per eye point (removes the DC)
STAY_LEVEL = 2.0       # keep listening while it hears at least this much
HOP_FRAMES = 10        # length of a hop (eyes shut)

_gray = {"id": None, "g": None}


def gray_blur(frame):
    if _gray["id"] is not frame:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _gray["g"] = cv2.blur(g, (7, 7)).astype(np.float32)
        _gray["id"] = frame
    return _gray["g"]


def make_bug_class(EnhancedBug, mode="resonator", omega_range=(0.1, 1.3)):
    assert mode in ("resonator", "window")

    class ResonatorBug(EnhancedBug):
        BRAIN = mode

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.traits.setdefault("omega", float(np.random.uniform(*omega_range)))
            self.traits.setdefault("r", float(np.random.uniform(0.85, 0.95)))
            self.z = np.zeros(len(EYE), complex)
            self.e = np.zeros(len(EYE))
            self.m = None
            self.activity = np.zeros(len(EYE))
            self.phase, self.phase_t = "listen", 0
            self.hop_turn, self.heard = 0.0, 0.0

        def inherit_traits(self, parent_traits, mutation_rate):
            super().inherit_traits(parent_traits, mutation_rate)
            self.traits["omega"] = float(np.clip(self.traits.get("omega", 0.5), 0.03, 2.8))
            self.traits["r"] = float(np.clip(self.traits.get("r", 0.9), 0.5, 0.985))

        def sample_eye(self, frame):
            g = gray_blur(frame)
            h, w = g.shape
            ang = self.angle + EYE[:, 1]
            px = np.clip((self.x + EYE[:, 0] * np.cos(ang)).astype(int), 0, w - 1)
            py = np.clip((self.y + EYE[:, 0] * np.sin(ang)).astype(int), 0, h - 1)
            return g[py, px]

        def process_frame(self, frame):
            x = self.sample_eye(frame)
            if self.m is None:
                self.m = x.copy()
            r, w = self.traits["r"], self.traits["omega"]
            listening = self.phase == "listen"
            if listening:
                d = x - self.m
                self.m += MEAN_ALPHA * d
            else:                      # saccadic suppression: eyes shut in flight
                d = np.zeros_like(x)
                self.m = x.copy()
            if self.BRAIN == "resonator":
                self.z = r * np.exp(1j * w) * self.z + (1 - r) * d
                a = np.abs(self.z)
            else:
                self.e = r * self.e + (1 - r) * np.abs(d)
                a = self.e
            self.activity = a
            heard = max(a.mean() - NOISE_FLOOR, 0.0)
            self.heard = heard
            lr = a[RIGHT].sum() - a[LEFT].sum()
            side = np.tanh(3 * lr / (a[RIGHT].sum() + a[LEFT].sum() + 1e-6))
            wander = self.random_direction * self.config.turn_speed   # body adds this

            self.phase_t += 1
            if listening:
                speed = 0.0
                rot = -wander + 0.05 * side * np.tanh(heard)   # efference copy: hold still
                settled = self.phase_t >= 1.5 / (1 - r)
                if settled and heard < STAY_LEVEL:
                    self.phase, self.phase_t = "hop", 0
                    self.hop_turn = 0.25 * side if heard > 0 else 0.0
                if listening and settled:
                    self.energy += min(EAT_GAIN * heard, 5.0)
            else:
                speed = self.traits["max_speed"] * 0.6
                rot = self.hop_turn
                if self.phase_t >= HOP_FRAMES:
                    self.phase, self.phase_t = "listen", 0
                    self.z[:] = 0
                    self.e[:] = 0

            self.energy -= self.config.energy_decay
            self.energy = min(self.energy, self.config.initial_energy * 1.5)
            self.is_mating = bool(self.energy > self.config.mating_threshold)

            m = self.config.momentum
            self.current_velocity = self.current_velocity * m + speed * (1 - m)
            self.current_rotation = self.current_rotation * m + rot * (1 - m)
            if listening:                  # no momentum on the brake / cancel
                self.current_velocity = 0.0
                self.current_rotation = rot
            return self.current_velocity, self.current_rotation

    ResonatorBug.__name__ = f"{mode.capitalize()}Bug"
    return ResonatorBug
