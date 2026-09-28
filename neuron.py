"""
neuron.py - a colony of resonator flies as one neuron, taught with a clicker.

The flies are the dendrite: each listening fly is a tuned temporal filter at
a place. The soma sums what they hear; the teacher's click is the modulator.

  weight      w_i = energy_i / 100 (clipped)            medium clock
  soma        V = sum_i w_i * heard_i  (settled flies)  fast clock
  spike       y = V > theta ; theta adapts so the neuron fires ~RATE of frames
  eligibility e_i <- LAM e_i + w_i heard_i (1 + BETA y)  "I was active, more so
                                                         if I made it fire"
  click       food_i = G * e_i, capped per fly at CAP and in total at BUDGET
              (absolute: a click when nobody was listening feeds nobody)
  no click    a spike not followed by a click within FA_WINDOW frames costs
              FA_COST energy in total, split by contribution (modulator -)
              (v2 charged every contributor per spike; with the soma firing
              ~10% of frames that cost 60% of all food and starved the colony)
  anchoring   a fly fed by a click is anchored for ANCHOR frames: it keeps
              listening where it is even when it hears nothing (a rewarded
              synapse is stabilised). Counts down only while learning.
              (v1 without it: every B segment drove the A-tuned flies off the
              lamp; the colony was 1-2 flies at the lamp.)
  growth      a fly with energy > BUD_E buds: splits in place into two, the
              child's tuning mutated a little. No mating. Unfed flies starve.
              (v0 used the original mating: fed flies rarely met, the colony
              starved to 5-10 random refills and learned nothing.)  slow clock

Flies eat nothing on their own (SELF_FEED = 0): only the teacher feeds them.
"""
import random
import numpy as np
import resonator_brain as rb

RATE = 0.10          # target firing rate of the soma (fraction of frames)
KAPPA = 0.05         # threshold adaptation speed
THETA_MIN = 1.0
LAM = 0.85           # eligibility trace decay (~7 frames)
BETA = 2.0           # extra credit for flies that were active when it fired
G = 2.0              # energy per unit of eligibility at a click
BUDGET = 120.0       # max energy handed out per click
CAP = 40.0           # max energy one fly gets from one click
FA_WINDOW = 30       # frames a spike waits for a click
FA_COST = 3.0        # energy lost in total per unconfirmed spike, split by contribution
BUD_E = 130.0        # energy at which a fly buds
MAX_POP = 60
ANCHOR = 400         # frames a click-fed fly stays put while quiet
W_CLIP = (0.2, 1.5)

OMEGA_MUT_SD = 0.12  # a bud's tuning mutates by about +-12% (log-normal)


def make_neuron_bug(EnhancedBug, self_feed=0.0):
    Base = rb.make_bug_class(EnhancedBug, "resonator")
    if not hasattr(Base, "SELF_FEED") or "anchor" not in open(rb.__file__).read():
        raise RuntimeError(
            "resonator_brain.py is older than neuron.py: copy the resonator_brain.py that came "
            "with neuron.py (it has SELF_FEED and anchoring). With the old one the flies feed "
            "themselves and the neuron never needs your clicks.")

    class NeuronBug(Base):
        SELF_FEED = self_feed

        def process_frame(self, frame):
            out = super().process_frame(frame)
            self.is_mating = False          # no mating in a neuron: growth is budding
            return out

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.elig = 0.0
            self.contrib = 0.0


    NeuronBug.__name__ = "NeuronBug"
    return NeuronBug


class Soma:
    def __init__(self, theta=5.0):
        self.theta = theta
        self.t = 0
        self.pending = []          # (t, {bug: contrib}) spikes waiting for a click
        self.V = 0.0
        self.y = False
        self.last_click_t = -10 ** 9
        self.born = 0

    def step(self, bugs, click=False, learn=True):
        self.t += 1
        V = 0.0
        for b in bugs:
            w = float(np.clip(b.energy / 100.0, *W_CLIP))
            b.contrib = w * b.heard if getattr(b, "settled", False) else 0.0
            V += b.contrib
        y = V > self.theta
        self.theta = max(THETA_MIN, self.theta * np.exp(KAPPA * ((1.0 if y else 0.0) - RATE)))
        self.V, self.y = V, y
        for b in bugs:
            b.elig = LAM * getattr(b, "elig", 0.0) + b.contrib * (1 + BETA * y)
        if not learn:
            return y
        for b in bugs:
            if getattr(b, "anchor", 0) > 0:
                b.anchor -= 1
        if y:
            self.pending.append((self.t, {b: b.contrib for b in bugs if b.contrib > 0}))
        if click:
            self.last_click_t = self.t
            want = {b: min(G * b.elig, CAP) for b in bugs if b.elig > 0}
            tot = sum(want.values())
            scale = min(1.0, BUDGET / tot) if tot > 0 else 0.0
            for b, f in want.items():
                b.energy = min(b.energy + f * scale, b.config.initial_energy * 1.5)
                if f * scale > 1.0:
                    b.anchor = ANCHOR
            self.pending = []                                   # confirmed
        keep = []
        for (ts, c) in self.pending:
            if self.t - ts >= FA_WINDOW:                        # never confirmed
                tot = sum(c.values())
                for b, v in c.items():
                    b.energy -= FA_COST * v / tot
            else:
                keep.append((ts, c))
        self.pending = keep
        self.bud(bugs)
        return y

    def bud(self, bugs):
        """Well-fed listeners split in place; the child's tuning mutates a little."""
        new = []
        for b in bugs:
            if b.energy > BUD_E and len(bugs) + len(new) < MAX_POP:
                tr = dict(b.traits)
                tr["omega"] = float(np.clip(tr["omega"] * np.exp(np.random.normal(0, OMEGA_MUT_SD)), 0.03, 2.8))
                tr["r"] = float(np.clip(tr["r"] + np.random.normal(0, 0.01), 0.5, 0.985))
                c = type(b)(x=b.x + np.random.uniform(-12, 12), y=b.y + np.random.uniform(-12, 12),
                            config=b.config, traits=tr, seed=random.randint(0, 10 ** 6))
                c.screen_width = getattr(b, "screen_width", 1280)
                c.screen_height = getattr(b, "screen_height", 720)
                c.angle = b.angle
                c.anchor = getattr(b, "anchor", 0)
                b.energy /= 2
                c.energy = b.energy
                new.append(c)
        bugs.extend(new)
        self.born += len(new)
