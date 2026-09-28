"""
arbor.py - a dendrite that grows toward what you taught (space colonization,
Runions et al. 2007; same rule as GeometricNeuronAndSapolskysFractal/index.html).

The soma is the seed. Every click drops "resource" where the flies that the
click fed are sitting (coloured by their tuning). Growing tips extend toward the
resource near them; resource a tip reaches is consumed. Resource nobody reaches
fades after LIFE frames. The tree only grows, so its shape is a record of
where, and on what rhythm, the neuron was taught.
"""
import numpy as np
import cv2

ATTRACT = 110.0     # a tip feels resource within this distance (px)
KILL = 9.0          # resource this close to a node is consumed
SEG = 4.0           # growth step (px)
NOISE = 0.15
LIFE = 900          # frames an unreached resource lasts
MAX_NODES = 6000


class Arbor:
    def __init__(self, soma_xy):
        self.nodes = np.array([soma_xy], float)        # (N,2)
        self.parent = [-1]
        self.color = [(255, 255, 255)]
        self.count = [1]                                # descendants incl. self (thickness)
        self.att = np.zeros((0, 2)); self.att_col = np.zeros((0, 3)); self.att_age = np.zeros(0)
        self.t = 0
        self._cache = None
        self._drawn = 0

    def feed(self, points, colors, per_point=3, jitter=22.0):
        """Drop resource around each fed fly."""
        if not len(points):
            return
        p = np.repeat(np.asarray(points, float), per_point, 0)
        p += np.random.normal(0, jitter, p.shape)
        c = np.repeat(np.asarray(colors, float), per_point, 0)
        self.att = np.vstack([self.att, p]); self.att_col = np.vstack([self.att_col, c])
        self.att_age = np.r_[self.att_age, np.zeros(len(p))]

    def step(self):
        self.t += 1
        if len(self.att) == 0:
            return 0
        self.att_age += 1
        keep = self.att_age < LIFE
        d = np.linalg.norm(self.att[:, None, :] - self.nodes[None, :, :], axis=2)   # (M,N)
        near = d.argmin(1); dmin = d[np.arange(len(d)), near]
        keep &= dmin > KILL                                            # consumed
        grow = keep & (dmin < ATTRACT)
        grew = 0
        if grow.any() and len(self.nodes) < MAX_NODES:
            for i in np.unique(near[grow]):
                sel = grow & (near == i)
                v = self.att[sel] - self.nodes[i]
                v = (v / np.linalg.norm(v, axis=1, keepdims=True)).mean(0)
                v += np.random.normal(0, NOISE, 2)
                v /= np.linalg.norm(v) + 1e-9
                self.nodes = np.vstack([self.nodes, self.nodes[i] + SEG * v])
                self.parent.append(int(i))
                col = self.att_col[sel].mean(0)
                self.color.append(tuple(int(0.7 * a + 0.3 * b) for a, b in zip(col, self.color[i])))
                self.count.append(1)
                j = int(i)
                while j >= 0:                                           # thicken the path to the soma
                    self.count[j] += 1
                    j = self.parent[j]
                grew += 1
        self.att, self.att_col, self.att_age = self.att[keep], self.att_col[keep], self.att_age[keep]
        return grew

    def render(self, frame_bgr, redraw_every=15):
        h, w = frame_bgr.shape[:2]
        bg = (cv2.cvtColor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.25).astype(np.uint8)
        if self._cache is None or self.t % redraw_every == 0 or self._cache.shape[:2] != (h, w):
            self._cache = np.zeros((h, w, 3), np.uint8)
            order = range(1, len(self.nodes))
            self._drawn = 0
        else:
            order = range(max(1, self._drawn), len(self.nodes))
        for k in order:
            p, q = self.nodes[k], self.nodes[self.parent[k]]
            th = 1 + int(np.log2(self.count[k]) / 1.6)
            cv2.line(self._cache, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.color[k], th, cv2.LINE_AA)
        self._drawn = len(self.nodes)
        img = cv2.add(bg, self._cache)
        for (x, y), c, a in zip(self.att, self.att_col, self.att_age):   # resource, fading
            k = 1 - a / LIFE
            cv2.circle(img, (int(x), int(y)), 2, tuple(int(v * k) for v in c), -1)
        s = self.nodes[0]
        cv2.circle(img, (int(s[0]), int(s[1])), 9, (255, 255, 255), 2)
        cv2.putText(img, f"arbor: {len(self.nodes) - 1} segments   resource {len(self.att)}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        return img
