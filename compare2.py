"""
compare2.py - the case where a transformer should win: a FIXED set of three
nonlinear regimes that recur (Markov switching, dwell 100-300 steps), so more
data really does teach more. Training size 500 -> 20000.
"""
import json, math
import numpy as np
import compare as C

REG = [  # each regime: (freqs, amps, waveshaping gain) - fixed for the whole series
    ((0.013, 0.071), (1.0, 0.6), 2.5),
    ((0.037,), (1.2,), 0.5),
    ((0.021, 0.113, 0.160), (0.7, 0.5, 0.4), 1.5),
]


def make_regimes(n, seed):
    r = np.random.default_rng(seed)
    out, t, k = [], 0, int(r.integers(3))
    while t < n:
        dwell = int(r.integers(100, 301))
        f, a, g = REG[k]
        for i in range(dwell):
            s = sum(ai * math.sin(2 * math.pi * fi * (t + i)) for fi, ai in zip(f, a))
            out.append(math.tanh(g * s) / math.tanh(g))            # nonlinear waveshaping
        t += dwell
        k = (k + int(r.integers(1, 3))) % 3
    return np.array(out[:n]) + r.normal(0, 0.15, n)


if __name__ == "__main__":
    R = {}
    for n in (500, 2000, 8000, 20000):
        for h in (1, 8):
            R[f"regimes_n{n}_h{h}"] = {f"seed{s}": C.evaluate(make_regimes(n, 20 + s), h, f"regimes n={n}", seed=s)
                                       for s in (0, 1)}
            json.dump(R, open("compare2_results.json", "w"), indent=1)
