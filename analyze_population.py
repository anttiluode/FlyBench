"""
analyze_population.py - read a population saved with "Save Population".

    python analyze_population.py pop.json

Children in dumbflies are clones of ONE parent (try_mate calls inherit_traits
twice and the second call overwrites the first) with rare, small mutations
(5% chance per gene, about +-5%). So a saved population is a set of clonal
families, and body size, which starts random per fly and is useless to a
resonator fly, works as a family name.

Prints: families (by size), their tuning, age range, energy; and how extreme
the winning families' tunings are compared with the random starting range.
"""
import json, sys, collections
import numpy as np

OMEGA0 = (0.1, 1.3)      # starting range for omega in resonator_brain.py
SIZE0 = (3.0, 6.0)       # starting range for size in dumbflies.py

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "pop.json"))
fps = float(sys.argv[2]) if len(sys.argv) > 2 else None
print(f"{len(d)} flies")

fam = collections.defaultdict(list)
for b in d:
    fam[round(b["traits"]["size"], 3)].append(b)

# merge families whose sizes differ only by a mutation step from a bigger family
sizes = sorted(fam, key=lambda s: -len(fam[s]))
print("\nfamily (size)   flies  tuning omega (median, range)        r        age (frames)   energy median  hungry(<50)")
tops = []
for s in sizes:
    bs = fam[s]
    w = np.array([b["traits"]["omega"] for b in bs])
    r = np.median([b["traits"]["r"] for b in bs])
    age = [b["age"] for b in bs]
    e = np.array([b["energy"] for b in bs])
    hz = f" = {np.median(w) * fps / (2 * np.pi):.2f} Hz" if fps else ""
    print(f"  {s:6.3f}       {len(bs):4d}   {np.median(w):.3f}{hz}  ({w.min():.3f}-{w.max():.3f})   {r:.3f}   "
          f"{min(age)}-{max(age)}     {np.median(e):6.1f}        {(e < 50).sum()}")
    tops.append((len(bs), np.median(w), s))

big = [t for t in tops if t[0] >= 0.05 * len(d)]
print(f"\nFamilies with >=5% of the colony: {len(big)}")
for n, w, s in big:
    qw = (w - OMEGA0[0]) / (OMEGA0[1] - OMEGA0[0])
    qs = (s - SIZE0[0]) / (SIZE0[1] - SIZE0[0])
    print(f"  size {s:.3f}: tuning sits at {qw:4.0%} of the starting range, size at {qs:4.0%}")
if big:
    q = np.array([(w - OMEGA0[0]) / (OMEGA0[1] - OMEGA0[0]) for _, w, _ in big])
    k = len(q)
    p_low = np.max(q) ** k
    print(f"\nIf winning had nothing to do with tuning, the chance that all {k} big families would be tuned")
    print(f"at or below {np.max(q):.0%} of the range is {p_low:.3f} ({np.max(q):.2f}^{k}). One run, so weak evidence either way.")
