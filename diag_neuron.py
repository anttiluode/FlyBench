import sys, json, numpy as np
import neuron as nr, neuron_bench as nb
rec = []
orig = nr.Soma.step
def step(self, bugs, click=False, learn=True):
    y = orig(self, bugs, click, learn)
    near = [b for b in bugs if np.hypot(b.x - nb.LAMP[0], b.y - nb.LAMP[1]) < 220]
    rec.append(dict(t=self.t - 1, pop=len(bugs), near=len(near),
                    near_om=[round(b.traits["omega"], 2) for b in near],
                    E=float(sum(b.energy for b in bugs)), y=int(y), click=int(click)))
    return y
nr.Soma.step = step
seed = int(sys.argv[1])
lg = nb.run("clicker", seed)
json.dump(rec, open(f"diag_s{seed}.json", "w"))
lab = np.array(lg["label"])
for a, b in ((3000, 3500), (3500, 4000), (4000, 4300), (4300, 4600), (4600, 4900)):
    R = rec[a:b]
    print(f"{a}-{b} pop {min(r['pop'] for r in R)}-{max(r['pop'] for r in R)}  near-lamp mean {np.mean([r['near'] for r in R]):.1f}"
          f"  total energy {R[0]['E']:.0f}->{R[-1]['E']:.0f}  clicks {sum(r['click'] for r in R)}")
    A = [r for r, l in zip(R, lab[a:b]) if l == "A"]; B = [r for r, l in zip(R, lab[a:b]) if l == "B"]
    print(f"     near-lamp during A {np.mean([r['near'] for r in A]) if A else float('nan'):.1f}, during B {np.mean([r['near'] for r in B]) if B else float('nan'):.1f};"
          f" near-lamp tunings at end: {rec[b-1]['near_om']}")
refills = [r["t"] for i, r in enumerate(rec[1:], 1) if r["pop"] - rec[i - 1]["pop"] >= 5]
print("refill events (pop jumped by >=5):", refills)
