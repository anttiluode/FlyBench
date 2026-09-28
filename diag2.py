import sys, numpy as np
import neuron as nr, neuron_bench as nb
nb.TRAIN, nb.TEST = 2500, 10
st = dict(food=0.0, food_near=0.0, pen=0.0, clicks=0, clicks_nobody=0, spikes=0, spikes_unconf=0, decay=0.0, elig_at_click=[], heard_near=[], heard_far=[])
orig_step = nr.Soma.step
def step(self, bugs, click=False, learn=True):
    before = {b: b.energy for b in bugs}
    near = {b for b in bugs if np.hypot(b.x - nb.LAMP[0], b.y - nb.LAMP[1]) < 220}
    if click:
        st["elig_at_click"].append(sorted([round(b.elig, 1) for b in bugs if b.elig > 0.5], reverse=True)[:5])
    y = orig_step(self, bugs, click, learn)
    for b in bugs:
        if b in before:
            d = b.energy - before[b]
            if d > 0:
                st["food"] += d; st["food_near"] += d * (b in near)
            elif d < 0:
                st["pen"] -= d
    st["clicks"] += click
    st["spikes"] += y
    for b in bugs:
        if getattr(b, "settled", False):
            (st["heard_near"] if b in near else st["heard_far"]).append(b.heard)
    return y
nr.Soma.step = step
lg = nb.run("clicker", int(sys.argv[1]))
print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items() if not isinstance(v, list)})
print("burn by decay over run ~", round(0.05 * np.sum(lg["pop"][:nb.TRAIN]), 0))
print("top eligibilities at first 8 clicks:", st["elig_at_click"][:8])
hn, hf = np.array(st["heard_near"]), np.array(st["heard_far"])
print(f"settled near lamp: {len(hn)} fly-frames, heard>0 in {np.mean(hn>0):.2f}, mean heard {hn.mean():.2f}")
print(f"settled elsewhere: {len(hf)} fly-frames, heard>0 in {np.mean(hf>0):.2f}, mean heard {hf.mean():.2f}")
