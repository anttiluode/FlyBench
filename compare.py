"""
compare.py - is a resonator model cheaper than a transformer, and when?

Task: forecast h steps ahead from the past (h = 1 and h = 8), scored on the last
20% of each series (never seen in training). Error is reported as MSE divided by
the error of a "same as last value" guess, so < 1 means better than naive.

Models
  naive       last value
  linear      ridge regression on the last 32 values (classic AR)
  res_random  64 windowed resonators  z <- r e^{iw} z + (1-r) x  (the fly unit),
              random w, r; features Re z, Im z, x; ridge readout
  res_grown   same bank, then 30 rounds of selection: resonators whose readout
              contribution is smallest die, the most useful ones bud mutated
              copies (the colony rule). Kept only if validation improves. No gradients.
  transformer 2 layers, d=32, 4 heads, window 32, Adam, early stopping on validation

Data
  sunspots   yearly sunspot number 1700-2008 (309 points)       statsmodels
  co2        weekly change in Mauna Loa CO2 (2283 points)        statsmodels
  switching  synthetic: rhythm that switches between 2 and 3 components every
             300-600 steps + noise; train size varied (250, 1000, 4000)
"""
import time, json, math
import numpy as np
import torch, torch.nn as nn
import statsmodels.api as sm

torch.set_num_threads(2)
RNG = np.random.default_rng(0)
L = 32


# ---------------------------------------------------------------- data
def load_sunspots():
    x = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].values.astype(float)
    return np.sqrt(x)


def load_co2():
    s = sm.datasets.co2.load_pandas().data["co2"].interpolate().bfill().values.astype(float)
    return np.diff(s)


def make_switching(n, seed=0):
    r = np.random.default_rng(seed)
    t, out, ph = 0, [], r.uniform(0, 6.28, 3)
    while t < n:
        seg = int(r.integers(300, 600))
        k = int(r.integers(2, 4))
        freqs = r.uniform(0.01, 0.2, k)
        amps = r.uniform(0.5, 1.5, k)
        for i in range(seg):
            out.append(sum(a * math.sin(2 * math.pi * f * (t + i) + p) for a, f, p in zip(amps, freqs, ph)))
        t += seg
    x = np.array(out[:n]) + r.normal(0, 0.3, n)
    return x


# ---------------------------------------------------------------- features
def resonator_features(x, w, r):
    """Run the bank over the whole series; row t = state after seeing x[t]."""
    lam = r * np.exp(1j * w)
    z = np.zeros(len(w), complex)
    F = np.empty((len(x), 2 * len(w) + 1))
    for t, v in enumerate(x):
        z = lam * z + (1 - r) * v
        F[t, :len(w)] = z.real
        F[t, len(w):2 * len(w)] = z.imag
        F[t, -1] = v
    return F


def lag_features(x):
    F = np.zeros((len(x), L))
    for k in range(L):
        F[k:, k] = x[:len(x) - k]
    return F


def ridge_fit(F, y, lam=1e-2):
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Z = (F - mu) / sd
    Z1 = np.c_[Z, np.ones(len(Z))]
    A = Z1.T @ Z1 + lam * len(Z) * np.eye(Z1.shape[1])
    W = np.linalg.solve(A, Z1.T @ y)
    return (mu, sd, W)


def ridge_pred(m, F):
    mu, sd, W = m
    return np.c_[(F - mu) / sd, np.ones(len(F))] @ W


# ---------------------------------------------------------------- models
def split_idx(n, h):
    """targets y[t] = x[t+h]; usable t in [L, n-h). train 0-64%, val 64-80%, test 80-100%."""
    t = np.arange(L, n - h)
    a, b = int(0.64 * n), int(0.80 * n)
    return t[t < a - h], t[(t >= a) & (t < b - h)], t[t >= b]


def run_linear(x, h):
    F = lag_features(x); tr, va, te = split_idx(len(x), h)
    t0 = time.perf_counter()
    m = ridge_fit(F[np.r_[tr, va]], x[np.r_[tr, va] + h])
    fit = time.perf_counter() - t0
    return ridge_pred(m, F[te]), fit, L + 1


def random_bank(n, rng):
    w = np.exp(rng.uniform(np.log(0.02), np.log(np.pi), n))
    r = rng.uniform(0.8, 0.99, n)
    return w, r


def run_res_random(x, h, n=64, seed=0):
    rng = np.random.default_rng(seed)
    w, r = random_bank(n, rng)
    tr, va, te = split_idx(len(x), h)
    t0 = time.perf_counter()
    F = resonator_features(x, w, r)
    m = ridge_fit(F[np.r_[tr, va]], x[np.r_[tr, va] + h])
    fit = time.perf_counter() - t0
    return ridge_pred(m, F[te]), fit, 2 * n + 2 + 2 * n


def run_res_grown(x, h, n=64, rounds=30, seed=0):
    rng = np.random.default_rng(seed)
    w, r = random_bank(n, rng)
    tr, va, te = split_idx(len(x), h)
    t0 = time.perf_counter()

    def score(w, r):
        F = resonator_features(x, w, r)
        m = ridge_fit(F[tr], x[tr + h])
        e = np.mean((ridge_pred(m, F[va]) - x[va + h]) ** 2)
        return e, m, F

    best_e, m, F = score(w, r)
    for _ in range(rounds):
        mu, sd, W = m
        contrib = np.abs(W[:n]) * sd[:n] + np.abs(W[n:2 * n]) * sd[n:2 * n]   # usefulness = "food"
        order = np.argsort(contrib)
        k = n // 4
        w2, r2 = w.copy(), r.copy()
        for dead, parent in zip(order[:k], order[-k:][::-1]):             # starve the weakest,
            w2[dead] = np.clip(w[parent] * np.exp(rng.normal(0, 0.15)), 0.005, np.pi)  # bud the strongest
            r2[dead] = np.clip(r[parent] + rng.normal(0, 0.02), 0.5, 0.995)
        e2, m2, F2 = score(w2, r2)
        if e2 < best_e:
            best_e, w, r, m, F = e2, w2, r2, m2, F2
    F = resonator_features(x, w, r)
    m = ridge_fit(F[np.r_[tr, va]], x[np.r_[tr, va] + h])
    fit = time.perf_counter() - t0
    return ridge_pred(m, F[te]), fit, 2 * n + 2 + 2 * n


class TinyTransformer(nn.Module):
    def __init__(self, d=32, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 1)

    def forward(self, s):
        return self.out(self.enc(self.inp(s.unsqueeze(-1)) + self.pos)[:, -1]).squeeze(-1)


def run_transformer(x, h, seed=0, max_epochs=300, patience=30):
    torch.manual_seed(seed)
    tr, va, te = split_idx(len(x), h)
    mu, sd = x[:int(0.64 * len(x))].mean(), x[:int(0.64 * len(x))].std() + 1e-8
    xn = (x - mu) / sd
    win = lambda idx: torch.tensor(np.stack([xn[i - L + 1:i + 1] for i in idx]), dtype=torch.float32)
    Xtr, ytr = win(tr), torch.tensor(xn[tr + h], dtype=torch.float32)
    Xva, yva = win(va), torch.tensor(xn[va + h], dtype=torch.float32)
    Xte = win(te)
    net = TinyTransformer()
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    t0 = time.perf_counter()
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = ((net(Xtr[b]) - ytr[b]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(Xva) - yva) ** 2).mean().item()
        if v < best:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    fit = time.perf_counter() - t0
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = net(Xte).numpy() * sd + mu
    return p, fit, sum(p_.numel() for p_ in net.parameters())


def step_cost():
    """Seconds per new sample in streaming use (update state + predict), single sample."""
    n = 64
    w, r = random_bank(n, np.random.default_rng(0)); lam = r * np.exp(1j * w)
    z = np.zeros(n, complex); Wt = np.random.randn(2 * n + 1)
    t0 = time.perf_counter()
    for i in range(20000):
        z = lam * z + (1 - r) * 0.5
        _ = z.real @ Wt[:n] + z.imag @ Wt[n:2 * n]
    res = (time.perf_counter() - t0) / 20000
    net = TinyTransformer().eval(); s = torch.zeros(1, L)
    with torch.no_grad():
        for _ in range(50): net(s)
        t0 = time.perf_counter()
        for _ in range(2000): net(s)
    tf = (time.perf_counter() - t0) / 2000
    return res, tf


def evaluate(x, h, name, seed=0):
    tr, va, te = split_idx(len(x), h)
    y = x[te + h]
    naive = np.mean((x[te] - y) ** 2)
    out = {}
    for mname, fn in (("linear", run_linear), ("res_random", run_res_random),
                      ("res_grown", run_res_grown), ("transformer", run_transformer)):
        args = (x, h) if mname == "linear" else (x, h, )
        p, fit, npar = fn(x, h) if mname == "linear" else fn(x, h, seed=seed)
        out[mname] = dict(rel_mse=float(np.mean((p - y) ** 2) / naive), fit_s=round(fit, 3), params=int(npar))
    print(name, f"h={h}", {k: round(v["rel_mse"], 3) for k, v in out.items()},
          {k: v["fit_s"] for k, v in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    R = {}
    rs, tf = step_cost()
    R["streaming_seconds_per_step"] = dict(resonator_bank=rs, transformer=tf)
    print("per-step cost", R["streaming_seconds_per_step"], flush=True)
    for name, x in (("sunspots", load_sunspots()), ("co2", load_co2())):
        for h in (1, 8):
            R[f"{name}_h{h}"] = {f"seed{s}": evaluate(x, h, name, seed=s) for s in (0, 1, 2)}
    for n in (250, 1000, 4000):
        for h in (1, 8):
            R[f"switching_n{n}_h{h}"] = {f"seed{s}": evaluate(make_switching(n, seed=10 + s), h, f"switching n={n}", seed=s)
                                          for s in (0, 1, 2)}
    json.dump(R, open("compare_results.json", "w"), indent=1)
