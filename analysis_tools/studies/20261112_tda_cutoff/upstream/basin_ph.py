"""Basin-graph H0 persistence: exact births (verified density maxima) + graph-estimated saddles.

Usage (results/modal_em, torch-venv):
  python basin_ph.py ibolt <data.npz> <out.pkl> [--n N] [--stride 8 | --states ..] [--samples 20000] [--k 15] [--seed 0]
  python basin_ph.py sac   <sac.npz>  <out.pkl> [...]
Why: on a plain sample kNN graph, any sample whose k neighbours are all lower is born as a "mode"; with
anisotropic / sharply peaked densities this creates spurious births (SAC, exactly unimodal, got K(1) ~ 2.7 on Hopper).
Here births can only be exact maxima:
  1. modes: iBOLT -- projected Modal EM from every point (fp32, tol 1e-4) -> dedup -> fp64 polishing -> Hessian/KKT
     verified local maxima (same code as modal_em.py). SAC -- exact product of 1D modes of tanh(N(mu, sigma^2)).
  2. basin label of every point = the verified mode its ascent reaches (iBOLT) / the side of each 1D antimode (SAC).
  3. graph = symmetric kNN on samples + modes + 8 points on each segment between the 6 heaviest modes;
     edge height h = min(f_i, f_j, f(midpoint)) with exact f = log p.
  4. saddle(A,B) estimate = max h over edges joining basin A and basin B. Each basin is joined to its mode by the
     ascent (monotone), so this is a path-realised lower bound of the true merge level -> no spurious births;
     persistence can only be over-estimated if no sampled path passes near the saddle (checked by sensitivity runs).
  5. H0 = union-find over basins (births = log p(mode)), elder rule. ToMATo at tau: merge if prominence < tau;
     cluster mass = share of POLICY samples (exploration points excluded).
"""
import argparse, itertools, math, pickle, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("kind", choices=["ibolt", "sac"]); p.add_argument("data"); p.add_argument("out")
p.add_argument("--n", type=int, default=0)
p.add_argument("--states", default="")
p.add_argument("--stride", type=int, default=1)
p.add_argument("--samples", type=int, default=20000)
p.add_argument("--k", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0]
SQ2 = math.sqrt(2.0)
ndtr = lambda x: 0.5 * torch.erfc(-x / SQ2)
gen = torch.Generator(device=dev).manual_seed(a.seed)
f = np.load(a.data)
if a.kind == "ibolt":
    MU, LS = f["mu"], f["log_std"]
    if a.n:
        MU, LS = MU[:, :a.n], LS[:, :a.n]
else:
    MU, SDs = f["mu"].astype(np.float64), f["sigma"].astype(np.float64)
S = MU.shape[0]
states = [int(x) for x in a.states.split(",")] if a.states else list(range(0, S, a.stride))


# ---------------- iBOLT mixture ----------------
class Mix:
    def __init__(s, mu, ls):
        s.mu, s.ls = mu, ls; s.sd = ls.exp(); s.lam = torch.exp(-2 * ls); s.lm = s.lam * mu
        s.c = (-math.log(mu.shape[0]) - torch.log(ndtr((1 - mu) / s.sd) - ndtr((-1 - mu) / s.sd)).sum(-1)
               - ls.sum(-1) - 0.5 * (s.lam * mu * mu).sum(-1) - 0.5 * mu.shape[1] * math.log(2 * math.pi))
        s.chunk = max(256, int(1.5e8 // mu.shape[0]))

    def logits(s, x):
        return s.c + x @ s.lm.T - 0.5 * (x * x) @ s.lam.T

    def logp(s, x):
        out = torch.empty(x.shape[0], dtype=x.dtype, device=dev)
        for k in range(0, x.shape[0], s.chunk):
            out[k:k + s.chunk] = torch.logsumexp(s.logits(x[k:k + s.chunk]), -1)
        return out

    def step(s, x):
        r = torch.softmax(s.logits(x), -1)
        return ((r @ s.lm) / (r @ s.lam)).clamp_(-1, 1)

    def sample(s, m):
        i = torch.randint(0, s.mu.shape[0], (m,), device=dev, generator=gen)
        mu, sd = s.mu[i], s.sd[i]
        lo, hi = ndtr((-1 - mu) / sd), ndtr((1 - mu) / sd)
        u = (lo + (hi - lo) * torch.rand(mu.shape, device=dev, generator=gen, dtype=mu.dtype)).clamp(1e-12, 1 - 1e-12)
        return (mu + sd * SQ2 * torch.erfinv(2 * u - 1)).clamp(-1, 1)


def ascend(mix, x, tol, maxit):
    x = x.clone(); done = torch.zeros(x.shape[0], dtype=torch.bool, device=dev)
    for _ in range(maxit):
        idx = (~done).nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        for k in range(0, idx.numel(), mix.chunk):
            j = idx[k:k + mix.chunk]
            xn = mix.step(x[j]); done[j[(xn - x[j]).abs().amax(-1) < tol]] = True; x[j] = xn
    return x


def dedup(x, lp, tol):
    """Greedy in descending density: keep a point unless it lies within tol (L_inf) of an already kept one.
    Streamed in chunks (same result as the all-pairs version, without the n x n matrix)."""
    order = torch.argsort(lp, descending=True); xs = x[order]; n = xs.shape[0]
    reps = xs[:0]
    for k in range(0, n, 4096):
        c = xs[k:k + 4096]
        if reps.shape[0]:
            c = c[~(torch.cdist(c, reps, p=float("inf")) < tol).any(1)]
        if c.shape[0] == 0:
            continue
        adj = (torch.cdist(c, c, p=float("inf")) < tol).cpu().numpy(); sup = np.zeros(c.shape[0], bool); keep = []
        for i in range(c.shape[0]):
            if not sup[i]:
                keep.append(i); sup |= adj[i]
        reps = torch.cat([reps, c[keep]])
    return reps


def local_max_mask(mix, X):
    out = []; lam, mu = mix.lam.double(), mix.mu.double()
    for k in range(0, X.shape[0], 16):
        x = X[k:k + 16]
        r = torch.softmax(mix.logits(x), -1).double()
        v = lam[None] * (mu[None] - x.double()[:, None])
        g = torch.einsum("bn,bnd->bd", r, v)
        H = torch.einsum("bn,bnd,bne->bde", r, v, v) - torch.diag_embed(r @ lam)
        act = x.abs() > 1 - 1e-6
        kkt = ~((g * torch.sign(x).double() < -1e-6) & act).any(-1)
        free = (~act).double()
        Hf = H * free[:, :, None] * free[:, None, :] - torch.diag_embed(1 - free)
        scale = torch.diagonal(H, dim1=1, dim2=2).abs().amax(-1).clamp_min(1e-12)
        out.append(kkt & (torch.linalg.eigvalsh(Hf)[:, -1] < -1e-7 * scale))
    return torch.cat(out)


def ibolt_modes_and_labels(mix32, mix64, X):
    """Verified modes (fp64, sorted by density) and the basin label of each row of X."""
    xc = ascend(mix32, X.float(), 1e-4, 3000)
    cand = dedup(xc, mix32.logp(xc), 1e-3)
    cand = ascend(mix64, cand.double(), 1e-10, 5000)
    cand = dedup(cand, mix64.logp(cand), 1e-3)
    ok = local_max_mask(mix64, cand)
    if ok.sum() == 0:
        ok[0] = True
    M = cand[ok]
    M = M[torch.argsort(mix64.logp(M), descending=True)]
    lab = torch.empty(X.shape[0], dtype=torch.long, device=dev); far = 0
    for k in range(0, X.shape[0], 4096):
        d = (xc[k:k + 4096].double()[:, None] - M[None]).abs().amax(-1)
        dm, am = d.min(1); lab[k:k + 4096] = am; far += int((dm > 1e-2).sum())
    return M, lab, far


# ---------------- SAC tanh-Gaussian ----------------
EPS = 1e-12


class TanhGauss:
    def __init__(s, mu, sd):
        s.mu, s.sd = mu, sd

    def logp(s, x):
        x = x.clamp(-1 + EPS, 1 - EPS); u = torch.atanh(x)
        return (-0.5 * ((u - s.mu) / s.sd) ** 2 - s.sd.log() - 0.5 * math.log(2 * math.pi) - torch.log1p(-x * x)).sum(-1)

    def sample(s, m):
        z = torch.randn((m, s.mu.shape[0]), device=dev, dtype=torch.float64, generator=gen)
        return torch.tanh(s.mu + s.sd * z).clamp(-1 + EPS, 1 - EPS)


def sac_1d(mu, sd):
    """1D maxima (a) and antimode (a) of tanh(N(mu, sd^2)) from a fine grid in u."""
    u = np.linspace(min(mu, -3) - 12 * sd - 3, max(mu, 3) + 12 * sd + 3, 400001)
    g = -(u - mu) ** 2 / (2 * sd * sd) + 2 * np.log(np.cosh(u))
    mx = np.where((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]))[0] + 1
    mn = np.where((g[1:-1] < g[:-2]) & (g[1:-1] <= g[2:]))[0] + 1
    return np.tanh(u[mx]), (np.tanh(u[mn[0]]) if len(mx) == 2 else None)


def sac_modes_and_labels(mu, sd, X):
    per = [sac_1d(m, s) for m, s in zip(mu, sd)]
    bim = [d for d, (mx, _) in enumerate(per) if len(mx) == 2]
    base = np.array([mx[0] if len(mx) == 1 else 0.0 for mx, _ in per])
    M = []
    for choice in itertools.product([0, 1], repeat=len(bim)):
        m = base.copy()
        for d, c in zip(bim, choice):
            m[d] = per[d][0][c]
        M.append(m)
    M = torch.tensor(np.array(M), device=dev, dtype=torch.float64)
    code = torch.zeros(X.shape[0], dtype=torch.long, device=dev)
    for i, d in enumerate(bim):  # label = binary code of the side of each 1D antimode (same order as itertools.product)
        code = code * 2 + (X[:, d] > per[d][1]).long()
    return M, code, len(bim)


# ---------------- graph + basin-level H0 ----------------
def knn_edges(X, k):
    n = X.shape[0]; nb = []
    for s0 in range(0, n, 2048):
        d = torch.cdist(X[s0:s0 + 2048].float(), X.float())
        d[torch.arange(d.shape[0]), torch.arange(s0, s0 + d.shape[0])] = float("inf")
        nb.append(d.topk(k, largest=False).indices)
    nb = torch.cat(nb).cpu().numpy()
    return np.unique(np.sort(np.stack([np.repeat(np.arange(n), k), nb.ravel()], 1), 1), axis=0)


def basin_filtration(births, be, bh, tau=None):
    K = len(births); par = np.arange(K); birth = births.copy(); pairs = []
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in np.argsort(-bh):
        x, y = find(be[i, 0]), find(be[i, 1])
        if x == y:
            continue
        if birth[x] < birth[y]:
            x, y = y, x
        if tau is None:
            pairs.append((birth[y], bh[i])); par[y] = x
        elif birth[y] - bh[i] < tau:
            par[y] = x
    return np.array(pairs).reshape(-1, 2), np.array([find(v) for v in range(K)])


res = []; t0 = time.time()
for si, s in enumerate(states):
    if a.kind == "ibolt":
        mix64 = Mix(torch.tensor(MU[s], device=dev, dtype=torch.float64), torch.tensor(LS[s], device=dev, dtype=torch.float64))
        mix32 = Mix(mix64.mu.float(), mix64.ls.float())
        X = mix64.sample(a.samples)
        M, lab, far = ibolt_modes_and_labels(mix32, mix64, X)
        dens = mix64
    else:
        pol = TanhGauss(torch.tensor(MU[s], device=dev, dtype=torch.float64), torch.tensor(SDs[s], device=dev, dtype=torch.float64))
        X = pol.sample(a.samples)
        M, lab, nb = sac_modes_and_labels(MU[s], SDs[s], X)
        far = 0; dens = pol
    n_pol = X.shape[0]; K = M.shape[0]
    births = dens.logp(M).cpu().numpy()
    qb = np.bincount(lab.cpu().numpy(), minlength=K) / n_pol          # basin masses (policy samples only)
    # exploration points: the modes themselves + segments between the 6 heaviest modes
    top = np.argsort(-qb)[:6]
    t = torch.linspace(0, 1, 10, device=dev, dtype=torch.float64)[1:-1, None]
    seg = [M[i] * (1 - t) + M[j] * t for i, j in itertools.combinations(top, 2)]
    E = torch.cat([M] + seg) if seg else M
    if a.kind == "ibolt":
        xcE = ascend(mix32, E.float(), 1e-4, 3000)
        lE = (xcE.double()[:, None] - M[None]).abs().amax(-1).argmin(1)
    else:
        _, lE, _ = sac_modes_and_labels(MU[s], SDs[s], E)
    P = torch.cat([X, E]); L = torch.cat([lab, lE]).cpu().numpy()
    fv = dens.logp(P)
    e = knn_edges(P, a.k)
    ei, ej = torch.tensor(e[:, 0], device=dev), torch.tensor(e[:, 1], device=dev)
    h = torch.minimum(torch.minimum(fv[ei], fv[ej]), dens.logp(0.5 * (P[ei] + P[ej]))).cpu().numpy()
    li, lj = L[e[:, 0]], L[e[:, 1]]
    cross = li != lj
    if cross.any():
        key = np.minimum(li, lj)[cross] * K + np.maximum(li, lj)[cross]
        o = np.lexsort((-h[cross], key)); key, hc = key[o], h[cross][o]
        first = np.r_[True, key[1:] != key[:-1]]
        be = np.stack([key[first] // K, key[first] % K], 1); bh = hc[first]
    else:
        be, bh = np.zeros((0, 2), int), np.zeros(0)
    pairs, root = basin_filtration(births, be, bh)
    n_comp = len(np.unique(root))
    tom = {}
    for tau in TAUS:
        _, rt = basin_filtration(births, be, bh, tau)
        q = np.bincount(rt, weights=qb, minlength=K); q = np.sort(q[q > 0])[::-1]
        tom[tau] = dict(q=q, K=len(q), K5=int((q >= 0.05).sum()), meff=float(np.exp(-(q * np.log(q)).sum())))
    pers = pairs[:, 0] - pairs[:, 1] if len(pairs) else np.zeros(0)
    res.append(dict(state=s, pairs=pairs.astype(np.float32), pers=pers.astype(np.float32), fmax=float(births.max()),
                    births=births, basin_mass=qb, n_modes=K, n_components=n_comp, far=far, tomato=tom,
                    n_points=P.shape[0], n_basin_edges=len(be)))
    if si < 2 or (si + 1) % 16 == 0:
        print(f"[{si + 1}/{len(states)}] state {s}: modes {K} comps {n_comp} far {far} | top pers "
              f"{np.round(np.sort(pers)[::-1][:5], 2).tolist()} | ToMATo K5 " + " ".join(f"{t}:{tom[t]['K5']}" for t in TAUS)
              + f" | {time.time() - t0:.0f}s", flush=True)
pickle.dump(dict(args=vars(a), taus=TAUS, res=res), open(a.out, "wb"))
HF = np.log(100.0)
Kt = lambda t: np.mean([1 + ((r["pers"] >= t) & (r["pairs"][:, 0] >= r["fmax"] - HF)).sum() + (r["n_components"] - 1) for r in res])
print(f"DONE {a.kind} {a.data}: K(0.25/0.5/1/2) {Kt(0.25):.2f} {Kt(0.5):.2f} {Kt(1):.2f} {Kt(2):.2f} | modes/state "
      f"{np.mean([r['n_modes'] for r in res]):.2f} | extra comps {np.mean([r['n_components'] - 1 for r in res]):.2f} | "
      f"far {np.mean([r['far'] for r in res]):.1f} | {time.time() - t0:.0f}s", flush=True)
