"""Exact H0 persistence / ToMATo-equivalent masses for SAC's tanh-Gaussian policy (no graph needed).

p(a) = prod_d p_d(a_d), p_d = density of tanh(N(mu_d, sigma_d^2)) on (-1,1):
  log p_d(a) = -(u - mu)^2 / (2 sigma^2) + 2 log cosh u + const,  u = atanh a
1D: one mode, or (sigma > 1/sqrt 2 and |mu| small enough) two modes h_d (higher), l_d (lower) with antimode u*_d,
  barrier_d = log p_d(l_d) - log p_d(u*_d) (nats), masses Phi((u*-mu)/sigma) / 1 - Phi(...).
Product density: modes = choices of h/l per bimodal dim. A mode m with lower-choice set L(m) merges into a higher mode
  through the cheapest single-dim switch, so its persistence = min_{d in L(m)} barrier_d (exact for separable densities).
  => K(tau) = #{L : all barriers_d >= tau, d in L, and peak >= 1% of the global max}.
ToMATo at tau: dims with barrier < tau are merged; cluster masses = products of 1D masses over the remaining dims.
Usage: python sac_exact.py ph_sac/*.npz   (prints a table; writes ph_sac/exact_summary.txt)
"""
import itertools, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq

HF = np.log(100.0)
TAUS = [0.25, 0.5, 1.0, 2.0]


def lp1(u, mu, s):
    return -(u - mu) ** 2 / (2 * s * s) + 2 * np.log(np.cosh(u))


def one_dim(mu, s):
    f = lambda u: (u - mu) / (s * s) - 2 * np.tanh(u)
    lo, hi = min(mu, -2.0) - 12 * s - 5, max(mu, 2.0) + 12 * s + 5
    g = np.linspace(lo, hi, 20001); fv = f(g)
    roots = sorted(set(np.round([brentq(f, g[i], g[i + 1]) for i in np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]], 9)))
    if len(roots) != 3:
        return None
    u1, um, u2 = roots
    m1 = ndtr((um - mu) / s)
    a, b = (u1, m1), (u2, 1 - m1)
    hi_, lo_ = (a, b) if lp1(u1, mu, s) >= lp1(u2, mu, s) else (b, a)
    return dict(barrier=lp1(lo_[0], mu, s) - lp1(um, mu, s), drop=lp1(hi_[0], mu, s) - lp1(lo_[0], mu, s),
                q=(hi_[1], lo_[1]))


def state_metrics(mu, sd):
    B = [x for x in (one_dim(m, s) for m, s in zip(mu, sd)) if x is not None]
    out = dict(n_bimodal=len(B))
    for t in TAUS:
        dims = [b for b in B if b["barrier"] >= t]
        # count modes (subsets of the kept dims) with peak >= 1% of the global max
        k = 0
        for L in itertools.product([0, 1], repeat=len(dims)):
            if sum(d["drop"] for d, l in zip(dims, L) if l) <= HF:
                k += 1
        q = np.array([np.prod([d["q"][l] for d, l in zip(dims, L)]) for L in itertools.product([0, 1], repeat=len(dims))]) \
            if dims else np.array([1.0])
        qn = q[q > 0]
        out[t] = dict(K=k, K5=int((q >= 0.05).sum()), meff=float(np.exp(-(qn * np.log(qn)).sum())))
    return out


L = ["SAC (tanh-Gaussian), exact. K(tau): modes with persistence >= tau and peak >= 1% of max. "
     "ToMATo@tau: mass >= 5% count, M_eff.",
     f"{'file':34s} {'states':>6s} {'bimodal dims/state':>18s} " + " ".join(f"K@{t:<5}" for t in TAUS)
     + "  K>=2@0.5  K>=2@1   K5@0.5 K5@1  Meff@0.5 Meff@1"]
for fn in sorted(sys.argv[1:]):
    d = np.load(fn)
    R = [state_metrics(m, s) for m, s in zip(d["mu"].astype(np.float64), d["sigma"].astype(np.float64))]
    g = lambda t, k: np.array([r[t][k] for r in R], float)
    L.append(f"{Path(fn).stem:34s} {len(R):6d} {np.mean([r['n_bimodal'] for r in R]):18.2f} "
             + " ".join(f"{g(t, 'K').mean():7.2f}" for t in TAUS)
             + f"  {(g(0.5, 'K') >= 2).mean():8.3f} {(g(1.0, 'K') >= 2).mean():7.3f}   {g(0.5, 'K5').mean():6.2f} {g(1.0, 'K5').mean():4.2f}"
             + f"  {g(0.5, 'meff').mean():8.2f} {g(1.0, 'meff').mean():6.2f}")
Path("ph_sac/exact_summary.txt").write_text("\n".join(L))
print("\n".join(L))
