"""Extreme-tail cluster designs (main text, Subsection 6.4).

Six designs, R replications each (default 600):
  tail designs   whole state blocks of the standardized residualized ADH regressor,
                 sampled with replacement to form G independent clusters
  ideal designs  independent standardized Laplace draws at the sampled cluster sizes
  u = sigma(s) {sqrt(.35) a_g + sqrt(.65) z},  x = cos(.25) s - sin(.25) u,  y = .7 x + u
  sigma = 1, or sigma^2(s) proportional to 1 + .6 tanh s (mean one).  MI holds exactly.

For each replication, family (P, B, L, R) and tested slope (0.7, 0.6, 0.8) the script
stores the analytic statistic S, its Satterthwaite degrees of freedom, the full-refit
jackknife statistic, its wild p-value and the frozen-first-step statistic.

Usage: python tail_sim.py [R] [cores]      writes output/r61/tail_sim.npz
"""
import sys, os
import numpy as np
from multiprocessing import get_context
import r61lib as L
from iiv.data import load
from iiv.core import Context

BETA0 = 0.7
BETAS = np.array([0.7, 0.6, 0.8])
FAMS = ['P', 'B', 'L', 'R']
DESIGNS = [('tail', 'const', 48), ('tail', 'vary', 48), ('ideal', 'const', 48), ('ideal', 'vary', 48),
           ('tail', 'const', 192), ('tail', 'vary', 192)]
B_WILD = 999
_POOL = {}


def adh_pool():
    """State blocks of the standardized residualized ADH regressor."""
    if not _POOL:
        C = Context(load('adh'))
        s = C.resid(C.x); s = s / np.sqrt(np.mean(s ** 2))
        _POOL['blocks'] = [s[C.cl_codes == g] for g in range(C.G)]
        _POOL['vnorm'] = np.mean(1 + 0.6 * np.tanh(s))
    return _POOL


def draw(shape, var, G, rng):
    pool = adh_pool()
    pick = rng.integers(0, len(pool['blocks']), G)
    blocks = [pool['blocks'][j] for j in pick]
    sizes = np.array([len(b) for b in blocks])
    cl = np.repeat(np.arange(G), sizes)
    n = int(sizes.sum())
    if shape == 'tail':
        s = np.concatenate(blocks); vnorm = pool['vnorm']
    else:
        s = rng.laplace(0.0, 1 / np.sqrt(2), n); vnorm = 1.0
    sig = np.ones(n) if var == 'const' else np.sqrt((1 + 0.6 * np.tanh(s)) / vnorm)
    u = sig * (np.sqrt(0.35) * rng.standard_normal(G)[cl] + np.sqrt(0.65) * rng.standard_normal(n))
    x = np.cos(0.25) * s - np.sin(0.25) * u
    y = BETA0 * x + u
    return y, x, cl


def one(args):
    d, seed = args
    shape, var, G = DESIGNS[d]
    rng = np.random.default_rng(seed)
    y, x, cl = draw(shape, var, G, rng)
    W = np.ones((len(y), 1))
    C = L.make_context(y, x, W, np.char.zfill(cl.astype(str), 4)); P = C.prep()
    m, M = L.jk_refit(W, y, x, C.cl_codes, C.G, BETAS, FAMS)
    res = np.empty((len(FAMS), len(BETAS), 5))
    for jf, f in enumerate(FAMS):
        for ib, b in enumerate(BETAS):
            S, nu, Tfr = L.analytic(C, P, b, L.FAMS[f])
            T, V, dd = L.jk_stat(m[ib, jf], M[:, ib, jf])
            res[jf, ib] = [S, nu, T, L.wild_p(T, V, dd, B_WILD, rng), Tfr]
    return d, res, len(y)


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
    ss = np.random.SeedSequence(20261110)
    seeds = ss.spawn(len(DESIGNS) * R)
    jobs = [(d, seeds[d * R + r]) for d in range(len(DESIGNS)) for r in range(R)]
    out = np.empty((len(DESIGNS), R, len(FAMS), len(BETAS), 5)); nobs = np.zeros((len(DESIGNS), R))
    cnt = np.zeros(len(DESIGNS), int)
    with get_context('spawn').Pool(cores) as pool:
        for k, (d, res, n) in enumerate(pool.imap(one, jobs, chunksize=4)):
            out[d, cnt[d]] = res; nobs[d, cnt[d]] = n; cnt[d] += 1
            if (k + 1) % 300 == 0:
                print('done', k + 1, 'of', len(jobs), flush=True)
    np.savez(L.out('tail_sim.npz'), stats=out, nobs=nobs, betas=BETAS, fams=np.array(FAMS),
             designs=np.array(['%s-%s-%d' % d for d in DESIGNS]))
    print('saved', L.out('tail_sim.npz'))
