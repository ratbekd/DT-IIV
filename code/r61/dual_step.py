"""Angrist-Krueger: mean and error-side tendencies with the degree step 1{educ = 16}.

Model: y = beta x + gamma D + u, D = 1{x = 16}, D endogenous like x. At a tested (b, g):
u_w = standardized residual of y - b x - g D (all residualized on the controls), s the
standardized part of x orthogonal to u_w, and the moment columns of dual_tendency.contrib
(mean tendencies, VT, error-side bounded and sieve). The jackknife refits the projections of
y, x and D on the controls (delete one state of birth). For each slope b the statistic of
each moment set is minimized over g on a grid of 41 values in [-0.2, 0.6]; the slope enters
the projection set if the minimum is below the Hotelling cutoff for m moments, and the J test
of the joint minimum has m - 2 degrees of freedom.

Usage: python dual_step.py <NEW7080.dta> [cores]     writes output/r61/dual_step_ak.json
"""
import sys, os, json
import numpy as np
from scipy.stats import chi2
from multiprocessing import get_context
from dual_tendency import COLS, ORDER, cut, comps, fmt
from dual_tendency import phi  # noqa: F401

GG = np.linspace(-0.2, 0.6, 41)


def contrib(xt, yt, qt, b, g):
    u = yt - b * xt - g * qt
    u = u - u.mean(); uw = u / np.sqrt(np.mean(u ** 2))
    xw = xt / np.sqrt(np.mean(xt ** 2))
    v = xw - np.mean(xw * uw) * uw
    v = v - v.mean(); s = v / np.sqrt(np.mean(v ** 2))
    ph = lambda z: np.exp(-z ** 2 / 2)
    return np.column_stack([uw * s ** 2, uw * s ** 3, uw * np.tanh(s), uw * ph(s),
                            s * uw ** 2, s * np.tanh(uw), s * np.tanh(uw / 2), s * ph(uw), s * uw * ph(uw),
                            s * uw ** 3, s * np.tanh(uw)])


class Data:
    def __init__(self, y, x, q, W, cl):
        self.W = W
        codes = np.unique(cl, return_inverse=True)[1]
        self.G = codes.max() + 1
        WW = W.T @ W; V = np.column_stack([y, x, q]); WV = W.T @ V
        Pi = np.linalg.solve(WW, WV); R = V - W @ Pi
        self.yt, self.xt, self.qt = R.T
        self.keep, self.d = [], []
        for gi in range(self.G):
            ix = codes == gi; Wg = W[ix]
            self.keep.append(~ix); self.d.append(Pi - np.linalg.solve(WW - Wg.T @ Wg, WV - Wg.T @ V[ix]))

    def T(self, b, g):
        m = contrib(self.xt, self.yt, self.qt, b, g).mean(0)
        Mg = np.empty((self.G, len(m)))
        for gi in range(self.G):
            k = self.keep[gi]; A = self.W[k] @ self.d[gi]
            Mg[gi] = contrib(self.xt[k] + A[:, 1], self.yt[k] + A[:, 0], self.qt[k] + A[:, 2], b, g).mean(0)
        dd = Mg - Mg.mean(0); Vj = (self.G - 1) / self.G * dd.T @ dd
        return {nm: float(m[c] @ np.linalg.solve(Vj[np.ix_(c, c)], m[c])) for nm, c in COLS.items()}


_D = {}


def _init(args):
    _D['d'] = Data(*args)


def _prof(b):
    d = _D['d']
    Ts = [d.T(b, g) for g in GG]
    return {nm: (min(t[nm] for t in Ts), float(GG[int(np.argmin([t[nm] for t in Ts]))])) for nm in ORDER} | \
           {'_lin': Ts[int(np.argmin(np.abs(GG)))]}


def main():
    import pandas as pd
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    d = pd.read_stata(sys.argv[1])
    d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
    cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
    cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
    W = np.column_stack(cols); y = d.v9.values.astype(float); x = d.v4.values.astype(float)
    q = (x == 16).astype(float); cl = d.v17.astype(int).values
    grid = np.linspace(0.046, 0.070, 25)
    with get_context('spawn').Pool(cores, initializer=_init, initargs=((y, x, q, W, cl),)) as pool:
        res = pool.map(_prof, list(grid))
    G = len(np.unique(cl))
    out = dict(grid=grid.tolist(), gamma_grid=GG.tolist(), G=G, sets={})
    print('Angrist-Krueger with the degree step 1{educ = 16}; slopes %.3f to %.3f, gamma grid %.2f to %.2f'
          % (grid[0], grid[-1], GG[0], GG[-1]))
    for nm in ORDER:
        m = len(COLS[nm])
        T = np.array([r[nm][0] for r in res]); gh = np.array([r[nm][1] for r in res]); i = int(np.argmin(T))
        rec = dict(m=m, T=T.tolist(), gamma=gh.tolist(), Tmin=float(T[i]), beta=float(grid[i]), gamma_hat=float(gh[i]),
                   p_J=float(chi2.sf(T[i], m - 2)) if m > 2 else None,
                   set_F=comps(T <= cut(m, G, 'F'), grid), set_chi=comps(T <= cut(m, G, 'chi'), grid))
        out['sets'][nm] = rec
        print('  %-10s m %d  min T %8.2f at beta %.4f, gamma %.2f  J p %s  set(F): %s' % (
            nm, m, rec['Tmin'], rec['beta'], rec['gamma_hat'], ' -- ' if rec['p_J'] is None else '%.4f' % rec['p_J'],
            fmt(rec['set_F'])), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61',
                                     'dual_step_ak.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
