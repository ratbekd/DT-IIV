"""Angrist-Krueger: why do the error-side tendencies reject with the degree step?

Three checks at the degree-step fit y = beta x + gamma 1{x = 16} + u:
  moments  t statistics of each of the eleven moment columns (delete-state jackknife) at
           (beta, gamma) = (0.058, 0.18), the mean-tendency minimum, and (0.060, 0.10), the
           bounded error-side minimum. Odd functions of u (tanh u, tanh(u/2), u^3) against even
           ones (phi(u), u phi(u) is odd; u^2): an even signal points to a scale that varies
           with schooling.
  spread   mean of u^2 by years of schooling at the step fit, and the slope of log mean u^2 on
           schooling.
  scaled   the profile of dual_step.py with u divided by sigma(x), the root mean of u^2 within
           each year of schooling, re-estimated at each (beta, gamma) and in each jackknife
           sample.

  collin   correlation and eigenvalues of the jackknife covariance of the bounded error-side
           moments, and the share of the statistic in each eigen-direction.

Usage: python dual_diag.py quick  <NEW7080.dta>           moments and spread
       python dual_diag.py collin <NEW7080.dta>           collinearity of the bounded moments
       python dual_diag.py scaled <NEW7080.dta> [cores]   writes output/r61/dual_diag_scaled.json
"""
import sys, os, json
import numpy as np
from scipy.stats import chi2
from multiprocessing import get_context
from dual_tendency import COLS, ORDER, cut, comps, fmt
from dual_step import Data, contrib

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
GG = np.linspace(-0.2, 0.6, 41)
NAMES = ['u s^2', 'u s^3', 'u tanh s', 'u phi(s)', 's u^2 (VT)', 's tanh u', 's tanh(u/2)', 's phi(u)',
         's u phi(u)', 's u^3', 's tanh u']
PARITY = ['s', 's', 's', 's', 'even', 'odd', 'odd', 'even', 'odd', 'odd', 'odd']


def load(path):
    import pandas as pd
    d = pd.read_stata(path)
    d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
    cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
    cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
    W = np.column_stack(cols); y = d.v9.values.astype(float); x = d.v4.values.astype(float)
    return y, x, (x == 16).astype(float), W, d.v17.astype(int).values


def scale_by_level(u, lev, L):
    s2 = np.bincount(lev, u ** 2, L) / np.maximum(np.bincount(lev, None, L), 1)
    return u / np.sqrt(s2[lev])


def contrib_scaled(xt, yt, qt, lev, L, b, g):
    u = scale_by_level(yt - b * xt - g * qt, lev, L)
    # contrib() recomputes u from (yt, b, xt, g, qt); pass the scaled error as yt with b = g = 0
    return contrib(xt, u, qt, 0.0, 0.0)


class ScaledData(Data):
    def __init__(self, y, x, q, W, cl):
        super().__init__(y, x, q, W, cl)
        self.lev = np.unique(x, return_inverse=True)[1]; self.L = self.lev.max() + 1

    def cols(self, b, g):
        m = contrib_scaled(self.xt, self.yt, self.qt, self.lev, self.L, b, g).mean(0)
        Mg = np.empty((self.G, len(m)))
        for gi in range(self.G):
            k = self.keep[gi]; A = self.W[k] @ self.d[gi]
            Mg[gi] = contrib_scaled(self.xt[k] + A[:, 1], self.yt[k] + A[:, 0], self.qt[k] + A[:, 2],
                                    self.lev[k], self.L, b, g).mean(0)
        dd = Mg - Mg.mean(0)
        return m, (self.G - 1) / self.G * dd.T @ dd

    def T(self, b, g):
        m, Vj = self.cols(b, g)
        return {nm: float(m[c] @ np.linalg.solve(Vj[np.ix_(c, c)], m[c])) for nm, c in COLS.items()}


def unscaled_cols(d, b, g):
    m = contrib(d.xt, d.yt, d.qt, b, g).mean(0)
    Mg = np.empty((d.G, len(m)))
    for gi in range(d.G):
        k = d.keep[gi]; A = d.W[k] @ d.d[gi]
        Mg[gi] = contrib(d.xt[k] + A[:, 1], d.yt[k] + A[:, 0], d.qt[k] + A[:, 2], b, g).mean(0)
    dd = Mg - Mg.mean(0)
    return m, (d.G - 1) / d.G * dd.T @ dd


def quick(path):
    y, x, q, W, cl = load(path)
    d = Data(y, x, q, W, cl)
    out = {}
    for lab, (b, g) in [('step fit (mean-tendency minimum)', (0.058, 0.18)),
                        ('bounded error-side minimum', (0.060, 0.10)), ('no step, VT root', (0.0599, 0.0))]:
        m, V = unscaled_cols(d, b, g)
        t = m / np.sqrt(np.diag(V))
        print('\n%s: beta %.4f, gamma %.2f' % (lab, b, g))
        for j in range(10):
            print('   %-14s %-5s  mean %+.5f  t %+7.2f' % (NAMES[j], PARITY[j], m[j], t[j]))
        out[lab] = dict(beta=b, gamma=g, mean=m[:10].tolist(), t=t[:10].tolist())
    # spread of the error by years of schooling at the step fit
    u = d.yt - 0.058 * d.xt - 0.18 * d.qt
    lv, inv = np.unique(x, return_inverse=True)
    n = np.bincount(inv); v = np.bincount(inv, u ** 2) / n
    print('\nmean of u^2 by years of schooling (step fit); overall %.4f' % np.mean(u ** 2))
    for a, nn, vv in zip(lv, n, v):
        print('   %4.0f  n %7d  E[u^2] %.4f' % (a, nn, vv))
    ok = n >= 1000
    sl = np.polyfit(lv[ok], np.log(v[ok]), 1, w=np.sqrt(n[ok]))[0]
    print('slope of log E[u^2|x] on x (cells with n >= 1000): %.4f per year' % sl)
    # same within 8..20 years, where most of the sample lies
    ok2 = ok & (lv >= 8)
    sl2 = np.polyfit(lv[ok2], np.log(v[ok2]), 1, w=np.sqrt(n[ok2]))[0]
    print('  within 8 to 20 years: %.4f per year' % sl2)
    out['spread'] = dict(levels=lv.tolist(), n=n.tolist(), Eu2=v.tolist(), slope=sl, slope_8_20=sl2)
    json.dump(out, open(os.path.join(OUT, 'dual_diag_quick.json'), 'w'), indent=1)


_D = {}


def _init(args):
    _D['d'] = ScaledData(*args)


def _prof(b):
    d = _D['d']
    Ts = [d.T(b, g) for g in GG]
    return {nm: (min(t[nm] for t in Ts), float(GG[int(np.argmin([t[nm] for t in Ts]))])) for nm in ORDER}


def scaled(path, cores):
    y, x, q, W, cl = load(path)
    grid = np.linspace(0.046, 0.070, 25)
    with get_context('spawn').Pool(cores, initializer=_init, initargs=((y, x, q, W, cl),)) as pool:
        res = pool.map(_prof, list(grid))
    G = len(np.unique(cl))
    out = dict(grid=grid.tolist(), gamma_grid=GG.tolist(), G=G, sets={})
    print('Angrist-Krueger, degree step, error scaled by sigma(years of schooling)')
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
    # moment t statistics at the scaled minimum of the bounded set
    d = ScaledData(y, x, q, W, cl); r = out['sets']['u_bounded']
    mm, V = d.cols(r['beta'], r['gamma_hat']); t = mm / np.sqrt(np.diag(V))
    print('\nscaled moments at the bounded minimum:')
    for j in range(10):
        print('   %-14s %-5s  t %+7.2f' % (NAMES[j], PARITY[j], t[j]))
    out['t_at_bounded_min'] = t[:10].tolist()
    json.dump(out, open(os.path.join(OUT, 'dual_diag_scaled.json'), 'w'), indent=1)


def collin(path):
    """Correlation of the bounded error-side moments in the jackknife, and the statistic by eigen-direction."""
    y, x, q, W, cl = load(path)
    d = Data(y, x, q, W, cl); c = COLS['u_bounded']; out = {}
    for lab, b, g in [('no step, VT root', 0.0599, 0.0), ('no step, bounded min', 0.063, 0.0), ('step fit', 0.058, 0.18)]:
        m, V = unscaled_cols(d, b, g)
        mc, Vc = m[c], V[np.ix_(c, c)]; sd = np.sqrt(np.diag(Vc)); C = Vc / np.outer(sd, sd)
        w, U = np.linalg.eigh(C); z = mc / sd; comp = (U.T @ z) ** 2 / w
        T = float(mc @ np.linalg.solve(Vc, mc))
        print('\n%s (beta %.4f, gamma %.2f): joint T %.1f, t %s' % (lab, b, g, T, np.round(z, 2)))
        print('  correlation\n', np.round(C, 4))
        print('  eigenvalues', w, ' condition %.0f' % (w[-1] / w[0]))
        print('  T by eigen-direction (small to large eigenvalue)', np.round(comp, 1))
        out[lab] = dict(T=T, t=z.tolist(), corr=C.tolist(), eig=w.tolist(), T_by_dir=comp.tolist())
    json.dump(out, open(os.path.join(OUT, 'dual_diag_collin.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'quick':
        quick(sys.argv[2])
    elif sys.argv[1] == 'collin':
        collin(sys.argv[2])
    else:
        scaled(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
