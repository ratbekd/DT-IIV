"""u-side tendencies E[s a(u)] = 0: fixed-slope tests and sets.

At a tested slope b (after partialling out the controls): u_w = standardized y - b x,
lambda = mean(x_w u_w), s = standardized x_w - lambda u_w. The moment columns are
    s-side (mean tendencies, the paper):  u_w * (s^2, s^3, tanh s, exp(-s^2/2))
    u-side bounded:                        s * (tanh u, tanh(u/2), phi(u), u phi(u))
    u-side sieve:                          s * (u^2, u^3, tanh u, phi(u), u phi(u))
    VT:                                    s * u^2
with phi(u) = exp(-u^2/2). Sets: 'VT', 'u_bounded', 'u_sieve', 's_four', and 'both'
(s_four and u_bounded together). The u-side tendencies hold when E[s0 | u0] = 0.
Covariance: the delete-group jackknife that refits the projections on the controls,
lambda and the scales (as in quad_feedback.py). Statistic T(b) is the Wald form, with
chi2_m and Hotelling F cutoffs; min_b T gives a J test on m - 1 degrees of freedom.

Usage:
  python dual_tendency.py check [reps]               synthetic designs
  python dual_tendency.py ak <NEW7080.dta> [cores]    Angrist-Krueger
  python dual_tendency.py apps [cores] [datasets]     the seven applications
"""
import sys, os, json
import numpy as np
from scipy.stats import chi2, f as fdist
from multiprocessing import get_context

phi = lambda u: np.exp(-u ** 2 / 2)
COLS = {'s_four': [0, 1, 2, 3], 'VT': [4], 'u_bounded': [5, 6, 7, 8], 'u_sieve': [4, 9, 10, 7, 8],
        'both': [0, 1, 2, 3, 5, 6, 7, 8]}
#        0..3 s-side; 4 s u^2; 5 s tanh u; 6 s tanh(u/2); 7 s phi(u); 8 s u phi(u); 9 s u^3; 10 = 5 (tanh u)
ORDER = ['VT', 'u_bounded', 'u_sieve', 's_four', 'both']


def contrib(xt, yt, b):
    u = yt - b * xt
    u = u - u.mean(); uw = u / np.sqrt(np.mean(u ** 2))
    xw = xt / np.sqrt(np.mean(xt ** 2))
    v = xw - np.mean(xw * uw) * uw
    v = v - v.mean(); s = v / np.sqrt(np.mean(v ** 2))
    return np.column_stack([uw * s ** 2, uw * s ** 3, uw * np.tanh(s), uw * phi(s),
                            s * uw ** 2, s * np.tanh(uw), s * np.tanh(uw / 2), s * phi(uw), s * uw * phi(uw),
                            s * uw ** 3, s * np.tanh(uw)])


class Data:
    def __init__(self, y, x, W, cl):
        self.W = W
        codes = np.unique(cl, return_inverse=True)[1]
        self.G = codes.max() + 1
        WW = W.T @ W; V = np.column_stack([y, x]); WV = W.T @ V
        Pi = np.linalg.solve(WW, WV); R = V - W @ Pi
        self.yt, self.xt = R.T
        self.keep, self.d = [], []
        for gi in range(self.G):
            ix = codes == gi; Wg = W[ix]
            self.keep.append(~ix); self.d.append(Pi - np.linalg.solve(WW - Wg.T @ Wg, WV - Wg.T @ V[ix]))

    def T(self, b):
        m = contrib(self.xt, self.yt, b).mean(0)
        Mg = np.empty((self.G, len(m)))
        for gi in range(self.G):
            k = self.keep[gi]; A = self.W[k] @ self.d[gi]
            Mg[gi] = contrib(self.xt[k] + A[:, 1], self.yt[k] + A[:, 0], b).mean(0)
        dd = Mg - Mg.mean(0); Vj = (self.G - 1) / self.G * dd.T @ dd
        return {nm: float(m[c] @ np.linalg.solve(Vj[np.ix_(c, c)], m[c])) for nm, c in COLS.items()}


def cut(m, G, kind):
    return chi2.ppf(0.95, m) if kind == 'chi' else m * (G - 1) / (G - m) * fdist.ppf(0.95, m, G - m)


def comps(acc, grid):
    res, i = [], 0
    while i < len(acc):
        if acc[i]:
            j = i
            while j + 1 < len(acc) and acc[j + 1]:
                j += 1
            res.append((float(grid[i]), float(grid[j]), bool(i == 0 or j == len(acc) - 1))); i = j + 1
        else:
            i += 1
    return res


fmt = lambda cs: ' U '.join('[%.4f, %.4f]%s' % (a, b, '+' if e else '') for a, b, e in cs) or 'empty'
_D = {}


def _init(args):
    _D['d'] = Data(*args)


def _T(b):
    return _D['d'].T(b)


def sets(y, x, W, cl, grid, cores, extra=None):
    with get_context('spawn').Pool(cores, initializer=_init, initargs=((y, x, W, cl),)) as pool:
        Ts = pool.map(_T, list(grid))
    G = len(np.unique(cl))
    out = dict(G=G, grid=list(map(float, grid)), res={})
    for nm in ORDER:
        T = np.array([t[nm] for t in Ts]); m = len(COLS[nm]); i = int(np.argmin(T))
        r = dict(m=m, T=T.tolist(), Tmin=float(T[i]), beta_min=float(grid[i]),
                 p_J=float(chi2.sf(T[i], m - 1)) if m > 1 else None,
                 set_chi=comps(T <= cut(m, G, 'chi'), grid), set_F=comps(T <= cut(m, G, 'F'), grid))
        if extra:
            r.update({k: bool(float(np.interp(v, grid, T)) <= cut(m, G, 'F')) for k, v in extra.items()})
        out['res'][nm] = r
        print('  %-10s m %d  min T %8.2f at %8.4f  J p %s  set(F): %s%s' % (
            nm, m, r['Tmin'], r['beta_min'], '  --  ' if m == 1 else '%.4f' % r['p_J'], fmt(r['set_F']),
            ''.join('  %s in: %s' % (k, r[k]) for k in (extra or {}))), flush=True)
    return out


def ols_grid(y, x, W, npts=121):
    xt = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]; yt = y - W @ np.linalg.lstsq(W, y, rcond=None)[0]
    b = xt @ yt / (xt @ xt); k = np.sqrt(np.mean((yt - b * xt) ** 2) / np.mean(xt ** 2))
    return b, b + np.tan(np.linspace(-np.pi / 4, np.pi / 4, npts)) * k


DESIGNS = {  # (name, s0 draw, u0 draw)
    'A: s Gaussian, u lognormal': (lambda r, n: r.standard_normal(n),
                                   lambda r, n: (np.exp(0.5 * r.standard_normal(n)) - np.exp(0.125)) / np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))),
    'B: s Gaussian, u t5': (lambda r, n: r.standard_normal(n), lambda r, n: r.standard_t(5, n) / np.sqrt(5 / 3)),
    'C: s chi2(4), u Gaussian': (lambda r, n: (r.chisquare(4, n) - 4) / np.sqrt(8), lambda r, n: r.standard_normal(n)),
    'D: s chi2(4), u t5': (lambda r, n: (r.chisquare(4, n) - 4) / np.sqrt(8), lambda r, n: r.standard_t(5, n) / np.sqrt(5 / 3)),
}


def synth(n, G, ds, du, rng):
    cl = np.repeat(np.arange(G), n // G)
    s0, u0, w = ds(rng, n), du(rng, n), rng.standard_normal(n)
    x = 1 + 0.5 * w + s0 + 0.4 * u0
    y = 2 + 1.0 * x + 0.3 * w + u0
    return y, x, np.column_stack([np.ones(n), w]), cl


def check(reps):
    """Rejection rates at the true slope (size) and at slopes 0.9 and 1.1 (power), n 5000, G 100."""
    rng = np.random.default_rng(20261116)
    out = {}
    for nm, (ds, du) in DESIGNS.items():
        rej = {k: np.zeros(3) for k in ORDER}
        for _ in range(reps):
            d = Data(*synth(5000, 100, ds, du, rng))
            for j, b in enumerate((1.0, 0.9, 1.1)):
                T = d.T(b)
                for k in ORDER:
                    rej[k][j] += T[k] > cut(len(COLS[k]), 100, 'F')
        out[nm] = {k: (rej[k] / reps).tolist() for k in ORDER}
        print('%s   (rejection: at truth | at 0.9 | at 1.1)' % nm)
        for k in ORDER:
            print('   %-10s %.3f | %.3f | %.3f' % ((k,) + tuple(out[nm][k])), flush=True)
    return out


if __name__ == '__main__':
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
    mode = sys.argv[1]
    if mode == 'check':
        res = check(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
        json.dump(res, open(os.path.join(OUT, 'dual_check.json'), 'w'), indent=1)
    elif mode == 'ak':
        import pandas as pd
        cores = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        d = pd.read_stata(sys.argv[2])
        d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
        cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
        cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
        W = np.column_stack(cols); y = d.v9.values.astype(float); x = d.v4.values.astype(float)
        bols, grid = ols_grid(y, x, W, 241)
        print('Angrist-Krueger: OLS %.4f; 2SLS 0.0990; VT root 0.0599' % bols, flush=True)
        out = sets(y, x, W, d.v17.astype(int).values, grid, cores, extra=dict(OLS=bols, TSLS=0.0990))
        out['ols'] = bols
        json.dump(out, open(os.path.join(OUT, 'dual_ak.json'), 'w'), indent=1)
    elif mode == 'apps':
        from iiv.data import load
        from iiv.core import Context
        cores = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        res = {}
        for ds in (sys.argv[3:] or ['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']):
            D = load(ds); y, x, W = D['y'], D['x'], D['W1']
            cl = (np.unique(D['cl'], return_inverse=True)[1] if D['cl'] is not None
                  else np.random.default_rng(20261114).permutation(np.arange(len(y)) % 50))
            C = Context(D); P = C.prep(); b_iv = float(C.tsls(P)[0])
            bols, grid = ols_grid(y, x, W, 121)
            print('==== %s: n %d, OLS %.4f, external IV %.4f' % (ds, len(y), bols, b_iv), flush=True)
            res[ds] = sets(y, x, W, cl, grid, cores, extra=dict(OLS=bols, IV=b_iv))
            res[ds].update(ols=bols, iv=b_iv)
            json.dump(res, open(os.path.join(OUT, 'dual_apps.json'), 'w'), indent=1)
