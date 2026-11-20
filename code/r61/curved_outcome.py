"""Mean tendencies with a curved outcome equation.

Model, after partialling out the controls:
    y = beta x + kappa q + u,   q = (x - mean x)^2,   x_w = v + lambda u_w,   E[u_w | v] = 0.
kappa = 0 is the linear model of the paper. At a tested slope b and curvature k,
    u_w(b, k) = standardized y - b x - k q (all residualized on the controls),
    v = x_w - lambda u_w,  lambda = mean(x_w u_w),  s = standardized v,
and the moments are mean(u_w h(s)), h = (s^2, s^3, tanh s, exp(-s^2/2)). Covariance: the
delete-cluster jackknife that refits the projections of y, x and q on the controls.
kappa is searched in standardized units k_std = kappa sd(q)/sd(y), |k_std| <= KBOUND,
and reported in raw units (log points per squared year for Angrist-Krueger).

Sets as in quad_feedback.py: projection (min_k T <= chi2_4), profiled (chi2_3), joint J (chi2_2).

Usage:
  python curved_outcome.py check [reps]
  python curved_outcome.py ak <NEW7080.dta> [cores]       quadratic q = (x - mean x)^2
  python curved_outcome.py akdeg <NEW7080.dta> [cores]    degree step q = 1{educ = 16}
"""
import sys, os, json
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from multiprocessing import get_context
from quad_feedback import basis, cuts, comps, fmt, M

KBOUND = 0.6
KGRID = np.linspace(-KBOUND, KBOUND, 25)


def mom(xt, yt, qt, b, k):
    u = yt - b * xt - k * qt
    u = u - u.mean(); uw = u / np.sqrt(np.mean(u ** 2))
    xw = xt / np.sqrt(np.mean(xt ** 2))
    v = xw - np.mean(xw * uw) * uw
    v = v - v.mean(); s = v / np.sqrt(np.mean(v ** 2))
    return (uw[:, None] * basis(s)).mean(0)


class Data:
    def __init__(self, y, x, W, cl, q=None):
        if q is None:
            q = (x - x.mean()) ** 2
        self.W = W
        codes = np.unique(cl, return_inverse=True)[1]
        self.G = codes.max() + 1
        WW = W.T @ W
        V = np.column_stack([y, x, q]); WV = W.T @ V
        Pi = np.linalg.solve(WW, WV)
        R = V - W @ Pi
        self.yt, self.xt, self.qt = R.T
        self.kscale = np.sqrt(np.mean(self.yt ** 2) / np.mean(self.qt ** 2))   # raw kappa per standardized unit
        self.keep, self.d = [], []
        for gi in range(self.G):
            ix = codes == gi; Wg = W[ix]
            Pg = np.linalg.solve(WW - Wg.T @ Wg, WV - Wg.T @ V[ix])
            self.keep.append(~ix); self.d.append(Pi - Pg)

    def T(self, b, ks):
        k = ks * self.kscale
        m = mom(self.xt, self.yt, self.qt, b, k)
        Mg = np.empty((self.G, M))
        for gi in range(self.G):
            kp = self.keep[gi]; A = self.W[kp] @ self.d[gi]
            Mg[gi] = mom(self.xt[kp] + A[:, 1], self.yt[kp] + A[:, 0], self.qt[kp] + A[:, 2], b, k)
        dd = Mg - Mg.mean(0)
        Vm = (self.G - 1) / self.G * dd.T @ dd
        return float(m @ np.linalg.solve(Vm, m))

    def profile(self, b):
        tg = [self.T(b, k) for k in KGRID]
        i = int(np.argmin(tg))
        lo, hi = KGRID[max(0, i - 1)], KGRID[min(len(KGRID) - 1, i + 1)]
        r = minimize_scalar(lambda k: self.T(b, k), bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
        return float(r.fun), float(r.x) * self.kscale, self.T(b, 0.0)


_D = {}


def _init(args):
    _D['d'] = Data(*args[:4], q=args[4] if len(args) > 4 else None)


def _prof(b):
    return _D['d'].profile(b)


def run(y, x, W, cl, grid, cores, label, q=None):
    with get_context('spawn').Pool(cores, initializer=_init, initargs=((y, x, W, cl, q),)) as pool:
        res = pool.map(_prof, list(grid))
    d = Data(y, x, W, cl, q=q)
    Tmin = np.array([r[0] for r in res]); kap = np.array([r[1] for r in res]); T0 = np.array([r[2] for r in res])
    c = cuts(d.G)
    i = int(np.argmin(Tmin))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    rb = minimize_scalar(lambda b: d.profile(b)[0], bounds=(lo, hi), method='bounded', options={'xatol': 1e-5})
    Jc, kc, _ = d.profile(rb.x)
    i0 = int(np.argmin(T0))
    out = dict(label=label, n=len(y), G=int(d.G), grid=grid.tolist(), Tmin=Tmin.tolist(), kappa=kap.tolist(), T0=T0.tolist(),
               cuts=c, kscale=d.kscale, kbound_raw=KBOUND * d.kscale, beta_curv=float(rb.x), kappa_curv=kc, J_curv=Jc,
               p_curv=float(chi2.sf(Jc, M - 2)), beta_lin=float(grid[i0]), J_lin=float(T0[i0]),
               p_lin=float(chi2.sf(T0[i0], M - 1)))
    for nm, T, key in (('linear (kappa = 0)', T0, 'lin'), ('curved, projection', Tmin, 'proj'), ('curved, profiled', Tmin, 'prof')):
        cc = c['chi_prof'] if key == 'prof' else c['chi_proj']
        cf = c['F_prof'] if key == 'prof' else c['F_proj']
        out['set_' + key] = dict(chi=comps(T <= cc, grid), F=comps(T <= cf, grid))
        print('  %-22s chi2: %-30s F: %s' % (nm, fmt(out['set_' + key]['chi']), fmt(out['set_' + key]['F'])), flush=True)
    print('  linear: min T %.2f at beta %.4f (J p, chi2_3, %.4f)' % (out['J_lin'], out['beta_lin'], out['p_lin']))
    print('  curved: beta %.4f, kappa %.5f (search bound +-%.4f), J %.2f (p, chi2_2, %.4f)' % (
        out['beta_curv'], kc, out['kbound_raw'], Jc, out['p_curv']), flush=True)
    return out


def synth(n, G, kappa0, rng):
    cl = np.repeat(np.arange(G), n // G)
    s0 = (rng.chisquare(4, n) - 4) / np.sqrt(8)
    u0 = -(rng.chisquare(3, n) - 3) / np.sqrt(6)
    w = rng.standard_normal(n)
    x = 1 + 0.5 * w + s0 + 0.4 * u0
    y = 2 + 1.0 * x + kappa0 * (x - x.mean()) ** 2 + 0.3 * w + u0
    return y, x, np.column_stack([np.ones(n), w]), cl


if __name__ == '__main__':
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
    if sys.argv[1] == 'check':
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        rng = np.random.default_rng(20261114)
        res = []
        for kappa0 in (0.0, 0.15):
            for r in range(reps):
                d = Data(*synth(20000, 100, kappa0, rng))
                Tm, kh, T0 = d.profile(1.0)
                Tmo, _, T0o = d.profile(0.85)
                print('kappa0 %.2f rep %d | true beta: T(k=0) %7.2f  min_k T %6.2f (kappa-hat %.3f) | beta 0.85: T(0) %7.2f  min_k T %7.2f'
                      % (kappa0, r, T0, Tm, kh, T0o, Tmo), flush=True)
                res.append(dict(kappa0=kappa0, T0=T0, Tmin=Tm, khat=kh, T0_off=T0o, Tmin_off=Tmo))
        print('cutoffs (G = 100):', {k: round(v, 2) for k, v in cuts(100).items()})
        json.dump(res, open(os.path.join(OUT, 'curved_check.json'), 'w'), indent=1)
    else:
        import pandas as pd
        cores = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        d = pd.read_stata(sys.argv[2])
        d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
        cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
        cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
        W = np.column_stack(cols); y = d.v9.values.astype(float); x = d.v4.values.astype(float)
        cl = d.v17.astype(int).values
        xt = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]; yt = y - W @ np.linalg.lstsq(W, y, rcond=None)[0]
        bols = xt @ yt / (xt @ xt); k = np.sqrt(np.mean((yt - bols * xt) ** 2) / np.mean(xt ** 2))
        grid = bols + np.tan(np.linspace(-np.pi / 4, np.pi / 4, 61)) * k
        if sys.argv[1] == 'akdeg':                         # degree step 1{educ = 16}, narrow slope grid
            q = (x == 16).astype(float)
            cq = np.linalg.lstsq(np.column_stack([W, x, q]), y, rcond=None)[0]
            print('Angrist-Krueger, degree term 1{educ = 16}: OLS beta %.4f, gamma %.4f' % (cq[-2], cq[-1]), flush=True)
            out = run(y, x, W, cl, np.linspace(0.048, 0.068, 21), cores, 'Angrist-Krueger, degree step', q=q)
            out['ols_curved'] = [float(cq[-2]), float(cq[-1])]
            json.dump(out, open(os.path.join(OUT, 'curved_ak_deg16.json'), 'w'), indent=1)
            sys.exit()
        q = (x - x.mean()) ** 2
        cq = np.linalg.lstsq(np.column_stack([W, x, q]), y, rcond=None)[0]
        print('Angrist-Krueger: n %d, OLS %.4f; OLS with x^2: beta %.4f, kappa %.5f; slope grid [%.4f, %.4f]' % (
            len(y), bols, cq[-2], cq[-1], grid[0], grid[-1]), flush=True)
        out = run(y, x, W, cl, grid, cores, 'Angrist-Krueger 1930-39, curved outcome')
        out['ols'] = float(bols); out['ols_curved'] = [float(cq[-2]), float(cq[-1])]
        json.dump(out, open(os.path.join(OUT, 'curved_ak.json'), 'w'), indent=1)
