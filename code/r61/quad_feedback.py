"""Mean tendencies under nonlinear feedback from the error to the regressor.

Model, after partialling out the controls and standardizing:
    y = beta x + u,     x_w = v + lambda u_w + gamma (u_w^2 - 1),     E[u_w | v] = 0.
gamma = 0 is the linear model of the paper. At a tested slope b and a value g of gamma,
    u_w(b) = standardized y - b x,
    v(b, g) = x_w - lambda u_w - g (u_w^2 - 1),  lambda = mean(x_w u_w) - g mean(u_w^3),
    s(b, g) = standardized v(b, g),
and the moments are the mean tendencies mean(u_w h(s)) for h = (s^2, s^3, tanh s,
exp(-s^2/2)). Their covariance is a delete-cluster jackknife that refits the projections
on the controls, the scales and lambda after deleting each cluster, at fixed (b, g).

Reported for each b: T(b, 0), the linear model; and min over g of T(b, g). The slope
enters the projection set if min_g T(b, g) is below the chi2_4 cutoff (valid whatever the
strength of identification of gamma) and the profiled set if it is below chi2_3 (needs
gamma strongly identified). Hotelling F cutoffs for G clusters are reported beside them.
The joint minimum gives (b, g) estimates and an overidentification statistic, chi2_2.

Usage:
  python quad_feedback.py check [reps]            synthetic check of the construction
  python quad_feedback.py ak <NEW7080.dta> [cores] Angrist-Krueger, men born 1930-39
"""
import sys, os, json
import numpy as np
from scipy.stats import chi2, f as fdist
from scipy.optimize import minimize_scalar, minimize
from multiprocessing import get_context

M = 4
GBOUND = 0.8
GGRID = np.linspace(-GBOUND, GBOUND, 17)


def basis(s):
    return np.column_stack([s ** 2, s ** 3, np.tanh(s), np.exp(-s ** 2 / 2)])


def mom(xt, yt, b, g):
    u = yt - b * xt
    u = u - u.mean(); uw = u / np.sqrt(np.mean(u ** 2))
    xw = xt / np.sqrt(np.mean(xt ** 2))
    lam = np.mean(xw * uw) - g * np.mean(uw ** 3)
    v = xw - lam * uw - g * (uw ** 2 - 1)
    v = v - v.mean(); s = v / np.sqrt(np.mean(v ** 2))
    return (uw[:, None] * basis(s)).mean(0)


class Data:
    """Residualized data and the delete-cluster refits of the projections."""

    def __init__(self, y, x, W, cl):
        self.y, self.x, self.W = y, x, W
        codes = np.unique(cl, return_inverse=True)[1]
        self.G = codes.max() + 1
        WW = W.T @ W; Wy = W.T @ y; Wx = W.T @ x
        pi_y = np.linalg.solve(WW, Wy); pi_x = np.linalg.solve(WW, Wx)
        self.xt = x - W @ pi_x; self.yt = y - W @ pi_y
        self.keep, self.dx, self.dy = [], [], []
        for gi in range(self.G):
            ix = codes == gi
            Wg = W[ix]
            WWg = WW - Wg.T @ Wg
            px = np.linalg.solve(WWg, Wx - Wg.T @ x[ix]); py = np.linalg.solve(WWg, Wy - Wg.T @ y[ix])
            self.keep.append(~ix); self.dx.append(pi_x - px); self.dy.append(pi_y - py)

    def T(self, b, g):
        m = mom(self.xt, self.yt, b, g)
        Mg = np.empty((self.G, M))
        for gi in range(self.G):
            k = self.keep[gi]; Wk = self.W[k]
            Mg[gi] = mom(self.xt[k] + Wk @ self.dx[gi], self.yt[k] + Wk @ self.dy[gi], b, g)
        d = Mg - Mg.mean(0)
        V = (self.G - 1) / self.G * d.T @ d
        return float(m @ np.linalg.solve(V, m))

    def profile(self, b):
        tg = [self.T(b, g) for g in GGRID]
        i = int(np.argmin(tg))
        lo, hi = GGRID[max(0, i - 1)], GGRID[min(len(GGRID) - 1, i + 1)]
        r = minimize_scalar(lambda g: self.T(b, g), bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
        return float(r.fun), float(r.x), self.T(b, 0.0)


_D = {}


def _init(args):
    _D['d'] = Data(*args)


def _prof(b):
    return _D['d'].profile(b)


def cuts(G):
    out = dict(chi_proj=chi2.ppf(0.95, M), chi_prof=chi2.ppf(0.95, M - 1))
    out['F_proj'] = M * (G - 1) / (G - M) * fdist.ppf(0.95, M, G - M)
    out['F_prof'] = (M - 1) * (G - 1) / (G - M + 1) * fdist.ppf(0.95, M - 1, G - M + 1)
    return out


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


def run(y, x, W, cl, grid, cores, label):
    ctx = get_context('spawn')
    with ctx.Pool(cores, initializer=_init, initargs=((y, x, W, cl),)) as pool:
        res = pool.map(_prof, list(grid))
    d = Data(y, x, W, cl)
    Tmin = np.array([r[0] for r in res]); gam = np.array([r[1] for r in res]); T0 = np.array([r[2] for r in res])
    c = cuts(d.G)
    i = int(np.argmin(Tmin))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    rb = minimize_scalar(lambda b: d.profile(b)[0], bounds=(lo, hi), method='bounded', options={'xatol': 1e-5})
    Jq, gq, _ = d.profile(rb.x)
    i0 = int(np.argmin(T0))
    out = dict(label=label, n=len(y), G=int(d.G), grid=grid.tolist(), Tmin=Tmin.tolist(), gamma=gam.tolist(), T0=T0.tolist(),
               cuts=c, beta_quad=float(rb.x), gamma_quad=gq, J_quad=Jq, p_quad=float(chi2.sf(Jq, M - 2)),
               beta_lin=float(grid[i0]), J_lin=float(T0[i0]), p_lin=float(chi2.sf(T0[i0], M - 1)))
    for nm, T, key in (('linear (gamma = 0)', T0, 'lin'), ('quadratic, projection', Tmin, 'proj'), ('quadratic, profiled', Tmin, 'prof')):
        cc = c['chi_proj'] if key in ('lin', 'proj') else c['chi_prof']
        cf = c['F_proj'] if key in ('lin', 'proj') else c['F_prof']
        out['set_' + key] = dict(chi=comps(T <= cc, grid), F=comps(T <= cf, grid))
        print('  %-22s chi2: %-30s F: %s' % (nm, fmt(out['set_' + key]['chi']), fmt(out['set_' + key]['F'])), flush=True)
    print('  linear: min T %.2f at beta %.4f (J p, chi2_3, %.4f)' % (out['J_lin'], out['beta_lin'], out['p_lin']))
    print('  quadratic: beta %.4f, gamma %.4f, J %.2f (p, chi2_2, %.4f)' % (out['beta_quad'], gq, Jq, out['p_quad']), flush=True)
    return out


def synth(n, G, gamma0, rng):
    cl = np.repeat(np.arange(G), n // G)
    s0 = (rng.chisquare(4, n) - 4) / np.sqrt(8)
    u0 = -(rng.chisquare(3, n) - 3) / np.sqrt(6)            # left-skewed error, like log wages
    w = rng.standard_normal(n)
    x = 1 + 0.5 * w + s0 + 0.4 * u0 + gamma0 * (u0 ** 2 - 1)
    y = 2 + 1.0 * x + 0.3 * w + u0
    return y, x, np.column_stack([np.ones(n), w]), cl


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, '..', '..', 'output', 'r61')
    if sys.argv[1] == 'check':
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        rng = np.random.default_rng(20261113)
        res = []
        for gamma0 in (0.0, 0.25):
            for r in range(reps):
                y, x, W, cl = synth(20000, 100, gamma0, rng)
                d = Data(y, x, W, cl)
                Tmin, gh, T0 = d.profile(1.0)
                Tmin_off, _, T0_off = d.profile(0.85)
                print('gamma0 %.2f rep %d | at true beta: T(gamma=0) %7.2f  min_g T %6.2f (gamma-hat %.3f) | at beta 0.85: T(0) %7.2f  min_g T %7.2f'
                      % (gamma0, r, T0, Tmin, gh, T0_off, Tmin_off), flush=True)
                res.append(dict(gamma0=gamma0, T0=T0, Tmin=Tmin, ghat=gh, T0_off=T0_off, Tmin_off=Tmin_off))
        print('cutoffs (G = 100):', {k: round(v, 2) for k, v in cuts(100).items()})
        json.dump(res, open(os.path.join(OUT, 'quad_check.json'), 'w'), indent=1)
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
        print('Angrist-Krueger: n %d, OLS %.4f, slope grid [%.4f, %.4f], 61 points' % (len(y), bols, grid[0], grid[-1]), flush=True)
        out = run(y, x, W, cl, grid, cores, 'Angrist-Krueger 1930-39')
        out['ols'] = float(bols)
        json.dump(out, open(os.path.join(OUT, 'quad_ak.json'), 'w'), indent=1)
