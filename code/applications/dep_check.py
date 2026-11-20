"""Cross-observation moment E[s_i u_j] = 0 for two observations of the same group
(port of dep_check.R). Part 1: simulation of three cases (beta = 1). Part 2:
applications in which the regressor varies within groups, with a cluster
bootstrap (percentile interval). Usage: python dep_check.py"""
import numpy as np
import _setup
from iiv.data import load
from iiv.core import Context, f3


def dep_fit(xw, rw, g):
    _, gi = np.unique(g, return_inverse=True)
    X = np.bincount(gi, xw); R = np.bincount(gi, rw); ng = np.bincount(gi); npairs = np.sum(ng * (ng - 1))
    A = np.sum(X * R) - np.sum(xw * rw); Bx = np.sum(X ** 2) - np.sum(xw ** 2); Br = np.sum(R ** 2) - np.sum(rw ** 2)
    return dict(theta=0.5 * np.arctan(2 * A / (Bx - Br)), rho_x=Bx / npairs, rho_r=Br / npairs, rho_xr=A / npairs)


rng = np.random.default_rng(3)


def sim(model, G=4000, m=4):
    g = np.repeat(np.arange(G), m); n = G * m
    s = np.sqrt(0.6) * rng.standard_normal(G)[g] + np.sqrt(0.4) * rng.standard_normal(n)
    a = np.sqrt(0.2) * rng.standard_normal(G)[g]; u = a + np.sqrt(0.8) * rng.standard_normal(n)
    x = {'whole': lambda: s + 0.5 * u, 'shared': lambda: s + 0.5 * a,
         'level': lambda: (rng.standard_normal(G) + 0.5 * a[::m] / np.sqrt(0.2))[g]}[model]()
    y = x + u; xt = x - x.mean(); yt = y - y.mean(); b = np.sum(xt * yt) / np.sum(xt ** 2); r = yt - b * xt
    sx, sr = np.sqrt(np.mean(xt ** 2)), np.sqrt(np.mean(r ** 2)); f = dep_fit(xt / sx, r / sr, g)
    return b, b + np.tan(f['theta']) * sr / sx


print('Simulation, true beta = 1 (medians over 200 samples):')
lab = {'whole': 'x loads on the whole error', 'shared': 'x loads on the shared part only', 'level': 'x constant within group'}
for md in ('whole', 'shared', 'level'):
    M = np.array([sim(md) for _ in range(200)])
    print('  %-34s OLS %.3f   cross-observation moment %.3f' % (lab[md], np.median(M[:, 0]), np.median(M[:, 1])))


def run(ds, unit=None, B=999):
    D = load(ds); C = Context(D); n = C.n
    g = C.cl if unit is None else D['dd'][unit].astype(str).values

    def est(id_, gid):
        Q = np.linalg.qr(C.W1[id_])[0]; xt = C.x[id_] - Q @ (Q.T @ C.x[id_]); yt = C.y[id_] - Q @ (Q.T @ C.y[id_])
        b = np.sum(xt * yt) / np.sum(xt ** 2); r = yt - b * xt; sx, sr = np.sqrt(np.mean(xt ** 2)), np.sqrt(np.mean(r ** 2))
        f = dep_fit(xt / sx, r / sr, gid); f['beta'] = b + np.tan(f['theta']) * sr / sx
        return f
    f0 = est(np.arange(n), g); idx = C._loo_setup()['groups']; rb = np.random.default_rng(77); bs = []
    for _ in range(B):
        pick = rb.integers(0, C.G, C.G); id_ = np.concatenate([idx[p] for p in pick])
        copy = np.repeat(np.arange(C.G), [len(idx[p]) for p in pick])
        bs.append(est(id_, np.char.add(np.char.add(copy.astype(str), ' '), g[id_].astype(str)))['beta'])
    P = C.prep(); b_iv = C.tsls(P)[0]; ci = np.nanquantile(bs, [0.025, 0.975])
    print('%-8s pairs within %-7s | within-group corr: x %.3f, OLS residual %.3f | OLS %s, 2SLS %s | estimate %s, '
          '95%% bootstrap [%s, %s]' % (ds, 'cluster' if unit is None else unit, f0['rho_x'], f0['rho_r'], f3(P.b), f3(b_iv),
                                       f3(f0['beta']), f3(ci[0]), f3(ci[1])))


print('\nApplications:')
for d in ('maim5v', 'maim4v', 'banerjee', 'adh'):
    run(d)
run('adh', 'czone')
