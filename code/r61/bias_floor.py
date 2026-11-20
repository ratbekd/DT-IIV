"""Check of the bias floor under approximate mean independence.

Claim (conditional homoscedasticity, one tendency E[u h(s)] = 0, or any GMM combination
of several): if E[u0 | s0] = m0(s0) with ||m0||_2 = eps, the pseudo-true angle is
displaced by, to first order,
      theta* - theta0 = - E[m0 H] / D(h),     H = h - E h - E[s0 h] s0,
so the worst case over departures of norm eps is  eps ||H|| / |D(h)|, and
      eps ||H|| / |D(h)|  >=  eps / sqrt(J - 1)  =  eps sqrt((1 - I*) / I*),
with J the Fisher information for location of s0 and I* = 1 - 1/J the bound of
Proposition 5.6. Equality holds for h equal to the location score of s0.

Part A  population check by quadrature: for several laws of s0 and functions h, the
        exact root of the moment under the worst-case departure against the
        first-order prediction and the floor.
Part B  the same in the estimation pipeline (controls, first step, rotation), in
        units of the slope.
Part C  the applications: sensitivity D'Q^{-1}D of the bounded pair at the OLS slope,
        the implied movement of the slope per unit of tolerance, and the observed
        movement of the endpoints of the sensitivity sets.

Usage: python bias_floor.py      writes output/r61/bias_floor.json
"""
import json, os
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import r61lib as L
from iiv.data import load
from iiv.core import Context

# ---------------------------------------------------------------------- quadrature
_x, _w = np.polynomial.legendre.leggauss(3000)
P_GRID, W_S = (_x + 1) / 2, _w / 2
_z, _v = np.polynomial.hermite_e.hermegauss(80)
E_GRID, W_E = _z, _v / np.sqrt(2 * np.pi)
A_LOG = np.sqrt(3) / np.pi

LAWS = {
    't5': dict(q=lambda p: stats.t.ppf(p, 5) / np.sqrt(5 / 3), psi=lambda s: 6 * s / (3 + s ** 2), J=1.25),
    'logistic': dict(q=lambda p: A_LOG * np.log(p / (1 - p)), psi=lambda s: np.tanh(s / (2 * A_LOG)) / A_LOG,
                     J=np.pi ** 2 / 9),
    'gamma6': dict(q=lambda p: (stats.gamma.ppf(p, 6) - 6) / np.sqrt(6), psi=None, J=1.5),
}
FUN = {
    'tanh': lambda s: np.tanh(s),
    'gauss': lambda s: np.exp(-s ** 2 / 2),
    's2': lambda s: s ** 2,
    's3': lambda s: s ** 3,
}


def Es(vals):
    """E over s0 of a function evaluated on the s0 grid."""
    return float(np.sum(W_S * vals))


def moment(phi, s0, h, eps=0.0, Hn=None):
    """M(phi) = E[u h(s)] for the rotation of (s0, u0) by phi,
    u0 = eps Hn(s0) + sqrt(1 - eps^2) e, e ~ N(0, 1) independent of s0."""
    u0 = np.sqrt(1 - eps ** 2) * E_GRID[None, :]
    if eps:
        u0 = u0 + eps * Hn[:, None]
    else:
        u0 = np.broadcast_to(u0, (len(s0), len(E_GRID)))
    s = np.cos(phi) * s0[:, None] + np.sin(phi) * u0
    u = np.cos(phi) * u0 - np.sin(phi) * s0[:, None]
    return float(W_S @ (u * h(s)) @ W_E)


def analyse(s0, h, eps_list=(0.01, 0.05)):
    hv = h(s0)
    a, c = Es(hv), Es(s0 * hv)
    H = hv - a - c * s0
    nH = np.sqrt(Es(H ** 2))
    d = 1e-4
    D = (moment(d, s0, h) - moment(-d, s0, h)) / (2 * d)
    out = dict(normH=nH, D=D, predicted=nH / abs(D), exact={})
    grid = np.linspace(-0.7, 0.7, 141)
    for eps in eps_list:
        f = lambda ph: moment(ph, s0, h, eps, H / nH)
        v = np.array([f(g) for g in grid])
        idx = np.where(np.diff(np.sign(v)) != 0)[0]
        if len(idx) == 0:
            out['exact']['%g' % eps] = None
            continue
        roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-10) for i in idx]
        r = min(roots, key=abs)
        out['exact']['%g' % eps] = abs(r) / eps
    return out, H, D


def best_combo(s0, names):
    """Combination c'h of a basis that minimizes the worst-case bias: c = Q^{-1} D."""
    Hs, Ds = [], []
    for nm in names:
        _, H, D = analyse(s0, FUN[nm], eps_list=())
        Hs.append(H); Ds.append(D)
    Hm = np.column_stack(Hs); Dv = np.array(Ds)
    Q = (Hm * W_S[:, None]).T @ Hm
    cvec = np.linalg.solve(Q, Dv)
    h = lambda s: sum(cj * FUN[nm](s) for cj, nm in zip(cvec, names))
    return h, float(1 / np.sqrt(Dv @ cvec))


res = dict(A=[], B=[], C=[])
print('PART A. Worst-case displacement of the angle per unit of the norm eps (population, quadrature)')
print('%-9s %-22s %8s %8s | %9s %9s %9s | %7s' % ('law', 'tendency h', '||H||', 'D(h)', 'predicted', 'eps=0.01', 'eps=0.05', 'floor'))
for lw, spec in LAWS.items():
    s0 = spec['q'](P_GRID)
    J = spec['J']; floor = 1 / np.sqrt(J - 1)
    chk = 'E s0^2 = %.4f' % Es(s0 ** 2)
    todo = []
    if spec['psi'] is not None:
        psi = spec['psi']
        chk += ', E psi^2 = %.4f (theory %.4f), E[s0 psi] = %.4f' % (Es(psi(s0) ** 2), J, Es(s0 * psi(s0)))
        todo.append(('location score', psi))
    todo.append(('tanh s', FUN['tanh']))
    if lw == 'gamma6':
        todo.append(('exp(-s^2/2)', FUN['gauss']))
    hb, _ = best_combo(s0, ['tanh', 'gauss'])
    todo.append(('bounded pair, best', hb))
    if lw != 't5':                      # E s0^6 is infinite for t5, so ||H|| is for s^3
        if lw == 'gamma6':
            todo.append(('s^2', FUN['s2']))
        todo.append(('s^3', FUN['s3']))
        hp, _ = best_combo(s0, ['s2', 's3'])
        todo.append(('powers, best', hp))
    print('%s: J = %.4f, I* = %.4f, floor = %.4f   [%s]' % (lw, J, 1 - 1 / J, floor, chk))
    for nm, h in todo:
        o, H, D = analyse(s0, h)
        extra = ''
        if spec['psi'] is not None:
            lam = spec['psi'](s0) - s0
            extra = '  E[lambda H] = %.4f' % Es(lam * H)
        e1, e5 = o['exact'].get('0.01'), o['exact'].get('0.05')
        print('%-9s %-22s %8.4f %8.4f | %9.4f %9s %9s | %7.4f%s' % (
            '', nm, o['normH'], o['D'], o['predicted'], '%.4f' % e1 if e1 else 'none', '%.4f' % e5 if e5 else 'none',
            floor, extra))
        res['A'].append(dict(law=lw, h=nm, normH=o['normH'], D=o['D'], predicted=o['predicted'],
                             exact_001=e1, exact_005=e5, floor=floor, J=J))

# ---------------------------------------------------------------------- Part B
print('\nPART B. Estimation pipeline: t5 exogenous component, location-score tendency, worst-case departure')
psi = LAWS['t5']['psi']
BETA0, TH0 = 1.0, -np.arctan(0.5)            # x = s0 + 0.5 u0: corr(x, u) = -sin(theta0)
DB_DTH = 1.0                                 # (sigma_r / sigma_x) / cos(theta0)^2 = 0.8 / 0.8
n, R = 5000, 400
rng = np.random.default_rng(20261113)
print('n = %d, %d replications; predicted slope bias = -2 eps (floor 1/sqrt(J-1) = 2, d beta / d theta = 1)' % (n, R))
print('%6s | %10s %10s %10s | %10s' % ('eps', 'mean bias', 'MC s.e.', 'sd', 'predicted'))
for eps in (0.0, 0.02, 0.05):
    est = []
    for r in range(R):
        s0 = rng.standard_t(5, n) / np.sqrt(5 / 3)
        lam = psi(s0) - s0
        u0 = eps * lam / 0.5 + np.sqrt(1 - eps ** 2) * rng.standard_normal(n)     # ||lambda|| = sqrt(J - 1) = 0.5
        w = rng.standard_normal(n)
        x = 2 + 0.5 * w + s0 + 0.5 * u0
        y = 1 + BETA0 * x + 0.3 * w + u0
        C = L.make_context(y, x, np.column_stack([np.ones(n), w])); P = C.prep()

        def g(th):
            s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
            return np.mean(u * psi(s))
        grid = np.linspace(-np.pi / 4, np.pi / 4, 161)
        v = np.array([g(t) for t in grid]); idx = np.where(np.diff(np.sign(v)) != 0)[0]
        if len(idx) == 0:
            continue
        th0_hat = C.theta_of(BETA0, P)
        roots = [brentq(g, grid[i], grid[i + 1]) for i in idx]
        est.append(C.beta_at(min(roots, key=lambda t: abs(t - th0_hat)), P) - BETA0)
    est = np.array(est)
    print('%6.2f | %10.4f %10.4f %10.4f | %10.4f' % (eps, est.mean(), est.std(ddof=1) / np.sqrt(len(est)),
                                                    est.std(ddof=1), -2 * eps * DB_DTH))
    res['B'].append(dict(eps=eps, mean_bias=float(est.mean()), se=float(est.std(ddof=1) / np.sqrt(len(est))),
                         sd=float(est.std(ddof=1)), predicted=-2 * eps * DB_DTH, n_roots=int(len(est))))

# ---------------------------------------------------------------------- Part C
print('\nPART C. Applications: sensitivity of the bounded pair at the OLS slope and movement of the slope per unit of tolerance')
print('%-9s %7s %9s %12s | %s' % ('sample', 'sr/sx', "D'Q^-1D", 'd beta/d eps', 'observed movement of interior endpoints of C_n(eps), per unit eps'))
for ds in ['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']:
    C = Context(load(ds)); P = C.prep()
    d = 1e-3
    t0, _ = L.ami_tV(C, P, P.b)
    tp, _ = L.ami_tV(C, P, C.beta_at(d, P)); tm, _ = L.ami_tV(C, P, C.beta_at(-d, P))
    D = (tp[:2] - tm[:2]) / (2 * d)
    Q = np.array([[t0[2], t0[3]], [t0[3], t0[4]]])
    S2 = float(D @ np.linalg.solve(Q, D))
    rate = float(P.sr / P.sx / np.sqrt(S2))
    obs = []
    pth = L.out('ami_apps_%s.json' % ds)
    if os.path.exists(pth):
        J = json.load(open(pth))['joint']
        for e1, e2 in ((0.025, 0.05), (0.05, 0.1), (0.1, 0.2)):
            a, b = J['%g' % e1], J['%g' % e2]
            if len(a) == 1 and len(b) == 1 and not a[0][2] and not b[0][2]:
                obs.append('eps %g-%g: lower %.2f, upper %.2f' % (e1, e2, (a[0][0] - b[0][0]) / (e2 - e1),
                                                                 (b[0][1] - a[0][1]) / (e2 - e1)))
    print('%-9s %7.3f %9.3f %12.2f | %s' % (ds, P.sr / P.sx, S2, rate, '; '.join(obs) or 'no interior single-interval pair'))
    res['C'].append(dict(dataset=ds, ratio=float(P.sr / P.sx), S2=S2, rate=rate, observed=obs))
json.dump(res, open(L.out('bias_floor.json'), 'w'), indent=1)
