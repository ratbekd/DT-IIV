"""Information of u-side tendencies E[s a(u)] = 0 (dual of the paper's mean tendencies).

Under E[s|u] = 0 and var(s|u) = 1, the corrected contribution of E[s a(u)] is
s {a(u) - E a - E[u a(u)] u}, its slope at the truth is D(a) = E[a(u)(u - psi_u(u))], and
    I(a) = D(a)^2 / var(a - E a - E[u a] u) <= J(f_u) - 1,
attained by a = psi_u, the location score of u. The VT moment (a = u^2) has
    I_VT = mu3^2 / (mu4 - 1 - mu3^2).
For comparison the s-side bound of Proposition 5.1 under CH is 1 - 1/J(f_s).

Part 1: quadrature for standardized error distributions: J(f_u) - 1, I_VT, I of the u-side
         bounded basis (tanh u, tanh(u/2), phi(u), u phi(u)) and of the sieve basis
         (u^2, u^3, tanh u, phi(u), u phi(u)), by numerical integration.
Part 2: Monte Carlo check that n var(angle estimate) from the one-moment u-side test with
         a = psi_u matches 1 / (J - 1).
Part 3 (optional, path to NEW7080.dta): Angrist-Krueger residual at the VT root, kernel
         estimate of J(f_u) - 1 and the basis informations, against I_VT.

Usage: python dual_info.py [NEW7080.dta]       writes output/r61/dual_info.json
"""
import sys, os, json
import numpy as np
from scipy import stats, integrate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
phi = lambda u: np.exp(-u ** 2 / 2)
BASES = {'bounded': lambda u: np.column_stack([np.tanh(u), np.tanh(u / 2), phi(u), u * phi(u)]),
         'sieve': lambda u: np.column_stack([u ** 2, u ** 3, np.tanh(u), phi(u), u * phi(u)]),
         'VT': lambda u: (u ** 2)[:, None]}


def std_dist(d):
    m, v = d.mean(), d.var()
    return m, np.sqrt(v)


def quad(dist, lo, hi, npt=400001):
    """Grid quadrature on the standardized density g(u) = sd f(m + sd u)."""
    m, sd = std_dist(dist)
    u = np.linspace(lo, hi, npt)
    g = sd * dist.pdf(m + sd * u)
    dg = np.gradient(g, u)
    w = np.gradient(u)
    ok = g > 1e-300
    psi = np.where(ok, -dg / np.where(ok, g, 1), 0.0)
    E = lambda f: float(np.sum(f * g * w))
    J = E(psi ** 2)
    mu3, mu4 = E(u ** 3), E(u ** 4)
    out = dict(J_minus_1=J - 1, I_VT=mu3 ** 2 / (mu4 - 1 - mu3 ** 2), mu3=mu3, mu4=mu4)
    for nm in ('bounded', 'sieve'):
        B = BASES[nm](u)
        if nm == 'sieve' and not np.isfinite(mu4 * 0 + E(u ** 6)):
            continue
        Bc = B - (B * (g * w)[:, None]).sum(0)
        c = (Bc * (u * g * w)[:, None]).sum(0)                    # E[u B]
        R = Bc - np.outer(u, c)                                     # B - EB - E[uB] u
        D = (B * ((u - psi) * g * w)[:, None]).sum(0)
        V = R.T @ (R * (g * w)[:, None])
        out['I_' + nm] = float(D @ np.linalg.solve(V, D))
    return out


def part1():
    cases = [('normal', stats.norm(), -12, 12),
             ('t(5)', stats.t(5), -60, 60), ('t(10)', stats.t(10), -60, 60),
             ('Laplace', stats.laplace(), -40, 40), ('logistic', stats.logistic(), -40, 40),
             ('skew-normal(4)', stats.skewnorm(4), -10, 12), ('skew-normal(-4)', stats.skewnorm(-4), -12, 10),
             ('gamma(5)', stats.gamma(5), -2.2, 20), ('gamma(10)', stats.gamma(10), -3.1, 25),
             ('lognormal(0.25)', stats.lognorm(0.25), -3.8, 25), ('lognormal(0.5)', stats.lognorm(0.5), -2.5, 40),
             ('Gumbel (left)', stats.gumbel_l(), -40, 8)]
    res = {}
    print('%-17s %8s %8s | %9s %8s %8s %8s' % ('error u', 'skew', 'kurt', 'J(f_u)-1', 'I_VT', 'I_bnd', 'I_sieve'))
    for nm, d, lo, hi in cases:
        r = quad(d, lo, hi); res[nm] = r
        print('%-17s %8.3f %8.3f | %9.4f %8.4f %8.4f %8.4f' % (nm, r['mu3'], r['mu4'], r['J_minus_1'], r['I_VT'],
                                                           r.get('I_bounded', np.nan), r.get('I_sieve', np.nan)), flush=True)
    return res


def part2(reps=400, n=4000):
    """Angle estimate from the one-moment tendency with a = psi_u - u, s Gaussian, u logistic.
    (Rotating the population-orthogonal pair, a = psi_u alone would give n var = J/(J-1)^2.)"""
    rng = np.random.default_rng(20261116)
    sc = np.sqrt(3) / np.pi                                         # standardized logistic scale
    psi = lambda u: np.tanh(u / (2 * sc)) / sc - u                  # its score minus u: the sample-whitened
                                                                    # construction removes the u-component
    J = np.pi ** 2 / 9
    th = []
    for _ in range(reps):
        s0 = rng.standard_normal(n); u0 = rng.logistic(0, sc, n)
        th0 = 0.3
        x = np.cos(th0) * s0 - np.sin(th0) * u0; r = np.sin(th0) * s0 + np.cos(th0) * u0
        f = lambda t: np.mean((np.cos(t) * x + np.sin(t) * r) * (lambda u: psi(u / u.std()))(np.cos(t) * r - np.sin(t) * x))
        g = np.linspace(th0 - 0.4, th0 + 0.4, 81); v = np.array([f(t) for t in g])
        k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        if len(k):
            j = k[np.argmin(np.abs(g[k] - th0))]
            a, b = g[j], g[j + 1]
            for _ in range(40):
                mid = (a + b) / 2
                if np.sign(f(mid)) == np.sign(f(a)): a = mid
                else: b = mid
            th.append((a + b) / 2 - th0)
    th = np.array(th)
    out = dict(reps=len(th), n=n, n_var=float(n * th.var()), bound=float(1 / (J - 1)), bias=float(th.mean()))
    print('Monte Carlo, s Gaussian, u logistic, a = psi_u - u, n %d: n var(angle) %.2f against 1/(J-1) = %.2f (bias %.4f, %d reps)'
          % (n, out['n_var'], out['bound'], out['bias'], len(th)), flush=True)
    return out


def part3(path):
    import pandas as pd
    d = pd.read_stata(path)
    d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
    cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
    cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
    W = np.column_stack(cols); y = d.v9.values.astype(float); x = d.v4.values.astype(float)
    xt = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]; yt = y - W @ np.linalg.lstsq(W, y, rcond=None)[0]
    u = yt - 0.0599 * xt; u = (u - u.mean()) / u.std()                 # residual at the VT root
    mu3, mu4 = np.mean(u ** 3), np.mean(u ** 4)
    out = dict(mu3=float(mu3), mu4=float(mu4), I_VT=float(mu3 ** 2 / (mu4 - 1 - mu3 ** 2)), kde={})
    grid = np.linspace(-9, 6, 30001)
    for bwf in (0.5, 1.0, 2.0):
        h = bwf * 1.06 * len(u) ** (-0.2)
        sub = np.random.default_rng(1).choice(u, 40000, replace=False)
        f = np.zeros_like(grid); df = np.zeros_like(grid)
        for c in np.array_split(sub, 40):
            z = (grid[:, None] - c[None, :]) / h
            k = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
            f += k.sum(1); df += (-z * k).sum(1) / h
        f /= len(sub) * h; df /= len(sub) * h
        psi_g = -df / np.maximum(f, 1e-300)
        psi = np.interp(u, grid, psi_g)
        J = float(np.mean(psi ** 2))
        res = dict(h=h, J_minus_1=J - 1)
        for nm in ('bounded', 'sieve'):
            B = BASES[nm](u); Bc = B - B.mean(0); c = (Bc * u[:, None]).mean(0); R = Bc - np.outer(u, c)
            D = (B * (u - psi)[:, None]).mean(0); V = R.T @ R / len(u)
            res['I_' + nm] = float(D @ np.linalg.solve(V, D))
        out['kde']['%.1f' % bwf] = res
    print('Angrist-Krueger residual at the VT root: skew %.3f kurt %.3f, I_VT %.4f' % (mu3, mu4, out['I_VT']))
    for k_, r in out['kde'].items():
        print('  bandwidth x%s (h %.3f): J(f_u)-1 %.4f, I_bounded %.4f, I_sieve %.4f' % (k_, r['h'], r['J_minus_1'], r['I_bounded'], r['I_sieve']))
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'mc':
        part2(); sys.exit()
    res = dict(quadrature=part1(), montecarlo=part2())
    if len(sys.argv) > 1:
        res['ak'] = part3(sys.argv[1])
    json.dump(res, open(os.path.join(OUT, 'dual_info.json'), 'w'), indent=1)
