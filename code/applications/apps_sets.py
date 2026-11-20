"""Confidence-set components, VT roots, standardized slopes L_j and root-screen
p-values (port of apps_sets_r47.R). Set endpoints are refined by root finding
between grid points; 'b' marks a component that reaches the edge |theta| = pi/4.
Usage: python apps_sets.py <dataset> [grid points]"""
import sys, numpy as np
from scipy.stats import chi2
import _setup
from iiv.data import load
from iiv.core import Context, comps, uniroot, optimize, f3

ds = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1441
C = Context(load(ds)); P = C.prep(); n = C.n
th = np.linspace(-np.pi / 4, np.pi / 4, N)
sets = {'DT': [1], 'one_sided': [2, 4], 'two_sided': [1, 2], 'four': [1, 2, 3, 4], 'five': [1, 2, 3, 4, 5],
        'bounded': [7, 8]}


def show(cm):
    if not cm:
        return 'empty'
    return ' U '.join('[%s, %s]%s' % (f3(C.beta_at(lo, P)), f3(C.beta_at(hi, P)), 'b' if edge else '')
                      for lo, hi, edge in cm)


print('\n==== %s : n %d ; G %d ; OLS %s ; beta at -pi/4 %s ; at +pi/4 %s ====' % (
    C.lab, n, C.G, f3(P.b), f3(C.beta_at(-np.pi / 4, P)), f3(C.beta_at(np.pi / 4, P))))
for nm, cols in sets.items():
    m = len(cols)
    Sb = lambda t: C.Q_c(None, P, cols, beta=C.beta_at(t, P), ext='rob', return_nu=True)
    c1 = comps(lambda t: Sb(t)[0] - chi2.ppf(0.95, m), th)
    c2 = comps(lambda t: (lambda S: S[0] - C.crit_S(m, S[1]))(Sb(t)), th)
    print('%-10s chi2: %s\n%-10s adj : %s' % (nm, show(c1), '', show(c2)))

# VT roots of mean(u^2 s) in the labelling domain, with the standardized slope
dt = lambda t: C.gbar(t, P, [1])[0]
v = np.array([dt(t) for t in th]); ix = np.where(np.diff(np.sign(v)) != 0)[0]
if not len(ix):
    print('DT roots in |theta| <= pi/4: none')
scr = [1, 3, 4, 5]
for i in ix:
    r = uniroot(dt, th[i], th[i + 1], tol=1e-12); h = 1e-5
    D = (dt(r + h) - dt(r - h)) / (2 * h)
    L = abs(D) / np.sqrt(C.meat(C.gcorr(r, P, [1])[1])[0, 0])
    a, _ = optimize(lambda a: C.Q_c(a, P, scr), max(r - 0.1, -np.pi / 2 + 1e-3), min(r + 0.1, np.pi / 2 - 1e-3))
    print('DT root: theta %.4f beta %s rho %.3f | L %.3f sqrt(n)L %.2f | screen p (chi2, 3 df) %.3f' % (
        r, f3(C.beta_at(r, P)), -np.sin(r), L, np.sqrt(n) * L, 1 - chi2.cdf(C.Q_c(a, P, scr), 3)))
