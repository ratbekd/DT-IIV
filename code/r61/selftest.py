"""Checks of r61lib against the package core and against brute force."""
import numpy as np
from scipy.stats import chi2
import r61lib as L
from iiv.data import load
from iiv.core import Context

rng = np.random.default_rng(1)
C = Context(load('adh')); P = C.prep()
print('ADH n', C.n, 'G', C.G, 'OLS', round(P.b, 4))
for b in (P.b - 0.1, P.b + 0.05, -0.303):
    s1 = L.analytic(C, P, b, L.fam_P)[0]; s0 = C.Q_c(None, P, [2, 4], beta=b)
    r1 = L.analytic(C, P, b, L.fam_R)[0]; r0 = C.Q_c(None, P, [7, 8], beta=b, ext='rob')
    print('beta %.3f  P: %.10f vs core %.10f | R: %.10f vs core %.10f' % (b, s1, s0, r1, r0))

# analytic derivative of the first-step correction against a jackknife in a large sample
n = 4000; G = 400
cl = np.repeat(np.arange(G), n // G)
s0 = (rng.chisquare(4, n) - 4) / np.sqrt(8); u0 = rng.standard_normal(n)
x = 2 + s0 + 0.5 * u0; y = 1 + x + u0
W = np.ones((n, 1))
Cs = L.make_context(y, x, W, cl.astype(str)); Ps = Cs.prep()
for fam in 'PBLR':
    S = L.analytic(Cs, Ps, 1.0, L.FAMS[fam])[0]
    m, M = L.jk_refit(W, y, x, Cs.cl_codes, Cs.G, [1.0], [fam])
    T = L.jk_stat(m[0, 0], M[:, 0, 0])[0]
    print('family %s  analytic S %.4f   full-refit T_JK %.4f' % (fam, S, T))

# explicit deletion check of jk_refit with controls
w = rng.standard_normal(n); W2 = np.column_stack([np.ones(n), w])
y2 = y + 0.3 * w; x2 = x + 0.5 * w
C2 = L.make_context(y2, x2, W2, cl.astype(str))
m, M = L.jk_refit(W2, y2, x2, C2.cl_codes, C2.G, [0.9, 1.0], ['P', 'R'])
g = 7; keep = C2.cl_codes != g
Cg = L.make_context(y2[keep], x2[keep], W2[keep], None); Pg = Cg.prep()
for ib, b in enumerate((0.9, 1.0)):
    th = Cg.theta_of(b, Pg); sg = np.cos(th) * Pg.xw + np.sin(th) * Pg.rw; ug = np.cos(th) * Pg.rw - np.sin(th) * Pg.xw
    for jf, f in enumerate('PR'):
        ref = (ug[:, None] * L.FAMS[f](sg, deriv=False)).mean(0)
        print('deletion check beta %.1f fam %s  max abs diff %.2e' % (b, f, np.max(np.abs(ref - M[g, ib, jf]))))

# reference check of the calibrations with Gaussian cluster vectors
G = 48; R = 2000; rej = np.zeros(3)
for r in range(R):
    Z = rng.standard_normal((G, 2)); mbar = Z.mean(0)
    Mg = (G * mbar - Z) / (G - 1)
    T, V, d = L.jk_stat(mbar, Mg)
    rej += [T > chi2.ppf(0.95, 2), T > L.crit_F(G), L.wild_p(T, V, d, 999, rng) <= 0.05]
print('Gaussian reference, G = 48: rejection chi2 %.3f  F %.3f  wild %.3f' % tuple(rej / R))

# projection distance: brute-force check on random problems
bad = 0
for r in range(40):
    A = rng.standard_normal((5, 5)); V = A @ A.T / 5 + 0.2 * np.eye(5)
    t = np.r_[rng.normal(0, 0.3, 2), 1.0 + 0.2 * rng.standard_normal(), 0.2 * rng.standard_normal(),
              1.0 + 0.2 * rng.standard_normal()]
    eps = rng.choice([0.0, 0.05, 0.1, 0.3]); n = 500
    D = L.ami_D(t, V, n, eps); Vi = np.linalg.inv(V)
    lo = np.inf
    for k in range(20000):
        Cm = np.array([[abs(rng.normal(1, 0.3)), 0], [rng.normal(0, 0.3), abs(rng.normal(1, 0.3))]])
        a = rng.standard_normal(2); a = a / np.linalg.norm(a) * eps * rng.uniform() ** 0.5
        Q = Cm @ Cm.T; mu = Cm @ a
        e = t - np.r_[mu, Q[0, 0], Q[0, 1], Q[1, 1]]
        lo = min(lo, n * e @ Vi @ e)
    if D > lo + 1e-6:
        bad += 1
        print('  projection not minimal: D %.4f random search %.4f eps %.2f' % (D, lo, eps))
print('projection check: %d of 40 problems where a random feasible point beat the solver' % bad)
