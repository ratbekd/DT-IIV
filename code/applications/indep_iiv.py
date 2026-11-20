"""Independence-based intrinsic IV on one application (port of the run block of
indep_iiv.R): OLS and 2SLS, screening moments, VT roots, and for moment sets
"3" (two-sided) and "34" (five moments) the local CUE minima, the corrected
J-test, component diagnostics, the S-set over the circle, and for each labelling
rule the analytic corrected SE and a pairs or cluster bootstrap.
Usage: python indep_iiv.py <dataset> [--legacy-card]
--legacy-card reproduces indep_iiv_card.R (S-set with theta free, uncentred SE, i.i.d.)."""
import sys, numpy as np
from scipy.stats import chi2
import _setup
from iiv.data import load
from iiv.core import Context, optimize, partner, estimate, f3

args = [a for a in sys.argv[1:] if not a.startswith('--')]; legacy = '--legacy-card' in sys.argv
ds = args[0] if args else 'card'
D = load(ds)
if legacy:
    D['cl'] = None
C = Context(D); P = C.prep(); n, k = C.n, C.k; B = D['B_boot']; A = 0.05
rng = np.random.default_rng(20260925)
xt0, r0, yt0 = P.xt, P.r, P.yt
se_ols = np.sqrt(np.sum(xt0 ** 2 * r0 ** 2)) / np.sum(xt0 ** 2) * np.sqrt(n / (n - k - 1))
xh = C.Zt @ np.linalg.lstsq(C.Zt, xt0, rcond=None)[0]; b_iv = np.sum(xh * yt0) / np.sum(xh * xt0)
se_iv = np.sqrt(np.sum(xh ** 2 * (yt0 - xt0 * b_iv) ** 2)) / abs(np.sum(xh * xt0)) * np.sqrt(n / (n - k - 1))
xw, rw = P.xw, P.rw
scr = dict(mu30=np.mean(xw ** 3), mu21=np.mean(xw ** 2 * rw), mu12=np.mean(xw * rw ** 2), mu03=np.mean(rw ** 3))
thd = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 20001); g1 = np.array([C.gbar(a, P, [1])[0] for a in thd])
ix = np.where(np.diff(np.sign(g1)) != 0)[0]
dt_roots = thd[ix] - g1[ix] * (thd[ix + 1] - thd[ix]) / (g1[ix + 1] - g1[ix])

if legacy:
    print('\n==== Independence-based intrinsic IV, Card: lwage on educ, n =', n, '====')
    print('OLS', f3(P.b), '(HC1', f3(se_ols), ');  2SLS nearc2, nearc4', f3(b_iv), '(HC1', f3(se_iv), ')')
else:
    print('\n==== Independence-based intrinsic IV,', C.lab, ', n =', n, (', %d clusters' % C.G) if C.clustered else '', '====')
    print('OLS', f3(P.b), '(HC1', f3(se_ols), ');  2SLS', ', '.join(D['z_vars']), f3(b_iv), '(HC1', f3(se_iv), ')')
    print('Screening (standardized): ', ', '.join('%s = %s' % (a, f3(v)) for a, v in scr.items()),
          '; distinct y =', len(np.unique(C.y)), ', distinct x =', len(np.unique(C.x)))
print('DT-only roots, t = k*delta:', ', '.join(f3(v) for v in np.tan(dt_roots)),
      ' -> beta:', ', '.join(f3(C.beta_at(r, P)) for r in dt_roots))

Cn = None
if C.clustered:
    Dn = dict(D); Dn['cl'] = None; Cn = Context(Dn)
groups = C._loo_setup()['groups']
for set_ in ('3', '34'):
    cols = [1, 2] if set_ == '3' else [1, 2, 3, 4, 5]; m = len(cols)
    sol = C.cue_minima(P, cols, ngrid=181); thf = sol[0][0]
    print('\n---- moment set', set_, '(', m, 'conditions ) ----')
    print('Local minima of the CUE criterion on |theta| <= pi/4 (partner in brackets):')
    for a, q in sol:
        if abs(abs(a) - np.pi / 4) < 1e-3:
            continue
        b = partner(a)
        print('   t = %7.3f  beta = %7.3f  [t = %8.3f  beta = %8.3f]  Q_u = %6.2f' % (np.tan(a), C.beta_at(a, P), np.tan(b),
                                                                                C.beta_at(b, P), q))
    a, J = optimize(lambda a: C.Q_c(a, P, cols), thf - 0.15, thf + 0.15)
    print('Corrected J = %.2f, df = %d, p = %.3f  (argmin t = %.3f vs CUE t = %.3f)' % (J, m - 1, 1 - chi2.cdf(J, m - 1),
                                                                                    np.tan(a), np.tan(thf)))
    if Cn is not None:
        _, Ji = optimize(lambda a: Cn.Q_c(a, P, cols), thf - 0.15, thf + 0.15)
        print('   (same J ignoring clustering = %.2f, p = %.3f)' % (Ji, 1 - chi2.cdf(Ji, m - 1)))
    s = np.cos(thf) * xw + np.sin(thf) * rw; u = np.cos(thf) * rw - np.sin(thf) * xw
    sk = lambda v: np.mean(v ** 3) / np.mean(v ** 2) ** 1.5; ku = lambda v: np.mean(v ** 4) / np.mean(v ** 2) ** 2 - 3
    X = np.column_stack([np.ones(n), s, s ** 2]); XXi = np.linalg.inv(X.T @ X); cf = XXi @ X.T @ (u ** 2); e = u ** 2 - X @ cf
    V = XXi @ (X * e[:, None] ** 2).T @ X @ XXi; tt = cf / np.sqrt(np.diag(V))
    print('Components: skew(s) = %.3f, skew(u) = %.3f, exkurt(s) = %.3f, exkurt(u) = %.3f' % (sk(s), sk(u), ku(s), ku(u)))
    print('Heteroscedasticity check, u^2 on s and s^2 (HC0 t): %.2f, %.2f' % (tt[1], tt[2]))
    thg = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 721); bg = C.beta_at(thg, P)
    Sv = np.array([C.Q_c(a, P, cols) if legacy else C.Q_b(C.beta_at(a, P), P, cols) for a in thg])
    acc = Sv <= chi2.ppf(1 - A, m); in_f = np.abs(thg) < np.pi / 4

    def rng_(v):
        if not v.any():
            return 'empty'
        if (v & (np.abs(thg) > np.pi / 2 - 0.01)).any():
            return 'unbounded'
        return '[%s, %s]' % (f3(bg[v].min()), f3(bg[v].max()))
    pcs = lambda v: int(np.sum(np.diff(np.r_[0, v.astype(int), 0]) == 1))
    print('S-set (5%): relevance domain', rng_(acc & in_f), '(', pcs(acc & in_f), 'piece(s)); partner domain',
          rng_(acc & ~in_f), '(', pcs(acc & ~in_f), 'piece(s))')
    for rule in ('relevance', 'beta_pos'):
        if rule == 'relevance':
            th = thf
        else:
            cand = np.array([thf, partner(thf)]); bb_ = C.beta_at(cand, P)
            th = cand[bb_ > 0][0] if np.sum(bb_ > 0) == 1 else np.nan
        if np.isnan(th):
            print('Label rule', rule, ': ambiguous (both or neither beta > 0)'); continue
        se = C.analytic_se(th, P, cols, centred=not legacy)[1]
        bh = C.beta_at(th, P); draws = []
        for _ in range(B):
            if C.clustered:
                id_ = np.concatenate([groups[g] for g in rng.integers(0, C.G, C.G)])
            else:
                id_ = rng.integers(0, n, n)
            draws.append(estimate(C.y[id_], C.x[id_], C.W1[id_], cols, rule))
        bb = np.array(draws); ok = ~np.isnan(bb[:, 1]); ci = np.quantile(bb[ok, 1], [A / 2, 1 - A / 2])
        print(('Label %-9s: beta = %.3f (t = %.3f, cor(s,x) = %.3f)\n   analytic corrected SE %.3f -> Wald [%.3f, %.3f]\n'
               '   bootstrap SE %.3f, 95%% percentile [%.3f, %.3f], labelled draws %d/%d, draws at the |t| = 1 edge %.1f%%') % (
            rule, bh, np.tan(th), abs(np.cos(th)), se, bh - 1.96 * se, bh + 1.96 * se, np.std(bb[ok, 1], ddof=1), ci[0], ci[1],
            ok.sum(), B, 100 * np.nanmean(bb[:, 2])))
