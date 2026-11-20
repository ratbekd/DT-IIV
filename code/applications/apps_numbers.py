"""OLS and 2SLS with cluster-robust SEs, standardized third moments with
bootstrap t-statistics, every VT root on the circle with its screen p-value,
root-specific naive, corrected and bootstrap SEs, and the endogeneity-correlation
table (port of apps_numbers_r47.R). Bootstrap seed 20261016.
Usage: python apps_numbers.py <dataset> [B]"""
import sys, numpy as np
from scipy.stats import chi2
import _setup
from iiv.data import load
from iiv.core import Context, comps, optimize, f3, coef_P, polyroot_real, cluster_draws, boot_prep

ds = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 999
C = Context(load(ds)); P = C.prep(); n = C.n; kk = P.sr / P.sx
print('====', C.lab, ': n', n, 'G', C.G, 'OLS', f3(P.b), 'sr/sx', '%.6f' % kk)

xt, yt = P.xt, P.yt
se_ols = np.sqrt(np.sum(C.rowsum(xt * (yt - P.b * xt)) ** 2)) / np.sum(xt ** 2)
b_iv, se_iv = C.tsls(P)
print('OLS %s (%s)  2SLS %s (%s)' % (f3(P.b), f3(se_ols), f3(b_iv), f3(se_iv)))

mom3 = lambda a, c: np.array([np.mean(a ** 3), np.mean(a ** 2 * c), np.mean(a * c ** 2), np.mean(c ** 3)])
m0 = mom3(P.xw, P.rw)
draws = [boot_prep(C.W1[id_], C.y[id_], C.x[id_]) for id_ in cluster_draws(C, B, 20261016)]
mb = np.array([mom3(d['xw'], d['rw']) for d in draws])
print('third moments (standardized), cluster-bootstrap t:')
print('        m30    m21    m12    m03')
print('value ' + ' '.join('%6.3f' % v for v in m0))
print('t     ' + ' '.join('%6.3f' % v for v in m0 / mb.std(0, ddof=1)))

# every VT root on the whole circle
th_r = np.arctan(polyroot_real(coef_P(P.xw, P.rw))); scr = [1, 3, 4, 5]


def scr_p(r, w):
    a, _ = optimize(lambda a: C.Q_c(a, P, scr), r - w, r + w)
    S, nu = C.Q_c(a, P, scr, return_nu=True)
    return 1 - chi2.cdf(S, 3), C.p_S(S, 3, nu), a


for j, r in enumerate(th_r):
    others = np.delete(th_r, j)
    w_near = min(0.1, 0.5 * (np.min(np.abs(others - r)) if len(others) else np.inf))
    p1 = scr_p(r, 0.1); p2 = scr_p(r, w_near)
    print('root %d: theta %.4f beta %s rho %.3f relevance %.3f in-domain %s | screen p (window 0.1) %.3f [min at %.4f] '
          '| (window %.3f) chi2 %.3f adj %.3f [min at %.4f]' % (
              j + 1, r, f3(C.beta_at(r, P)), -np.sin(r), np.cos(r), 'TRUE' if abs(r) <= np.pi / 4 else 'FALSE',
              p1[0], p1[2], w_near, p2[0], p2[1], p2[2]))

# root-specific standard errors for the in-domain root nearest theta = 0
inr = th_r[np.abs(th_r) <= np.pi / 4]
if len(inr):
    r = inr[np.argmin(np.abs(inr))]; cs, sn = np.cos(r), np.sin(r)
    g, gc, s, u = C.gcorr(r, P, [1]); gc = gc[:, 0]
    D = np.mean(u ** 2 * u - 2 * u * s * s)            # d mean(g)/d theta = E[g_s u - g_u s]
    psi_th = -gc / D; tn = np.tan(r)
    psi_beta = kk / cs ** 2 * psi_th + P.psi_b + (tn / P.sx) * P.psi_sr - (tn * P.sr / P.sx ** 2) * P.psi_sx
    psi_gap = psi_beta - P.psi_b
    se_c = np.sqrt(np.sum(C.rowsum(psi_beta) ** 2)) / n; se_gap = np.sqrt(np.sum(C.rowsum(psi_gap) ** 2)) / n
    b_r = C.beta_at(r, P)
    sd_ = s * P.sx / cs; e_r = yt - b_r * xt
    se_n = np.sqrt(np.sum(C.rowsum(sd_ * e_r) ** 2)) / abs(np.sum(sd_ * xt))
    bb = []
    for d in draws:
        tt = polyroot_real(coef_P(d['xw'], d['rw'])); tt = tt[np.abs(tt) <= 1]
        bb.append(np.nan if not len(tt) else d['b'] + tt[np.argmin(np.abs(tt))] * d['k'])
    bb = np.array(bb); ok = ~np.isnan(bb)
    q = np.quantile(bb[ok], [0.025, 0.975])
    print('in-domain root nearest 0: beta %s | naive se %s | corrected se %s, Wald [%s, %s] | bootstrap se %s, '
          'percentile [%s, %s], draws with a root %d/%d | gap %s (se %s)' % (
              f3(b_r), f3(se_n), f3(se_c), f3(b_r - 1.96 * se_c), f3(b_r + 1.96 * se_c), f3(np.std(bb[ok], ddof=1)),
              f3(q[0]), f3(q[1]), ok.sum(), B, f3(b_r - P.b), f3(se_gap)))

# endogeneity correlation: estimates, image of the chi2 sets, prior |rho| <= 0.3
th = np.linspace(-np.pi / 4, np.pi / 4, 1441); lim = np.arcsin(0.3)
for nm, cols in (('one_sided', [2, 4]), ('two_sided', [1, 2])):
    f = C.fit_cue(P, cols)
    cm = comps(lambda t: C.Q_c(None, P, cols, beta=C.beta_at(t, P)) - chi2.ppf(0.95, len(cols)), th)
    rho_s = 'empty' if not cm else ' U '.join('[%.3f, %.3f]' % (-np.sin(hi), -np.sin(lo)) for lo, hi, _ in cm)
    ii = [(max(lo, -lim), min(hi, lim)) for lo, hi, _ in cm]; ii = [v for v in ii if v[0] <= v[1]]
    pri = 'empty' if not ii else ' U '.join('[%s, %s]' % (f3(C.beta_at(a, P)), f3(C.beta_at(b, P))) for a, b in ii)
    print('%-9s rho_hat %.3f | C_rho %s | C_beta(D) %s' % (nm, -np.sin(f['theta']), rho_s, pri))
print('prior-only bound B(D): [%s, %s]' % (f3(C.beta_at(-lim, P)), f3(C.beta_at(lim, P))))
