"""Full VT diagnostics for Angrist and Krueger (1991), 1980 Census, men born 1930-39:
log weekly wage on years of schooling, 51 state-of-birth clusters, quarter-of-birth 2SLS.

  1. VT roots on the whole circle, with the standardized slope L and sqrt(n) L
  2. fourth-order screen of each root (chi2_3, Bonferroni over the roots), chi2 and adjusted
  3. CUE estimates and corrected J-tests of the two-sided, four-moment, five-moment and
     variance-augmented sets, chi2 and adjusted p-values
  4. fixed-slope confidence sets of the VT moment and of every moment set, chi2 and adjusted
  5. the mixed third moments mu21 and mu12 (the heteroscedasticity channels) and the
     skewness of the residual, the source that identifies the VT moment
  6. root stability: shares of state-cluster resamples with 0 / 1 / 2+ VT roots in the
     labelling domain, and the distribution of the admissible root

Usage: python ak_vt.py [path to NEW7080.dta] [B]      writes output/r61/ak_vt.json
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import chi2
import r61lib as L
from iiv.core import Context, comps, uniroot, optimize, coef_P, polyroot_real, boot_prep

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(L.HERE, '..', '..', '..', 'NEW7080_1', 'NEW7080.dta')
B = int(sys.argv[2]) if len(sys.argv) > 2 else 999
d = pd.read_stata(PATH)
d = d[(d.v16 == 80) & (d.v27 >= 30) & (d.v27 <= 39)].reset_index(drop=True)
cols = [np.ones(len(d))] + [(d.v27 == t).astype(float).values for t in range(31, 40)]
cols += [d[v].values.astype(float) for v in ('v19', 'v20', 'v10', 'v5', 'v6', 'v11', 'v12', 'v13', 'v21', 'v24', 'v25')]
C = Context(dict(y=d.v9.values.astype(float), x=d.v4.values.astype(float), W1=np.column_stack(cols),
                 Z=np.column_stack([(d.v18 == q).astype(float).values for q in (1, 2, 3)]),
                 cl=d.v17.astype(int).astype(str).values, lab='Angrist-Krueger 1930-39'))
P = C.prep(); n = C.n
b_iv, se_iv = C.tsls(P)
out = dict(n=n, G=C.G, ols=P.b, tsls=b_iv, tsls_se=se_iv, ratio=P.sr / P.sx, rho_tsls=-np.sin(C.theta_of(b_iv, P)))
print('==== %s: n %d, %d clusters; OLS %.4f; 2SLS %.4f (%.4f), implied rho_xu %.3f; sr/sx %.3f ====' % (
    C.lab, n, C.G, P.b, b_iv, se_iv, out['rho_tsls'], P.sr / P.sx), flush=True)

# 5. identifying source and heteroscedasticity channels (cluster-robust t, no first-step correction)
def mom_t(v):
    m = v.mean(); s = np.sqrt(C.meat(v)[0, 0] / n); return m, m / s
xw, rw = P.xw, P.rw
src = {}
for nm, v in (('mu30', xw ** 3), ('mu21', xw ** 2 * rw), ('mu12', xw * rw ** 2), ('mu03', rw ** 3)):
    src[nm] = mom_t(v)
print('third moments of (x_w, r_w) with cluster t: ' + '  '.join('%s %.3f (t %.1f)' % (k, a, t) for k, (a, t) in src.items()))
out['third_moments'] = {k: dict(value=a, t=t) for k, (a, t) in src.items()}

# 1-2. VT roots on the circle, slope, screen
f = lambda a: np.mean((np.cos(a) * rw - np.sin(a) * xw) ** 2 * (np.cos(a) * xw + np.sin(a) * rw))
roots = C.vt_roots(P)
R = len(roots); scr = [1, 3, 4, 5]
out['roots'] = []
print('VT roots on the circle: %d (Bonferroni level for the screen 0.05/%d = %.4f)' % (R, R, 0.05 / R))
for r in roots:
    h = 1e-5
    D = (f(r + h) - f(r - h)) / (2 * h)
    sd = np.sqrt(C.meat(C.gcorr(r, P, [1])[1])[0, 0])
    Lr = abs(D) / sd
    lo, hi = max(r - 0.1, -np.pi / 2 + 1e-3), min(r + 0.1, np.pi / 2 - 1e-3)
    others = [o for o in roots if o != r]
    if others:
        gap = min(abs(o - r) for o in others)
        if gap < 0.2:
            lo, hi = max(lo, r - gap / 2), min(hi, r + gap / 2)
    a, _ = optimize(lambda a: C.Q_c(a, P, scr), lo, hi)
    J, nu = C.Q_c(a, P, scr, return_nu=True)
    rec = dict(theta=r, beta=C.beta_at(r, P), rho=-np.sin(r), admissible=bool(abs(r) <= np.pi / 4), L=Lr,
               sqrtnL=np.sqrt(n) * Lr, screen_J=J, screen_p_chi2=1 - chi2.cdf(J, 3), screen_p_adj=C.p_S(J, 3, nu), nu=nu)
    out['roots'].append(rec)
    print('  beta %9.4f  rho %6.3f  %-10s L %.4f  sqrt(n)L %7.2f | screen J %.2f  p chi2 %.4f  p adj %.4f (nu %.0f)' % (
        rec['beta'], rec['rho'], 'admissible' if rec['admissible'] else 'outside', Lr, rec['sqrtnL'], J,
        rec['screen_p_chi2'], rec['screen_p_adj'], nu), flush=True)

# 3. CUE and corrected J-tests
SETS = {'VT': [1], 'two-sided': [1, 2], 'four': [1, 2, 3, 4], 'five': [1, 2, 3, 4, 5],
        'variance-augmented': [1, 2, 5, 6], 'one-sided': [2, 4]}
out['fits'] = {}
print('CUE estimates and corrected J-tests:')
for nm, cs in SETS.items():
    fr = C.fit_cue(P, cs)
    out['fits'][nm] = dict(beta=fr['beta'], rho=-np.sin(fr['theta']), J=fr['J'], p_chi2=fr['J_p_chi2'], p_adj=fr['J_p'])
    print('  %-19s beta %.4f  rho %6.3f  J %s  p chi2 %s  p adj %s' % (
        nm, fr['beta'], -np.sin(fr['theta']), '--' if np.isnan(fr['J']) else '%.2f' % fr['J'],
        '--' if np.isnan(fr['J_p_chi2']) else '%.4f' % fr['J_p_chi2'], '--' if np.isnan(fr['J_p']) else '%.4f' % fr['J_p']),
          flush=True)

# 4. fixed-slope confidence sets
th = np.linspace(-np.pi / 4, np.pi / 4, 241)
show = lambda cm: ' U '.join('[%.4f, %.4f]%s' % (C.beta_at(lo, P), C.beta_at(hi, P), '+' if e else '') for lo, hi, e in cm) or 'empty'
out['sets'] = {}
print('Fixed-slope confidence sets (241 angles, crossings refined):')
for nm, cs in SETS.items():
    m = len(cs)
    Sb = lambda t, cs=cs: C.Q_c(None, P, cs, beta=C.beta_at(t, P), return_nu=True)
    vals = [Sb(t) for t in th]
    cache = dict(zip(th, vals))
    get = lambda t: cache[t] if t in cache else Sb(t)
    c1 = comps(lambda t: get(t)[0] - chi2.ppf(0.95, m), th)
    c2 = comps(lambda t: (lambda S: S[0] - C.crit_S(m, S[1]))(get(t)), th)
    inn = lambda b: (lambda S: (S[0] <= chi2.ppf(0.95, m), S[0] <= C.crit_S(m, S[1])))(
        C.Q_c(None, P, cs, beta=b, return_nu=True))
    io, ii = inn(P.b), inn(b_iv)
    out['sets'][nm] = dict(chi2=[(C.beta_at(a, P), C.beta_at(b, P), e) for a, b, e in c1],
                           adj=[(C.beta_at(a, P), C.beta_at(b, P), e) for a, b, e in c2],
                           ols_in=io, tsls_in=ii)
    print('  %-19s chi2: %-34s adj: %-34s OLS in %s  2SLS in %s' % (nm, show(c1), show(c2), io, ii), flush=True)

# 6. root stability
def adm_roots(W, y, x):
    dd = boot_prep(W, y, x); tau = polyroot_real(coef_P(dd['xw'], dd['rw'])); tau = tau[np.abs(tau) <= 1]
    return dd['b'] + tau * dd['k']
full = adm_roots(C.W1, C.y, C.x)
groups = [np.where(C.cl_codes == g)[0] for g in range(C.G)]     # whole state clusters
rng = np.random.default_rng(20261016)
br = []
for _ in range(B):
    i = np.concatenate([groups[g] for g in rng.integers(0, C.G, C.G)])
    br.append(adm_roots(C.W1[i], C.y[i], C.x[i]))
cnt = np.array([len(r) for r in br])
one = np.array([r[0] for r in br if len(r) == 1])
out['stability'] = dict(B=B, full=full.tolist(), zero=float(np.mean(cnt == 0)), one=float(np.mean(cnt == 1)),
                        two_plus=float(np.mean(cnt >= 2)),
                        one_root_q=np.quantile(one, [0.025, 0.5, 0.975]).tolist() if len(one) else None,
                        one_root_sd=float(one.std(ddof=1)) if len(one) > 1 else None)
print('Root stability, %d state-cluster resamples: full-sample admissible roots %s; zero %.1f%%  one %.1f%%  two+ %.1f%%' % (
    B, ', '.join('%.4f' % v for v in full), 100 * np.mean(cnt == 0), 100 * np.mean(cnt == 1), 100 * np.mean(cnt >= 2)))
if len(one):
    print('  single-root resamples: median %.4f, 95%% range [%.4f, %.4f], sd %.4f' % (
        *np.quantile(one, [0.5, 0.025, 0.975]), one.std(ddof=1)))
json.dump(out, open(L.out('ak_vt.json'), 'w'), indent=1, default=float)
