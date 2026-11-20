"""Small-sample accuracy of the S-sets, J-tests and root screen under the chi2 and
F(nu) critical values, and null draws for calibrated p-values (port of
sim_semi_crit.R). Designs: hom (u resampled from the pooled residuals; every
restriction holds) and het (u resampled within deciles of s0; coverage only).
Usage: python sim_semi_crit.py <dataset> [replications] [gamma] [beta0]
Writes sim_semi_crit_<ds>[tag].csv and the raw draws sim_semi_crit_<ds>[tag]_<design>.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import Semi, replicate, vt_roots, screen_J, cue_J, cdist, nanmean, ALPHA

args = sys.argv[1:]
ds = args[0] if args else 'banerjee'; R = int(args[1]) if len(args) > 1 else 1000
gam = float(args[2]) if len(args) > 2 and args[2] != 'NA' else None
b0 = float(args[3]) if len(args) > 3 else None
C = Context(load(ds)); S = Semi(C, beta0=b0, gamma=gam); beta0 = S.beta0
sets = {'two_sided': [1, 2], 'one_sided': [2, 4], 'four': [1, 2, 3, 4], 'five': [1, 2, 3, 4, 5],
        'two_sided_var': [1, 2, 5, 6]}


def one_rep(design):
    def f(rng):
        y, x = S.data(S.draw(design, rng)); P = C.prep(y, x); r = {}
        for nm, cols in sets.items():
            v = dict(S=np.nan, S_nu=np.nan, J=np.nan, J_nu=np.nan)
            try:
                v['S'], v['S_nu'] = C.Q_c(None, P, cols, beta=beta0, return_nu=True)
                if design == 'hom':
                    _, _, a = cue_J(C, P, cols); v['J'], v['J_nu'] = C.Q_c(a, P, cols, return_nu=True)
            except (np.linalg.LinAlgError, ValueError):
                pass
            r.update({nm + '.' + k: val for k, val in v.items()})
        r['screen.J'] = r['screen.J_nu'] = np.nan
        if design == 'hom':
            try:
                th0 = C.theta_of(beta0, P); rts = vt_roots(P)
                if len(rts):
                    k = int(np.argmin(cdist(rts, th0)))
                    if cdist(rts[k], th0) <= 0.12:
                        r['screen.J'], r['screen.J_nu'] = screen_J(C, P, rts[k], return_nu=True)
            except (np.linalg.LinAlgError, ValueError):
                pass
        return r
    return f


tag = ('_' + '_'.join(args[2:])) if len(args) > 2 else ''
print('\n====', ds, ': n =', C.n, '; clusters', C.G if C.clustered else 'none', '; beta0', f3(beta0), '; OLS',
      f3(S.b_ols), '; gamma', f3(S.gamma), '; replications', R, '====')
rows = []
for d, seed in (('hom', 20261011), ('het', 20261012)):
    M = pd.DataFrame(replicate(one_rep(d), R, seed)); M.to_csv(out('sim_semi_crit_%s%s_%s.csv' % (ds, tag, d)), index=False)
    print('\n---- design', d, '----')
    print('%-14s %2s | %14s | %14s | %5s' % ('', '', 'coverage', 'J rejects', ''))
    print('%-14s %2s | %6s %7s | %6s %7s | %5s' % ('set', 'm', 'chi2', 'F(nu)', 'chi2', 'F(nu)', 'nu'))
    for s in list(sets) + (['screen'] if d == 'hom' else []):
        m = 4 if s == 'screen' else len(sets[s])
        Sv = M[s + '.S'].values if s != 'screen' else np.full(R, np.nan)
        Snu = M[s + '.S_nu'].values if s != 'screen' else np.full(R, np.nan)
        J = M[s + '.J'].values; Jnu = M[s + '.J_nu'].values
        okS = ~np.isnan(Sv); okJ = ~np.isnan(J)
        v = dict(c_chi=nanmean(np.where(okS, Sv <= chi2.ppf(1 - ALPHA, m), np.nan)),
                 c_Fnu=nanmean(np.where(okS, [sv <= C.crit_S(m, nu) if ok else False for sv, nu, ok in zip(Sv, Snu, okS)], np.nan)),
                 j_chi=nanmean(np.where(okJ, J > chi2.ppf(1 - ALPHA, m - 1), np.nan)),
                 j_Fnu=nanmean(np.where(okJ, [C.p_S(j, m - 1, nu) < ALPHA if ok else False for j, nu, ok in zip(J, Jnu, okJ)], np.nan)),
                 nu=np.nanmedian(np.r_[Snu, Jnu]) if np.isfinite(np.r_[Snu, Jnu]).any() else np.nan)
        print('%-14s %2d | %6s %7s | %6s %7s | %5.0f' % (s, m, f3(v['c_chi']), f3(v['c_Fnu']), f3(v['j_chi']), f3(v['j_Fnu']), v['nu']))
        rows.append(dict(dataset=ds, design=d, set=s, m=m, **v, reps=R,
                         no_true_root=np.mean(np.isnan(J)) if s == 'screen' else np.nan))
pd.DataFrame(rows).to_csv(out('sim_semi_crit_%s%s.csv' % (ds, tag)), index=False)
