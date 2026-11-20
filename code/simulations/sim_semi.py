"""Semi-synthetic Monte Carlo calibrated to an application sample (port of sim_semi.R).
Real regressor, controls, clusters and 2SLS residuals; beta0 = sample 2SLS; gamma
so that the OLS bias equals OLS - 2SLS. Designs hom and het (u resampled within
deciles of s0). Estimators: VT root by the relevance label; VT root after the
kurtosis screen (Bonferroni); CUE on six moment sets.
Usage: python sim_semi.py <dataset> [replications]   (writes sim_semi_<ds>.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import Semi, replicate, vt_roots, screen_J, fit_set, safe, cdist, mad, nanmedian, nanmean, ALPHA

ds = sys.argv[1] if len(sys.argv) > 1 else 'maim4v'; R = int(sys.argv[2]) if len(sys.argv) > 2 else 300
C = Context(load(ds)); S = Semi(C); beta0 = S.beta0
sets = {'two_sided': [1, 2], 'two_sided_var': [1, 2, 5, 6], 'one_sided': [2, 4], 'kurt_pair': [3, 4],
        'all4': [1, 2, 3, 4], 'five': [1, 2, 3, 4, 5]}
KEYS = ['est', 'cover', 'excl_ols', 'J_rej']


def one_rep(design):
    def f(rng):
        y, x = S.data(S.draw(design, rng)); P = C.prep(y, x); th0 = C.theta_of(beta0, P)
        rts = vt_roots(P); nr = len(rts); pv = []
        for r in rts:
            try:
                pv.append(1 - chi2.cdf(screen_J(C, P, r), 3))
            except (np.linalg.LinAlgError, ValueError):
                pv.append(np.nan)
        pv = np.array(pv)

        def pick(keep):
            c = np.where(keep & (np.abs(rts) <= np.pi / 4))[0]
            return c[np.argmin(np.abs(rts[c]))] if len(c) else None
        i1 = pick(np.ones(nr, bool)); i2 = pick(~(pv < ALPHA / max(nr, 1)))
        r = dict(ols=P.b, tsls=C.tsls(P)[0], rho0=-np.sin(th0),
                 **{'DT_label.est': np.nan if i1 is None else C.beta_at(rts[i1], P),
                    'DT_label.hit': 0.0 if i1 is None else float(cdist(rts[i1], th0) < 0.12),
                    'DT_screen.est': np.nan if i2 is None else C.beta_at(rts[i2], P),
                    'DT_screen.hit': 0.0 if i2 is None else float(cdist(rts[i2], th0) < 0.12)},
                 all_rej=float(nr > 0 and bool(np.all(pv[~np.isnan(pv)] < ALPHA))))
        for nm, cols in sets.items():
            r.update({nm + '.' + k: v for k, v in safe(lambda: fit_set(C, P, cols, beta0), KEYS).items()})
        return r
    return f


vr = np.array([np.mean(S.e_real[ix] ** 2) for ix in S.idx_bin])
print('\n====', ds, ': n =', C.n, '; real OLS', f3(S.b_ols), '; beta0 = real 2SLS', f3(beta0), '; gamma', f3(S.gamma),
      '; replications', R, '====')
print('residual variance by decile of the regressor, relative to the mean:\n ', ' '.join(f3(v) for v in vr / vr.mean()))
rows = []
for d, seed in (('hom', 20261005), ('het', 20261006)):
    M = pd.DataFrame(replicate(one_rep(d), R, seed)); M.to_csv(out('sim_semi_%s_%s.csv' % (ds, d)), index=False)
    print('\n---- design', d, ': median OLS', f3(np.median(M.ols)), ', median 2SLS', f3(np.median(M.tsls)), ', true rho_xu',
          f3(np.median(M.rho0)), '----')
    print('DT, label rule    : median %s, MAD %s, true root %s, none %s' % (f3(nanmedian(M['DT_label.est'])), f3(mad(M['DT_label.est'])),
          f3(M['DT_label.hit'].mean()), f3(M['DT_label.est'].isna().mean())))
    print('DT, screened      : median %s, MAD %s, true root %s, none %s; all roots rejected %s' % (
        f3(nanmedian(M['DT_screen.est'])), f3(mad(M['DT_screen.est'])), f3(M['DT_screen.hit'].mean()),
        f3(M['DT_screen.est'].isna().mean()), f3(M.all_rej.mean())))
    print('%-14s %8s %8s %8s %9s %9s' % ('set', 'median', 'MAD', 'cover', 'excl OLS', 'J reject'))
    for s in sets:
        v = dict(dataset=ds, design=d, set=s, median=nanmedian(M[s + '.est']), mad=mad(M[s + '.est']),
                 cover=nanmean(M[s + '.cover']), excl_ols=nanmean(M[s + '.excl_ols']), J_reject=nanmean(M[s + '.J_rej']))
        print('%-14s %8s %8s %8s %9s %9s' % (s, f3(v['median']), f3(v['mad']), f3(v['cover']), f3(v['excl_ols']), f3(v['J_reject'])))
        rows.append(v)
    for s in ('DT_label', 'DT_screen'):
        rows.append(dict(dataset=ds, design=d, set=s, median=nanmedian(M[s + '.est']), mad=mad(M[s + '.est']),
                         true_root=M[s + '.hit'].mean(), none=M[s + '.est'].isna().mean()))
pd.DataFrame(rows).to_csv(out('sim_semi_%s.csv' % ds), index=False)
