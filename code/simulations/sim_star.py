"""Semi-synthetic Monte Carlo on the STAR kindergarten sample (port of sim_star.R).
Designs D0-D3; moment sets VT, two-sided, one-sided, five; 2SLS benchmark.
Per set: median and MAD of the estimate, coverage of beta0 by the corrected,
clustered S-set, share excluding OLS, J-test rejection rate.
Usage: python sim_star.py [replications]   (writes sim_star.csv)"""
import sys, numpy as np, pandas as pd
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import StarDesign, replicate, fit_set, tsls_rep, safe, mad, nanmedian, nanmean

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
C = Context(load('star')); S = StarDesign(C); beta0 = S.beta0
sets = {'DT': [1], 'two_sided': [1, 2], 'one_sided': [2, 4], 'five': [1, 2, 3, 4, 5]}
KEYS = ['est', 'cover', 'excl_ols', 'J_rej']


def one_rep(design):
    def f(rng):
        y, x = S.data(design, rng); P = C.prep(y, x); r = {}
        for nm, cols in sets.items():
            r.update({nm + '.' + k: v for k, v in safe(lambda: fit_set(C, P, cols, beta0), KEYS).items()})
        r.update({'tsls.' + k: v for k, v in tsls_rep(C, P, beta0).items()}); r['ols'] = P.b
        return r
    return f


res = {}
for i, d in enumerate(('D0', 'D1', 'D2', 'D3')):
    res[d] = pd.DataFrame(replicate(one_rep(d), R, 20260925 + i)); print(d, 'done')
print('\nbeta0 =', beta0, '; replications per design =', R, '; n =', C.n, '; clusters =', C.G)
rows = []
for d, M in res.items():
    print('\n====', d, '==== median OLS', f3(np.median(M.ols)))
    print('%-10s %9s %9s %9s %9s %9s %8s' % ('method', 'median', 'MAD', 'cover', 'excl OLS', 'J reject', 'NA est'))
    for mth in list(sets) + ['tsls']:
        e = M[mth + '.est']; jr = M[mth + '.J_rej'] if mth + '.J_rej' in M else None
        v = dict(design=d, method=mth, median=nanmedian(e), mad=mad(e), cover=nanmean(M[mth + '.cover']),
                 excl_ols=nanmean(M[mth + '.excl_ols']), J_reject=np.nan if jr is None else nanmean(jr), na_est=int(e.isna().sum()))
        print('%-10s %9s %9s %9s %9s %9s %8d' % (mth, f3(v['median']), f3(v['mad']), f3(v['cover']), f3(v['excl_ols']),
                                                '--' if np.isnan(v['J_reject']) else f3(v['J_reject']), v['na_est']))
        rows.append(v)
pd.DataFrame(rows).to_csv(out('sim_star.csv'), index=False)
