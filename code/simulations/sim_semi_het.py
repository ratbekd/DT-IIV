"""Variance-index and group-covariance moments in semi-synthetic designs calibrated
to an application (port of sim_semi_het.R). Designs: hom; hetW (error variance
following the real residual variance given W); blk (cluster residual paths) when
clustered. Optional single control h_one (for example t2 in ADH).
Usage: python sim_semi_het.py <dataset> [replications] [h_one|NA] [beta0]
Writes sim_semi_het_<ds>.csv."""
import sys, numpy as np, pandas as pd
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import Semi, replicate, safe, mad, nanmedian, nanmean, ALPHA

a = sys.argv[1:]
ds = a[0] if a else 'card'; R = int(a[1]) if len(a) > 1 else 300
h_one = a[2] if len(a) > 2 and a[2] != 'NA' else None; b0 = float(a[3]) if len(a) > 3 else None
D = load(ds); C = Context(D); S = Semi(C, beta0=b0); beta0 = S.beta0; E = 'het'
sets = {'one_sided': [2, 4], 'het_index': [C.idx_col], 'het_index_one': [2, 4, C.idx_col]}
if C.q_fix and (not C.clustered or C.q_fix <= C.G / 20):
    sets['het_controls'] = C.h_cols
if h_one is not None:
    j1 = 7 + D['W1_names'][1:].index(h_one)
    sets['het_' + h_one] = [j1]; sets['het_' + h_one + '_one'] = [2, 4, j1]
KEYS = ['est', 'cover', 'excl_ols', 'J_rej']


def one_rep(design):
    def f(rng):
        eps = S.draw('blk' if design == 'blk' else 'hom', rng)
        u = np.sqrt(S.v_w) * eps if design == 'hetW' else eps
        y, x = S.data(u); P = C.prep(y, x, index=True); r = dict(ols=P.b, tsls=C.tsls(P)[0])
        for nm, cols in sets.items():
            def fit():
                fc = C.fit_cue(P, cols, E, ngrid=91)
                return dict(est=fc['beta'], cover=float(C.in_S(P, cols, beta0, E)), excl_ols=float(not C.in_S(P, cols, P.b, E)),
                            J_rej=np.nan if np.isnan(fc['J_p']) else float(fc['J_p'] < ALPHA))
            r.update({nm + '.' + k: v for k, v in safe(fit, KEYS).items()})
        return r
    return f


print('\n====', ds, ': n =', C.n, ('; %d clusters' % C.G) if C.clustered else '', '; real OLS', f3(S.b_ols),
      '; beta0 = real 2SLS', f3(beta0), '; gamma', f3(S.gamma), '; replications', R, '====')
rows = []
for d in ['hom', 'hetW'] + (['blk'] if C.clustered else []):
    M = pd.DataFrame(replicate(one_rep(d), R, 20261009 + len(rows)))
    print('\n---- design', d, ': median OLS', f3(np.median(M.ols)), '; median 2SLS', f3(np.median(M.tsls)), '----')
    print('%-15s %8s %8s %8s %9s %9s' % ('set', 'median', 'MAD', 'cover', 'excl OLS', 'J reject'))
    for s, cols in sets.items():
        v = dict(dataset=ds, design=d, set=s, m=len(cols), median=nanmedian(M[s + '.est']), mad=mad(M[s + '.est']),
                 cover=nanmean(M[s + '.cover']), excl_ols=nanmean(M[s + '.excl_ols']), J_reject=nanmean(M[s + '.J_rej']))
        print('%-15s %8s %8s %8s %9s %9s' % (s, f3(v['median']), f3(v['mad']), f3(v['cover']), f3(v['excl_ols']),
                                            '--' if np.isnan(v['J_reject']) else f3(v['J_reject'])))
        rows.append(v)
pd.DataFrame(rows).to_csv(out('sim_semi_het_%s.csv' % ds), index=False)
