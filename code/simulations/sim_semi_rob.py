"""The one-sided and bounded sets in semi-synthetic designs calibrated to an
application (port of sim_semi_rob.R). Designs hom, het and, when clustered, blk.
Usage: python sim_semi_rob.py <dataset> [replications] [beta0]   (writes sim_semi_rob_<ds>.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import Semi, replicate, safe, mad, nanmedian, nanmean, ALPHA

a = sys.argv[1:]
ds = a[0] if a else 'adh'; R = int(a[1]) if len(a) > 1 else 500; b0 = float(a[2]) if len(a) > 2 else None
C = Context(load(ds)); S = Semi(C, beta0=b0); beta0 = S.beta0; E = 'rob'
sets = {'one_sided': [2, 4], 'bounded': [7, 8], 'both': [2, 4, 7, 8]}
KEYS = ['est', 'c_chi', 'c_F', 'excl', 'j_chi', 'j_F']


def one_rep(design):
    def f(rng):
        y, x = S.data(S.draw(design, rng)); P = C.prep(y, x); r = {}
        for nm, cols in sets.items():
            def fit():
                m = len(cols); fc = C.fit_cue(P, cols, E, ngrid=91)
                Sb, nub = C.Q_c(None, P, cols, beta=beta0, ext=E, return_nu=True)
                So, nuo = C.Q_c(None, P, cols, beta=P.b, ext=E, return_nu=True)
                return dict(est=fc['beta'], c_chi=float(Sb <= chi2.ppf(1 - ALPHA, m)), c_F=float(Sb <= C.crit_S(m, nub)),
                            excl=float(So > C.crit_S(m, nuo)), j_chi=float(fc['J'] > chi2.ppf(1 - ALPHA, m - 1)),
                            j_F=float(fc['J_p'] < ALPHA))
            r.update({nm + '.' + k: v for k, v in safe(fit, KEYS).items()})
        return r
    return f


print('\n====', ds, ': n =', C.n, '; clusters', C.G if C.clustered else 'none', '; beta0', f3(beta0), '; OLS', f3(S.b_ols),
      '; replications', R, '====')
rows = []
for i, d in enumerate(['hom', 'het'] + (['blk'] if C.clustered else [])):
    M = pd.DataFrame(replicate(one_rep(d), R, 20261012 + i))
    print('\n---- design', d, '----')
    print('%-10s %8s %7s | %6s %6s | %8s | %6s %6s' % ('set', 'median', 'MAD', 'cover', 'cover', 'excl OLS', 'J rej', 'J rej'))
    print('%-10s %8s %7s | %6s %6s | %8s | %6s %6s' % ('', '', '', 'chi2', 'F(nu)', 'F(nu)', 'chi2', 'F(nu)'))
    for s in sets:
        g = lambda k: M[s + '.' + k]
        v = [nanmedian(g('est')), mad(g('est')), nanmean(g('c_chi')), nanmean(g('c_F')), nanmean(g('excl')), nanmean(g('j_chi')),
             nanmean(g('j_F'))]
        print('%-10s %8s %7s | %6s %6s | %8s | %6s %6s' % ((s,) + tuple(f3(x) for x in v)))
        rows.append(dict(dataset=ds, design=d, set=s, median=v[0], mad=v[1], cover_chi2=v[2], cover_F=v[3], excl_ols=v[4],
                         J_rej_chi2=v[5], J_rej_F=v[6], reps=R))
pd.DataFrame(rows).to_csv(out('sim_semi_rob_%s.csv' % ds), index=False)
