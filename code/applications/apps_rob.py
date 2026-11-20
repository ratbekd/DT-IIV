"""Bounded mean tendencies on the applications (port of apps_rob.R).
Sets: one-sided {u s^2, u s^3}; bounded {u tanh(z), u exp(-z^2/2)}; both.
Usage: python apps_rob.py [datasets]   (writes apps_rob.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, f3, pieces, zs

datasets = sys.argv[1:] or ['card', 'mroz', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']
rows = []
for ds in datasets:
    C = Context(load(ds)); P = C.prep(); b_iv = C.tsls(P)[0]; th_iv = C.theta_of(b_iv, P); E = 'rob'
    print('\n====', C.lab, ': n =', C.n, ('; %d clusters' % C.G) if C.clustered else '', '; OLS', f3(P.b),
          '; 2SLS', f3(b_iv), '====')
    print('%-10s %7s %7s | %-19s | %-19s %3s %4s | %6s %6s' % ('set', 'beta', 'rho_xu', 'chi2: S-set', 'F(nu): S-set',
                                                             'OLS', '2SLS', 'J chi2', 'J F'))
    thg = np.linspace(-np.pi / 4, np.pi / 4, 241); bg = C.beta_at(thg, P)
    for nm, cols in {'one_sided': [2, 4], 'bounded': [7, 8], 'both': [2, 4, 7, 8]}.items():
        m = len(cols); f = C.fit_cue(P, cols, E)
        S = [C.Q_c(None, P, cols, beta=b, ext=E, return_nu=True) for b in bg]
        a1 = np.array([s <= chi2.ppf(0.95, m) for s, _ in S]); a2 = np.array([s <= C.crit_S(m, v) for s, v in S])
        st = lambda a: ('[%s, %s]%s' % (f3(bg[a].min()), f3(bg[a].max()), '*' if pieces(a) > 1 else '')) if a.any() else 'empty'
        o_in = bool(C.in_S(P, cols, P.b, E)); i_in = bool(abs(th_iv) <= np.pi / 4 and C.in_S(P, cols, b_iv, E))
        yn = lambda v: 'in' if v else 'out'
        print('%-10s %7s %7s | %-19s | %-19s %3s %4s | %6s %6s' % (nm, f3(f['beta']), f3(-np.sin(f['theta'])), st(a1), st(a2),
                                                                 yn(o_in), yn(i_in), f3(f['J_p_chi2']), f3(f['J_p'])))
        rows.append(dict(dataset=ds, set=nm, beta=f['beta'], rho_xu=-np.sin(f['theta']),
                         chi_lo=bg[a1].min() if a1.any() else np.nan, chi_hi=bg[a1].max() if a1.any() else np.nan,
                         chi_pieces=pieces(a1), F_lo=bg[a2].min() if a2.any() else np.nan,
                         F_hi=bg[a2].max() if a2.any() else np.nan, F_pieces=pieces(a2), ols_in=o_in, tsls_in=i_in,
                         J_p_chi2=f['J_p_chi2'], J_p_F=f['J_p']))
    print('(scale of s at OLS: standard deviation 1, robust scale tau %.3f)' % zs(P.xw)[1])
pd.DataFrame(rows).to_csv(out('apps_rob.csv'), index=False)
