"""Group-covariance (heteroscedasticity) moments h(W) s u on the applications
(port of apps_het.R). One h at a time (variance index, then each control), and
moment sets combining them with the one-sided and two-sided tendencies.
Usage: python apps_het.py [datasets]   (writes apps_het.csv)"""
import sys, numpy as np, pandas as pd
from _setup import out
from iiv.data import load
from iiv.core import Context, f3, comps

datasets = sys.argv[1:] or ['card', 'mroz', 'banerjee', 'maim5v', 'maim4v', 'adh', 'star']
fset = lambda S: 'empty' if np.isnan(S['lo']) else '[%s, %s]' % (f3(S['lo']), f3(S['hi']))
TH = np.linspace(-np.pi / 4, np.pi / 4, 1441)


def S_set(C, P, cols, E):
    """95% set with the adjusted critical value on |theta| <= pi/4: grid of 1,441
    angles, crossings refined by root finding (as in apps_sets.py)."""
    m = len(cols)
    f = lambda t: (lambda S: S[0] - C.crit_S(m, S[1]))(C.Q_c(None, P, cols, beta=C.beta_at(t, P), ext=E, return_nu=True))
    cm = [(C.beta_at(a, P), C.beta_at(b, P)) for a, b, _ in comps(f, TH)]
    return dict(lo=min(a for a, _ in cm) if cm else np.nan, hi=max(b for _, b in cm) if cm else np.nan, pieces=len(cm))
rows = []
for ds in datasets:
    D = load(ds); C = Context(D); P = C.prep(index=True); b_iv = C.tsls(P)[0]; E = 'het'
    print('\n====', C.lab, ': n =', C.n, ('; %d clusters' % C.G) if C.clustered else '', '; OLS', f3(P.b),
          '; 2SLS', f3(b_iv), '====')
    if C.W1.shape[1] > 31:
        print('CAUTION: W holds group effects; the index test over-rejects in this case.')
    if C.loo_singular:
        print('NOTE:', C.loo_singular, 'cluster(s) have no leave-out fit and do not enter the index.')
    print('%-11s %8s %8s %9s %19s %6s' % ('h', 'beta', 'rho_xu', 'signal p', '95% S-set', 'pieces'))
    sg = C.h_signal(P, P.H[:, -1]); S = S_set(C, P, [C.idx_col], E); fi = C.fit_cue(P, [C.idx_col], E)
    print('%-11s %8s %8s %9s %19s %6d' % ('index', f3(fi['beta']), f3(-np.sin(fi['theta'])), f3(sg['p']), fset(S), S['pieces']))
    names = D.get('W1_names')
    for j in range(C.q_fix):
        sg = C.h_signal(P, C.H_fix[:, j]); S = S_set(C, P, [7 + j], E)
        nm = (names[j + 1] if names is not None else 'h%d' % (j + 1))[:11]
        print('%-11s %8s %8s %9s %19s %6d' % (nm, f3(C.beta_at(sg['theta'], P)), f3(-np.sin(sg['theta'])), f3(sg['p']),
                                              fset(S), S['pieces']))
    sets = {'one_sided': [2, 4], 'two_sided': [1, 2], 'het_index': [C.idx_col], 'het_index_one': [2, 4, C.idx_col]}
    if C.q_fix and (not C.clustered or C.q_fix <= C.G / 20):
        sets['het_controls'] = C.h_cols; sets['het_ctl_one'] = [2, 4] + C.h_cols
    elif C.q_fix:
        print('(all-controls sets skipped:', C.q_fix, 'moments with', C.G, 'clusters)')
    print('\n%-14s %3s %8s %19s %6s %7s %7s %8s' % ('set', 'm', 'beta', '95% S-set', 'pieces', 'J p', 'OLS in', '2SLS in'))
    for nm, cols in sets.items():
        m = len(cols); f = C.fit_cue(P, cols, E); S = S_set(C, P, cols, E)
        o_in = bool(C.in_S(P, cols, P.b, E)); i_in = bool(abs(C.theta_of(b_iv, P)) <= np.pi / 4 and C.in_S(P, cols, b_iv, E))
        print('%-14s %3d %8s %19s %6d %7s %7s %8s' % (nm, m, f3(f['beta']), fset(S), S['pieces'],
                                                     f3(f['J_p']) if m > 1 else '--', str(o_in).upper(), str(i_in).upper()))
        rows.append(dict(dataset=ds, set=nm, m=m, n=C.n, clusters=C.G if C.clustered else np.nan, ols=P.b, tsls=b_iv,
                         beta=f['beta'], rho_xu=-np.sin(f['theta']), J_p=f['J_p'], S_lo=S['lo'], S_hi=S['hi'],
                         S_pieces=S['pieces'], ols_in_S=o_in, tsls_in_S=i_in))
pd.DataFrame(rows).to_csv(out('apps_het.csv'), index=False)
