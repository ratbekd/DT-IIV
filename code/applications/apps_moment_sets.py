"""Moment sets on the applications: two-sided, one-sided and five-moment sets
(apps_summary.R), the kurtosis sets (apps_kurtosis.R) and the variance-tendency
sets (apps_variance.R). CUE estimate on |theta| <= pi/4, corrected J-test,
corrected chi2 S-set for beta (with the angle channel), analytic corrected SE.
Usage: python apps_moment_sets.py [summary|kurtosis|variance ...]
Writes apps_summary.csv, apps_kurtosis.csv, apps_variance.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, optimize, pieces

GROUPS = {
    'summary': (['mroz', 'card', 'star', 'banerjee', 'ditella_cl', 'burde_enr', 'ae'],
                {'two_sided': [1, 2], 'one_sided': [2, 4], 'five': [1, 2, 3, 4, 5]}),
    'kurtosis': (['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh'],
                 {'kurt': [3, 4], 'skew_kurt_u': [1, 3], 'all4': [1, 2, 3, 4]}),
    'variance': (['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh'],
                 {'var12': [1, 5], 'var123': [1, 5, 6], 'two_sided': [1, 2], 'two_sided_var': [1, 2, 5, 6]}),
}


def run_ds(ds, sets):
    C = Context(load(ds)); P = C.prep(); n = C.n
    b_iv, _ = C.tsls(P)
    xh = C.Zt @ np.linalg.lstsq(C.Zt, P.xt, rcond=None)[0]
    se_iv = np.sqrt(C.meat(xh * (P.yt - P.xt * b_iv))[0, 0] * n) / abs(np.sum(xh * P.xt))
    rows = []
    for nm, cols in sets.items():
        m = len(cols)
        th = np.linspace(-np.pi / 4, np.pi / 4, 181)
        Qv = [C.Q_u(a, P, cols) for a in th]; i = int(np.argmin(Qv))
        thh, _ = optimize(lambda a: C.Q_u(a, P, cols), th[max(0, i - 1)], th[min(180, i + 1)])
        bh = C.beta_at(thh, P)
        _, J = optimize(lambda a: C.Q_c(a, P, cols), thh - 0.1, thh + 0.1)
        se = C.analytic_se(thh, P, cols)[1]
        thg = np.linspace(-np.pi / 4, np.pi / 4, 241); bg = C.beta_at(thg, P)
        crit = chi2.ppf(0.95, m)
        acc = np.array([C.Q_b(b, P, cols) <= crit for b in bg])
        th_iv = C.theta_of(b_iv, P)
        rows.append(dict(dataset=ds, set=nm, n=n, ols=P.b, tsls=b_iv, tsls_se=se_iv, beta=bh, t=np.tan(thh),
                         cor_sx=np.cos(thh), se=se, J=J, J_df=m - 1,
                         J_p=1 - chi2.cdf(J, m - 1) if m > 1 else np.nan,
                         S_lo=bg[acc].min() if acc.any() else np.nan, S_hi=bg[acc].max() if acc.any() else np.nan,
                         S_pieces=pieces(acc), S_hits_edge=bool(acc[0] or acc[-1]),
                         ols_in_S=C.Q_b(P.b, P, cols) <= crit,
                         tsls_in_S=bool(abs(th_iv) <= np.pi / 4 and C.Q_b(b_iv, P, cols) <= crit)))
    return rows


if __name__ == '__main__':
    groups = sys.argv[1:] or list(GROUPS)
    for gname in groups:
        datasets, sets = GROUPS[gname]; rows = []
        for ds in datasets:
            try:
                rows += run_ds(ds, sets); print('done', ds)
            except FileNotFoundError as e:
                print('skipped', ds, '(data file not found:', e.filename, ')')
        tab = pd.DataFrame(rows); tab.to_csv(out('apps_%s.csv' % gname), index=False)
        show = tab[['dataset', 'set', 'beta', 'se', 'J_p', 'S_lo', 'S_hi', 'S_pieces', 'ols_in_S', 'tsls', 'tsls_in_S']]
        print(show.round(3).to_string(index=False))
