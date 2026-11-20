"""S-sets and J-tests under the chi2 and F(nu) critical values, calibrated J
p-values from the null draws of sim_semi_crit.py, and the root screen with
Bonferroni level (port of apps_crit.R).
Usage: python apps_crit.py [datasets]   (writes apps_crit.csv)"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out, OUT
from iiv.data import load
from iiv.core import Context, optimize, uniroot, f3, pieces

datasets = sys.argv[1:] or ['card', 'mroz', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']
A = 0.05; rows = []
for ds in datasets:
    C = Context(load(ds)); P = C.prep(); b_iv = C.tsls(P)[0]; th_iv = C.theta_of(b_iv, P)
    cands = [os.path.join(OUT, '..', 'simulations', 'sim_semi_crit_%s_hom.csv' % ds), out('sim_semi_crit_%s_hom.csv' % ds)]
    f0 = next((c for c in cands if os.path.exists(c)), None); M0 = pd.read_csv(f0) if f0 else None
    calib = lambda J, nm: np.nan if M0 is None else np.nanmean(np.where(M0[nm + '.J'].isna(), np.nan, M0[nm + '.J'] >= J))
    print('\n====', C.lab, ': n =', C.n, ('; %d clusters' % C.G) if C.clustered else '', '; OLS', f3(P.b), '; 2SLS', f3(b_iv), '====')
    print('%-14s %7s | %-18s %3s %4s | %-18s %3s %4s | %6s %6s %6s | %4s' % ('set', 'beta', 'chi2: S-set', 'OLS', '2SLS',
          'F(nu): S-set', 'OLS', '2SLS', 'J chi2', 'J F', 'J cal', 'nu'))
    sets = {'two_sided': [1, 2], 'one_sided': [2, 4], 'four': [1, 2, 3, 4], 'five': [1, 2, 3, 4, 5], 'two_sided_var': [1, 2, 5, 6]}
    thg = np.linspace(-np.pi / 4, np.pi / 4, 241); bg = C.beta_at(thg, P)
    for nm, cols in sets.items():
        m = len(cols); f = C.fit_cue(P, cols)
        S = [C.Q_c(None, P, cols, beta=b, return_nu=True) for b in bg]; nu = np.array([v for _, v in S])
        a1 = np.array([s <= chi2.ppf(1 - A, m) for s, _ in S]); a2 = np.array([s <= C.crit_S(m, v) for s, v in S])

        def inn(b, rule):
            s, v = C.Q_c(None, P, cols, beta=b, return_nu=True)
            return bool(s <= (chi2.ppf(1 - A, m) if rule == 1 else C.crit_S(m, v)))
        a, _ = optimize(lambda a: C.Q_c(a, P, cols), f['theta'] - 0.05, f['theta'] + 0.05)
        J, Jnu = C.Q_c(a, P, cols, return_nu=True)
        st = lambda a: ('[%s, %s]%s' % (f3(bg[a].min()), f3(bg[a].max()), '*' if pieces(a) > 1 else '')) if a.any() else 'empty'
        yn = lambda v: 'in' if v else 'out'
        iv1 = abs(th_iv) <= np.pi / 4 and inn(b_iv, 1); iv2 = abs(th_iv) <= np.pi / 4 and inn(b_iv, 2)
        p1 = 1 - chi2.cdf(J, m - 1); p2 = C.p_S(J, m - 1, Jnu); p3 = calib(J, nm)
        print('%-14s %7s | %-18s %3s %4s | %-18s %3s %4s | %6s %6s %6s | %4.0f' % (nm, f3(f['beta']), st(a1), yn(inn(P.b, 1)),
              yn(iv1), st(a2), yn(inn(P.b, 2)), yn(iv2), f3(p1), f3(p2), f3(p3), np.median(nu)))
        rows.append(dict(dataset=ds, set=nm, m=m, clusters=C.G if C.clustered else np.nan, beta=f['beta'], admissible=np.nan, J=J,
                         chi_lo=bg[a1].min() if a1.any() else np.nan, chi_hi=bg[a1].max() if a1.any() else np.nan,
                         chi_pieces=pieces(a1), chi_ols_in=inn(P.b, 1), chi_tsls_in=bool(iv1), chi_J_p=p1,
                         F_lo=bg[a2].min() if a2.any() else np.nan, F_hi=bg[a2].max() if a2.any() else np.nan,
                         F_pieces=pieces(a2), F_ols_in=inn(P.b, 2), F_tsls_in=bool(iv2), F_J_p=p2, calibrated_J_p=p3,
                         median_nu=np.median(nu)))
    print('(* = the set has more than one piece; the hull is shown)')
    dt_m = lambda a: C.gbar(a, P, [1])[0]
    th = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 2001); m1 = np.array([dt_m(a) for a in th])
    roots = [uniroot(dt_m, th[i], th[i + 1]) for i in np.where(np.diff(np.sign(m1)) != 0)[0]]
    print('Root screen; Bonferroni level', f3(A / len(roots)))
    print('%9s %8s %10s %8s %8s %8s %8s %5s' % ('beta', 'rho_xu', 'admissible', 'J', 'p chi2', 'p F(nu)', 'p cal', 'nu'))
    for r in roots:
        a, _ = optimize(lambda a: C.Q_c(a, P, [1, 3, 4, 5]), max(r - 0.1, -np.pi / 2 + 1e-3), min(r + 0.1, np.pi / 2 - 1e-3))
        J, nu = C.Q_c(a, P, [1, 3, 4, 5], return_nu=True)
        p1 = 1 - chi2.cdf(J, 3); p2 = C.p_S(J, 3, nu); p3 = calib(J, 'screen')
        print('%9s %8s %10s %8s %8s %8s %8s %5.0f' % (f3(C.beta_at(r, P)), f3(-np.sin(r)), 'yes' if abs(r) <= np.pi / 4 else 'no',
                                                     f3(J), f3(p1), f3(p2), f3(p3), nu))
        rows.append(dict(dataset=ds, set='screen', m=4, clusters=C.G if C.clustered else np.nan, beta=C.beta_at(r, P),
                         admissible=abs(r) <= np.pi / 4, J=J, chi_J_p=p1, F_J_p=p2, calibrated_J_p=p3, median_nu=nu))
pd.DataFrame(rows).to_csv(out('apps_crit.csv'), index=False)
