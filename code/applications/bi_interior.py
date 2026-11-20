"""Banerjee-Iyer: do the bounds of the outcome drive the results? (supplement,
Subsection S4.2). Re-runs the third moments, the VT root and screen, the
one-sided, two-sided, four-moment and bounded sets, and the covariance tendency
(variance index) on the full sample and on the district-years with an outcome
strictly between 0 and 1.
Usage: python bi_interior.py    (writes bi_interior.csv)"""
import numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, comps, f3, optimize, uniroot

N = 1441
th = np.linspace(-np.pi / 4, np.pi / 4, N)


def show(cm):
    return 'empty' if not cm else ' U '.join('[%s, %s]%s' % (f3(a), f3(b), 'b' if e else '') for a, b, e in cm)


def sets(C, P, cols, ext='base'):
    m = len(cols)
    Sb = lambda t: C.Q_c(None, P, cols, beta=C.beta_at(t, P), ext=ext, return_nu=True)
    tob = lambda cm: [(C.beta_at(a, P), C.beta_at(b, P), e) for a, b, e in cm]
    c1 = tob(comps(lambda t: Sb(t)[0] - chi2.ppf(0.95, m), th))
    c2 = tob(comps(lambda t: (lambda S: S[0] - C.crit_S(m, S[1]))(Sb(t)), th))
    return c1, c2


def mean_t(C, v):
    """Mean of v and its cluster-robust t-statistic."""
    m = v.mean(); se = np.sqrt(C.meat(v)[0, 0] / C.n); return m, m / se


rows = []
for ds in ('banerjee', 'banerjee_int'):
    D = load(ds); C = Context(D); P = C.prep(index=True); b_iv, se_iv = C.tsls(P)
    xo = P.xw; ro = P.rw
    print('\n==== %s : n = %d ; %d districts ; OLS %s ; 2SLS %s (%s) ====' % (C.lab, C.n, C.G, f3(P.b), f3(b_iv), f3(se_iv)))
    r = dict(sample=ds, n=C.n, G=C.G, ols=P.b, tsls=b_iv, tsls_se=se_iv)
    mom = {'mu30': xo ** 3, 'mu21': xo ** 2 * ro, 'mu12': xo * ro ** 2, 'mu03': ro ** 3}
    print('  third moments (cluster t):', '  '.join('%s %s (%.2f)' % (k, f3(mean_t(C, v)[0]), mean_t(C, v)[1]) for k, v in mom.items()))
    for k, v in mom.items():
        r[k], r[k + '_t'] = mean_t(C, v)
    # VT roots in the labelling domain and the screen
    dt = lambda t: C.gbar(t, P, [1])[0]
    v = np.array([dt(t) for t in th]); ix = np.where(np.diff(np.sign(v)) != 0)[0]
    roots = []
    for i in ix:
        rt = uniroot(dt, th[i], th[i + 1], tol=1e-12)
        a, _ = optimize(lambda a: C.Q_c(a, P, [1, 3, 4, 5]), max(rt - 0.1, -np.pi / 2 + 1e-3), min(rt + 0.1, np.pi / 2 - 1e-3))
        roots.append((C.beta_at(rt, P), -np.sin(rt), 1 - chi2.cdf(C.Q_c(a, P, [1, 3, 4, 5]), 3)))
    print('  VT roots in domain:', ', '.join('beta %s (rho %s, screen p %s)' % tuple(map(f3, q)) for q in roots) or 'none')
    r['vt_roots'] = '; '.join('%.3f' % q[0] for q in roots)
    for nm, cols, ext in (('one_sided', [2, 4], 'base'), ('two_sided', [1, 2], 'base'), ('four', [1, 2, 3, 4], 'base'),
                          ('bounded', [7, 8], 'rob'), ('index', [C.idx_col], 'het'), ('index_one', [2, 4, C.idx_col], 'het')):
        fit = C.fit_cue(P, cols, ext=ext); c1, c2 = sets(C, P, cols, ext)
        ins = lambda b: abs(C.theta_of(b, P)) <= np.pi / 4 and bool(C.in_S(P, cols, b, ext, rule='chi2'))
        print('  %-10s m=%d CUE %s  J p %s | chi2 %s | adj %s | OLS in %s, 2SLS in %s' % (
            nm, len(cols), f3(fit['beta']), f3(fit['J_p_chi2']) if len(cols) > 1 else '--', show(c1), show(c2),
            ins(P.b), ins(b_iv)))
        r.update({nm + '_cue': fit['beta'], nm + '_Jp': fit['J_p_chi2'], nm + '_chi2': show(c1), nm + '_adj': show(c2),
                  nm + '_ols_in': ins(P.b), nm + '_tsls_in': ins(b_iv)})
    sg = C.h_signal(P, P.H[:, -1]); r['index_signal_p'] = sg['p']
    names = D.get('W1_names') or []
    single = []
    for j, nm in enumerate(names[1:C.q_fix + 1]):
        if nm in ('alt', 'lat', 'coastal'):
            s_ = C.h_signal(P, C.H_fix[:, j]); single.append('%s %s (p %s)' % (nm, f3(C.beta_at(s_['theta'], P)), f3(s_['p'])))
    print('  index signal p %s ; single controls: %s' % (f3(sg['p']), ', '.join(single)))
    r['single_controls'] = '; '.join(single)
    rows.append(r)
pd.DataFrame(rows).to_csv(out('bi_interior.csv'), index=False)
