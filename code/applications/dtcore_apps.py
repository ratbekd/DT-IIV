"""VT-IIV estimator core on the Mroz and Card applications: the rows of the
supplement's inference table (naive, corrected and bootstrap standard errors)
and the Card two-regressor table. Settings as in the application drivers of
the R code:
  Mroz  hours (thousands) on lwage; controls educ, age, kidslt6, kidsge6,
        nwifeinc; domain [0.001, 8] in SD units; relevance floor 0.25
  Card  lwage on educ and exper, both endogenous; controls black, smsa, south,
        smsa66, reg662-669, married; domain [0.05, 5] in SD units; floor 0.01;
        external IV nearc2, nearc4, fatheduc, motheduc
Pairs bootstrap, B = 1999 by default, sign fixed at the sample fit, BCa intervals.
Usage: python dtcore_apps.py [mroz|card ...] [--B=1999]   (writes dtcore_apps.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2, norm, f as fdist
from _setup import out
from iiv.data import _first
from iiv.dtcore import run_dt_iiv, siv_scalar_profile

B = next((int(a[4:]) for a in sys.argv[1:] if a.startswith('--B=')), 1999)
apps = [a for a in sys.argv[1:] if not a.startswith('--')] or ['mroz', 'card']


def design(name):
    if name == 'mroz':
        dd = pd.read_csv(_first('mroz.csv', 'mroz_clean.csv')); dd['lwage'] = pd.to_numeric(dd.lwage, errors='coerce')
        dd = dd[(dd.inlf == 1) & dd.lwage.notna()].reset_index(drop=True)
        return dict(dd=dd, y=dd.hours.values / 1000, xn=['lwage'], wn=['educ', 'age', 'kidslt6', 'kidsge6', 'nwifeinc'],
                    zn=['exper', 'expersq'], dom=(0.001, 8), floor=0.25)
    dd = pd.read_csv(_first('card.csv')); dd = dd[dd.married.notna()].reset_index(drop=True)
    dd['married1'] = (dd.married == 1).astype(float)
    return dict(dd=dd, y=dd.lwage.values, xn=['educ', 'exper'],
                wn=['black', 'smsa', 'south', 'smsa66'] + ['reg66%d' % i for i in range(2, 10)] + ['married1'],
                zn=['nearc2', 'nearc4', 'fatheduc', 'motheduc'], dom=(0.05, 5), floor=0.01)


def resid(v, W1):
    return v - W1 @ np.linalg.lstsq(W1, v, rcond=None)[0]


def iv_hc1(y, X, Wc, Z):
    """2SLS of y on X with controls Wc (constant included) and excluded instruments Z:
    HC1 standard errors, first-stage F (classical and robust), Hansen J, RMSE, adjusted R2."""
    n = len(y); W1 = np.column_stack([np.ones(n), Wc]); R = np.column_stack([W1, X]); Zf = np.column_stack([W1, Z])
    PZ = Zf @ np.linalg.lstsq(Zf, R, rcond=None)[0]
    b = np.linalg.solve(PZ.T @ R, PZ.T @ y); e = y - R @ b; kk = R.shape[1]
    A = np.linalg.inv(PZ.T @ R); V = A @ (PZ * e[:, None] ** 2).T @ PZ @ A.T * n / (n - kk)
    p = X.shape[1]; q = Z.shape[1]; se = np.sqrt(np.diag(V))[-p:]
    Fc, Fr = [], []
    for j in range(p):
        xj = X[:, j]; full = np.column_stack([W1, Z]); bf = np.linalg.lstsq(full, xj, rcond=None)[0]; ef = xj - full @ bf
        er = resid(xj, W1); df = n - full.shape[1]
        Fc.append(((er @ er - ef @ ef) / q) / (ef @ ef / df))
        XtXi = np.linalg.inv(full.T @ full); Vr = XtXi @ (full * ef[:, None] ** 2).T @ full @ XtXi * n / df
        bz = bf[-q:]; Fr.append(float(bz @ np.linalg.solve(Vr[-q:, -q:], bz)) / q)
    g = Zf.T @ e / n; Sg = (Zf * e[:, None] ** 2).T @ Zf / n; J = float(n * g @ np.linalg.solve(Sg, g)); dfJ = q - p
    tss = np.sum((y - y.mean()) ** 2); r2a = 1 - (np.sum(e ** 2) / (n - kk)) / (tss / (n - 1))
    return dict(b=b[-p:], se=se, F_classical=np.array(Fc), F_robust=np.array(Fr), J=J, J_df=dfJ,
                J_p=1 - chi2.cdf(J, dfJ) if dfJ > 0 else np.nan, rmse=np.sqrt(np.mean(e ** 2)), adj_r2=r2a, n=n)


rows = []
for name in apps:
    D = design(name); dd = D['dd']; y = D['y']; X = dd[D['xn']].values.astype(float); W = dd[D['wn']].values.astype(float)
    n, p = X.shape; W1 = np.column_stack([np.ones(n), W])
    ols = iv_hc1(y, X, W, X)                      # Z = X gives OLS with HC1
    keep = dd[D["zn"]].notna().all(axis=1).values
    iv = iv_hc1(y[keep], X[keep], W[keep], dd.loc[keep, D['zn']].values.astype(float))
    fit = run_dt_iiv(y, X, W, delta_lower=D['dom'][0], delta_upper=D['dom'][1], scale='sd', relevance_c=D['floor'],
                     inference='both', B=B, n_cores=2, names=D['xn'])
    a, bt = fit['analytic'], fit['bootstrap']
    xt = fit['X_tilde']; yt = fit['y_tilde']; S = fit['S_raw']; u = yt - xt @ fit['beta']
    Q = np.linalg.inv(S.T @ xt); Vn = Q @ (S * u[:, None] ** 2).T @ S @ Q.T * n / (n - W1.shape[1] - p)
    se_naive = np.sqrt(np.diag(Vn)); cor_sx = [np.corrcoef(S[:, j], xt[:, j])[0, 1] for j in range(p)]
    print('\n==== %s: n = %d ; domain [%g, %g] SD units ; relevance floor %g ; B = %d ====' % (
        name, n, D['dom'][0], D['dom'][1], D['floor'], B))
    print('status %s ; sign k = %s ; S-set share of the grid %.0f%%' % (', '.join(fit['status']), fit['k_hat'],
                                                                     100 * fit['gate']['id_set']['frac_inside']))
    if p == 1:
        pr = siv_scalar_profile(fit)
        print('roots on the domain: %d (%d admissible)' % (len(pr['roots']), sum(r['admissible'] for r in pr['roots'])))
    for j, v in enumerate(D['xn']):
        lo_hi = (fit['ci_analytic'][j, 0], fit['ci_analytic'][j, 1])
        print('%-6s OLS %.3f (%.3f) | IV %.3f (%.3f), F %.2f / %.2f | VT-IIV %.3f, delta %.3f, cor(S,x) %.3f' % (
            v, ols['b'][j], ols['se'][j], iv['b'][j], iv['se'][j], iv['F_classical'][j], iv['F_robust'][j], fit['beta'][j],
            fit['delta_hat'][j], cor_sx[j]))
        print('       naive se %.3f (p %.3f) | corrected se %.3f (p %.3f), Wald [%.3f, %.3f] | bootstrap se %.3f (p %.3f), '
              '%s [%.3f, %.3f], draws %d/%d | ratio %.1f | gap %.3f (%.3f)' % (
                  se_naive[j], 2 * norm.cdf(-abs(fit['beta'][j] / se_naive[j])), a['se_beta'][j], fit['p_analytic'][j],
                  *lo_hi, bt['se_beta'][j], bt['p_beta'][j], bt['ci_type_used_beta'][j], bt['ci_beta'][j, 0], bt['ci_beta'][j, 1],
                  bt['n_success'], B, bt['se_beta'][j] / se_naive[j], a['ols_gap']['difference'][j], a['ols_gap']['se'][j]))
        rows.append(dict(app=name, variable=v, n=n, ols=ols['b'][j], ols_se=ols['se'][j], iv=iv['b'][j], iv_se=iv['se'][j],
                         iv_n=iv['n'], F_classical=iv['F_classical'][j], F_robust=iv['F_robust'][j], beta=fit['beta'][j],
                         k=fit['k_hat'][j], delta_sd=fit['delta_hat'][j], cor_sx=cor_sx[j], se_naive=se_naive[j],
                         se_analytic=a['se_beta'][j], wald_lo=lo_hi[0], wald_hi=lo_hi[1], se_boot=bt['se_beta'][j],
                         p_boot=bt['p_beta'][j], boot_lo=bt['ci_beta'][j, 0], boot_hi=bt['ci_beta'][j, 1],
                         draws=bt['n_success'], B=B, gap=a['ols_gap']['difference'][j], gap_se=a['ols_gap']['se'][j],
                         S_share=fit['gate']['id_set']['frac_inside']))
    print('IV: overidentification p (%d df) %.3f ; RMSE OLS %.3f IV %.3f ; adjusted R2 OLS %.3f IV %.3f ; IV n = %d' % (
        iv['J_df'], iv['J_p'], ols['rmse'], iv['rmse'], ols['adj_r2'], iv['adj_r2'], iv['n']))
pd.DataFrame(rows).to_csv(out('dtcore_apps.csv'), index=False)
