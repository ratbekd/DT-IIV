"""VT-IIV on the Mroz (1987) labour-supply data (port of mroz_DT_IIV_v22.R).
hours (thousands) on lwage; controls educ, age, kidslt6, kidsge6, nwifeinc;
comparison IV exper, expersq. Stages: screen, closed-form roots, estimate,
corrected three-step SE, S-set, pairs bootstrap with the sign fixed, BCa interval.
Usage: python mroz_dtiiv.py [B]   (default 1999 replications, seed 20260924)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2, norm
import _setup
from iiv.data import _first
from iiv.dtk import M_n, moments_raw, fit_dtiiv, D_SD, REL_FLOOR

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1999
n_grid = 300; alpha = 0.05
dd = pd.read_csv(_first('mroz.csv', 'mroz_clean.csv'))
dd['lwage'] = pd.to_numeric(dd.lwage, errors='coerce'); dd['hours'] = dd.hours / 1000
dd = dd[(dd.inlf == 1) & dd.lwage.notna()].reset_index(drop=True)
Wn = ['educ', 'age', 'kidslt6', 'kidsge6', 'nwifeinc']
Xm = dd.lwage.values; ym = dd.hours.values; n = len(dd)
W1 = np.column_stack([np.ones(n), dd[Wn].values.astype(float)]); Zm = dd[['exper', 'expersq']].values.astype(float)
res = lambda v: v - W1 @ np.linalg.lstsq(W1, v, rcond=None)[0]

fit = fit_dtiiv(ym, W1, Xm); x, yt, r = fit['x'], fit['y'], fit['r']; k_ctrl = len(Wn) + 1
se_ols = np.sqrt(np.sum(x ** 2 * (yt - x * fit['bols']) ** 2)) / np.sum(x ** 2) * np.sqrt(n / (n - k_ctrl - 1))
Zt = res(Zm); pi_h = np.linalg.lstsq(Zt, x, rcond=None)[0]; xh = Zt @ pi_h
b_iv = np.sum(xh * yt) / np.sum(xh * x); u_iv = yt - x * b_iv
se_iv = np.sqrt(np.sum(xh ** 2 * u_iv ** 2)) / abs(np.sum(xh * x)) * np.sqrt(n / (n - k_ctrl - 1))
v1 = x - xh; q = Zm.shape[1]; df1 = n - k_ctrl - q
F_cl = (np.sum(xh ** 2) / q) / (np.sum(v1 ** 2) / df1)
ZZi = np.linalg.inv(Zt.T @ Zt); V_rob = ZZi @ (Zt * v1[:, None] ** 2).T @ Zt @ ZZi * n / df1
F_rb = float(pi_h @ np.linalg.solve(V_rob, pi_h)) / q
Z1 = np.column_stack([np.ones(n), Zt]); e = u_iv - Z1 @ np.linalg.lstsq(Z1, u_iv, rcond=None)[0]
sarg = n * (1 - np.sum(e ** 2) / np.sum((u_iv - u_iv.mean()) ** 2)); p_sarg = 1 - chi2.cdf(sarg, q - 1)
y_raw = dd.hours.values
skew_y = np.mean((y_raw - y_raw.mean()) ** 3) / np.std(y_raw, ddof=1) ** 3
mo = moments_raw(x, r); xrate = np.sqrt(mo['B'] / mo['A']); b_hat = fit['mom']['mu03'] / fit['mom']['mu30']

rt = fit['roots']; sdx, sdr = np.sqrt(mo['A']), np.sqrt(mo['B'])
dgrid = np.exp(np.linspace(np.log(D_SD[0]), np.log(D_SD[1]), n_grid))
n_cross = {k: int(np.sum(np.diff(np.sign([M_n(k * d * sdx / sdr, x, r) for d in dgrid])) != 0)) for k in (-1, 1)}
cf_check = max(abs(M_n(t, x, r)) for t in rt.tau)
sel = fit['sel']; k_hat = int(sel.k); tau_h = sel.tau
s_hat = x + tau_h * r
b_iiv_iv = np.sum(s_hat * yt) / np.sum(s_hat * x); b_iiv = fit['bols'] + tau_h * mo['B'] / mo['A']
tau_til = -mo['A'] ** 2 * mo['m03'] / (mo['B'] ** 2 * mo['m30'])
beta_til = fit['bols'] + tau_til * mo['B'] / mo['A']; dsd_til = abs(tau_til) * sdr / sdx


def corrected_scores(tau, beta_iiv):
    A, Bv = mo['A'], mo['B']; s = x + tau * r; D = np.mean(s ** 2)
    g = np.sum(x * s) / np.sum(s * s); ee = x - g * s; m = (ee ** 2 - np.mean(ee ** 2)) * s; h = 1e-6
    J_tau = (M_n(tau + h, x, r) - M_n(tau - h, x, r)) / (2 * h)
    J_b = (M_n(tau, x, r - x * h) - M_n(tau, x, r + x * h)) / (2 * h)
    J_b_cf = -tau ** 3 / D ** 2 * (tau ** 2 * Bv ** 2 * mo['m30'] - 4 * tau * A * Bv * mo['m21']
                                   + (3 * A ** 2 - 2 * tau ** 2 * A * Bv) * mo['m12'] + 2 * tau * A ** 2 * mo['m03'])
    J_G = -2 * np.mean(ee * s ** 2)
    mt = m + J_b * (x * r / A) + J_G * (s * ee / D)
    gt = s * (yt - x * beta_iiv) - tau * (x * (fit['bols'] - beta_iiv)) * r
    return dict(mt=mt, psi=gt / A - Bv / (A * J_tau) * mt, J_tau=J_tau, J_b=J_b, J_b_cf=J_b_cf, m=m)


cs = corrected_scores(tau_h, b_iiv); se_corr = np.sqrt(np.mean(cs['psi'] ** 2) / n)
psi_unc = s_hat * (yt - x * b_iiv) / mo['A'] - mo['B'] / (mo['A'] * cs['J_tau']) * cs['m']
se_unc = np.sqrt(np.mean(psi_unc ** 2) / n)
se_naive = np.sqrt(np.sum(s_hat ** 2 * (yt - x * b_iiv) ** 2)) / abs(np.sum(s_hat * x)) * np.sqrt(n / (n - k_ctrl - 1))
se_delta_sd = np.sqrt(np.mean((cs['mt'] / cs['J_tau']) ** 2) / n) * sdr / sdx
S_stat = []
for d in dgrid:
    tau = k_hat * d * sdx / sdr; cc = corrected_scores(tau, fit['bols'] + tau * mo['B'] / mo['A'])
    S_stat.append(n * np.mean(cc['m']) ** 2 / np.var(cc['mt'], ddof=1))
keep = np.array(S_stat) <= chi2.ppf(1 - alpha, 1)
beta_grid = fit['bols'] + k_hat * dgrid * sdx / sdr * mo['B'] / mo['A']
S_runs = int(np.sum(np.diff(np.r_[0, keep.astype(int), 0]) == 1))

rng = np.random.default_rng(20260924); boot = np.full((B, 7), np.nan); nroots = np.zeros(B, int)
for b in range(B):
    i = rng.integers(0, n, n); fb = fit_dtiiv(ym[i], W1[i], Xm[i], k_fix=k_hat)
    boot[b, 2:6] = list(fb['mom'].values())
    rtb = fb['roots']; nroots[b] = int(np.sum(rtb.in_D & (rtb.k == k_hat))) if len(rtb) else 0
    if fb['sel'] is not None:
        boot[b, [0, 1, 6]] = fb['sel'].beta, fb['sel'].delta_sd, fb['sel'].beta - fb['bols']
ok = ~np.isnan(boot[:, 0]); bb = boot[ok, 0]
se_boot = np.std(bb, ddof=1); ci_pct = np.quantile(bb, [alpha / 2, 1 - alpha / 2])
idx = np.arange(n); jk = np.array([fit_dtiiv(ym[idx != i], W1[idx != i], Xm[idx != i], k_fix=k_hat)['beta'] for i in range(n)])
jk = jk[~np.isnan(jk)]
z0 = norm.ppf(np.mean(bb < b_iiv)); a_ = np.sum((jk.mean() - jk) ** 3) / (6 * np.sum((jk.mean() - jk) ** 2) ** 1.5)
za = norm.ppf([alpha / 2, 1 - alpha / 2]); ci_bca = np.quantile(bb, norm.cdf(z0 + (z0 + za) / (1 - a_ * (z0 + za))))
t_mom = np.array(list(fit['mom'].values())) / np.nanstd(boot[:, 2:6], 0, ddof=1)

f3 = lambda v: '%.3f' % v
print('\n==== VT-IIV, Mroz: hours (thousands) on lwage, n =', n, '====')
print('\nStage 1: screening')
for (nm, v), t in zip(fit['mom'].items(), t_mom):
    print('  %-5s = %7s  (t = %6s)' % (nm, f3(v), f3(t)))
print('  b-hat = mu03/mu30 = %s ; exchange rate sd(r)/sd(x) = %s ; se(OLS)/rate = %.4f' % (f3(b_hat), f3(xrate), se_ols / xrate))
print('  outcome: distinct = %d ; min = %s ; max = %s ; share at min/max = %s / %s ; skewness = %s' % (
    len(np.unique(y_raw)), f3(y_raw.min()), f3(y_raw.max()), f3(np.mean(y_raw == y_raw.min())), f3(np.mean(y_raw == y_raw.max())), f3(skew_y)))
print('\nStage 2: roots of the VT moment (closed-form cubic)')
show = rt.copy(); show[['tau', 'delta_raw']] = show[['tau', 'delta_raw']].round(4)
show[['delta_sd', 'cor_sx', 'beta']] = show[['delta_sd', 'cor_sx', 'beta']].round(3)
print(show.to_string(index=False))
print('  grid sign changes, k = -1: %d ; k = +1: %d ; max |M| at closed-form roots = %.2g' % (n_cross[-1], n_cross[1], cf_check))
print('  selected: k = %d , delta_sd = %s , cor(s,x) = %s  (floor %s )' % (k_hat, f3(sel.delta_sd), f3(sel.cor_sx), REL_FLOOR))
print('\nStage 3: estimates')
print('  OLS            %s  (HC1 %s )' % (f3(fit['bols']), f3(se_ols)))
print('  2SLS exper,exper^2 %s  (HC1 %s );  F classical %s , F robust %s , Sargan p %s' % (f3(b_iv), f3(se_iv), f3(F_cl), f3(F_rb), f3(p_sarg)))
print('  VT-IIV         %s   [IV form %s ]' % (f3(b_iiv), f3(b_iiv_iv)))
print('  sensitivity (mixed moments = 0): delta_sd = %s , beta = %s' % (f3(dsd_til), f3(beta_til)))
print('\nStage 4: inference for VT-IIV')
print('  naive SE (s-hat as data)   %s' % f3(se_naive))
print('  uncorrected two-step SE    %s' % f3(se_unc))
print('  corrected three-step SE    %s   (J_b numeric %.4g vs closed form %.4g )' % (f3(se_corr), cs['J_b'], cs['J_b_cf']))
print('  pairs bootstrap SE         %s   admissible %d / %d ; resamples with 2+ roots in D at k-hat: %s' % (
    f3(se_boot), ok.sum(), B, f3(np.mean(nroots >= 2))))
print('  95%% percentile interval    [ %s , %s ]' % (f3(ci_pct[0]), f3(ci_pct[1])))
print('  95%% BCa interval           [ %s , %s ]  (z0 = %s , a = %s )' % (f3(ci_bca[0]), f3(ci_bca[1]), f3(z0), f3(a_)))
print('  95%% corrected Wald         [ %s , %s ]' % (f3(b_iiv - 1.96 * se_corr), f3(b_iiv + 1.96 * se_corr)))
print('  delta_sd = %s  corrected SE %s' % (f3(sel.delta_sd), f3(se_delta_sd)))
print('  S-set: share of D = %s ; pieces = %d ; beta range [ %s , %s ]' % (f3(keep.mean()), S_runs, f3(beta_grid[keep].min()), f3(beta_grid[keep].max())))
print('  IIV - OLS gap = %s  (corrected SE %s ; bootstrap SE %s )' % (f3(b_iiv - fit['bols']), f3(se_corr), f3(np.nanstd(boot[ok, 6], ddof=1))))
