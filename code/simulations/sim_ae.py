"""Does VT-IIV track the truth in the Angrist-Evans sample? (port of sim_ae.R)
Part 1: the real data. Part 2: linear and censored semi-synthetic designs.
Needs ae.csv (not included). Usage: python sim_ae.py [replications]   (writes sim_ae.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from _setup import out
from iiv.data import _p
from iiv.dtk import fit_dtiiv, REL_FLOOR
from iiv.sim import mad, nanmedian

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
dd = pd.read_csv(_p('ae.csv')); w_vars = ['agem1', 'black', 'hispan', 'othrace']
n = len(dd); Wm = dd[w_vars].values.astype(float); W1 = np.column_stack([np.ones(n), Wm])
x = dd.morekids.values.astype(float); y_real = dd.weeksm1.values.astype(float); Q = np.linalg.qr(W1)[0]
ss = dd.samesex.values.astype(float); Zt = ss - Q @ (Q.T @ ss)
tsls_of = lambda ft: np.sum(Zt * ft['y']) / np.sum(Zt * ft['x'])
fit = fit_dtiiv(y_real, W1, x, Q=Q)
print('\n==== Angrist-Evans, real data: n =', n, '====')
print('share at 0 weeks %.3f ; share at 52 weeks %.3f ; mean of morekids %.3f' % (np.mean(y_real == 0), np.mean(y_real == 52), x.mean()))
print('OLS %.3f ; 2SLS (samesex) %.3f' % (fit['bols'], tsls_of(fit)))
print(fit['roots'].round(3).to_string(index=False))
print('standardized third moments:', '; '.join('%s %.3f' % kv for kv in fit['mom'].items()))
XZ = np.column_stack([W1, ss]); v = x - XZ @ np.linalg.lstsq(XZ, x, rcond=None)[0]; v = v / np.std(v, ddof=1)
b_iv = tsls_of(fit); g_lin = np.linalg.lstsq(W1, y_real - b_iv * x, rcond=None)[0]; base_lin = W1 @ g_lin
e_pool = y_real - b_iv * x - base_lin; sd_e = np.std(e_pool, ddof=1)
p0, p52 = np.mean(y_real == 0), np.mean(y_real == 52)
sig = 52 / (norm.ppf(1 - p52) - norm.ppf(p0)); mu = -sig * norm.ppf(p0)
Wc = Wm @ g_lin[1:]; Wc = Wc - Wc.mean()
rng = np.random.default_rng(20261007); rows = []
for ty in ('linear', 'censored'):
    for lam in (-0.1, 0, 0.1, 0.3):
        for b0 in (-6, -20):
            res = []
            for _ in range(R):
                if ty == 'linear':
                    y = base_lin + b0 * x + lam * sd_e * v + rng.choice(e_pool, n, replace=True)
                else:
                    y = np.clip(mu - b0 * x.mean() + Wc + b0 * x + lam * sig * v + sig * rng.standard_normal(n), 0, 52)
                ft = fit_dtiiv(y, W1, x, Q=Q); rt = ft['roots']
                res.append((ft['bols'], tsls_of(ft), ft['beta'], int(np.sum(rt.in_D & (rt.cor_sx >= REL_FLOOR))) if len(rt) else 0))
            M = np.array(res)
            rows.append(dict(design=ty, beta0=b0, **{'lambda': lam}, OLS=np.median(M[:, 0]), TSLS=np.median(M[:, 1]),
                             IIV=nanmedian(M[:, 2]), IIV_mad=mad(M[:, 2]), no_root=np.mean(np.isnan(M[:, 2])), roots_adm=M[:, 3].mean()))
tab = pd.DataFrame(rows); tab.to_csv(out('sim_ae.csv'), index=False)
print(tab.round(2).to_string(index=False))
