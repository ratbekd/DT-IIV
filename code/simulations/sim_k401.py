"""Does VT-IIV track the truth with a binary outcome and a binary regressor?
Semi-synthetic check on the 401(k) sample (port of sim_k401.R).
Pr(y = 1) = clip(W'g + beta0 x + lambda v), v the part of p401k that eligibility
does not explain. Usage: python sim_k401.py [replications]   (writes sim_k401.csv)"""
import sys, numpy as np, pandas as pd
from _setup import out
from iiv.data import _p
from iiv.dtk import fit_dtiiv, REL_FLOOR
from iiv.sim import mad, nanmedian

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
w_vars = ['inc', 'incsq', 'age', 'agesq', 'marr', 'fsize']
dd = pd.read_csv(_p('k401.csv')).dropna(subset=['pira', 'p401k', 'e401k'] + w_vars).reset_index(drop=True)
n = len(dd); W1 = np.column_stack([np.ones(n), dd[w_vars].values.astype(float)]); x_real = dd.p401k.values.astype(float)
Q = np.linalg.qr(W1)[0]; e401 = dd.e401k.values.astype(float)
XZ = np.column_stack([W1, e401]); v = x_real - XZ @ np.linalg.lstsq(XZ, x_real, rcond=None)[0]
base = W1 @ np.linalg.lstsq(W1, dd.pira.values - 0.017 * x_real, rcond=None)[0]
Zt = e401 - Q @ (Q.T @ e401)
rng = np.random.default_rng(20261007); rows = []
for lam in (-0.15, 0, 0.05, 0.15, 0.30):
    for b0 in (0.017, 0.15):
        p = np.clip(base + b0 * x_real + lam * v, 0.005, 0.995); res = []
        for _ in range(R):
            y = rng.binomial(1, p).astype(float); fit = fit_dtiiv(y, W1, x_real, Q=Q); rt = fit['roots']
            res.append((fit['bols'], np.sum(Zt * fit['y']) / np.sum(Zt * fit['x']), fit['beta'],
                        int(np.sum(rt.in_D & (rt.cor_sx >= REL_FLOOR))) if len(rt) else 0))
        M = np.array(res)
        rows.append(dict(beta0=b0, lambda_=lam, clipped=np.mean((p <= 0.005) | (p >= 0.995)), OLS=np.median(M[:, 0]),
                         TSLS=np.median(M[:, 1]), IIV=nanmedian(M[:, 2]), IIV_mad=mad(M[:, 2]),
                         IIV_minus_OLS=nanmedian(M[:, 2] - M[:, 0]), no_root=np.mean(np.isnan(M[:, 2])), roots_adm=M[:, 3].mean()))
tab = pd.DataFrame(rows).rename(columns={'lambda_': 'lambda'}); tab.to_csv(out('sim_k401.csv'), index=False)
print(tab.round(3).to_string(index=False))
