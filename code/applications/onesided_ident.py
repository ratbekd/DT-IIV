"""Identification check for the one-sided set {E[u s^2], E[u s^3]} (port of
onesided_ident.R): component skewness and kurtosis at the one-sided estimate,
the spurious roots they imply, the zeros of the two sample moments, and the
pieces of the corrected 95% S-set on the whole circle.
Usage: python onesided_ident.py   (writes onesided_ident.csv)"""
import numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, optimize, uniroot, f3

rows = []
wrap = lambda a: np.mod(a + np.pi / 2, np.pi) - np.pi / 2
for ds in ['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']:
    C = Context(load(ds)); P = C.prep(); bet = lambda a: C.beta_at(a, P)
    g = np.linspace(-np.pi / 4, np.pi / 4, 181); Qv = [C.Q_u(a, P, [2, 4]) for a in g]; i = int(np.argmin(Qv))
    th, _ = optimize(lambda a: C.Q_u(a, P, [2, 4]), g[max(0, i - 1)], g[min(180, i + 1)])
    s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
    k3s, k3u, k4s, k4u = np.mean(s ** 3), np.mean(u ** 3), np.mean(s ** 4) - 3, np.mean(u ** 4) - 3
    ph1 = np.arctan(k3s / k3u); ph2 = np.sign(ph1) * np.arctan(np.sqrt(k4s / k4u)) if k4s / k4u > 0 else np.nan
    cir = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 2001)

    def zeros(col):
        f = lambda a: C.gbar(a, P, [col])[0]; m = np.array([f(a) for a in cir])
        return np.array([uniroot(f, cir[j], cir[j + 1]) for j in np.where(np.diff(np.sign(m)) != 0)[0]])
    z2, z4 = zeros(2), zeros(4)
    e = np.mod(np.abs(z2[:, None] - z4[None, :]), np.pi); d = np.minimum(e, np.pi - e)
    far = (np.abs(wrap(z2 - th)) > 0.05)[:, None] & (np.abs(wrap(z4 - th)) > 0.05)[None, :]
    dmin = d[far].min() if far.any() else np.nan
    wm = np.argwhere((d == dmin) & far)[0] if far.any() else None
    acc = np.array([C.Q_b(bet(a), P, [2, 4]) <= chi2.ppf(0.95, 2) for a in cir]).astype(int)
    st = np.where(np.diff(np.r_[0, acc]) == 1)[0]; en = np.where(np.diff(np.r_[acc, 0]) == -1)[0]
    print('\n====', ds, '==== one-sided beta', f3(bet(th)), '(OLS', f3(P.b), ')')
    print('zeros of mean(u s^2), beta:', ', '.join(f3(v) for v in bet(z2)))
    print('zeros of mean(u s^3), beta:', ', '.join(f3(v) for v in bet(z4)))
    print('95% S-set pieces (whole circle):', ' '.join('[%s, %s]' % (f3(bet(cir[a])), f3(bet(cir[b]))) for a, b in zip(st, en)))
    rows.append(dict(dataset=ds, beta_one=bet(th), k3s=k3s, k3u=k3u, k4s=k4s, k4u=k4u, ratio_s=k3s ** 2 / k4s,
                     ratio_u=k3u ** 2 / k4u, spur_us2=bet(wrap(th + ph1)),
                     spur_us3=np.nan if np.isnan(ph2) else bet(wrap(th + ph2)),
                     gap_rad=np.nan if np.isnan(ph2) else abs(ph1 - ph2), nearest_shared_gap=dmin,
                     shared_beta=np.nan if wm is None else bet(z2[wm[0]]),
                     shared_rho=np.nan if wm is None else -np.sin(z2[wm[0]]), n_pieces=len(st)))
tab = pd.DataFrame(rows); tab.to_csv(out('onesided_ident.csv'), index=False)
print(); print(tab.round(3).to_string(index=False))
