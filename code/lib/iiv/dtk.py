"""Closed-form VT-root helpers of dtiiv_k401.R (fit_dtiiv and friends), shared by
dtiiv_k401.py, sim_k401.py and sim_ae.py."""
import numpy as np, pandas as pd

D_SD = (0.001, 8); REL_FLOOR = 0.25


def resid_on(v, W1):
    """Residual of v on W1 (W1 already includes the constant)."""
    return v - W1 @ np.linalg.lstsq(W1, v, rcond=None)[0]


def M_n(tau, x, r):
    s = x + tau * r; g = np.sum(x * s) / np.sum(s * s); e = x - g * s
    return np.mean((e ** 2 - np.mean(e ** 2)) * s)


def moments_raw(x, r):
    return dict(A=np.mean(x ** 2), B=np.mean(r ** 2), m30=np.mean(x ** 3), m21=np.mean(x ** 2 * r),
                m12=np.mean(x * r ** 2), m03=np.mean(r ** 3))


def P_roots(mo):
    A, B = mo['A'], mo['B']
    cf = [A ** 2 * mo['m12'], A ** 2 * mo['m03'] - 2 * A * B * mo['m21'], B ** 2 * mo['m30'] - 2 * A * B * mo['m12'],
          B ** 2 * mo['m21']]
    z = np.roots(cf[::-1]); return np.sort(np.real(z[np.abs(np.imag(z)) < 1e-8]))


def root_table(x, r, y, D_sd=D_SD):
    mo = moments_raw(x, r); sdx, sdr = np.sqrt(mo['A']), np.sqrt(mo['B']); bols = np.sum(x * y) / np.sum(x * x)
    tau = P_roots(mo); tau = tau[np.abs(tau) > 1e-12]
    d_sd = np.abs(tau) * sdr / sdx
    return pd.DataFrame(dict(tau=tau, k=np.sign(tau).astype(int), delta_raw=np.abs(tau), delta_sd=d_sd,
                             cor_sx=1 / np.sqrt(1 + d_sd ** 2), beta=bols + tau * mo['B'] / mo['A'],
                             in_D=(d_sd >= D_sd[0]) & (d_sd <= D_sd[1])))


def select_root(rt, k_fix=None, rel_floor=REL_FLOOR):
    """Fixed sign (optional), inside D, above the relevance floor, then the highest relevance."""
    if not len(rt):
        return None
    ok = rt.in_D & (rt.cor_sx >= rel_floor)
    if k_fix is not None:
        ok &= rt.k == k_fix
    if not ok.any():
        return None
    cand = rt[ok]; return cand.loc[cand.cor_sx.idxmax()]


def std_moments(x, r):
    zx = x / np.sqrt(np.mean(x ** 2)); zr = r / np.sqrt(np.mean(r ** 2))
    return dict(mu30=np.mean(zx ** 3), mu21=np.mean(zx ** 2 * zr), mu12=np.mean(zx * zr ** 2), mu03=np.mean(zr ** 3))


def fit_dtiiv(y, W1, X, k_fix=None, Q=None):
    """Full DT-IIV fit. W1 includes the constant. Q, an orthonormal basis of W1,
    speeds up repeated fits on the same controls."""
    if Q is not None:
        res = lambda v: v - Q @ (Q.T @ v)
    else:
        res = lambda v: resid_on(v, W1)
    x = res(X); yt = res(y); bols = np.sum(x * yt) / np.sum(x * x); r = yt - x * bols
    rt = root_table(x, r, yt); sel = select_root(rt, k_fix)
    return dict(x=x, y=yt, r=r, roots=rt, sel=sel, bols=bols, beta=np.nan if sel is None else sel.beta,
                mom=std_moments(x, r))
