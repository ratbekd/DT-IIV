"""Optimal (sieve) mean tendencies and the cross-fitted one-degree-of-freedom test.

Moments: g_i(theta) = u_i(theta) * B(s_i(theta)) for a basis B.
Corrected for first-step estimation exactly as in eqs. (27)-(29) of the paper,
including the angle channel, by the same chain-rule code as oneside.py.
"""
import numpy as np
from scipy.stats import chi2, norm
from oneside import OneSided

def B_poly(s):  return np.column_stack([s**2, s**3])
def B_sieve(s):
    e = np.exp(-s**2 / 2)
    return np.column_stack([s**2, s**3, np.tanh(s), e, s * e])
def B_bounded(s):
    e = np.exp(-s**2 / 2)
    return np.column_stack([np.tanh(s), np.tanh(s / 2), e, s * e])

class Tend(OneSided):
    def __init__(self, y, x, W, cl=None, basis=B_sieve):
        super().__init__(y, x, W, cl)
        self.B = basis
        k = self.k
        px, py = self.eta[:k], self.eta[k:2*k]
        self.xt = self.x - self.W @ px; self.yt = self.y - self.W @ py
        if self.cl is not None:
            _, self.inv = np.unique(self.cl, return_inverse=True)
            self.G = self.inv.max() + 1
        else:
            self.inv = None; self.G = self.n

    def _gth(self, th, xt, yt, b, sx, sr):
        r = yt - b * xt; xw, rw = xt / sx, r / sr
        s = np.cos(th) * xw + np.sin(th) * rw
        u = np.cos(th) * rw - np.sin(th) * xw
        return u[:, None] * self.B(s)

    def _gb(self, b0, xt, yt, b, sx, sr):
        return self._gth(np.arctan((b0 - b) * sx / sr), xt, yt, b, sx, sr)

    def parts(self, b0, h=1e-6):
        """Mean moment, cluster sums of centred corrected contributions, and dgbar/dtheta."""
        b, sx, sr = self.eta[-3:]; xt, yt = self.xt, self.yt
        g = self._gb(b0, xt, yt, b, sx, sr); gbar = g.mean(0)
        dgx = (self._gb(b0, xt + h, yt, b, sx, sr) - self._gb(b0, xt - h, yt, b, sx, sr)) / (2*h)
        dgy = (self._gb(b0, xt, yt + h, b, sx, sr) - self._gb(b0, xt, yt - h, b, sx, sr)) / (2*h)
        Gs = []
        for j, val in enumerate((b, sx, sr)):
            st = h * max(1.0, abs(val)); a1 = [b, sx, sr]; a2 = [b, sx, sr]
            a1[j] += st; a2[j] -= st
            Gs.append((self._gb(b0, xt, yt, *a1).mean(0) - self._gb(b0, xt, yt, *a2).mean(0)) / (2*st))
        Geta = np.column_stack([-dgx.T @ self.W / self.n, -dgy.T @ self.W / self.n, np.column_stack(Gs)])
        gt = g + self.psi @ Geta.T
        gt = gt - gt.mean(0)
        if self.inv is not None:
            gt = np.array([np.bincount(self.inv, weights=gt[:, m], minlength=self.G) for m in range(gt.shape[1])]).T
        th = np.arctan((b0 - b) * sx / sr)
        D = (self._gth(th + 1e-5, xt, yt, b, sx, sr).mean(0) - self._gth(th - 1e-5, xt, yt, b, sx, sr).mean(0)) / 2e-5
        return gbar, gt, D

    def S(self, b0):
        gbar, gt, _ = self.parts(b0)
        Om = gt.T @ gt / self.n
        return self.n * gbar @ np.linalg.solve(Om, gbar)

    def info(self, b0):
        """Information index I(b0) = D' Omega^{-1} D per observation, and the slope
        distance at which a one-degree-of-freedom 5% test has 80% power."""
        gbar, gt, D = self.parts(b0)
        Om = gt.T @ gt / self.n
        I = D @ np.linalg.solve(Om, D)
        b, sx, sr = self.eta[-3:]
        th = np.arctan((b0 - b) * sx / sr)
        db_dth = (sr / sx) / np.cos(th)**2
        mde = (norm.ppf(.975) + norm.ppf(.8)) / np.sqrt(self.n * I) * db_dth
        return I, mde

def folds_by_cluster(n, cl, K=2, seed=12345):
    rng = np.random.default_rng(seed)
    ids = np.arange(n) if cl is None else np.asarray(cl)
    u = np.unique(ids); perm = rng.permutation(len(u))
    lab = dict(zip(u[perm], np.arange(len(u)) % K))
    return np.array([lab[i] for i in ids])

class CrossFit:
    """Cross-fitted 1-df test. In fold k the direction c_k = Omega_{-k}^{-1} D_{-k}
    is estimated from the other folds only (their own first step), and
    Z_k = sqrt(n_k) c_k' gbar_k / sqrt(c_k' Omega_k c_k); T = (sum_k w_k Z_k)^2,
    w_k = sqrt(n_k/n), is asymptotically chi2_1 under H0."""
    def __init__(self, y, x, W, cl=None, basis=B_sieve, K=2, seed=12345, design=None):
        y, x, W = np.asarray(y), np.asarray(x), np.asarray(W)
        f = folds_by_cluster(len(y), cl, K, seed)
        mk = (lambda Wk: Wk) if design is None else design
        self.ins, self.outs, self.w = [], [], []
        for k in range(K):
            a, o = f == k, f != k
            self.ins.append(Tend(y[a], x[a], mk(W[a]), None if cl is None else np.asarray(cl)[a], basis))
            self.outs.append(Tend(y[o], x[o], mk(W[o]), None if cl is None else np.asarray(cl)[o], basis))
            self.w.append(np.sqrt(a.mean()))
    def Z(self, b0):
        z = []
        for mi, mo in zip(self.ins, self.outs):
            _, gto, Do = mo.parts(b0)
            c = np.linalg.solve(gto.T @ gto / mo.n, Do)
            gbar, gti, _ = mi.parts(b0)
            v = c @ (gti.T @ gti / mi.n) @ c
            z.append(np.sqrt(mi.n) * (c @ gbar) / np.sqrt(v))
        return np.array(z)
    def T(self, b0):
        return float(np.dot(self.w, self.Z(b0))**2)

def info_xf(cf, b0):
    """Cross-fitted information index: D_k' Omega_{-k}^{-1} D_{-k}, averaged over folds
    with weights n_k/n. Unbiased to first order, unlike the in-sample index."""
    vals = []
    for mi, mo, w in zip(cf.ins, cf.outs, cf.w):
        _, gto, Do = mo.parts(b0); _, _, Di = mi.parts(b0)
        vals.append(w**2 * (Di @ np.linalg.solve(gto.T @ gto / mo.n, Do)))
    return float(np.sum(vals))

def mde80(I, n, sx, sr, th=0.0):
    from scipy.stats import norm
    if I <= 0: return np.inf
    return (norm.ppf(.975) + norm.ppf(.8)) / np.sqrt(n * I) * (sr / sx) / np.cos(th)**2
