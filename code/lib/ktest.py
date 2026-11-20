"""Optimal-tendency K and KJ tests (Kleibergen-type) for the fixed-slope null.

For a basis B, moments g_i = u_i B(s_i). Per-observation derivative contributions
d_i = dg_i/dtheta are stacked with g_i, and both are corrected for the first step
(eqs. 27-29 incl. the angle channel). Under H0:
  S = n g' W g,            W = Omega_gg^{-1}                    -> chi2_m
  K = n (Dt' W g)^2 / (Dt' W Dt),  Dt = D - Omega_dg W g          -> chi2_1
  J = S - K                                                       -> chi2_{m-1}, indep. of K
whatever the strength of identification (Kleibergen 2005).
"""
import numpy as np
from scipy.stats import chi2
from opttend import Tend, B_poly, B_sieve, B_bounded


class KTest(Tend):
    EPS = 1e-4

    def _stack(self, b0, xt, yt, b, sx, sr):
        th = np.arctan((b0 - b) * sx / sr)
        g = self._gth(th, xt, yt, b, sx, sr)
        d = (self._gth(th + self.EPS, xt, yt, b, sx, sr)
             - self._gth(th - self.EPS, xt, yt, b, sx, sr)) / (2 * self.EPS)
        return np.column_stack([g, d])

    def stacked(self, b0, h=1e-5):
        b, sx, sr = self.eta[-3:]
        xt, yt = self.xt, self.yt
        f = self._stack
        g = f(b0, xt, yt, b, sx, sr)
        gbar = g.mean(0)
        dgx = (f(b0, xt + h, yt, b, sx, sr) - f(b0, xt - h, yt, b, sx, sr)) / (2 * h)
        dgy = (f(b0, xt, yt + h, b, sx, sr) - f(b0, xt, yt - h, b, sx, sr)) / (2 * h)
        Gs = []
        for j, val in enumerate((b, sx, sr)):
            st = h * max(1.0, abs(val))
            a1 = [b, sx, sr]; a2 = [b, sx, sr]
            a1[j] += st; a2[j] -= st
            Gs.append((f(b0, xt, yt, *a1).mean(0) - f(b0, xt, yt, *a2).mean(0)) / (2 * st))
        Geta = np.column_stack([-dgx.T @ self.W / self.n, -dgy.T @ self.W / self.n,
                                np.column_stack(Gs)])
        gt = g + self.psi @ Geta.T
        gt = gt - gt.mean(0)
        if self.inv is not None:
            gt = np.array([np.bincount(self.inv, weights=gt[:, m], minlength=self.G)
                           for m in range(gt.shape[1])]).T
        V = gt.T @ gt / self.n
        return gbar, V

    def SKJ(self, b0):
        z, V = self.stacked(b0)
        m = z.size // 2
        g, D = z[:m], z[m:]
        Vgg, Vdg = V[:m, :m], V[m:, :m]
        Wg = np.linalg.solve(Vgg, g)
        Dt = D - Vdg @ Wg
        WDt = np.linalg.solve(Vgg, Dt)
        S = self.n * g @ Wg
        K = self.n * (Dt @ Wg) ** 2 / (Dt @ WDt)
        return S, K, S - K


def nmom(model):
    return model.B(np.zeros(3)).shape[1]


def accept(model, grid, alpha=0.05, aK=0.04, aJ=0.01):
    m = nmom(model)
    vals = np.array([model.SKJ(b) for b in grid])
    accS = vals[:, 0] <= chi2.ppf(1 - alpha, m)
    accK = vals[:, 1] <= chi2.ppf(1 - alpha, 1)
    accKJ = (vals[:, 1] <= chi2.ppf(1 - aK, 1)) & (vals[:, 2] <= chi2.ppf(1 - aJ, m - 1))
    return vals, accS, accK, accKJ


def intervals(grid, acc):
    out, i = [], 0
    while i < len(grid):
        if acc[i]:
            j = i
            while j + 1 < len(grid) and acc[j + 1]:
                j += 1
            out.append((grid[i], grid[j], i == 0, j == len(grid) - 1))
            i = j + 1
        else:
            i += 1
    return out


def info_index(model, b0):
    """Bias-corrected information index (eq. info_index) and Delta_80 (eq. mde)."""
    from scipy.stats import norm
    z, V = model.stacked(b0)
    m = z.size // 2
    D = z[m:]
    Vgg, Vdd = V[:m, :m], V[m:, m:]
    I = D @ np.linalg.solve(Vgg, D) - np.trace(np.linalg.solve(Vgg, Vdd)) / model.n
    b, sx, sr = model.eta[-3:]
    th = np.arctan((b0 - b) * sx / sr)
    d80 = (norm.ppf(.975) + norm.ppf(.8)) / np.sqrt(model.n * I) * (sr / sx) / np.cos(th) ** 2 if I > 0 else np.inf
    return I, d80
