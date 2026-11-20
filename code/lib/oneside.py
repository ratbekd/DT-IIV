"""Fixed-slope identification-robust set for the one-sided mean tendencies
E[u s^2] = E[u s^3] = 0, with first-step and angle corrections (eqs. 27-29),
optional clustering, and the pooled cell statistic of Proposition 5.4."""
import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq, minimize_scalar

class OneSided:
    def __init__(self, y, x, W, cl=None):
        self.y, self.x, self.W = map(np.asarray, (y, x, W))
        self.n, self.k = self.W.shape
        self.cl = None if cl is None else np.asarray(cl)
        Q = self.W.T @ self.W / self.n
        self.Qi = np.linalg.inv(Q)
        px = np.linalg.solve(self.W.T @ self.W, self.W.T @ self.x)
        py = np.linalg.solve(self.W.T @ self.W, self.W.T @ self.y)
        xt = self.x - self.W @ px; yt = self.y - self.W @ py
        b = xt @ yt / (xt @ xt); r = yt - b * xt
        sx = np.sqrt(np.mean(xt**2)); sr = np.sqrt(np.mean(r**2))
        self.eta = np.concatenate([px, py, [b, sx, sr]])
        self.bols, self.sx, self.sr = b, sx, sr
        # influence functions of the first step, eq. (27)
        psi = np.column_stack([
            (self.W * xt[:, None]) @ self.Qi, (self.W * yt[:, None]) @ self.Qi,
            xt * r / sx**2, (xt**2 - sx**2) / (2 * sx), (r**2 - sr**2) / (2 * sr)])
        self.psi = psi

    def theta(self, b0, eta=None):
        eta = self.eta if eta is None else eta
        b, sx, sr = eta[-3:]
        return np.arctan((b0 - b) * sx / sr)

    def contrib(self, b0, eta=None):
        eta = self.eta if eta is None else eta
        k = self.k
        px, py = eta[:k], eta[k:2*k]; b, sx, sr = eta[-3:]
        xt = self.x - self.W @ px; yt = self.y - self.W @ py
        r = yt - b * xt; xw, rw = xt / sx, r / sr
        th = np.arctan((b0 - b) * sx / sr)
        s = np.cos(th) * xw + np.sin(th) * rw
        u = np.cos(th) * rw - np.sin(th) * xw
        return np.column_stack([u * s**2, u * s**3])

    def _g(self, b0, xt, yt, b, sx, sr):
        r = yt - b * xt; xw, rw = xt / sx, r / sr
        th = np.arctan((b0 - b) * sx / sr)
        s = np.cos(th) * xw + np.sin(th) * rw
        u = np.cos(th) * rw - np.sin(th) * xw
        return np.column_stack([u * s**2, u * s**3])

    def S(self, b0, h=1e-6):
        """Corrected fixed-slope statistic; projection derivatives by the pointwise chain rule."""
        k = self.k
        px, py = self.eta[:k], self.eta[k:2*k]; b, sx, sr = self.eta[-3:]
        xt = self.x - self.W @ px; yt = self.y - self.W @ py
        g = self._g(b0, xt, yt, b, sx, sr); gbar = g.mean(0)
        dgx = (self._g(b0, xt + h, yt, b, sx, sr) - self._g(b0, xt - h, yt, b, sx, sr)) / (2*h)
        dgy = (self._g(b0, xt, yt + h, b, sx, sr) - self._g(b0, xt, yt - h, b, sx, sr)) / (2*h)
        Gs = []
        for jj, val in enumerate((b, sx, sr)):
            st = h * max(1.0, abs(val)); a1 = [b, sx, sr]; a2 = [b, sx, sr]
            a1[jj] += st; a2[jj] -= st
            Gs.append((self._g(b0, xt, yt, *a1).mean(0) - self._g(b0, xt, yt, *a2).mean(0)) / (2*st))
        G = np.column_stack([-dgx.T @ self.W / self.n, -dgy.T @ self.W / self.n, np.column_stack(Gs)])
        gt = g + self.psi @ G.T
        gt = gt - gt.mean(0)
        if self.cl is not None:
            _, inv = np.unique(self.cl, return_inverse=True)
            gt = np.array([np.bincount(inv, weights=gt[:, m]) for m in range(gt.shape[1])]).T
        Om = gt.T @ gt / self.n
        return self.n * gbar @ np.linalg.solve(Om, gbar)

    def S_slow(self, b0, h=1e-6):
        g = self.contrib(b0); gbar = g.mean(0)
        G = np.empty((g.shape[1], self.eta.size))
        for j in range(self.eta.size):
            e1 = self.eta.copy(); e2 = self.eta.copy()
            st = h * max(1.0, abs(self.eta[j]))
            e1[j] += st; e2[j] -= st
            G[:, j] = (self.contrib(b0, e1).mean(0) - self.contrib(b0, e2).mean(0)) / (2 * st)
        gt = g + self.psi @ G.T
        gt = gt - gt.mean(0)
        if self.cl is not None:
            _, inv = np.unique(self.cl, return_inverse=True)
            gt = np.array([np.bincount(inv, weights=gt[:, m]) for m in range(gt.shape[1])]).T
        Om = gt.T @ gt / self.n
        return self.n * gbar @ np.linalg.solve(Om, gbar)

    def bgrid(self, ngrid=1441):
        th = np.linspace(-np.pi/4, np.pi/4, ngrid)
        return self.bols + np.tan(th) * self.sr / self.sx

    def cue(self):
        def Q(th):
            b0 = self.bols + np.tan(th) * self.sr / self.sx
            g = self.contrib(b0); gb = g.mean(0); V = np.cov(g.T, bias=True)
            return self.n * gb @ np.linalg.solve(V, gb)
        th = np.linspace(-np.pi/4, np.pi/4, 721)
        q = np.array([Q(t) for t in th]); i = int(np.argmin(q))
        lo, hi = th[max(i-1, 0)], th[min(i+1, len(th)-1)]
        res = minimize_scalar(Q, bounds=(lo, hi), method='bounded')
        return self.bols + np.tan(res.x) * self.sr / self.sx, res.fun

def invert(Sfun, grid, crit):
    """Accepted set {b: Sfun(b) <= crit} on the grid, crossings refined."""
    v = np.array([Sfun(b) for b in grid]) - crit
    acc = v <= 0; out = []; i = 0
    while i < len(grid):
        if acc[i]:
            j = i
            while j + 1 < len(grid) and acc[j+1]: j += 1
            lo = grid[i] if i == 0 else brentq(lambda b: Sfun(b) - crit, grid[i-1], grid[i])
            hi = grid[j] if j == len(grid)-1 else brentq(lambda b: Sfun(b) - crit, grid[j], grid[j+1])
            out.append((lo, hi, i == 0, j == len(grid)-1)); i = j + 1
        else:
            i += 1
    return out, v + crit

def fmt(sets):
    if not sets: return "empty"
    return " U ".join("[%.3f, %.3f]%s" % (a, b, "+" if (l or h) else "") for a, b, l, h in sets)
