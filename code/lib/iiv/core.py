"""Core estimation machinery (port of indep_iiv.R, sim_star.R, het_lib.R, rob_lib.R).

A Context holds one estimation sample: the controls W1, the clusters, the
projections and, when needed, the leave-one-cluster-out machinery of het_lib.R.
All statistics are functions of a "prep" object P, so the same code serves the
applications (real y, x) and the semi-synthetic simulations (generated y, x).

Moment columns (1-indexed, as in the R code):
  1  u^2 s            variance tendency (VT)
  2  u s^2            mean tendency
  3  u^3 s
  4  u s^3            mean tendency
  5  u^2 s^2 - 1
  6  (u^2 - 1) s^3
  ext='het': 7..      h(W) s u for the columns of H (controls, then the variance index)
  ext='rob': 7        u tanh(z),  8  u exp(-z^2/2),  z = (s - med s) / (1.4826 MAD s)
"""
import numpy as np
from scipy.stats import chi2, f as fdist
from scipy.optimize import minimize_scalar, brentq

TOL = np.finfo(float).eps ** 0.25          # R's default tolerance in optimize()
ALPHA = 0.05


def optimize(fun, lo, hi):
    """Brent minimization on [lo, hi], tolerance as in R's optimize()."""
    r = minimize_scalar(fun, bounds=(min(lo, hi), max(lo, hi)), method='bounded', options={'xatol': TOL})
    return r.x, r.fun


def zs(s):
    med = np.median(s)
    tau = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / tau, tau


class Prep(dict):
    __getattr__ = dict.__getitem__


class Context:
    def __init__(self, D):
        self.D = D
        self.W1 = D['W1']; self.y = D['y']; self.x = D['x']; self.Z = D['Z']
        self.lab = D['lab']
        self.n, self.k = self.W1.shape
        self.clustered = D['cl'] is not None
        self.cl = D['cl'] if self.clustered else np.arange(self.n).astype(str)
        self.cl_codes = np.unique(self.cl, return_inverse=True)[1]
        self.G = int(self.cl_codes.max() + 1)
        self.Qw = np.linalg.qr(self.W1)[0]
        self.A_W = self.W1 @ np.linalg.inv(self.W1.T @ self.W1 / self.n)
        self.Zt = self.resid(self.Z)
        self._loo = None
        k1 = self.W1.shape[1]
        if k1 <= 31:
            Hf = self.W1[:, 1:]
            self.H_fix = (Hf - Hf.mean(0)) / Hf.std(0, ddof=1)
        else:
            self.H_fix = None
        self.q_fix = 0 if self.H_fix is None else self.H_fix.shape[1]
        self.h_cols = list(range(7, 7 + self.q_fix))
        self.idx_col = 7 + self.q_fix

    # ------------------------------------------------------------------ basics
    def resid(self, v):
        return v - self.Qw @ (self.Qw.T @ v)

    def rowsum(self, M):
        M = np.asarray(M)
        if M.ndim == 1:
            return np.bincount(self.cl_codes, weights=M, minlength=self.G)
        return np.array([np.bincount(self.cl_codes, weights=M[:, j], minlength=self.G) for j in range(M.shape[1])]).T

    def prep(self, y=None, x=None, index=False):
        y = self.y if y is None else y; x = self.x if x is None else x
        xt = self.resid(x); yt = self.resid(y)
        b = np.sum(xt * yt) / np.sum(xt ** 2); r = yt - xt * b
        sx = np.sqrt(np.mean(xt ** 2)); sr = np.sqrt(np.mean(r ** 2))
        P = Prep(xt=xt, yt=yt, r=r, b=b, sx=sx, sr=sr, xw=xt / sx, rw=r / sr,
                 psi_b=xt * r / np.mean(xt ** 2), psi_sx=(xt ** 2 - sx ** 2) / (2 * sx),
                 psi_sr=(r ** 2 - sr ** 2) / (2 * sr), H=None, eps=None)
        if index:
            a = P.rw ** 2 - P.xw ** 2
            ah = self.loo_fit(a)
            sc = 1 / np.std(ah, ddof=1)
            P['eps'] = (a - ah) * sc
            idx = ((ah - ah.mean()) * sc)[:, None]
            P['H'] = idx if self.H_fix is None else np.column_stack([self.H_fix, idx])
        return P

    @staticmethod
    def beta_at(th, P):
        return P.b + np.tan(th) * P.sr / P.sx

    @staticmethod
    def theta_of(beta, P):
        return np.arctan((beta - P.b) * P.sx / P.sr)

    # ------------------------------------------------------------------ leave-one-cluster-out
    def _loo_setup(self):
        if self._loo is not None:
            return self._loo
        Qw, n = self.Qw, self.n
        pii = np.sum(Qw ** 2, 1)
        groups = [np.where(self.cl_codes == g)[0] for g in range(self.G)]
        if self.clustered:
            inv = []
            B = Qw.copy()
            for ix in groups:
                A = np.eye(len(ix)) - Qw[ix] @ Qw[ix].T
                rc = 1 / np.linalg.cond(A, 1)
                if rc < 1e-8:
                    inv.append(None); B[ix] = 0
                else:
                    Ai = np.linalg.inv(A); inv.append(Ai); B[ix] = Ai @ Qw[ix]
            singular = sum(v is None for v in inv)
        else:
            inv = None
            B = Qw / (1 - pii)[:, None]
            B[pii > 1 - 1e-8] = 0
            singular = int(np.sum(pii > 1 - 1e-8))
        self._loo = dict(pii=pii, groups=groups, inv=inv, B=B, singular=singular)
        return self._loo

    def loo_fit(self, tg):
        L = self._loo_setup()
        tg = np.asarray(tg, float)
        one = tg.ndim == 1
        T = tg[:, None] if one else tg
        E = T - self.Qw @ (self.Qw.T @ T)
        if not self.clustered:
            out = T - E / (1 - L['pii'])[:, None]
            bad = L['pii'] > 1 - 1e-8
            if bad.any():
                out[bad] = T.mean(0)
        else:
            out = T.copy()
            for ix, Ai in zip(L['groups'], L['inv']):
                out[ix] = T.mean(0) if Ai is None else T[ix] - Ai @ E[ix]
        return out[:, 0] if one else out

    @property
    def loo_singular(self):
        return self._loo_setup()['singular']

    def idx_extra(self, P, s, u):
        L = self._loo_setup()
        RV = self.rowsum((s * u)[:, None] * L['B'])
        RE = self.rowsum(P.eps[:, None] * self.Qw)
        M = RE.T @ RV
        return max(np.sum(M * M.T) - np.sum(np.sum(RV * RE, 1) ** 2), 0.0)

    # ------------------------------------------------------------------ moments
    @staticmethod
    def _gmat(s, u, cols, ext='base', H=None, deriv=False):
        """Moment columns, and with deriv=True also their derivatives with respect
        to s (gs) and to u (gu)."""
        g = [u ** 2 * s, u * s ** 2, u ** 3 * s, u * s ** 3, u ** 2 * s ** 2 - 1, (u ** 2 - 1) * s ** 3]
        if deriv:
            gs = [u ** 2, 2 * u * s, u ** 3, 3 * u * s ** 2, 2 * u ** 2 * s, 3 * (u ** 2 - 1) * s ** 2]
            gu = [2 * u * s, s ** 2, 3 * u ** 2 * s, s ** 3, 2 * u * s ** 2, 2 * u * s ** 3]
        if ext in ('het', 'hetx') and H is not None:
            for j in range(H.shape[1]):
                g.append(H[:, j] * s * u)
                if deriv:
                    gs.append(H[:, j] * u); gu.append(H[:, j] * s)
        elif ext == 'rob':
            z, tau = zs(s); ez = np.exp(-z ** 2 / 2); th = np.tanh(z)
            g += [u * th, u * ez]
            if deriv:
                gs += [u * (1 - th ** 2) / tau, -u * z * ez / tau]
                gu += [th, ez]
        idx = [c - 1 for c in cols]
        G0 = np.column_stack([g[i] for i in idx])
        if not deriv:
            return G0
        return G0, np.column_stack([gs[i] for i in idx]), np.column_stack([gu[i] for i in idx])

    def _H(self, P, cols, ext):
        if ext in ('het', 'hetx') and max(cols) > 6:
            return P.H
        return None

    def Q_u(self, th, P, cols, ext='base'):
        s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
        g = self._gmat(s, u, cols, ext, self._H(P, cols, ext))
        m = g.mean(0); S = (g - m).T @ (g - m) / self.n
        return self.n * float(m @ np.linalg.solve(S, m))

    def gcorr(self, th, P, cols, beta=None, ext='base'):
        """Moments g and first-step corrected contributions gc (n x m, not centred).
        With beta, theta follows beta and the angle channel enters gc."""
        if beta is not None:
            th = self.theta_of(beta, P)
        cs, sn = np.cos(th), np.sin(th)
        s = cs * P.xw + sn * P.rw; u = cs * P.rw - sn * P.xw
        H = self._H(P, cols, ext)
        g, gs, gu = self._gmat(s, u, cols, ext, H, deriv=True)
        return g, self.correct(P, th, beta, s, u, g, gs, gu), s, u

    def correct(self, P, th, beta, s, u, g, gs, gu):
        """First-step corrected contributions of moment functions g(s, u) with
        derivatives gs, gu (n x m arrays), at angle th (with beta: the angle channel)."""
        cs, sn = np.cos(th), np.sin(th)
        d_xw = gs * cs - gu * sn; d_rw = gs * sn + gu * cs
        d_xt = d_xw / P.sx; d_r = d_rw / P.sr
        G_px = self.W1.T @ (-d_xt + P.b * d_r) / self.n
        G_py = self.W1.T @ (-d_r) / self.n
        G_b = np.mean(-d_r * P.xt[:, None], 0)
        G_sx = np.mean(-d_xw * P.xw[:, None], 0) / P.sx
        G_sr = np.mean(-d_rw * P.rw[:, None], 0) / P.sr
        if beta is not None:
            G_th = np.mean(gs * u[:, None] - gu * s[:, None], 0); c2 = cs ** 2
            G_b = G_b + G_th * (-(P.sx / P.sr) * c2)
            G_sx = G_sx + G_th * (((beta - P.b) / P.sr) * c2)
            G_sr = G_sr + G_th * (-((beta - P.b) * P.sx / P.sr ** 2) * c2)
        gc = (g + P.xt[:, None] * (self.A_W @ G_px) + P.yt[:, None] * (self.A_W @ G_py)
              + np.outer(P.psi_b, G_b) + np.outer(P.psi_sx, G_sx) + np.outer(P.psi_sr, G_sr))
        return gc

    def meat(self, M):
        """Variance of a mean of contributions times n (clustered when the sample is)."""
        M = np.asarray(M, float)
        M = M[:, None] if M.ndim == 1 else M
        M = self.rowsum(M - M.mean(0))
        return M.T @ M / self.n

    def Q_c(self, th, P, cols, beta=None, ext='base', return_nu=False):
        """Corrected, clustered S statistic. With beta, tests H0: beta and adds the
        angle channel. Returns (statistic, nu) when return_nu."""
        g, gc, s, u = self.gcorr(th, P, cols, beta, ext)
        m = g.mean(0)
        gc = self.rowsum(gc - gc.mean(0))
        Om = gc.T @ gc / self.n
        if ext == 'het' and self.idx_col in cols:
            j = cols.index(self.idx_col)
            Om[j, j] += self.idx_extra(P, s, u) / self.n
        w = np.linalg.solve(Om, m)
        stat = self.n * float(m @ w)
        if not return_nu:
            return stat
        z = gc @ w
        nu = min(self.G - 1, 3 * np.sum(z ** 2) ** 2 / np.sum(z ** 4))
        return stat, nu

    def Q_b(self, beta, P, cols, ext='base'):
        return self.Q_c(None, P, cols, beta=beta, ext=ext)

    def gbar(self, th, P, cols, ext='base'):
        s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
        return self._gmat(s, u, cols, ext, self._H(P, cols, ext)).mean(0)

    def analytic_se(self, th, P, cols, centred=True):
        """Sandwich for theta and beta: CUE with the uncorrected weight, corrected meat
        (analytic_se of indep_iiv.R)."""
        s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
        gi = self._gmat(s, u, cols); Wt = np.linalg.inv((gi - gi.mean(0)).T @ (gi - gi.mean(0)) / self.n)
        hh = 1e-6
        Gd = (self.gbar(th + hh, P, cols) - self.gbar(th - hh, P, cols)) / (2 * hh)
        gc = self.gcorr(th, P, cols)[1]
        A = 1 / float(Gd @ Wt @ Gd)
        psi_th = -A * (gc @ (Wt @ Gd))
        psi_b = (P.psi_b + (1 / np.cos(th) ** 2) * (P.sr / P.sx) * psi_th
                 + np.tan(th) * (P.psi_sr / P.sx - P.sr * P.psi_sx / P.sx ** 2))
        if not centred:                       # indep_iiv_card.R: mean(psi^2), i.i.d.
            return np.sqrt(np.mean(psi_th ** 2) / self.n), np.sqrt(np.mean(psi_b ** 2) / self.n)
        return (np.sqrt(self.meat(psi_th)[0, 0] / self.n), np.sqrt(self.meat(psi_b)[0, 0] / self.n))

    def cue_minima(self, P, cols, ngrid=91, ext='base'):
        """All local minima of Q_u on |theta| <= pi/4, refined, sorted by Q (cue_fit of indep_iiv.R)."""
        th = np.linspace(-np.pi / 4, np.pi / 4, ngrid)
        Qv = np.array([self.Q_u(a, P, cols, ext) for a in th])
        ext_v = np.r_[np.inf, Qv, np.inf]
        loc = np.where(np.diff(np.sign(np.diff(ext_v))) == 2)[0]
        sols = []
        for i in loc:
            t, q = optimize(lambda a: self.Q_u(a, P, cols, ext), th[max(0, i - 1)], th[min(ngrid - 1, i + 1)])
            sols.append((t, q))
        return sorted(sols, key=lambda v: v[1])

    # ------------------------------------------------------------------ critical values
    def crit_S(self, m, nu=None):
        nu = self.G - 1 if nu is None else nu
        if nu - m + 1 < 1:
            return np.inf
        return self.G / (self.G - 1) * nu * m / (nu - m + 1) * fdist.ppf(1 - ALPHA, m, nu - m + 1)

    def p_S(self, stat, m, nu=None):
        nu = self.G - 1 if nu is None else nu
        if m < 1 or nu - m + 1 < 1:
            return np.nan
        return 1 - fdist.cdf(stat * (self.G - 1) / self.G * (nu - m + 1) / (nu * m), m, nu - m + 1)

    # ------------------------------------------------------------------ estimation (het_lib / rob_lib)
    def fit_cue(self, P, cols, ext='base', ngrid=181):
        m = len(cols)
        th = np.linspace(-np.pi / 4, np.pi / 4, ngrid)
        Qv = np.array([self.Q_u(a, P, cols, ext) for a in th]); i = int(np.argmin(Qv))
        tmin, _ = optimize(lambda a: self.Q_u(a, P, cols, ext), th[max(0, i - 1)], th[min(ngrid - 1, i + 1)])
        J = J_p = J_chi = np.nan
        if m > 1:
            tj, _ = optimize(lambda a: self.Q_c(a, P, cols, ext=ext), tmin - 0.05, tmin + 0.05)
            J, nu = self.Q_c(tj, P, cols, ext=ext, return_nu=True)
            J_p = self.p_S(J, m - 1, nu); J_chi = 1 - chi2.cdf(J, m - 1)
        return dict(theta=tmin, beta=self.beta_at(tmin, P), J=J, J_p=J_p, J_p_chi2=J_chi)

    def in_S(self, P, cols, beta, ext='base', rule='F'):
        S, nu = self.Q_c(None, P, cols, beta=beta, ext=ext, return_nu=True)
        crit = self.crit_S(len(cols), nu) if rule == 'F' else chi2.ppf(1 - ALPHA, len(cols))
        return S <= crit

    def S_set(self, P, cols, ext='base', ngrid=241, rule='F'):
        th = np.linspace(-np.pi / 4, np.pi / 4, ngrid); bg = self.beta_at(th, P)
        acc = np.array([self.in_S(P, cols, b, ext, rule) for b in bg])
        return dict(lo=bg[acc].min() if acc.any() else np.nan, hi=bg[acc].max() if acc.any() else np.nan,
                    pieces=int(np.sum(np.diff(np.r_[0, acc.astype(int), 0]) == 1)), acc=acc, grid=bg)

    def h_signal(self, P, h):
        A = np.column_stack([h * P.xw * P.rw, h * (P.xw ** 2 - P.rw ** 2)]); m2 = A.mean(0)
        V = self.rowsum(A - m2); V = V.T @ V / self.n
        return dict(theta=0.5 * np.arctan(2 * m2[0] / m2[1]), p=self.p_S(self.n * float(m2 @ np.linalg.solve(V, m2)), 2))

    def tsls(self, P):
        xh = self.Zt @ np.linalg.lstsq(self.Zt, P.xt, rcond=None)[0]
        b = np.sum(xh * P.yt) / np.sum(xh * P.xt)
        e = P.yt - P.xt * b
        se = np.sqrt(np.sum(self.rowsum(xh * e) ** 2)) / abs(np.sum(xh * P.xt))
        return b, se

    def vt_roots(self, P, lo=-np.pi / 2 + 1e-3, hi=np.pi / 2 - 1e-3, ngrid=2001):
        """All roots of the sample VT moment mean(u^2 s) on (lo, hi)."""
        f = lambda a: np.mean((np.cos(a) * P.rw - np.sin(a) * P.xw) ** 2 * (np.cos(a) * P.xw + np.sin(a) * P.rw))
        th = np.linspace(lo, hi, ngrid); v = np.array([f(a) for a in th])
        idx = np.where(np.diff(np.sign(v)) != 0)[0]
        return np.array([brentq(f, th[i], th[i + 1], xtol=TOL) for i in idx])


def uniroot(f, lo, hi, tol=TOL):
    """R's uniroot (Brent's zeroin) with its default tolerance."""
    return brentq(f, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps)


def comps(fun, th):
    """Components of {t on the grid th: fun(t) <= 0}, with interior crossings refined by
    uniroot. Returns a list of (lo, hi, touches_boundary)."""
    v = np.array([fun(t) for t in th]); acc = v <= 0; N = len(th)
    out, i = [], 0
    while i < N:
        if acc[i]:
            j = i
            while j + 1 < N and acc[j + 1]:
                j += 1
            lo, hi = th[i], th[j]
            if i > 0:
                lo = uniroot(fun, th[i - 1], th[i])
            if j < N - 1:
                hi = uniroot(fun, th[j], th[j + 1])
            out.append((lo, hi, i == 0 or j == N - 1)); i = j + 1
        else:
            i += 1
    return out


def pieces(acc):
    return int(np.sum(np.diff(np.r_[0, np.asarray(acc, int), 0]) == 1))


def coef_P(a, c):
    """Coefficients (increasing order) of the VT cubic in tau = tan(theta)."""
    return np.array([np.mean(c ** 2 * a), np.mean(c ** 3) - 2 * np.mean(a ** 2 * c),
                     np.mean(a ** 3) - 2 * np.mean(a * c ** 2), np.mean(a ** 2 * c)])


def polyroot_real(cf, tol=1e-7):
    z = np.roots(cf[::-1])
    return np.sort(np.real(z[np.abs(np.imag(z)) < tol]))


def f3(v):
    return '--' if v is None or (isinstance(v, float) and np.isnan(v)) else '%.3f' % v


def components(acc, grid):
    """Connected components of an acceptance vector on a grid."""
    out, i = [], 0
    acc = np.asarray(acc, bool)
    while i < len(acc):
        if acc[i]:
            j = i
            while j + 1 < len(acc) and acc[j + 1]:
                j += 1
            out.append((grid[i], grid[j])); i = j + 1
        else:
            i += 1
    return out


def cluster_draws(C, B, seed):
    """B cluster (or pairs) bootstrap index vectors."""
    rng = np.random.default_rng(seed)
    groups = C._loo_setup()['groups']
    for _ in range(B):
        pick = rng.integers(0, C.G, C.G)
        yield np.concatenate([groups[g] for g in pick])


def boot_prep(W1, y, x):
    """Residualize y, x on W1 and whiten (one bootstrap draw)."""
    Q = np.linalg.qr(W1)[0]
    a = x - Q @ (Q.T @ x); yy = y - Q @ (Q.T @ y)
    b = np.sum(a * yy) / np.sum(a ** 2); r = yy - b * a
    return dict(xw=a / np.sqrt(np.mean(a ** 2)), rw=r / np.sqrt(np.mean(r ** 2)), b=b,
                k=np.sqrt(np.mean(r ** 2) / np.mean(a ** 2)))


def partner(theta):
    return theta - np.sign(theta) * np.pi / 2


def estimate(y, x, W, cols, rule, ngrid=91):
    """Full estimator on one (re)sample: first step, CUE on |theta| <= pi/4, labelling
    (estimate of indep_iiv.R). Returns (theta, beta, edge); NaN when degenerate."""
    try:
        coef = lambda v: np.linalg.lstsq(W, v, rcond=None)[0]
        xt = x - W @ coef(x); yt = y - W @ coef(y)
        b = np.sum(xt * yt) / np.sum(xt ** 2); r = yt - xt * b
        sx, sr = np.sqrt(np.mean(xt ** 2)), np.sqrt(np.mean(r ** 2))
        P = Prep(xw=xt / sx, rw=r / sr); n = len(y)

        def Qu(th):
            s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
            g = Context._gmat(s, u, cols); m = g.mean(0); S = (g - m).T @ (g - m) / n
            return n * float(m @ np.linalg.solve(S, m))
        th = np.linspace(-np.pi / 4, np.pi / 4, ngrid); Qv = np.array([Qu(a) for a in th])
        loc = np.where(np.diff(np.sign(np.diff(np.r_[np.inf, Qv, np.inf]))) == 2)[0]
        sols = sorted((optimize(Qu, th[max(0, i - 1)], th[min(ngrid - 1, i + 1)]) for i in loc), key=lambda v: v[1])
        thf = sols[0][0]; bet = lambda a: b + np.tan(a) * sr / sx
        if rule == 'relevance':
            tl = thf
        else:
            cand = np.array([thf, partner(thf)]); bb = bet(cand)
            tl = cand[bb > 0][0] if np.sum(bb > 0) == 1 else np.nan
        return tl, (np.nan if np.isnan(tl) else bet(tl)), float(abs(abs(thf) - np.pi / 4) < 0.01)
    except (np.linalg.LinAlgError, ValueError, IndexError):
        return np.nan, np.nan, np.nan
