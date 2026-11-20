"""Shared simulation machinery (calibration of sim_semi*.R, designs, fit_set of
sim_star.R, parallel replications).

Random numbers come from NumPy (PCG64) streams spawned from one seed per run,
one stream per replication, so results do not depend on the number of cores.
They differ draw by draw from the R runs; summaries agree within simulation error.
"""
import os
import numpy as np
from multiprocessing import get_context
from scipy.stats import chi2
from .core import Context, optimize, uniroot

ALPHA = 0.05


def mad(v):
    """R's mad(): 1.4826 * median absolute deviation from the median, NAs removed."""
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v):
        return np.nan
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def nanmedian(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return np.median(v) if len(v) else np.nan


def nanmean(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return np.mean(v) if len(v) else np.nan


def cdist(a, b):
    d = np.mod(np.asarray(a) - b, np.pi)
    return np.minimum(d, np.pi - d)


def deciles(s0, nb=10):
    """cut(rank(s0, ties.method = 'first'), 10, labels = FALSE) of R, 0-based."""
    n = len(s0)
    rk = np.empty(n); rk[np.argsort(s0, kind='stable')] = np.arange(1, n + 1)
    br = np.linspace(1, n, nb + 1); dx = (n - 1) / 1000; br[0] -= dx; br[-1] += dx
    return np.clip(np.searchsorted(br, rk, side='left') - 1, 0, nb - 1)


class Semi:
    """Calibration of the semi-synthetic designs to an application sample:
    real regressor, controls and clusters; beta0 = sample 2SLS (or given);
    gamma so that the OLS bias equals OLS - beta0 (or given)."""

    def __init__(self, C, beta0=None, gamma=None):
        self.C = C; n = C.n
        s0 = C.resid(C.x); self.sd_xt = np.sqrt(np.mean(s0 ** 2)); self.s0 = s0 / self.sd_xt
        yt = C.resid(C.y); xt = self.s0 * self.sd_xt
        xh = C.Zt @ np.linalg.lstsq(C.Zt, xt, rcond=None)[0]
        self.b_ols = np.sum(xt * yt) / np.sum(xt ** 2)
        self.beta0 = np.sum(xh * yt) / np.sum(xh * xt) if beta0 is None else beta0
        e = yt - xt * self.beta0; self.e_real = e
        self.sd_u = np.std(e, ddof=1); self.e_pool = (e - e.mean()) / self.sd_u
        self.bin = deciles(self.s0)
        self.idx_bin = [np.where(self.bin == b)[0] for b in range(10)]
        eb = e.copy()
        for ix in self.idx_bin:
            eb[ix] -= e[ix].mean()
        self.e_bin = eb / np.sqrt(np.mean(eb ** 2))
        g = np.linalg.lstsq(C.W1, C.y - self.beta0 * C.x, rcond=None)[0]
        self.Wg = C.W1 @ g
        target = (self.b_ols - self.beta0) * self.sd_xt / self.sd_u
        self.gamma = ((1 - np.sqrt(1 - 4 * target ** 2)) / (2 * target) if abs(target) < 0.5 else np.sign(target)) \
            if gamma is None else gamma
        self.idx_cl = C._loo_setup()['groups'] if C.clustered else None
        v_w = C.W1 @ np.linalg.lstsq(C.W1, e ** 2, rcond=None)[0]
        v_w = np.maximum(v_w, 0.2 * np.mean(e ** 2)); self.v_w = v_w / v_w.mean()

    def draw(self, design, rng):
        n = self.C.n
        if design == 'hom':
            return rng.choice(self.e_pool, n, replace=True)
        if design == 'het':
            v = np.empty(n)
            for ix in self.idx_bin:
                v[ix] = self.e_bin[rng.choice(ix, len(ix), replace=True)]
            return v
        if design == 'hetW':
            return np.sqrt(self.v_w) * rng.choice(self.e_pool, n, replace=True)
        if design == 'blk':
            return self.draw_blk(rng)
        raise ValueError(design)

    def draw_blk(self, rng):
        """Each cluster receives the residual path of a random donor cluster, in order,
        after one circular shift common to all clusters."""
        idx = self.idx_cl; G = len(idx); v = np.empty(self.C.n)
        don = rng.integers(0, G, G)
        k = int(rng.integers(1, max(1, max(len(i) for i in idx) - 1) + 1))
        for g in range(G):
            ix = idx[g]; src = idx[don[g]]
            src = src[(np.arange(len(src)) + k) % len(src)]
            v[ix] = self.e_pool[np.resize(src, len(ix))]
        return v

    def data(self, u):
        x = self.C.x + self.gamma * self.sd_xt * u
        y = self.beta0 * x + self.Wg + self.sd_u * u
        return y, x


# ---------------------------------------------------------------------- per-replication fits
def vt_roots(P, ngrid=721):
    """Roots of mean(u^2 s) on (-pi/2, pi/2), grid then uniroot (as in sim_screen.R)."""
    f = lambda a: np.mean((np.cos(a) * P.rw - np.sin(a) * P.xw) ** 2 * (np.cos(a) * P.xw + np.sin(a) * P.rw))
    th = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, ngrid)
    v = np.array([f(a) for a in th]); ix = np.where(np.diff(np.sign(v)) != 0)[0]
    return np.array([uniroot(f, th[i], th[i + 1]) for i in ix])


def screen_J(C, P, r, cols=(1, 3, 4, 5), return_nu=False):
    lo, hi = max(r - 0.1, -np.pi / 2 + 1e-3), min(r + 0.1, np.pi / 2 - 1e-3)
    a, J = optimize(lambda a: C.Q_c(a, P, list(cols)), lo, hi)
    if return_nu:
        return C.Q_c(a, P, list(cols), return_nu=True)
    return J


def cue_J(C, P, cols, ngrid=91, ext='base'):
    """CUE on |theta| <= pi/4 and the corrected J near it (fit_set of sim_star.R)."""
    th = np.linspace(-np.pi / 4, np.pi / 4, ngrid)
    Qv = [C.Q_u(a, P, cols, ext) for a in th]; i = int(np.argmin(Qv))
    t, _ = optimize(lambda a: C.Q_u(a, P, cols, ext), th[max(0, i - 1)], th[min(ngrid - 1, i + 1)])
    a, J = optimize(lambda a: C.Q_c(a, P, cols, ext=ext), t - 0.05, t + 0.05)
    return t, J, a


def fit_set(C, P, cols, beta0):
    """sim_star.R fit_set: estimate, coverage of beta0, exclusion of OLS, J rejection."""
    m = len(cols)
    if m == 1:
        th = np.linspace(-np.pi / 4, np.pi / 4, 181)
        f = lambda a: np.mean((np.cos(a) * P.rw - np.sin(a) * P.xw) ** 2 * (np.cos(a) * P.xw + np.sin(a) * P.rw))
        gv = np.array([f(a) for a in th]); i = np.where(np.diff(np.sign(gv)) != 0)[0]
        if not len(i):
            est = np.nan
        else:
            rts = np.array([uniroot(f, th[j], th[j + 1]) for j in i])
            est = C.beta_at(rts[np.argmin(np.abs(rts))], P)
        J = np.nan
    else:
        t, J, _ = cue_J(C, P, cols)
        est = C.beta_at(t, P)
    crit = chi2.ppf(1 - ALPHA, m)
    return dict(est=est, cover=float(C.Q_b(beta0, P, cols) <= crit), excl_ols=float(C.Q_b(P.b, P, cols) > crit),
                J_rej=np.nan if np.isnan(J) else float(J > chi2.ppf(1 - ALPHA, m - 1)))


def safe(fun, keys):
    try:
        return fun()
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError):
        return {k: np.nan for k in keys}


def tsls_rep(C, P, beta0):
    from scipy.stats import norm
    b, se = C.tsls(P); z = norm.ppf(1 - ALPHA / 2)
    return dict(est=b, cover=float(abs(b - beta0) <= z * se), excl_ols=float(abs(b - P.b) > z * se))


# ---------------------------------------------------------------------- parallel replications
_G = {}


def _worker(args):
    i, ss = args
    rng = np.random.default_rng(ss)
    return _G['fun'](rng)


def replicate(fun, R, seed, cores=None):
    """Run fun(rng) R times in parallel; replication i uses stream i of SeedSequence(seed)."""
    cores = cores or int(os.environ.get('IIV_CORES', os.cpu_count() or 1))
    seeds = np.random.SeedSequence(seed).spawn(R)
    _G['fun'] = fun
    if cores <= 1:
        return [_worker((i, s)) for i, s in enumerate(seeds)]
    with get_context('fork').Pool(cores) as pool:
        return pool.map(_worker, list(enumerate(seeds)), chunksize=max(1, R // (4 * cores)))


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + '.'))
        else:
            out[prefix + k] = v
    return out


class StarDesign:
    """Semi-synthetic STAR design of sim_star.R: real class size, controls and
    class clusters; error shape from the real 2SLS residuals.
      u = sigma(s0) eps + c q(s0),  x = x_real + gamma sd(x~) u,  y = beta0 x + W g + sd_u u
    D0 exogenous; D1 endogenous; D2 heteroscedastic (sigma^2 ~ exp(a s0)); D3 curvature;
    D1t t(5)-type tails; D1w error variance following the real residual variance given W."""
    beta0, gamma, a_het, c_curv = -0.98, 0.3, 0.6, 0.3

    def __init__(self, C):
        self.C = C; n = C.n
        s0 = C.resid(C.x); self.sd_xt = np.sqrt(np.mean(s0 ** 2)); self.s0 = s0 / self.sd_xt
        g = np.linalg.lstsq(C.W1, C.y - self.beta0 * C.x, rcond=None)[0]; self.Wg = C.W1 @ g
        yt = C.resid(C.y); xt = self.s0 * self.sd_xt
        xh = C.Zt @ np.linalg.lstsq(C.Zt, xt, rcond=None)[0]; self.b_2s = np.sum(xh * yt) / np.sum(xh * xt)
        e = yt - xt * self.b_2s; self.e_raw = e
        self.sd_u = np.std(e, ddof=1); self.e_pool = (e - e.mean()) / self.sd_u
        X1 = np.column_stack([np.ones(n), self.s0]); q0 = self.s0 ** 2
        q0 = q0 - X1 @ np.linalg.lstsq(X1, q0, rcond=None)[0]; self.q0 = q0 / np.std(q0, ddof=1)
        v_w = C.W1 @ np.linalg.lstsq(C.W1, e ** 2, rcond=None)[0]
        v_w = np.maximum(v_w, 0.2 * np.mean(e ** 2)); self.v_w = v_w / v_w.mean()

    def data(self, design, rng):
        n = self.C.n
        eps = rng.choice(self.e_pool, n, replace=True)
        if design == 'D1t':
            eps = eps * np.sqrt(3 / rng.chisquare(5, n))
        gam = 0 if design == 'D0' else self.gamma
        if design == 'D2':
            v = np.exp(self.a_het * self.s0); sig = np.sqrt(v / v.mean())
        elif design == 'D1w':
            sig = np.sqrt(self.v_w)
        else:
            sig = 1
        u = sig * eps + (self.c_curv * self.q0 if design == 'D3' else 0)
        x = self.C.x + gam * self.sd_xt * u
        return self.beta0 * x + self.Wg + self.sd_u * u, x
