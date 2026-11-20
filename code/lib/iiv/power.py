"""Sets between the one-sided and the CH sets (Remark rem:power of the main paper;
Subsection S5.7 of the supplement).

The one-sided moments E[u s^2] = E[u s^3] = 0 hold exactly; the variance tendencies
of orders one and two are bounded, |E[(u^2-1) h(s)]| <= c sd(u^2) sd(h(s)) for
h(s) = s, s^2. bounded_test() gives the statistic minimized over the bounds and a
least-favourable critical value with moment selection. It reuses the first-step
corrected contributions of Context.gcorr().
"""
import itertools, numpy as np
from scipy.stats import chi2
from .core import ALPHA

ONE_SIDED = [2, 4]          # E[u s^2], E[u s^3]
CH_COLS = [1, 5]            # E[(u^2-1) s] = E[u^2 s], E[(u^2-1) s^2] = E[u^2 s^2] - 1


# ---------------------------------------------------------------- small box-constrained quadratic programmes
def box_min(Z, Wt, lo, hi):
    """min over tau with lo <= tau <= hi of (Z - E tau)' Wt (Z - E tau), where tau
    enters only the last k coordinates (k = len(lo)). Z is (R, m). Bounds may be
    infinite. Exact, by enumeration of the 3^k faces; vectorized over the rows of Z."""
    R, m = Z.shape; k = len(lo); j0 = m - k
    best = np.full(R, np.inf)
    for face in itertools.product((0, 1, 2), repeat=k):      # 0 free, 1 at lo, 2 at hi
        fixed = [i for i in range(k) if face[i]]
        if any(not np.isfinite(lo[i] if face[i] == 1 else hi[i]) for i in fixed):
            continue
        tau = np.zeros((R, k))
        for i in fixed:
            tau[:, i] = lo[i] if face[i] == 1 else hi[i]
        free = [i for i in range(k) if not face[i]]
        if free:
            fi = [j0 + i for i in free]
            Zr = Z.copy(); Zr[:, j0:] -= tau
            A = Wt[np.ix_(fi, fi)]; b = Zr @ Wt[:, fi]
            tf = np.linalg.solve(A, b.T).T
            tau[:, free] = tf
            ok = np.all((tf >= np.asarray(lo)[free] - 1e-12) & (tf <= np.asarray(hi)[free] + 1e-12), axis=1)
        else:
            ok = np.ones(R, bool)
        D = Z.copy(); D[:, j0:] -= tau
        val = np.einsum('ij,jk,ik->i', D, Wt, D)
        best = np.where(ok & (val < best), val, best)
    return best


# ---------------------------------------------------------------- bounded variance tendencies
def moments(C, P, beta):
    """Estimates v = (E[u s^2], E[u s^3], rho_1, rho_2) at H0: beta and their
    first-step corrected influence contributions (n x 4), where
    rho_1 = corr(u^2, s) = E[u^2 s] / sd(u^2) and rho_2 = corr(u^2, s^2) =
    {E[u^2 s^2] - 1} / {sd(u^2) sd(s^2)}. In the sample E[s] = 0 and
    E[s^2] = E[u^2] = 1 exactly, so sd(u^2)^2 = E[u^4] - 1 and sd(s^2)^2 = E[s^4] - 1."""
    th = C.theta_of(beta, P); cs, sn = np.cos(th), np.sin(th)
    s = cs * P.xw + sn * P.rw; u = cs * P.rw - sn * P.xw
    g = np.column_stack([u * s ** 2, u * s ** 3, u ** 2 * s, u ** 2 * s ** 2 - 1, u ** 4, s ** 4])
    gs = np.column_stack([2 * u * s, 3 * u * s ** 2, u ** 2, 2 * u ** 2 * s, 0 * s, 4 * s ** 3])
    gu = np.column_stack([s ** 2, s ** 3, 2 * u * s, 2 * u * s ** 2, 4 * u ** 3, 0 * u])
    gc = C.correct(P, th, beta, s, u, g, gs, gu)
    m = g.mean(0); a, b = m[4] - 1, m[5] - 1                 # var(u^2), var(s^2)
    psi = gc - gc.mean(0)
    s1, s2 = np.sqrt(a), np.sqrt(a * b)
    r1, r2 = m[2] / s1, m[3] / s2
    p1 = psi[:, 2] / s1 - r1 * psi[:, 4] / (2 * a)
    p2 = psi[:, 3] / s2 - r2 * (psi[:, 4] / (2 * a) + psi[:, 5] / (2 * b))
    return np.array([m[0], m[1], r1, r2]), np.column_stack([psi[:, 0], psi[:, 1], p1, p2])


def envelope(v, se, c, n, kappa):
    """Box for the critical value, on the sqrt(n) scale of tau = sqrt(n)(t - v0).
    For each bounded coordinate, [m_lo, m_hi] is the set of correlations in [-c, c]
    within kappa standard errors of the estimate; the box
    [-sqrt(n)(c + m_lo), sqrt(n)(c - m_hi)] is contained in the true box whenever the
    true correlation lies in [m_lo, m_hi], so its quantile is conservative."""
    lo, hi = np.zeros(2), np.zeros(2)
    for j in range(2):
        r, d = v[2 + j], kappa * se[j] / np.sqrt(n)
        a, b = max(-c, r - d), min(c, r + d)
        if a > b:                                            # estimate far outside the bounds
            a = b = float(np.clip(r, -c, c))
        lo[j], hi[j] = -np.sqrt(n) * (c + a), np.sqrt(n) * (c - b)
    return lo, hi


def bounded_test(C, P, beta, c, Zstd, kappa=None):
    """Test of H0: beta with E[u s^2] = E[u s^3] = 0 and |corr(u^2, h_j(s))| <= c for
    h_1(s) = s, h_2(s) = s^2. Returns (T, critical value).
    T = min over |t_j| <= c of n (v - t)' Sigma^{-1} (v - t), with t zero in the two
    one-sided rows. The critical value is the (1 - alpha) quantile of the same
    minimum for N(0, Sigma) draws over the envelope box (Proposition S5.1);
    kappa = sqrt(log G), G clusters (G = n without clusters).
    c = 0 gives a test under CH with chi2_4 critical value; c = inf gives the
    one-sided S test with chi2_2. Zstd: (R, 4) standard normal draws, fixed across
    beta and c (common random numbers)."""
    n = C.n; kappa = np.sqrt(np.log(C.G)) if kappa is None else kappa
    v, psi = moments(C, P, beta)
    Sig = C.meat(psi); Wt = np.linalg.inv(Sig); z = np.sqrt(n) * v
    if np.isinf(c):
        return float(z[:2] @ np.linalg.solve(Sig[:2, :2], z[:2])), chi2.ppf(1 - ALPHA, 2)
    T = float(box_min(z[None, :], Wt, -np.sqrt(n) * c * np.ones(2), np.sqrt(n) * c * np.ones(2))[0])
    if c == 0:
        return T, chi2.ppf(1 - ALPHA, 4)
    lo, hi = envelope(v, np.sqrt(np.diag(Sig))[2:], c, n, kappa)
    Zs = Zstd @ np.linalg.cholesky(Sig).T
    return T, float(np.quantile(box_min(Zs, Wt, lo, hi), 1 - ALPHA))


def breakdown(C, P, b, Zstd, c_max=1.0, tol=1e-3, ngrid=201):
    """Breakdown value for the conclusion 'beta != b': the smallest c at which b
    enters the set. Returns 0 if b is in the set at c = 0 and inf if b is outside
    the set at every c up to c_max and outside the one-sided set."""
    acc = lambda c: (lambda r: r[0] <= r[1])(bounded_test(C, P, b, c, Zstd))
    if acc(0.0):
        return 0.0
    grid = np.linspace(0, c_max, ngrid)
    hit = next((c for c in grid[1:] if acc(c)), None)
    if hit is None:
        return np.inf if not acc(np.inf) else c_max
    lo, hi = hit - grid[1], hit
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if acc(mid) else (mid, hi)
    return hi


def c_nonempty(C, P, Zstd, c_max=1.0, tol=1e-3, ngrid=241):
    """Smallest bound c at which the set is nonempty on the labelling domain:
    the least heteroscedasticity, in the units of c, that the data require
    together with MI. Bisection on c; returns 0 if the set under CH is nonempty."""
    bg = C.beta_at(np.linspace(-np.pi / 4, np.pi / 4, ngrid), P)
    ne = lambda c: any((lambda r: r[0] <= r[1])(bounded_test(C, P, b, c, Zstd)) for b in bg)
    if ne(0.0):
        return 0.0
    if not ne(c_max):
        return np.inf
    lo, hi = 0.0, c_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ne(mid) else (mid, hi)
    return hi
