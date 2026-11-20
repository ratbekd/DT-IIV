"""Shared machinery for the R61 additions.

1. Fixed two-function families of mean tendencies E[u h(s)] = 0:
     P  h = (s^2, s^3)                         polynomial benchmark
     B  h = (tanh s, exp(-s^2/2))              fixed bounded pair
     L  h = (s e_c, e_c), e_c = exp(-s^2/2c^2) localized pair, c = 1.5
     R  h = (tanh z, exp(-z^2/2)),             median/MAD-scaled bounded pair
        z = (s - med s) / (1.4826 MAD s)       (eq. 31 of the main text)
2. Calibrations of the fixed-slope test:
     analytic covariance (first-step and angle corrections) with the chi2 cutoff
       and with the adjusted cutoff of eq. (30);
     full-refit delete-cluster jackknife with the chi2 and the Hotelling F cutoff;
     a linearized wild calibration of the jackknife statistic;
     a frozen-first-step jackknife (diagnostic only).
3. Approximate mean independence: the joint statistic t_b = (mu_b, vech Q_b), its
   corrected covariance, and the projection distance D_n(b; eps).
"""
import os, sys
import numpy as np
from scipy.stats import chi2, f as fdist
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from iiv.core import Context  # noqa: E402

ALPHA = 0.05
C_LOC = 1.5
OUT = os.environ.get('IIV_OUT', os.path.join(HERE, '..', '..', 'output', 'r61'))
os.makedirs(OUT, exist_ok=True)


def out(name):
    return os.path.join(OUT, name)


# ---------------------------------------------------------------------- families
def fam_P(s, deriv=True):
    h = np.stack([s ** 2, s ** 3], -1)
    return (h, np.stack([2 * s, 3 * s ** 2], -1)) if deriv else h


def fam_B(s, deriv=True):
    t = np.tanh(s); e = np.exp(-s ** 2 / 2)
    h = np.stack([t, e], -1)
    return (h, np.stack([1 - t ** 2, -s * e], -1)) if deriv else h


def fam_L(s, deriv=True):
    c2 = C_LOC ** 2
    e = np.exp(-s ** 2 / (2 * c2))
    h = np.stack([s * e, e], -1)
    return (h, np.stack([e * (1 - s ** 2 / c2), -s * e / c2], -1)) if deriv else h


def fam_R(s, deriv=True):
    """Median/MAD-scaled bounded pair; location and scale along the last axis.
    The derivative holds them fixed (Proposition 5.5(iv))."""
    med = np.median(s, axis=-1, keepdims=True)
    tau = 1.4826 * np.median(np.abs(s - med), axis=-1, keepdims=True)
    z = (s - med) / tau
    t = np.tanh(z); e = np.exp(-z ** 2 / 2)
    h = np.stack([t, e], -1)
    return (h, np.stack([(1 - t ** 2) / tau, -z * e / tau], -1)) if deriv else h


FAMS = {'P': fam_P, 'B': fam_B, 'L': fam_L, 'R': fam_R}


def make_context(y, x, W1, cl=None):
    return Context(dict(y=np.asarray(y, float), x=np.asarray(x, float), W1=np.asarray(W1, float),
                        Z=np.asarray(x, float)[:, None], cl=cl, lab=''))


# ---------------------------------------------------------------------- analytic covariance
def analytic(C, P, beta, fam):
    """Corrected fixed-slope statistic S_n(beta) for u h(s), the Satterthwaite degrees
    of freedom nu of eq. (30), and the frozen-first-step jackknife statistic."""
    th = C.theta_of(beta, P)
    cs, sn = np.cos(th), np.sin(th)
    s = cs * P.xw + sn * P.rw; u = cs * P.rw - sn * P.xw
    h, d = fam(s)
    g = u[:, None] * h
    gc = C.correct(P, th, beta, s, u, g, u[:, None] * d, h)
    m = g.mean(0)
    gcl = C.rowsum(gc - gc.mean(0))
    Om = gcl.T @ gcl / C.n
    w = np.linalg.solve(Om, m)
    stat = C.n * float(m @ w)
    z = gcl @ w
    nu = min(C.G - 1, 3 * np.sum(z ** 2) ** 2 / np.sum(z ** 4))
    # frozen first step: delete clusters from the raw moment average only
    gsum = C.rowsum(g); ng = np.bincount(C.cl_codes, minlength=C.G)
    Mg = (C.n * m[None, :] - gsum) / (C.n - ng)[:, None]
    T_fr = jk_stat(m, Mg)[0]
    return stat, nu, T_fr


def crit_adj(G, m, nu):
    """Adjusted critical value c_{m,alpha} of eq. (30)."""
    if nu - m + 1 < 1:
        return np.inf
    return G / (G - 1) * nu * m / (nu - m + 1) * fdist.ppf(1 - ALPHA, m, nu - m + 1)


def crit_F(G, m=2):
    """Hotelling critical value for m moments and G clusters, eq. (tail_f)."""
    return m * (G - 1) / (G - m) * fdist.ppf(1 - ALPHA, m, G - m)


# ---------------------------------------------------------------------- full-refit jackknife
def first_step(W, y, x):
    if W.shape[1] == 1:
        xt = x - x.mean(); yt = y - y.mean()
    else:
        co = np.linalg.lstsq(W, np.column_stack([x, y]), rcond=None)[0]
        res = np.column_stack([x, y]) - W @ co
        xt, yt = res[:, 0], res[:, 1]
    b = xt @ yt / (xt @ xt); r = yt - b * xt
    sx = np.sqrt(np.mean(xt ** 2)); sr = np.sqrt(np.mean(r ** 2))
    return xt / sx, r / sr, b, sx, sr


def moments_at(xw, rw, b, sx, sr, betas, fams):
    """Moment means of u h(s) at each tested slope: array (len(betas), len(fams), 2)."""
    th = np.arctan((betas - b) * sx / sr)
    cs = np.cos(th)[:, None]; sn = np.sin(th)[:, None]
    s = cs * xw[None, :] + sn * rw[None, :]
    u = cs * rw[None, :] - sn * xw[None, :]
    return np.stack([(u[..., None] * FAMS[f](s, deriv=False)).mean(1) for f in fams], 1)


def jk_refit(W, y, x, codes, G, betas, fams):
    """Full-sample moments m (nb, F, 2) and delete-cluster moments M (G, nb, F, 2).
    Every deletion re-estimates the projections on the controls, the OLS slope and
    the scales (and, for family R, the median and MAD), holding the slope fixed."""
    betas = np.atleast_1d(np.asarray(betas, float))
    m = moments_at(*first_step(W, y, x), betas, fams)
    M = np.empty((G,) + m.shape)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(G + 1))
    keep = np.ones(len(y), bool)
    for g in range(G):
        ix = order[bounds[g]:bounds[g + 1]]
        keep[ix] = False
        M[g] = moments_at(*first_step(W[keep], y[keep], x[keep]), betas, fams)
        keep[ix] = True
    return m, M


def jk_stat(m, Mg):
    """T_JK = m' V_JK^{-1} m with V_JK = (G-1)/G sum (m_-g - mbar)(m_-g - mbar)'."""
    G = len(Mg)
    d = Mg - Mg.mean(0)
    V = (G - 1) / G * d.T @ d
    return float(m @ np.linalg.solve(V, m)), V, d


def wild_p(T, V, d, B, rng):
    """Linearized wild calibration of T_JK: Rademacher signs on the pseudo-values
    p_g = -(G-1)(m_-g - mbar); T* = q / (1 - q/(G-1)), q = m*' V_JK^{-1} m*."""
    G = len(d)
    p = -(G - 1) * d
    w = rng.integers(0, 2, (B, G)) * 2.0 - 1.0
    ms = w @ p / G
    q = np.einsum('bi,ij,bj->b', ms, np.linalg.inv(V), ms)
    Ts = q / (1 - q / (G - 1))
    return (1 + np.sum(Ts >= T)) / (B + 1)


# ---------------------------------------------------------------------- approximate MI
def ami_tV(C, P, beta, fam=fam_B):
    """t_b = (mu_b, vech Q_b) of eq. (ami_projection) for K = 2 fixed functions and the
    covariance of sqrt(n)(t_hat - t), with every first-step term and the angle
    channel, evaluated at the possibly nonzero moment means (eqs. ami_stack, ami_if)."""
    th = C.theta_of(beta, P)
    cs, sn = np.cos(th), np.sin(th)
    s = cs * P.xw + sn * P.rw; u = cs * P.rw - sn * P.xw
    h, d = fam(s)
    h1, h2, d1, d2 = h[:, 0], h[:, 1], d[:, 0], d[:, 1]
    z0 = np.zeros_like(s)
    g = np.column_stack([u * h1, u * h2, h1, h2, s * h1, s * h2, h1 * h1, h1 * h2, h2 * h2])
    gs = np.column_stack([u * d1, u * d2, d1, d2, h1 + s * d1, h2 + s * d2,
                          2 * h1 * d1, d1 * h2 + h1 * d2, 2 * h2 * d2])
    gu = np.column_stack([h1, h2, z0, z0, z0, z0, z0, z0, z0])
    gc = C.correct(P, th, beta, s, u, g, gs, gu)
    ze = g.mean(0)
    a, c = ze[2:4], ze[4:6]
    t = np.array([ze[0], ze[1], ze[6] - a[0] ** 2 - c[0] ** 2, ze[7] - a[0] * a[1] - c[0] * c[1],
                  ze[8] - a[1] ** 2 - c[1] ** 2])
    J = np.zeros((5, 9))
    J[0, 0] = J[1, 1] = 1
    J[2, 6] = 1; J[2, 2] = -2 * a[0]; J[2, 4] = -2 * c[0]
    J[3, 7] = 1; J[3, 2] = -a[1]; J[3, 3] = -a[0]; J[3, 4] = -c[1]; J[3, 5] = -c[0]
    J[4, 8] = 1; J[4, 3] = -2 * a[1]; J[4, 5] = -2 * c[1]
    psi = C.rowsum((gc - gc.mean(0)) @ J.T)
    return t, psi.T @ psi / C.n


def _feasible(t, eps):
    Q = np.array([[t[2], t[3]], [t[3], t[4]]])
    ev = np.linalg.eigvalsh(Q)
    if ev[0] <= 0:
        return False
    return float(t[:2] @ np.linalg.solve(Q, t[:2])) <= eps ** 2


def ami_D(t, V, n, eps, cut=None):
    """D_n(b; eps) = inf over the admissible set A_eps of n (t_hat - t)' V^{-1} (t_hat - t).
    A_eps = {(mu, Q): [[Q, mu], [mu', eps^2]] >= 0}, parametrized by Q = C C' and
    mu = C a with |a| <= eps. With cut, the search stops once the comparison of
    D_n with cut is settled; the returned value is then on the correct side of cut."""
    if _feasible(t, eps):
        return 0.0
    Vi = np.linalg.inv(V)

    def obj_t(tt):
        e = t - tt
        return n * float(e @ Vi @ e)

    best = np.inf
    # eps = 0, or a start for eps > 0: mu = 0 and Q profiled out
    Vmm, VQm = V[:2, :2], V[2:, :2]
    tq = t[2:] - VQm @ np.linalg.solve(Vmm, t[:2])
    Q0 = np.array([[tq[0], tq[1]], [tq[1], tq[2]]])
    if np.linalg.eigvalsh(Q0)[0] > 0:
        best = n * float(t[:2] @ np.linalg.solve(Vmm, t[:2]))
        if eps == 0:
            return best

    def unpack(p):
        Cm = np.array([[p[0], 0.0], [p[1], p[2]]])
        Q = Cm @ Cm.T
        mu = Cm @ p[3:5]
        return np.array([mu[0], mu[1], Q[0, 0], Q[0, 1], Q[1, 1]])

    Qh = np.array([[t[2], t[3]], [t[3], t[4]]])
    starts = []
    for Qs in (Qh, Q0):
        if np.linalg.eigvalsh(Qs)[0] > 0:
            L = np.linalg.cholesky(Qs)
            a0 = np.linalg.solve(L, t[:2])
            na = np.linalg.norm(a0)
            if na > eps:
                a0 = a0 * (eps / na if na > 0 else 0.0)
            starts.append(np.r_[L[0, 0], L[1, 0], L[1, 1], a0])
    if not starts:
        starts.append(np.r_[1.0, 0.0, 1.0, 0.0, 0.0])
    # the solution has |a| = eps when t_hat is infeasible: also start on the boundary
    base = starts[0]
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        starts.append(np.r_[base[:3], eps * np.cos(ang), eps * np.sin(ang)])
    scale = max(obj_t(unpack(base)), 1e-8)
    cons = [{'type': 'ineq', 'fun': lambda p: eps ** 2 - p[3] ** 2 - p[4] ** 2}]
    for k, p0 in enumerate(starts):
        r = minimize(lambda p: obj_t(unpack(p)) / scale, p0, method='SLSQP', constraints=cons,
                     options={'ftol': 1e-14, 'maxiter': 1000})
        p = r.x.copy()
        na = np.hypot(p[3], p[4])
        if na > eps:                       # put a marginally infeasible iterate on the boundary
            p[3:5] *= eps / na
        val = obj_t(unpack(p))
        if val < best:
            best = float(val)
        if cut is not None:
            # the decision D <= cut is all that is needed: a feasible value below the
            # cutoff settles it, and so does a value far above it after two starts
            if best <= cut or (k >= 1 and best > 4 * cut):
                break
    return best


def ami_box(t, V, n, eps, bound=(1.0, 1.0)):
    """Simpler conservative statistic: distance of mu_hat from the box |mu_j| <= eps M_j."""
    Vi = np.linalg.inv(V[:2, :2]); mu = t[:2]
    lim = eps * np.asarray(bound)
    if np.all(np.abs(mu) <= lim):
        return 0.0
    f = lambda v: n * float((mu - v) @ Vi @ (mu - v))
    r = minimize(f, np.clip(mu, -lim, lim), method='L-BFGS-B', bounds=[(-lim[0], lim[0]), (-lim[1], lim[1])])
    return float(r.fun)


def eps_enter(t, V, n, cut, tol=1e-4):
    """Smallest tolerance in [0, 1] at which the slope enters the set; nan if never."""
    if ami_D(t, V, n, 0.0, cut) <= cut:
        return 0.0
    if ami_D(t, V, n, 1.0, cut) > cut:
        return np.nan
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if ami_D(t, V, n, mid, cut) <= cut:
            hi = mid
        else:
            lo = mid
    return hi


def comps(acc, grid):
    """Connected components of an acceptance vector: list of (lo, hi, touches_edge)."""
    acc = np.asarray(acc, bool); res = []; i = 0; N = len(acc)
    while i < N:
        if acc[i]:
            j = i
            while j + 1 < N and acc[j + 1]:
                j += 1
            res.append((float(grid[i]), float(grid[j]), bool(i == 0 or j == N - 1)))
            i = j + 1
        else:
            i += 1
    return res
