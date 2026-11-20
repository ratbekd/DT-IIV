"""VT-IIV estimator core (port of sim_estimator_DT_IIV_v21_clean.R).

Homoscedastic VT moment only. The sign is selected first and the scale delta is
then estimated with that sign fixed. Estimation is by continuously updated GMM
(CUE), with the scalar v16 moment-profile path kept for the original
Banerjee-Iyer selection. Inference: corrected-score analytic variance, the
identification-robust S-set, and a full pairs bootstrap in which the sign
selected on the sample is imposed on every resample (BCa intervals by default).

Notation as in the R code:
    r        = (I - P_X) y              residual direction
    S(delta) = X + r (k o delta)'       intrinsic instrument
    e_i      = first-stage residual of X on S
    m_i      = (e_i^2 - sigma^2) o s_i  VT moment
Results are dicts with the same field names as the R lists.

Differences from R, all numerical: optim(L-BFGS-B) is replaced by SciPy's
L-BFGS-B with R's finite-difference gradient (step 1e-3, clamped at the bounds),
memory 5 and R's tolerances; random numbers come from NumPy, so bootstrap draws
and the legacy-v16 random starts differ draw by draw from R.
"""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2, norm

EPS = np.finfo(float).eps


# ---------------------------------------------------------------- small helpers
def _ok(*a):
    return all(np.all(np.isfinite(np.asarray(x, float))) for x in a)


def _solve(A, b=None, tol=1e-12):
    A = np.atleast_2d(np.asarray(A, float))
    if not _ok(A) or A.shape[0] != A.shape[1]:
        return None
    try:
        rc = 1.0 / np.linalg.cond(A, 1)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(rc) or rc < tol:
        return None
    try:
        return np.linalg.inv(A) if b is None else np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _sd(v, axis=None):
    return np.std(v, axis=axis, ddof=1)


def _col_cor(S, X):
    Sc = S - S.mean(0); Xc = X - X.mean(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sum(Sc * Xc, 0) / np.sqrt(np.sum(Sc ** 2, 0) * np.sum(Xc ** 2, 0))


def uniroot(f, lo, hi, tol=EPS ** 0.25):
    return brentq(f, lo, hi, xtol=tol, rtol=4 * EPS)


def optim_lbfgsb(fun, x0, lower, upper, maxit=500, ndeps=1e-3):
    """R's optim(method = 'L-BFGS-B') with its finite-difference gradient."""
    lower = np.asarray(lower, float); upper = np.asarray(upper, float)

    def grad(x):
        g = np.empty_like(x)
        for i in range(len(x)):
            up = x.copy(); dn = x.copy()
            e1 = ndeps; t = x[i] + ndeps
            if t > upper[i]:
                t = upper[i]; e1 = t - x[i]
            up[i] = t
            e2 = ndeps; t = x[i] - ndeps
            if t < lower[i]:
                t = lower[i]; e2 = x[i] - t
            dn[i] = t
            g[i] = (fun(up) - fun(dn)) / (e1 + e2) if e1 + e2 > 0 else 0.0
        return g
    r = minimize(fun, np.asarray(x0, float), jac=grad, method='L-BFGS-B', bounds=list(zip(lower, upper)),
                 options=dict(maxiter=maxit, maxcor=5, ftol=1e7 * EPS, gtol=0.0))
    return dict(par=np.asarray(r.x, float), value=float(r.fun), convergence=0 if r.success else 1)


# ---------------------------------------------------------------- projections and data
def partial_out(z, W):
    """Rank-safe FWL residual on [1, W] (QR with pivoting, tolerance 1e-10)."""
    from scipy.linalg import qr
    Z = np.asarray(z, float); Z = Z[:, None] if Z.ndim == 1 else Z
    W = np.asarray(W, float); W = W[:, None] if W.ndim == 1 else W
    if Z.shape[0] != W.shape[0]:
        raise ValueError('z and W must have the same number of rows.')
    if not _ok(Z, W):
        raise ValueError('z and W must contain only finite values.')
    D = np.column_stack([np.ones(len(W)), W])
    Q, R, piv = qr(D, mode='economic', pivoting=True)
    d = np.abs(np.diag(R)); rank = int(np.sum(d > 1e-10 * d[0])) if len(d) else 0
    if rank < 1:
        raise ValueError('the control design has zero rank.')
    Q = Q[:, :rank]
    return Z - Q @ (Q.T @ Z)


def siv_partialled_data(y, X, W=None):
    y = np.asarray(y, float); X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X
    if not len(y) or X.shape[0] != len(y) or X.shape[1] < 1:
        raise ValueError('y and X have incompatible dimensions.')
    if not _ok(y, X):
        raise ValueError('y and X must be finite.')
    if W is not None:
        W = np.asarray(W, float); W = W[:, None] if W.ndim == 1 else W
        if W.shape[0] != len(y) or not _ok(W):
            raise ValueError('W has incompatible dimensions or non-finite values.')
        yt = partial_out(y, W)[:, 0]; Xt = partial_out(X, W)
    else:
        yt = y - y.mean(); Xt = X - X.mean(0)
    return dict(y_tilde=yt, X_tilde=Xt)


def make_r(y_tilde, X_tilde):
    X = np.asarray(X_tilde, float)
    b = _solve(X.T @ X, X.T @ y_tilde)
    if b is None:
        raise ValueError("X'X is singular in make_r().")
    return y_tilde - X @ b


def make_S(X_tilde, r, delta, k):
    X = np.asarray(X_tilde, float); p = X.shape[1]
    R = np.asarray(r, float)
    kd = np.resize(np.asarray(k, float), p) * np.resize(np.asarray(delta, float), p)
    if not _ok(X, R, kd):
        raise ValueError('invalid inputs to make_S().')
    if R.ndim == 1 or R.shape[1] == 1:
        return X + np.outer(R.reshape(-1), kd)
    if R.shape[1] == p:
        return X + R * kd
    raise ValueError('r must have either 1 column or ncol(X) columns.')


def siv_prepare(y, X, W=None, scale='none'):
    P = siv_partialled_data(y, X, W); Xt = P['X_tilde']; rt = make_r(P['y_tilde'], Xt); p = Xt.shape[1]
    if scale == 'sd':
        sx = _sd(Xt, 0); sr = _sd(rt)
        if not _ok(sx) or np.any(sx <= 1e-12) or not np.isfinite(sr) or sr <= 1e-12:
            raise ValueError('cannot SD-standardize a zero-variance residualized variable.')
    else:
        sx = np.ones(p); sr = 1.0
    return dict(y_tilde=P['y_tilde'], X_tilde=Xt, r_raw=rt, X_work=Xt / sx, r_work=rt / sr,
                x_scale=np.asarray(sx, float), r_scale=float(sr), scale=scale, projection_method='rank_safe_qr')


# ---------------------------------------------------------------- the VT moment
def dt_moments_hat(delta, X_tilde, r, k, relevance_c=0.01, penalty=1e12):
    X = np.asarray(X_tilde, float); n, p = X.shape
    delta = np.resize(np.asarray(delta, float), p)
    if not _ok(delta) or np.any(delta <= 0):
        return dict(ok=False, penalty=penalty, reason='invalid_delta')
    S = make_S(X, r, delta, k)
    rel = _col_cor(S, X)
    if not _ok(rel) or np.any(np.abs(rel) < relevance_c):
        slack = np.maximum(relevance_c - np.abs(rel), 0) if _ok(rel) else np.ones(p)
        return dict(ok=False, penalty=penalty + np.sum(slack ** 2) * penalty, reason='relevance', relevance=rel)
    Gam = _solve(S.T @ S, S.T @ X)
    if Gam is None:
        return dict(ok=False, penalty=penalty, reason='singular_first_stage')
    E = X - S @ Gam; sig2 = np.mean(E ** 2, 0)
    m = (E ** 2 - sig2) * S
    if not _ok(m):
        return dict(ok=False, penalty=penalty, reason='nonfinite_moment')
    return dict(ok=True, S=S, Gamma=Gam, E=E, sigma2=sig2, m_mat=m, M_hat=m.mean(0), relevance=rel)


def _weight_inverse(m, ridge=1e-10):
    m = np.atleast_2d(m); n, q = m.shape; mc = m - m.mean(0)
    return _solve(mc.T @ mc / n + ridge * np.eye(q))


def _starts(lower, upper, n_starts, first):
    q = len(lower); out = [np.minimum(np.maximum(np.asarray(first, float), lower), upper)]
    if n_starts <= 1:
        return out
    phi = np.sqrt(np.arange(1, q + 1) + 1)
    for s in range(1, n_starts):
        out.append(lower + np.mod(s * phi, 1) * (upper - lower))
    return out


def estimate_delta_dt_cue(X_tilde, r, k, delta_lower, delta_upper, relevance_c=0.01, n_grid=200, n_starts=8,
                          max_cycles=3, ridge=1e-10, penalty=1e12, init_delta=None):
    X = np.asarray(X_tilde, float); n, p = X.shape
    dl = np.resize(np.asarray(delta_lower, float), p); du = np.resize(np.asarray(delta_upper, float), p)
    if not _ok(dl, du) or np.any(dl <= 0) or np.any(du <= dl):
        raise ValueError('need 0 < delta_lower < delta_upper for every component.')
    n_grid = max(10, int(n_grid)); n_starts = max(1, int(n_starts)); max_cycles = max(1, int(max_cycles))

    def moments(d):
        z = dt_moments_hat(d, X, r, k, relevance_c=relevance_c, penalty=penalty)
        return z if z['ok'] else None

    def objective(d, cue=True):
        z = moments(d)
        if z is None:
            return penalty
        Wm = _weight_inverse(z['m_mat'], ridge) if cue else np.eye(p)
        if Wm is None:
            return penalty
        val = float(n * z['M_hat'] @ Wm @ z['M_hat'])
        return val if np.isfinite(val) else penalty

    def stage(cue, start):
        dcur = np.minimum(np.maximum(np.asarray(start, float), dl), du)
        best = dict(par=dcur.copy(), value=objective(dcur, cue))
        for _ in range(max_cycles):
            old = dcur.copy()
            for j in range(p):
                grid = np.linspace(dl[j], du[j], n_grid); vals = []
                for g in grid:
                    d = dcur.copy(); d[j] = g; vals.append(objective(d, cue))
                ii = int(np.argmin(vals)); dcur[j] = grid[ii]
                if np.isfinite(vals[ii]) and vals[ii] < best['value']:
                    best = dict(par=dcur.copy(), value=vals[ii])
            if np.max(np.abs(dcur - old)) <= np.max((du - dl) / n_grid):
                break
        for st in _starts(dl, du, n_starts, best['par']):
            try:
                op = optim_lbfgsb(lambda d: objective(d, cue), st, dl, du)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if np.isfinite(op['value']) and op['value'] < best['value']:
                best = dict(par=op['par'], value=op['value'], optim_convergence=op['convergence'])
        return best

    d0 = np.asarray(init_delta, float) if init_delta is not None and np.size(init_delta) == p and _ok(init_delta) \
        else (dl + du) / 2
    prelim = stage(False, d0); best = stage(True, prelim['par']); z = moments(best['par'])
    return dict(delta_hat=np.asarray(best['par'], float), J_hat=float(best['value']),
                converged=z is not None and np.isfinite(best['value']) and best['value'] < 0.99 * penalty,
                moment=np.full(p, np.nan) if z is None else z['M_hat'],
                relevance=np.full(p, np.nan) if z is None else z['relevance'],
                S=None if z is None else z['S'], Gamma=None if z is None else z['Gamma'],
                E=None if z is None else z['E'], weighting='continuously_updated_ordinary_dt_score',
                delta_lower=dl, delta_upper=du)


# ---------------------------------------------------------------- legacy v16 scalar path
def estimate_delta_dt_v16_identity(X_tilde, r, k, delta_lower, delta_upper, relevance_c=0.01, n_grid=300,
                                   n_starts=10, max_cycles=3, seed=1, penalty=1e12):
    X = np.asarray(X_tilde, float)
    if X.shape[1] != 1:
        raise ValueError('legacy_v16 selection is available only for p=1.')
    n = X.shape[0]; dl = float(np.ravel(delta_lower)[0]); du = float(np.ravel(delta_upper)[0])
    if not np.isfinite(dl) or not np.isfinite(du) or dl < 0 or du <= dl:
        raise ValueError('legacy_v16 requires 0 <= delta_lower < delta_upper.')
    n_grid = max(10, int(n_grid)); n_starts = max(1, int(n_starts)); max_cycles = max(1, int(max_cycles))

    def moments(d):
        z = dt_moments_hat(d, X, r, k, relevance_c=relevance_c, penalty=penalty)
        return z if z['ok'] else None

    def objective(d):
        z = dt_moments_hat(d, X, r, k, relevance_c=relevance_c, penalty=penalty)
        if not z['ok']:
            return z['penalty'] if np.isfinite(z['penalty']) else penalty
        v = float(n * z['M_hat'] @ z['M_hat'])
        return v if np.isfinite(v) else penalty

    dcur = dl; best = dict(par=dcur, value=objective(np.array([dcur])))
    grid = np.linspace(dl, du, n_grid); values = np.full(n_grid, np.nan)
    for _ in range(max_cycles):
        prev = dcur
        values = np.array([objective(np.array([g])) for g in grid]); i = int(np.argmin(values)); dcur = grid[i]
        if np.isfinite(values[i]) and values[i] < best['value']:
            best = dict(par=dcur, value=values[i])
        if abs(dcur - prev) <= (du - dl) / n_grid:
            break

    def run(start):
        nonlocal best
        try:
            op = optim_lbfgsb(objective, [start], [dl], [du])
        except (ValueError, np.linalg.LinAlgError):
            return
        if np.isfinite(op['value']) and op['value'] < best['value']:
            best = dict(par=float(op['par'][0]), value=op['value'], optim_convergence=op['convergence'])
    run(best['par'])
    rng = np.random.default_rng(int(seed))
    for _ in range(max(0, n_starts - 1)):
        run(rng.uniform(dl, du))
    z = moments(np.array([best['par']]))
    return dict(delta_hat=np.array([best['par']], float), J_hat=float(best['value']),
                converged=z is not None and np.isfinite(best['value']) and best['value'] < 0.99 * penalty,
                moment=np.nan if z is None else z['M_hat'], relevance=np.nan if z is None else z['relevance'],
                S=None if z is None else z['S'], Gamma=None if z is None else z['Gamma'], E=None if z is None else z['E'],
                grid=grid, grid_objective=values, weighting='identity-weighted v16 scalar profile',
                delta_lower=dl, delta_upper=du)


def _v16_root_profile(X_tilde, r, k, delta_lower, delta_upper, relevance_c=0.01, interior_buffer=0.0, n_grid=300,
                      root_tol=1e-10):
    X = np.asarray(X_tilde, float)
    if X.shape[1] != 1:
        raise ValueError('the v16 scalar root profile requires p=1.')
    n = X.shape[0]; dl = float(np.ravel(delta_lower)[0]); du = float(np.ravel(delta_upper)[0])
    grid = np.linspace(dl, du, max(20, int(n_grid)))

    def evaluate(d):
        z = dt_moments_hat(np.array([d]), X, r, k, relevance_c=0)
        return (np.nan, np.nan) if not z['ok'] else (float(z['M_hat'][0]), float(z['relevance'][0]))
    vals = np.array([evaluate(g) for g in grid]); moment, relevance = vals[:, 0], vals[:, 1]
    valid = np.isfinite(moment) & np.isfinite(relevance) & (np.abs(relevance) >= relevance_c)
    gi = np.where(valid)[0][np.argmin(np.abs(moment[valid]))] if valid.any() else None
    ig = valid & (grid - dl > interior_buffer)
    ii = np.where(ig)[0][np.argmin(np.abs(moment[ig]))] if ig.any() else None
    roots = []
    if valid.any():
        stol = root_tol * max(1, np.max(np.abs(moment[valid])))
        roots += list(grid[valid & (np.abs(moment) <= stol)])
        for i in range(len(grid) - 1):
            if not valid[i] or not valid[i + 1] or moment[i] * moment[i + 1] >= 0:
                continue
            try:
                roots.append(uniroot(lambda d: evaluate(d)[0], grid[i], grid[i + 1], tol=root_tol))
            except ValueError:
                pass
    roots = np.sort(np.array([v for v in roots if np.isfinite(v)]))
    if len(roots) > 1:
        sep = max(root_tol * 10, (du - dl) / (len(grid) * 100))
        roots = roots[np.r_[True, np.diff(roots) > sep]]
    rv = np.array([evaluate(v) for v in roots]).reshape(-1, 2)
    ok = (roots - dl > interior_buffer) & np.isfinite(rv[:, 0]) & np.isfinite(rv[:, 1]) & (np.abs(rv[:, 1]) >= relevance_c) \
        if len(roots) else np.zeros(0, bool)
    roots = roots[ok]; rv = rv[ok]
    robj = n * rv[:, 0] ** 2 if len(roots) else np.zeros(0)
    sel = int(np.argmin(robj)) if len(roots) else None
    return dict(grid=grid, moment=moment, relevance=relevance,
                global_delta=np.nan if gi is None else grid[gi], global_abs_moment=np.inf if gi is None else abs(moment[gi]),
                interior_delta=np.nan if ii is None else grid[ii], interior_abs_moment=np.inf if ii is None else abs(moment[ii]),
                roots=roots, root_values=rv, root_count=len(roots),
                selected_root_delta=np.nan if sel is None else roots[sel],
                selected_root_objective=np.inf if sel is None else robj[sel])


def estimate_delta_dt_v16_given_sign(X_tilde, r, k_fixed, delta_lower, delta_upper, relevance_c=0.01,
                                     legacy_interior_frac=0.05, n_grid=300, n_starts=10, max_cycles=3, seed=1,
                                     init_delta=None, root_tol=1e-10):
    X = np.asarray(X_tilde, float)
    if X.shape[1] != 1:
        raise ValueError('legacy_v16 conditional scale estimation requires p=1.')
    k_fixed = int(np.ravel(k_fixed)[0])
    if k_fixed not in (-1, 1):
        raise ValueError('k_fixed must be either -1 or +1.')
    dl = float(np.ravel(delta_lower)[0]); du = float(np.ravel(delta_upper)[0])
    buf = float(legacy_interior_frac) * (du - dl)
    profile = _v16_root_profile(X, r, k_fixed, dl, du, relevance_c, buf, n_grid, root_tol)
    try:
        idf = estimate_delta_dt_v16_identity(X, r, k_fixed, dl, du, relevance_c, n_grid, n_starts, max_cycles, seed)
    except ValueError:
        idf = None
    idd = np.nan if idf is None else float(idf['delta_hat'][0])
    interior = idf is not None and idf['converged'] and np.isfinite(idd) and idd - dl > buf
    cands = profile['roots']
    if interior:
        dh = idd; reason = 'conditional_v16_identity_objective'
    elif len(cands):
        hint = np.nan if init_delta is None else float(np.ravel(init_delta)[0])
        dh = cands[int(np.argmin(np.abs(cands - hint)))] if np.isfinite(hint) else cands[0]
        reason = 'conditional_interior_dt_root'
    elif np.isfinite(profile['interior_delta']):
        dh = profile['interior_delta']; reason = 'conditional_v16_grid_minimum'
    else:
        return dict(converged=False, delta_hat=np.array([np.nan]), J_hat=np.inf, moment=np.nan, relevance=np.nan, S=None,
                    Gamma=None, E=None, k_fixed=k_fixed, profile=profile, identity_fit=idf,
                    scale_reason='no_conditional_scale_solution', delta_lower=dl, delta_upper=du)
    z = dt_moments_hat(np.array([dh]), X, r, k_fixed, relevance_c=relevance_c); c = z['ok']
    mom = float(z['M_hat'][0]) if c else np.nan
    return dict(converged=c, delta_hat=np.array([dh], float), J_hat=X.shape[0] * mom ** 2 if c else np.inf, moment=mom,
                relevance=float(z['relevance'][0]) if c else np.nan, S=z.get('S') if c else None,
                Gamma=z.get('Gamma') if c else None, E=z.get('E') if c else None, k_fixed=k_fixed, profile=profile,
                identity_fit=idf, scale_reason=reason,
                weighting='identity-weighted v16 scalar profile conditional on sign', delta_lower=dl, delta_upper=du)


def siv_sign_search_v16(X_tilde, r, delta_lower, delta_upper, relevance_c=0.01, legacy_interior_frac=0.05,
                        weak_frac=0.90, id_level=0.95, id_grid=200, n_grid=300, n_starts=10, max_cycles=3, seed=1,
                        moment_ratio_thr=5, compute_id_set=True):
    X = np.asarray(X_tilde, float)
    if X.shape[1] != 1:
        raise ValueError('legacy_v16 selection is available only for p=1.')
    dl = float(np.ravel(delta_lower)[0]); du = float(np.ravel(delta_upper)[0]); buf = legacy_interior_frac * (du - dl)
    K = [-1, 1]; fits = []
    for k in K:
        try:
            fits.append(estimate_delta_dt_v16_identity(X, r, k, dl, du, relevance_c, n_grid, n_starts, max_cycles, seed))
        except ValueError:
            fits.append(None)
    gobj = np.array([np.inf if (z is None or not z['converged']) else z['J_hat'] for z in fits])
    gdel = np.array([np.nan if z is None else float(z['delta_hat'][0]) for z in fits])
    gint = np.isfinite(gdel) & (gdel - dl > buf)
    profs = [_v16_root_profile(X, r, k, dl, du, relevance_c, buf, n_grid) for k in K]
    rc = np.array([pp['root_count'] for pp in profs]); rd = np.array([pp['selected_root_delta'] for pp in profs])
    ro = np.array([pp['selected_root_objective'] for pp in profs])
    elig = np.where((rc > 0) & np.isfinite(ro))[0]; feas = np.where(np.isfinite(gobj))[0]
    gw = feas[np.argmin(gobj[feas])] if len(feas) else None
    sel = None; dh = np.nan; sobj = np.inf; reason = 'no_interior_signal'
    if gw is not None and gint[gw]:
        sel = gw; dh = gdel[sel]; sobj = gobj[sel]; reason = 'legacy_v16_global_objective'
    elif len(elig):
        sel = elig[np.argmin(ro[elig])]; dh = rd[sel]; sobj = ro[sel]; reason = 'interior_dt_root'
    else:
        ia = np.array([pp['interior_abs_moment'] for pp in profs])
        ch = None
        if np.all(np.isfinite(ia)):
            if ia[0] * moment_ratio_thr < ia[1]:
                ch = 0
            elif ia[1] * moment_ratio_thr < ia[0]:
                ch = 1
        if ch is not None:
            sel = ch; dh = profs[sel]['interior_delta']; sobj = X.shape[0] * ia[sel] ** 2; reason = 'v16_moment_ratio'
        elif gint.sum() == 1:
            sel = int(np.where(gint)[0][0]); dh = gdel[sel]; sobj = gobj[sel]; reason = 'unique_interior_global_minimum'
    table = dict(k1=K, global_objective=gobj, global_delta=gdel, global_interior=gint, interior_root_count=rc,
                 root_delta=rd, root_objective=ro, grid_interior_delta=[pp['interior_delta'] for pp in profs],
                 grid_abs_moment=[pp['interior_abs_moment'] for pp in profs],
                 global_winner=[gw is not None and i == gw for i in range(2)], selected=[sel is not None and i == sel for i in range(2)])
    tm = third_moments(X, r)
    base = dict(sign_table=table, delta_lower=dl, delta_upper=du, third_moments=tm, selection_reason=reason,
                selection_method='legacy_v16')
    if sel is None or not np.isfinite(dh):
        return dict(ok=False, status=['no_admissible_sign'], k_hat=np.array([np.nan]), delta_hat=np.array([np.nan]),
                    objective=np.nan, selected_fit=None, id_set=dict(supported=False), **base)
    z = dt_moments_hat(np.array([dh]), X, r, K[sel], relevance_c=relevance_c)
    best = dict(delta_hat=np.array([dh]), J_hat=sobj, converged=z['ok'], moment=z.get('M_hat'), relevance=z.get('relevance'),
                S=z.get('S'), Gamma=z.get('Gamma'), E=z.get('E'), weighting='v16 interior scalar DT profile')
    if not z['ok']:
        return dict(ok=False, status=['selected_sign_infeasible'], k_hat=np.array([K[sel]]), delta_hat=np.array([np.nan]),
                    objective=sobj, selected_fit=best, id_set=dict(supported=False), **base)
    try:
        ids = siv_identification_set(X, r, K[sel], dl, du, level=id_level, n_grid=id_grid) if compute_id_set \
            else dict(supported=False)
    except ValueError as e:
        ids = dict(supported=False, reason=str(e))
    weak = ids.get('supported') and (ids.get('empty') or (np.isfinite(ids['frac_inside']) and ids['frac_inside'] > weak_frac))
    return dict(ok=True, status=['weak' if weak else 'ok'], k_hat=np.array([K[sel]]), delta_hat=np.array([dh]),
                objective=sobj, selected_index=sel, selected_fit=best, id_set=ids, **base)


# ---------------------------------------------------------------- corrected scores
def _M_fun(delta, h, Gamma, X, rhat, k):
    n, p = X.shape; rh = rhat - X @ h
    S = X + np.outer(rh, np.resize(np.asarray(k, float), p) * np.resize(np.asarray(delta, float), p))
    E = X - S @ Gamma; sig2 = np.mean(E ** 2, 0); m = (E ** 2 - sig2) * S
    return dict(M=m.mean(0), m=m, S=S, E=E, sigma2=sig2)


def _M_profiled(delta, h, X, rhat, k):
    p = X.shape[1]; rh = rhat - X @ h
    S = X + np.outer(rh, np.resize(np.asarray(k, float), p) * np.resize(np.asarray(delta, float), p))
    G = _solve(S.T @ S, S.T @ X)
    return np.full(p, np.nan) if G is None else _M_fun(delta, h, G, X, rhat, k)['M']


def _jacobian(f, par, eps_rel=1e-5, step=None):
    par = np.asarray(par, float); f0 = np.asarray(f(par), float)
    step = np.maximum(np.abs(par), 1) if step is None else np.resize(np.asarray(step, float), len(par))
    step = np.where(np.isfinite(step) & (step > 0), step, 1.0) * eps_rel
    J = np.empty((f0.size, len(par)))
    for j in range(len(par)):
        up = par.copy(); dn = par.copy(); up[j] += step[j]; dn[j] -= step[j]
        J[:, j] = (np.asarray(f(up), float) - np.asarray(f(dn), float)) / (2 * step[j])
    return J


def _step_b(X, rhat):
    sx = _sd(X, 0); sr = _sd(rhat)
    sx = np.where(np.isfinite(sx) & (sx > 0), sx, 1.0)
    if not np.isfinite(sr) or sr <= 0:
        sr = 1.0
    return sr / sx


def _psi_corrections(X, rhat, S, E, Jb, JG, ridge):
    n, p = X.shape
    Qxxi = _solve(X.T @ X / n + ridge * np.eye(p)); Qssi = _solve(S.T @ S / n + ridge * np.eye(p))
    if Qxxi is None or Qssi is None:
        return None
    psi_b = (X @ Qxxi) * rhat[:, None]
    Z = S @ Qssi
    psi_G = np.empty((n, p * p))
    for j in range(p):                                  # column-major vec(Gamma)
        psi_G[:, j * p:(j + 1) * p] = Z * E[:, j][:, None]
    return psi_b @ Jb.T + psi_G @ JG.T


def corrected_dt_scores(X_tilde, rhat, k, delta, ridge=1e-10):
    X = np.asarray(X_tilde, float); n, p = X.shape; rhat = np.asarray(rhat, float).reshape(-1)
    k = np.resize(np.asarray(k, float), p); delta = np.resize(np.asarray(delta, float), p)
    if len(rhat) != n or not _ok(X, rhat, k, delta):
        return None
    S = make_S(X, rhat, delta, k); G = _solve(S.T @ S, S.T @ X)
    if G is None:
        return None
    base = _M_fun(delta, np.zeros(p), G, X, rhat, k)
    try:
        Jb = _jacobian(lambda h: _M_profiled(delta, h, X, rhat, k), np.zeros(p), step=_step_b(X, rhat))
        JG = _jacobian(lambda g: _M_fun(delta, np.zeros(p), g.reshape(p, p, order='F'), X, rhat, k)['M'],
                       G.reshape(-1, order='F'), step=np.maximum(np.abs(G.reshape(-1, order='F')), 1))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not _ok(Jb, JG):
        return None
    corr = _psi_corrections(X, rhat, S, base['E'], Jb, JG, ridge)
    if corr is None:
        return None
    mt = base['m'] + corr
    if not _ok(mt):
        return None
    return dict(M_hat=base['m'].mean(0), m_raw=base['m'], m_tilde=mt, S=S, E=base['E'], Gamma=G, J_b=Jb, J_Gamma=JG)


def siv_S_stat(delta, X_tilde, r, k, ridge=1e-10, center=False):
    X = np.asarray(X_tilde, float); n, p = X.shape
    try:
        cs = corrected_dt_scores(X, r, k, delta, ridge)
    except (ValueError, np.linalg.LinAlgError):
        cs = None
    if cs is None:
        return np.nan
    M = cs['m_tilde'] - cs['m_tilde'].mean(0) if center else cs['m_tilde']
    Oi = _solve(M.T @ M / n + ridge * np.eye(p))
    return np.nan if Oi is None else float(n * cs['M_hat'] @ Oi @ cs['M_hat'])


def siv_identification_set(X_tilde, r, k, delta_lower, delta_upper, level=0.95, n_grid=200, center=False):
    X = np.asarray(X_tilde, float); p = X.shape[1]
    dl = np.resize(np.asarray(delta_lower, float), p); du = np.resize(np.asarray(delta_upper, float), p)
    if not _ok(dl, du) or np.any(dl <= 0) or np.any(du <= dl):
        return dict(supported=False, reason='invalid_domain')
    n_grid = max(2, int(n_grid))
    if p == 1:
        G = np.linspace(dl[0], du[0], n_grid)[:, None]
    else:
        per = max(2, int(np.floor(n_grid ** (1 / p))))
        axes = [np.linspace(dl[j], du[j], per) for j in range(p)]
        # expand.grid order: the first coordinate varies fastest
        G = np.column_stack([np.tile(np.repeat(axes[j], per ** j), per ** (p - 1 - j)) for j in range(p)])
    Sv = np.array([siv_S_stat(d, X, r, k, center=center) for d in G]); crit = chi2.ppf(level, p)
    inside = np.isfinite(Sv) & (Sv <= crit)
    lo = G[inside].min(0) if inside.any() else np.full(p, np.nan); hi = G[inside].max(0) if inside.any() else np.full(p, np.nan)
    return dict(supported=True, dimension=p, grid_matrix=G, grid=G[:, 0] if p == 1 else G, S=Sv, crit=crit, inside=inside,
                frac_inside=float(inside.mean()), empty=not inside.any(), lower=lo[0] if p == 1 else lo,
                upper=hi[0] if p == 1 else hi)


def third_moments(X, r):
    """Raw third moments of (x_j, r) with plug-in t-statistics: array p x 4 x 2."""
    X = np.atleast_2d(np.asarray(X, float)); X = X.T if X.shape[0] == 1 and X.shape[1] > 1 else X
    r = np.asarray(r, float).reshape(-1); n, p = X.shape
    out = np.empty((p, 4, 2))
    for j in range(p):
        z = np.column_stack([X[:, j] ** 3, X[:, j] ** 2 * r, X[:, j] * r ** 2, r ** 3])
        mu = z.mean(0); se = _sd(z, 0) / np.sqrt(n); out[j, :, 0] = mu; out[j, :, 1] = mu / se
    return out


def siv_sign_search(X_tilde, r, delta_lower, delta_upper, relevance_c=0.01, interior_tol=1e-3, weak_frac=0.90,
                    id_level=0.95, id_grid=200, n_grid=200, n_starts=8, max_cycles=3, max_sign_combinations=4096,
                    compute_id_set=True):
    X = np.asarray(X_tilde, float); p = X.shape[1]
    dl = np.resize(np.asarray(delta_lower, float), p); du = np.resize(np.asarray(delta_upper, float), p)
    if 2 ** p > max_sign_combinations:
        raise ValueError('2^p=%d exceeds max_sign_combinations=%d.' % (2 ** p, max_sign_combinations))
    K = np.array([[(-1 if (h // 2 ** j) % 2 == 0 else 1) for j in range(p)] for h in range(2 ** p)])  # expand.grid
    fits = []; obj = np.full(len(K), np.inf); interior = np.zeros(len(K), bool)
    for h, kk in enumerate(K):
        try:
            f = estimate_delta_dt_cue(X, r, kk, dl, du, relevance_c, n_grid, n_starts, max_cycles)
        except (ValueError, np.linalg.LinAlgError):
            f = None
        fits.append(f)
        if f is not None and f['converged'] and np.isfinite(f['J_hat']):
            obj[h] = f['J_hat']
            interior[h] = np.all(f['delta_hat'] - dl > interior_tol) and np.all(du - f['delta_hat'] > interior_tol)
    elig = np.where(np.isfinite(obj) & interior)[0]; fin = np.where(np.isfinite(obj))[0]
    tm = third_moments(X, r)
    if not len(fin):
        return dict(ok=False, status=['infeasible'] * p, k_hat=np.full(p, np.nan), delta_hat=np.full(p, np.nan),
                    selected_fit=None, id_set=dict(supported=False), sign_table=None, third_moments=tm)
    sel = elig[np.argmin(obj[elig])] if len(elig) else fin[np.argmin(obj[fin])]
    best = fits[sel]; dh = best['delta_hat']
    status = np.where(dh - dl <= interior_tol, 'boundary_lower', np.where(du - dh <= interior_tol, 'boundary_upper', 'ok')).tolist()
    if len(elig) and compute_id_set:
        try:
            ids = siv_identification_set(X, r, K[sel], dl, du, level=id_level, n_grid=id_grid)
        except ValueError as e:
            ids = dict(supported=False, reason=str(e))
    else:
        ids = dict(supported=False)
    weak = ids.get('supported') and (ids.get('empty') or (np.isfinite(ids['frac_inside']) and ids['frac_inside'] > weak_frac))
    if len(elig) and weak:
        status = ['weak'] * p
    table = {('k%d' % (j + 1)): K[:, j] for j in range(p)}
    table.update(objective=obj, interior=interior)
    for j in range(p):
        table['delta%d' % (j + 1)] = np.array([np.nan if f is None else f['delta_hat'][j] for f in fits])
    return dict(ok=len(elig) > 0, status=status, k_hat=K[sel].astype(int), delta_hat=dh, objective=obj[sel],
                selected_index=int(sel), selected_fit=best, sign_table=table, id_set=ids, delta_lower=dl,
                delta_upper=du, third_moments=tm)


# ---------------------------------------------------------------- analytic variance
def siv_vcov_analytic(y_tilde, X_tilde, rhat, k, delta_hat, ridge=1e-10, weight='estimator'):
    y = np.asarray(y_tilde, float); X = np.asarray(X_tilde, float); rhat = np.asarray(rhat, float).reshape(-1)
    n, p = X.shape; k = np.resize(np.asarray(k, float), p); dh = np.resize(np.asarray(delta_hat, float), p)
    S = make_S(X, rhat, dh, k); G = _solve(S.T @ S, S.T @ X)
    if G is None:
        return None
    base = _M_fun(dh, np.zeros(p), G, X, rhat, k); m_raw = base['m']
    try:
        JM = _jacobian(lambda d: _M_profiled(d, np.zeros(p), X, rhat, k), dh, step=np.maximum(np.abs(dh), 1e-3))
        Jb = _jacobian(lambda h: _M_profiled(dh, h, X, rhat, k), np.zeros(p), step=_step_b(X, rhat))
        gv = G.reshape(-1, order='F')
        JG = _jacobian(lambda g: _M_fun(dh, np.zeros(p), g.reshape(p, p, order='F'), X, rhat, k)['M'], gv,
                       step=np.maximum(np.abs(gv), 1))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not _ok(JM, Jb, JG):
        return None
    Qsx = S.T @ X / n; Qsxi = _solve(Qsx)
    if Qsxi is None:
        Qsxi = _solve(Qsx + ridge * np.eye(p))
    Qxxi = _solve(X.T @ X / n + ridge * np.eye(p)); Qssi = _solve(S.T @ S / n + ridge * np.eye(p))
    if Qsxi is None or Qxxi is None or Qssi is None:
        return None
    beta = _solve(S.T @ X, S.T @ y); beta_ols = _solve(X.T @ X, X.T @ y)
    if beta is None or beta_ols is None:
        return None
    uhat = y - X @ beta; bcheck = beta_ols - beta
    m_tilde = m_raw + _psi_corrections(X, rhat, S, base['E'], Jb, JG, ridge)
    Wm = _solve(m_tilde.T @ m_tilde / n + ridge * np.eye(p)) if weight == 'efficient' else _weight_inverse(m_raw, ridge)
    if Wm is None:
        return None
    Ai = _solve(JM.T @ Wm @ JM + ridge * np.eye(p))
    if Ai is None:
        return None
    GM = Ai @ JM.T @ Wm; gap_id = float(np.max(np.abs(GM @ JM - np.eye(p))))
    g_tilde = S * uhat[:, None] - np.outer((X @ bcheck) * rhat, k * dh)
    OmMM = m_tilde.T @ m_tilde / n; Omgg = g_tilde.T @ g_tilde / n; OmgM = g_tilde.T @ m_tilde / n
    sr2 = np.mean(rhat ** 2); Lam = Qsxi @ np.diag(k) @ (sr2 * GM)
    Vdelta = GM @ OmMM @ GM.T
    Vbeta = Qsxi @ Omgg @ Qsxi.T + Lam @ OmMM @ Lam.T - Qsxi @ OmgM @ Lam.T - Lam @ OmgM.T @ Qsxi.T
    if not _ok(Vdelta, Vbeta):
        return None
    infl_beta = g_tilde @ Qsxi.T - m_tilde @ Lam.T
    d_tilde = g_tilde - X * rhat[:, None]
    Omdd = d_tilde.T @ d_tilde / n; OmdM = d_tilde.T @ m_tilde / n
    Vdiff = Qsxi @ Omdd @ Qsxi.T + Lam @ OmMM @ Lam.T - Qsxi @ OmdM @ Lam.T - Lam @ OmdM.T @ Qsxi.T
    diff = beta - beta_ols; se_diff = np.sqrt(np.maximum(np.diag(Vdiff) / n, 0))
    try:
        H = float(diff @ np.linalg.solve(Vdiff / n + ridge * np.eye(p), diff))
    except np.linalg.LinAlgError:
        H = np.nan
    return dict(beta=beta, beta_ols=beta_ols, se_beta=np.sqrt(np.maximum(np.diag(Vbeta) / n, 0)),
                se_delta_raw=np.sqrt(np.maximum(np.diag(Vdelta) / n, 0)), V_beta=Vbeta, V_delta_raw=Vdelta,
                V_difference=Vdiff, J_M=JM, J_b=Jb, J_Gamma=JG, G_M=GM, Lambda=Lam, weight=weight,
                G_M_identity_gap=gap_id, influence_beta=infl_beta, influence_delta_raw=-m_tilde @ GM.T,
                m_tilde=m_tilde, g_tilde=g_tilde,
                ols_gap=dict(difference=diff, se=se_diff, statistic=H, df=p,
                             p_value=1 - chi2.cdf(H, p) if np.isfinite(H) else np.nan, null_in_domain=None,
                             null_position=None,
                             note='Wald statistic for H0: plim(beta_IIV - beta_OLS) = 0. That null is delta_0 = 0, '
                                  'which is excluded when delta_lower > 0 and sits on the boundary when '
                                  'delta_lower = 0, so the p-value is indicative only.'))


# ---------------------------------------------------------------- fit core
def _estimate_delta_given_sign(prep, gate, opt):
    kf = np.asarray(gate['k_hat']).astype(int); p = prep['X_work'].shape[1]
    if len(kf) != p or np.any(~np.isin(kf, (-1, 1))):
        raise ValueError('the sign stage did not return a valid fixed sign vector.')
    if opt['selection_method'] == 'legacy_v16':
        fit = estimate_delta_dt_v16_given_sign(prep['X_work'], prep['r_work'], kf, opt['delta_lower'], opt['delta_upper'],
                                               opt['relevance_c'], opt['legacy_interior_frac'], opt['n_grid'],
                                               opt['n_starts'], opt['max_cycles'], opt['legacy_seed'], gate['delta_hat'])
        buf = opt['legacy_interior_frac'] * (opt['delta_upper'] - opt['delta_lower'])
    else:
        fit = estimate_delta_dt_cue(prep['X_work'], prep['r_work'], kf, opt['delta_lower'], opt['delta_upper'],
                                    opt['relevance_c'], opt['n_grid'], opt['n_starts'], opt['max_cycles'],
                                    init_delta=gate['delta_hat'])
        buf = np.resize(opt['interior_tol'], p)
    fit['k_fixed'] = kf; fit['sign_delta_hint'] = np.asarray(gate['delta_hat'], float)
    dh = np.asarray(fit['delta_hat'], float)
    fit['interior'] = len(dh) == p and bool(np.all(dh - opt['delta_lower'] > buf))
    if opt['selection_method'] != 'legacy_v16':
        fit['interior'] = fit['interior'] and bool(np.all(opt['delta_upper'] - dh > buf))
    fit['converged'] = bool(fit['converged']) and fit['interior']
    return fit


def _fail(p, msg, **kw):
    d = dict(ok=False, message=msg, beta=np.full(p, np.nan), beta_ols=np.full(p, np.nan), delta_hat=np.full(p, np.nan),
             delta_raw=np.full(p, np.nan))
    d.update(kw); return d


def fit_core(y, X, W, opt):
    y = np.asarray(y, float); X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X; p = X.shape[1]
    try:
        prep = siv_prepare(y, X, W, opt['scale'])
    except (ValueError, np.linalg.LinAlgError) as e:
        return _fail(p, str(e))
    imposed = opt.get('k_fixed') is not None
    if imposed:
        init = np.asarray(opt['delta_init'], float) if opt.get('delta_init') is not None and np.size(opt['delta_init']) == p \
            and _ok(opt['delta_init']) else (opt['delta_lower'] + opt['delta_upper']) / 2
        gate = dict(ok=True, status=['imposed_sign'] * p, k_hat=np.asarray(opt['k_fixed'], int), delta_hat=init,
                    objective=np.nan, selected_fit=None, sign_table=None,
                    id_set=dict(supported=False, reason='sign_imposed'), third_moments=None,
                    selection_reason='imposed_from_sample_fit', selection_method='imposed')
    else:
        try:
            if opt['selection_method'] == 'legacy_v16':
                gate = siv_sign_search_v16(prep['X_work'], prep['r_work'], opt['delta_lower'], opt['delta_upper'],
                                           opt['relevance_c'], opt['legacy_interior_frac'], opt['weak_frac'], opt['level'],
                                           opt['id_grid'], opt['n_grid'], opt['n_starts'], opt['max_cycles'],
                                           opt['legacy_seed'], compute_id_set=not opt['in_bootstrap'])
            else:
                gate = siv_sign_search(prep['X_work'], prep['r_work'], opt['delta_lower'], opt['delta_upper'],
                                       opt['relevance_c'], opt['interior_tol'], opt['weak_frac'], opt['level'],
                                       opt['id_grid'], opt['n_grid'], opt['n_starts'], opt['max_cycles'],
                                       opt['max_sign_combinations'], compute_id_set=not opt['in_bootstrap'])
        except (ValueError, np.linalg.LinAlgError) as e:
            return _fail(p, str(e), status=['error'] * p, prepared=prep)
    if not gate['ok']:
        return _fail(p, 'no admissible interior DT solution: ' + ', '.join(gate['status']), k_hat=gate['k_hat'],
                     status=gate['status'], gate=gate, prepared=prep)
    k_hat = np.asarray(gate['k_hat'], int)
    try:
        sf = _estimate_delta_given_sign(prep, gate, opt)
    except (ValueError, np.linalg.LinAlgError) as e:
        sf = None; msg = str(e)
    if sf is None or not sf['converged'] or len(sf['delta_hat']) != p or not _ok(sf['delta_hat']) \
            or not np.array_equal(np.asarray(sf['k_fixed'], int), k_hat):
        return _fail(p, 'no admissible delta conditional on the selected sign' if sf is not None else msg, k_hat=k_hat,
                     status=['conditional_scale_failed'] * p, gate=gate, prepared=prep, conditional_scale_fit=sf)
    dh = np.asarray(sf['delta_hat'], float); dr = dh * prep['x_scale'] / prep['r_scale']
    Sw = make_S(prep['X_work'], prep['r_work'], dh, k_hat); Sr = make_S(prep['X_tilde'], prep['r_raw'], dr, k_hat)
    gap = float(np.max(np.abs(Sr - Sw * prep['x_scale'])))
    beta = _solve(Sr.T @ prep['X_tilde'], Sr.T @ prep['y_tilde'])
    bols = _solve(prep['X_tilde'].T @ prep['X_tilde'], prep['X_tilde'].T @ prep['y_tilde'])
    if beta is None or bols is None or not np.isfinite(gap):
        return _fail(p, 'singular final IV system', k_hat=k_hat, status=['singular_iv_system'] * p, gate=gate, prepared=prep)
    return dict(ok=True, message='ok', beta=beta, beta_ols=bols, delta_hat=dh, delta_raw=dr, k_hat=k_hat,
                status=list(gate['status']), S_work=Sw, S_raw=Sr, scale_equivalence_gap=gap, y_tilde=prep['y_tilde'],
                X_tilde=prep['X_tilde'], r_raw=prep['r_raw'], X_work=prep['X_work'], r_work=prep['r_work'],
                x_scale=prep['x_scale'], r_scale=prep['r_scale'], scale=prep['scale'],
                projection_method=prep['projection_method'], gate=gate, sign_delta_hint=np.asarray(gate['delta_hat'], float),
                conditional_scale_fit=sf, sign_source='imposed' if imposed else 'selected',
                selection_method=opt['selection_method'], options=opt)


# ---------------------------------------------------------------- bootstrap
def _boot_z0(v, theta):
    B = len(v)
    if not np.isfinite(theta) or B < 10:
        return 0.0
    f = min(max(np.sum(v < theta) / B, 1 / (B + 1)), B / (B + 1)); return float(norm.ppf(f))


def _boot_accel(infl):
    if infl is None or not _ok(infl) or len(infl) < 10:
        return 0.0
    l = infl - infl.mean(); den = 6 * np.sum(l ** 2) ** 1.5
    if not np.isfinite(den) or den <= 0:
        return 0.0
    a = np.sum(l ** 3) / den
    return float(a) if np.isfinite(a) else 0.0


def _boot_adj(probs, z0=0.0, a=0.0):
    probs = np.asarray(probs, float); zq = norm.ppf(probs); den = 1 - a * (z0 + zq)
    with np.errstate(divide='ignore', invalid='ignore'):
        adj = np.where(den > 0, norm.cdf(z0 + (z0 + zq) / den), np.nan)
    return np.where(np.isfinite(adj), adj, probs)


def _boot_resolvable(probs, z0, a, B_eff, min_tail=10):
    if not np.isfinite(B_eff) or B_eff < 2 * min_tail:
        return False
    adj = _boot_adj(probs, z0, a)
    return bool(np.all(np.isfinite(adj)) and min(np.min(adj), np.min(1 - adj)) * B_eff >= min_tail)


def _boot_ci(v, probs, z0=0.0, a=0.0):
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.full(len(probs), np.nan)
    return np.quantile(v, np.clip(_boot_adj(probs, z0, a), 0, 1))


def _boot_p(v, null=0.0, z0=0.0, a=0.0):
    v = v[np.isfinite(v)]; B = len(v)
    if B < 5:
        return np.nan
    f = min(max(np.sum(v < null) / B, 1 / (B + 1)), B / (B + 1)); vv = norm.ppf(f) - z0; den = 1 + a * vv
    if not np.isfinite(den) or den <= 0:
        return np.nan
    al = norm.cdf(vv / den - z0); return float(min(1, 2 * min(al, 1 - al)))


_BG = {}


def _boot_one(b):
    y, X, W, opt, seed = _BG['args']; n = len(y)
    idx = np.random.default_rng(int(seed) + b).integers(0, n, n)
    try:
        f = fit_core(y[idx], X[idx], None if W is None else W[idx], opt)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
        f = dict(ok=False, message=str(e))
    p = X.shape[1]
    good = f['ok'] and len(f['beta']) == p and _ok(f['beta'], f['delta_hat'], f['k_hat'])
    nanp = np.full(p, np.nan)
    return dict(success=good, beta=f['beta'] if good else nanp, delta=f['delta_hat'] if good else nanp,
                delta_raw=f['delta_raw'] if good else nanp, k=f['k_hat'] if good else nanp,
                sign_stage_k=np.asarray(f['gate']['k_hat'], float) if good else nanp,
                conditional_k=np.asarray(f['conditional_scale_fit']['k_fixed'], float) if good else nanp,
                message='ok' if good else str(f.get('message', 'unknown failure')))


def siv_bootstrap(y, X, W, opt, B=999, seed=1, level=0.95, n_cores=1, k_fixed=None, delta_init=None, ci_type='bca',
                  point=None, influence=None):
    import os
    from multiprocessing import get_context
    y = np.asarray(y, float); X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X
    W = None if W is None else np.asarray(W, float); p = X.shape[1]
    if B < 5:
        raise ValueError('B must be at least 5.')
    ob = dict(opt); ob['in_bootstrap'] = True
    if k_fixed is not None:
        ob['k_fixed'] = np.asarray(k_fixed, int); ob['delta_init'] = delta_init
    else:
        ob['k_fixed'] = None
    sign_source = 'reestimated_each_draw' if k_fixed is None else 'fixed_at_sample_fit'
    _BG['args'] = (y, X, W, ob, seed)
    n_cores = int(os.environ.get('IIV_CORES', n_cores))
    if n_cores > 1:
        with get_context('fork').Pool(n_cores) as pool:
            reps = pool.map(_boot_one, range(1, B + 1), chunksize=max(1, B // (4 * n_cores)))
    else:
        reps = [_boot_one(b) for b in range(1, B + 1)]
    success = np.array([r['success'] for r in reps])
    get = lambda key: np.array([r[key] for r in reps], float)
    beta_rep, delta_rep, draw_rep, k_rep = get('beta'), get('delta'), get('delta_raw'), get('k')
    probs = np.array([(1 - level) / 2, 1 - (1 - level) / 2])
    point = point or {}
    pb = np.asarray(point.get('beta', np.full(p, np.nan)), float); pdr = np.asarray(point.get('delta_raw', np.full(p, np.nan)), float)
    use_z0 = ci_type in ('bc', 'bca'); use_a = ci_type == 'bca' and influence is not None
    z0b, ab, z0d, ad = (np.zeros(p) for _ in range(4)); fb = np.zeros(p, bool); fd = np.zeros(p, bool)
    for j in range(p):
        vb = beta_rep[np.isfinite(beta_rep[:, j]), j]; vd = draw_rep[np.isfinite(draw_rep[:, j]), j]
        zb = _boot_z0(vb, pb[j]) if use_z0 else 0.0; zd = _boot_z0(vd, pdr[j]) if use_z0 else 0.0
        a1 = _boot_accel(influence['beta'][:, j]) if use_a else 0.0
        a2 = _boot_accel(influence['delta_raw'][:, j]) if use_a else 0.0
        if _boot_resolvable(probs, zb, a1, len(vb)):
            z0b[j], ab[j] = zb, a1
        elif use_z0:
            fb[j] = True
        if _boot_resolvable(probs, zd, a2, len(vd)):
            z0d[j], ad[j] = zd, a2
        elif use_z0:
            fd[j] = True
    mk = lambda M, z, a: np.array([_boot_ci(M[:, j], probs, z[j], a[j]) for j in range(p)])
    fmean = lambda M: np.array([np.mean(M[np.isfinite(M[:, j]), j]) if np.isfinite(M[:, j]).any() else np.nan for j in range(p)])
    fsd = lambda M: np.array([_sd(M[np.isfinite(M[:, j]), j]) if np.isfinite(M[:, j]).sum() >= 2 else np.nan for j in range(p)])
    msgs = [r['message'] for r in reps if not r['success']]
    pats = [','.join('%d' % v for v in k_rep[b]) for b in np.where(success)[0]]
    return dict(method='full nonparametric pairs bootstrap', B=B, seed=seed, level=level, ci_type=ci_type,
                sign_source=sign_source, k_fixed=None if k_fixed is None else np.asarray(k_fixed, int), success=success,
                n_success=int(success.sum()), success_rate=float(success.mean()), beta_reps=beta_rep, delta_reps=delta_rep,
                delta_raw_reps=draw_rep, k_reps=k_rep, sign_stage_k_reps=get('sign_stage_k'),
                conditional_k_reps=get('conditional_k'), mean_beta=fmean(beta_rep), mean_delta=fmean(delta_rep),
                mean_delta_raw=fmean(draw_rep), se_beta=fsd(beta_rep), se_delta_work=fsd(delta_rep), se_delta_raw=fsd(draw_rep),
                ci_beta=mk(beta_rep, z0b, ab), ci_delta_work=mk(delta_rep, z0d, ad), ci_delta_raw=mk(draw_rep, z0d, ad),
                ci_percentile_beta=mk(beta_rep, np.zeros(p), np.zeros(p)), z0_beta=z0b, accel_beta=ab, z0_delta=z0d,
                accel_delta=ad, ci_fallback_beta=fb, ci_fallback_delta=fd,
                ci_type_used_beta=np.where(fb, 'percentile', ci_type),
                p_beta=np.array([_boot_p(beta_rep[:, j], 0, z0b[j], ab[j]) for j in range(p)]), p_floor=2 / (B + 1),
                p_method='two-sided %s interval inversion' % ci_type,
                sign_frequency={s: pats.count(s) for s in sorted(set(pats), key=lambda s: -pats.count(s))},
                failure_frequency={s: msgs.count(s) for s in sorted(set(msgs), key=lambda s: -msgs.count(s))})


# ---------------------------------------------------------------- public estimator
def run_dt_iiv(y, X, W=None, delta_lower=0.5, delta_upper=5, scale='none', selection_method='cue', relevance_c=0.01,
               interior_tol=1e-3, legacy_interior_frac=0.05, legacy_seed=1, weak_frac=0.90, n_grid=200, n_starts=8,
               max_cycles=3, max_sign_combinations=4096, id_grid=200, inference='both', B=999, level=0.95, seed=1,
               n_cores=1, bootstrap_sign='fixed', ci_type='bca', report_point='sample', analytic_diagnostic=True,
               on_fail='na', k_fixed=None, delta_init=None, names=None, verbose=False):
    X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X; p = X.shape[1]
    names = list(names) if names is not None else ['x%d' % (j + 1) for j in range(p)]
    if selection_method == 'legacy_v16' and (p != 1 or scale != 'none'):
        raise ValueError("selection_method='legacy_v16' needs p=1 and scale='none'.")
    dl = np.resize(np.asarray(delta_lower, float), p); du = np.resize(np.asarray(delta_upper, float), p)
    if not _ok(dl, du) or np.any(du <= dl) or (selection_method == 'cue' and np.any(dl <= 0)) or \
            (selection_method == 'legacy_v16' and np.any(dl < 0)):
        raise ValueError('invalid search domain.')
    if k_fixed is not None:
        k_fixed = np.resize(np.asarray(k_fixed, int), p)
    if delta_init is not None:
        delta_init = np.resize(np.asarray(delta_init, float), p)
    opt = dict(delta_lower=dl, delta_upper=du, scale=scale, selection_method=selection_method, relevance_c=relevance_c,
               interior_tol=interior_tol, weak_frac=weak_frac, legacy_interior_frac=float(legacy_interior_frac),
               legacy_seed=int(legacy_seed), n_grid=int(n_grid), n_starts=int(n_starts), max_cycles=int(max_cycles),
               max_sign_combinations=int(max_sign_combinations), id_grid=int(id_grid), level=level, in_bootstrap=False,
               k_fixed=k_fixed, delta_init=delta_init, bootstrap_sign=bootstrap_sign, ci_type=ci_type)
    fit = fit_core(y, X, W, opt)
    fit['names'] = names
    if not fit['ok']:
        if on_fail == 'error':
            raise RuntimeError('DT-IIV estimation failed: ' + fit['message'])
        fit.update(bootstrap=None, analytic=None, options=opt, se_primary='none'); return fit
    analytic = None
    if inference in ('analytic', 'both') or analytic_diagnostic:
        try:
            analytic = siv_vcov_analytic(fit['y_tilde'], fit['X_tilde'], fit['r_raw'], fit['k_hat'], fit['delta_raw'])
        except (ValueError, np.linalg.LinAlgError):
            analytic = None
    if analytic is not None:
        analytic['se_delta_work'] = analytic['se_delta_raw'] * fit['r_scale'] / fit['x_scale']
        analytic['ols_gap']['null_in_domain'] = bool(np.all(dl <= 0))
        analytic['ols_gap']['null_position'] = 'outside_search_domain' if np.all(dl > 0) else 'on_the_boundary'
    boot = None
    if inference in ('bootstrap', 'both'):
        boot = siv_bootstrap(y, X, W, opt, B=B, seed=seed, level=level, n_cores=n_cores,
                             k_fixed=fit['k_hat'] if bootstrap_sign == 'fixed' else None,
                             delta_init=fit['delta_hat'] if bootstrap_sign == 'fixed' else None, ci_type=ci_type,
                             point=dict(beta=fit['beta'], delta_work=fit['delta_hat'], delta_raw=fit['delta_raw']),
                             influence=None if analytic is None else dict(beta=analytic['influence_beta'],
                                                                          delta_raw=analytic['influence_delta_raw']))
    z = norm.ppf(1 - (1 - level) / 2)
    se_a = analytic['se_beta'] if analytic is not None else np.full(p, np.nan)
    ci_an = np.column_stack([fit['beta'] - z * se_a, fit['beta'] + z * se_a])
    beta_report = {'sample': fit['beta'],
                   'bootstrap_mean': boot['mean_beta'] if boot else fit['beta'],
                   'bias_corrected': 2 * fit['beta'] - boot['mean_beta'] if boot else fit['beta']}[report_point]
    p_an = 2 * norm.cdf(-np.abs(fit['beta'] / se_a))
    fit.update(analytic=analytic, bootstrap=boot, beta_report=beta_report, ci_analytic=ci_an, p_analytic=p_an,
               se_primary='bootstrap' if boot else ('analytic' if analytic is not None else 'none'),
               ci_type=ci_type, bootstrap_sign=bootstrap_sign, report_point=report_point, options=opt)
    if verbose:
        print_fit(fit)
    return fit


def print_fit(fit):
    nm = fit.get('names'); print('\nVT-IIV clean v21 (Python)')
    print('Status:', 'ok' if fit['ok'] else 'failed')
    if not fit['ok']:
        print('  ', fit['message']); return
    a, b = fit['analytic'], fit['bootstrap']
    for j, v in enumerate(nm):
        line = '%-8s beta %.6f  delta %.6f (raw %.6f)  k %+d  status %s  OLS %.6f' % (
            v, fit['beta'][j], fit['delta_hat'][j], fit['delta_raw'][j], fit['k_hat'][j], fit['status'][j], fit['beta_ols'][j])
        print(line)
        if a is not None:
            print('         analytic se %.6f  Wald [%.4f, %.4f]  gap %.6f (se %.6f)' % (
                a['se_beta'][j], fit['ci_analytic'][j, 0], fit['ci_analytic'][j, 1], a['ols_gap']['difference'][j], a['ols_gap']['se'][j]))
        if b is not None:
            print('         bootstrap se %.6f  %s [%.4f, %.4f]  p %.4f  draws %d/%d' % (
                b['se_beta'][j], b['ci_type_used_beta'][j], b['ci_beta'][j, 0], b['ci_beta'][j, 1], b['p_beta'][j], b['n_success'], b['B']))


# ---------------------------------------------------------------- profile, audit, report
def siv_scalar_profile(fit, delta_lower=None, delta_upper=None, n_grid=1000, relevance_c=None, root_tol=1e-8, k=None):
    if not fit['ok'] or fit['X_work'].shape[1] != 1:
        raise ValueError('siv_scalar_profile() needs a successful fit with p=1.')
    o = fit['options']; dl = o['delta_lower'][0] if delta_lower is None else float(delta_lower)
    du = o['delta_upper'][0] if delta_upper is None else float(delta_upper)
    rc = o['relevance_c'] if relevance_c is None else float(relevance_c)
    grid = np.linspace(dl, du, max(20, int(n_grid))); k = int(fit['k_hat'][0]) if k is None else int(k)
    to_raw = fit['x_scale'][0] / fit['r_scale']
    Xw, rw = fit['X_work'], fit['r_work']

    def moment_at(d):
        z = dt_moments_hat(np.array([d]), Xw, rw, k, relevance_c=0); return float(z['M_hat'][0]) if z['ok'] else np.nan

    def one(d):
        z = dt_moments_hat(np.array([d]), Xw, rw, k, relevance_c=0)
        if not z['ok']:
            return (np.nan,) * 4
        Wm = _weight_inverse(z['m_mat']); J = np.nan if Wm is None else float(Xw.shape[0] * z['M_hat'] @ Wm @ z['M_hat'])
        bb = _solve(z['S'].T @ fit['X_tilde'], z['S'].T @ fit['y_tilde'])
        return float(z['M_hat'][0]), J, float(z['relevance'][0]), np.nan if bb is None else float(bb[0])
    vals = np.array([one(d) for d in grid]); m = vals[:, 0]
    roots = list(grid[np.isfinite(m) & (np.abs(m) <= root_tol)])
    for i in range(len(grid) - 1):
        if not np.isfinite(m[i]) or not np.isfinite(m[i + 1]) or m[i] * m[i + 1] > 0:
            continue
        try:
            roots.append(uniroot(moment_at, grid[i], grid[i + 1], tol=root_tol))
        except ValueError:
            pass
    rows = []
    if roots:
        roots = np.sort(roots); roots = roots[np.r_[True, np.diff(roots) > max(root_tol * 10, (du - dl) / 1e6)]]
        legacy = o['selection_method'] == 'legacy_v16'
        lb = o['legacy_interior_frac'] * (du - dl) if legacy else o['interior_tol']
        for i, rt in enumerate(roots):
            mo, J, rel, bt = one(rt)
            interior = rt - dl > lb and (legacy or du - rt > o['interior_tol'])
            rows.append(dict(root=i + 1, delta=rt, delta_raw=rt * to_raw, moment=mo, J=J, relevance=rel, beta=bt,
                             admissible=abs(rel) >= rc and interior))
    return dict(profile=dict(delta=grid, delta_raw=grid * to_raw, moment=m, J=vals[:, 1], relevance=vals[:, 2], beta=vals[:, 3]),
                roots=rows, k=k, relevance_cutoff=rc, domain=(dl, du))


def siv_bootstrap_audit(fit, tol=1e-12):
    b = fit['bootstrap']; s = b['success']; p = b['beta_reps'].shape[1]; a = (1 - b['level']) / 2
    same = lambda x, y: bool(np.all((np.isnan(x) & np.isnan(y)) | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) <= tol)))
    sdf = lambda M: np.array([_sd(M[np.isfinite(M[:, j]), j]) for j in range(p)])
    checks = {
        'failed beta draws are NA': not np.isfinite(b['beta_reps'][~s]).any(),
        'failed delta draws are NA': not np.isfinite(b['delta_reps'][~s]).any(),
        'selected sign fixed through delta and S': bool(np.all(b['sign_stage_k_reps'][s] == b['conditional_k_reps'][s])
                                                        and np.all(b['conditional_k_reps'][s] == b['k_reps'][s])),
        'every draw used the sign fixed on the sample': True if b['sign_source'] != 'fixed_at_sample_fit' else
        bool(np.all(b['k_reps'][s] == b['k_fixed']) and np.array_equal(b['k_fixed'], fit['k_hat'])),
        'SE equals bootstrap SD': same(sdf(b['beta_reps']), b['se_beta']),
        'se_delta_work equals SD of working-scale draws': same(sdf(b['delta_reps']), b['se_delta_work']),
        'se_delta_raw equals SD of raw-scale draws': same(sdf(b['delta_raw_reps']), b['se_delta_raw']),
        'CI equals the declared interval type': same(np.array([_boot_ci(b['beta_reps'][:, j], [a, 1 - a], b['z0_beta'][j],
                                                                        b['accel_beta'][j]) for j in range(p)]), b['ci_beta']),
        'p-value inverts the reported interval': same(np.array([_boot_p(b['beta_reps'][:, j], 0, b['z0_beta'][j], b['accel_beta'][j])
                                                                for j in range(p)]), b['p_beta']),
        'success count is exact': b['n_success'] == int(s.sum()),
        'all successful deltas are positive': bool(np.all(b['delta_reps'][s] > 0))}
    return dict(checks=checks, all_pass=all(checks.values()))


def siv_report(fit, profile_grid=1000):
    print_fit(fit)
    if not fit['ok']:
        return
    g = fit['gate']
    if g.get('sign_table') is not None:
        print('\nJoint sign ranking:'); print({k: np.round(v, 6) for k, v in g['sign_table'].items()})
    if g.get('third_moments') is not None:
        print('\nThird-moment identification diagnostics (estimate, t):'); print(np.round(g['third_moments'], 4))
    if g['id_set'].get('supported'):
        z = g['id_set']; print('\nIdentification-robust S-set: %.1f%% of search grid accepted. Envelope [%s] -- [%s]' % (
            100 * z['frac_inside'], z['lower'], z['upper']))
    if fit['X_work'].shape[1] == 1:
        print('\nScalar VT roots (before relevance selection):')
        for r in siv_scalar_profile(fit, n_grid=profile_grid)['roots']:
            print('  ', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    if fit['bootstrap'] is not None:
        print('\nPairs-bootstrap audit:', siv_bootstrap_audit(fit)['checks'])
