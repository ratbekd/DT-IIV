"""Monte Carlo evidence for the VT-IIV estimator (port of DT_IIV_montecarlo_v21.R).

  dgp_siv()                  aligned design of the appendix ("aligned", "aligned_hetero")
  check_identification()     pre-flight check: third moments, interior VT root
  dgp_population_moments()   table tab:dgp_params
  run_mc()                   table tab:simulation_results (n = 1,000, 2,000, 4,000)
  run_het()                  tables tab:het_channels and tab:fgls
  mc_selftest()              checks of the DGP, the closed forms and the variance routes

Usage: python sim_montecarlo.py [quick] [--no-het]
Writes the four LaTeX fragments to output/simulations/tables/ and the raw results
to output/simulations/sim_montecarlo_*.csv. Random numbers come from NumPy, so the
draws differ from the R runs; the design is the same.
"""
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import get_context
from scipy.stats import norm
from scipy.optimize import brentq
from _setup import out, OUT
from iiv.dtcore import run_dt_iiv, dt_moments_hat, make_S, siv_vcov_analytic, _solve

CFG = dict(n_populations=50, n_draws=10, sample_sizes=(1000, 2000, 4000), population_size=100000, beta_true=2.0,
           alpha=0.05, dgp_variant='aligned', delta0_target=1.0, eps_x=0.8, kappa_x=0.5, hetero_theta=1.2,
           hetero_driver='x', nu_range=15, delta_lower=0.5, delta_upper=5.0, scale='sd', relevance_c=0.01, n_grid=120,
           n_starts=4, max_cycles=3, id_grid=60, weak_frac=0.90, het_N=100000, het_draws=9, het_theta=(0.5, 1.5),
           het_theta_x=(0.5, 1.0, 1.5), het_degree=2, seed_base=1001, n_cores=int(os.environ.get('IIV_CORES', os.cpu_count() or 1)),
           tab_dir=os.path.join(OUT, 'tables'))


def quick_cfg(cfg=CFG):
    c = dict(cfg); c.update(n_populations=3, n_draws=2, population_size=20000, sample_sizes=(1000, 2000), het_N=20000,
                            het_draws=2, het_theta=(1.5,), het_theta_x=(1.5,)); return c


# ---------------------------------------------------------------- DGP
SAS = lambda z, eps, kappa: np.sinh(kappa * np.arcsinh(z) - eps)
_std = lambda v: (v - v.mean()) / v.std(ddof=1)


def _resid(v, *cols):
    Z = np.column_stack([np.ones(len(v))] + list(cols)); return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def _solve_lambda(h, z, a, target):
    f = lambda l: np.mean(_std(h * (np.sqrt(1 - l ** 2) * z + l * a)) ** 3) - target
    try:
        return brentq(f, -0.99, 0.99, xtol=np.finfo(float).eps ** 0.25)
    except ValueError:
        return np.nan


def dgp_siv(N=100000, cfg=CFG, seed=None):
    variant = cfg['dgp_variant']
    if variant not in ('aligned', 'aligned_hetero'):
        raise ValueError("dgp_variant must be 'aligned' or 'aligned_hetero'.")
    rng = np.random.default_rng(seed)
    nu = np.linspace(-cfg['nu_range'], cfg['nu_range'], N)
    w = rng.normal(20, 10, N)
    x = _std(7 * SAS(nu, cfg['eps_x'], cfg['kappa_x']) + 1.1 * rng.uniform(-1.01, 1.01, N)); mu30 = np.mean(x ** 3)
    a = _std(rng.uniform(-0.5, 0.5, N) + SAS(rng.standard_normal(N), 5, 1.2)); z = rng.standard_normal(N)
    if variant != 'aligned_hetero' or cfg['hetero_theta'] == 0:
        h = np.ones(N)
    else:
        drv = x if cfg['hetero_driver'] == 'x' else _std(w)
        h = np.sqrt(0.3 + np.exp(cfg['hetero_theta'] * drv)); h = h / np.sqrt(np.mean(h ** 2))
    b = cfg['delta0_target']; lam = _solve_lambda(h, z, a, b * mu30)
    if not np.isfinite(lam):
        raise ValueError('dgp_siv: the skewness needed for alignment is not attainable.')
    eps = _std(h * (np.sqrt(1 - lam ** 2) * z + lam * a)); u = b * x + eps
    y = 1 + cfg['beta_true'] * x + 0.5 * w + u
    y0 = _std(_resid(y, w)); x0 = _std(_resid(x, w)); R = _std(_resid(y0, x0))
    return dict(y=y, x=x, w=w, x0=x0, y0=y0, R=R, u=u, lam=lam, mu30=mu30, b=b, variant=variant)


def _tstat(v):
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def _third(x0, r):
    vals = dict(mu30=x0 ** 3, mu21=x0 ** 2 * r, mu12=x0 * r ** 2, mu03=r ** 3)
    return {k: v.mean() for k, v in vals.items()}, {k: _tstat(v) for k, v in vals.items()}


def _M_on_grid(x0, r, k, grid):
    out_ = []
    for d in grid:
        z = dt_moments_hat(np.array([d]), x0[:, None], r, k, relevance_c=0)
        out_.append(float(z['M_hat'][0]) if z['ok'] else np.nan)
    return np.array(out_)


def check_identification(cfg=CFG, N=50000, verbose=True):
    p = dgp_siv(N, cfg, cfg['seed_base']); mu, tt = _third(p['x0'], p['R'])
    grid = np.linspace(cfg['delta_lower'], cfg['delta_upper'], 150)
    Mm, Mp = _M_on_grid(p['x0'], p['R'], -1, grid), _M_on_grid(p['x0'], p['R'], 1, grid)
    has = lambda M: (M[np.isfinite(M)] > 0).any() and (M[np.isfinite(M)] < 0).any()
    ok = max(abs(v) for v in tt.values()) > 3 and (has(Mm) or has(Mp))
    if verbose:
        print('-' * 78); print('PRE-FLIGHT IDENTIFICATION CHECK  (dgp_variant = %s, N = %s)' % (cfg['dgp_variant'], format(N, ',')))
        print('-' * 78)
        print('  moment ' + '  '.join('%s %.4f' % kv for kv in mu.items())); print('  t      ' + '  '.join('%s %.2f' % kv for kv in tt.items()))
        print('  VT root on [%g, %g]:   k = -1  %s      k = +1  %s' % (cfg['delta_lower'], cfg['delta_upper'],
                                                                     'yes' if has(Mm) else 'NO ', 'yes' if has(Mp) else 'NO'))
        c = np.cov(p['x'], p['u'])[0, 1]
        print('  cov(x,u) = %+.4f   =>   k = %+d,  delta_0 = %.4f' % (c, -np.sign(c), cfg['delta0_target']))
        print('  identification conditions look satisfied.' if ok else '  *** VT identification looks unavailable here. ***')
        print('-' * 78 + '\n')
    return dict(ok=ok, moments=mu, t=tt, lam=p['lam'])


def dgp_population_moments(cfg=CFG, N=100000, seed=None):
    seed = cfg['seed_base'] if seed is None else seed
    p = dgp_siv(N, cfg, seed); mu, tt = _third(p['x0'], p['R'])
    grid = np.linspace(0.05, 5, 2000); k = -np.sign(np.cov(p['x'], p['u'])[0, 1])
    Mv = _M_on_grid(p['x0'], p['R'], k, grid); root = np.nan
    s = np.where(np.isfinite(Mv[1:]) & np.isfinite(Mv[:-1]) & (Mv[1:] * Mv[:-1] < 0))[0]
    if len(s):
        i = s[np.argmin(np.abs(grid[s] - cfg['delta0_target']))]
        root = grid[i] - Mv[i] * (grid[i + 1] - grid[i]) / (Mv[i + 1] - Mv[i])
    Z = np.column_stack([np.ones(N), p['x'], p['w']]); bols = np.linalg.lstsq(Z, p['y'], rcond=None)[0][1]
    return dict(N=N, lam=p['lam'], moments=mu, t=tt, k=int(k), beta_ols=bols, dt_root=root,
                target_delta0=cfg['delta0_target'], eps_x=cfg['eps_x'], kappa_x=cfg['kappa_x'], beta_true=cfg['beta_true'])


# ---------------------------------------------------------------- the three variance routes
def dt_variance_routes(fit, hc1=True):
    """Corrected (from the core), uncorrected two-step and naive IV variances, all on
    the raw FWL scale at the same (k, delta_raw)."""
    if not fit['ok']:
        return None
    X = fit['X_tilde']; y = fit['y_tilde']; r = fit['r_raw']; n, p = X.shape
    k = np.asarray(fit['k_hat'], float); d = np.asarray(fit['delta_raw'], float)
    S = make_S(X, r, d, k); Qsxi = _solve(S.T @ X / n); beta = _solve(S.T @ X, S.T @ y)
    if Qsxi is None or beta is None:
        return None
    uhat = y - X @ beta
    try:
        an = siv_vcov_analytic(y, X, r, k, d)
    except (ValueError, np.linalg.LinAlgError):
        an = None
    z = dt_moments_hat(d, X, r, k, relevance_c=0); V_unc = None
    if z['ok'] and an is not None:
        GM = _solve(an['J_M'])
        if GM is not None:
            m_raw = z['m_mat']; g_raw = S * uhat[:, None]
            Lam = Qsxi @ np.diag(k) @ (np.mean(r ** 2) * GM)
            OmMM = m_raw.T @ m_raw / n; Omgg = g_raw.T @ g_raw / n; OmgM = g_raw.T @ m_raw / n
            V_unc = dict(beta=Qsxi @ Omgg @ Qsxi.T + Lam @ OmMM @ Lam.T - Qsxi @ OmgM @ Lam.T - Lam @ OmgM.T @ Qsxi.T,
                         delta=GM @ OmMM @ GM.T)
    meat = (S * uhat[:, None] ** 2).T @ S / n
    V_nai = Qsxi @ meat @ Qsxi.T * (n / (n - p) if hc1 else 1)
    se = lambda V: np.full(p, np.nan) if V is None else np.sqrt(np.maximum(np.diag(V) / n, 0))
    return dict(beta=beta, se=dict(corrected=np.full(p, np.nan) if an is None else an['se_beta'],
                                   uncorrected=se(None if V_unc is None else V_unc['beta']), naive=se(V_nai)),
                se_delta_raw=dict(corrected=np.full(p, np.nan) if an is None else an['se_delta_raw'],
                                  uncorrected=se(None if V_unc is None else V_unc['delta'])),
                delta_raw=d, k=k.astype(int), analytic=an)


# ---------------------------------------------------------------- one fit and the nested Monte Carlo
def fit_one(y, x, w, cfg):
    n = len(y); Z = np.column_stack([np.ones(n), x, w]); co = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ co; Vo = np.linalg.inv(Z.T @ Z) * (e @ e / (n - 3))
    o = dict(beta_ols=co[1], se_ols=np.sqrt(Vo[1, 1]), beta=np.nan, se=np.nan, delta=np.nan, delta_raw=np.nan, k=np.nan,
             status=None, weak=None, se_corrected=np.nan, se_uncorrected=np.nan, se_naive=np.nan, S_frac=np.nan, error=None)
    try:
        f = run_dt_iiv(y, x[:, None], w[:, None], delta_lower=cfg['delta_lower'], delta_upper=cfg['delta_upper'],
                       scale=cfg['scale'], selection_method='cue', relevance_c=cfg['relevance_c'], weak_frac=cfg['weak_frac'],
                       n_grid=cfg['n_grid'], n_starts=cfg['n_starts'], max_cycles=cfg['max_cycles'], id_grid=cfg['id_grid'],
                       inference='analytic', analytic_diagnostic=True, on_fail='na', names=['x'])
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as err:
        o['error'] = str(err); return o
    if not f['ok']:
        o['status'] = 'gate_failed'; return o
    st = f['status'][0]; o.update(status=st, delta=f['delta_hat'][0], delta_raw=f['delta_raw'][0], k=int(f['k_hat'][0]),
                                  weak=st == 'weak', S_frac=f['gate']['id_set'].get('frac_inside', np.nan))
    if st.startswith('boundary'):
        return o
    vr = dt_variance_routes(f); o['beta'] = f['beta'][0]
    if vr is not None:
        o.update(se_corrected=vr['se']['corrected'][0], se_uncorrected=vr['se']['uncorrected'][0],
                 se_naive=vr['se']['naive'][0], se=vr['se']['corrected'][0])
    return o


_G = {}


def _population(l):
    cfg = _G['cfg']; pop = dgp_siv(cfg['population_size'], cfg, cfg['seed_base'] + l); rows = []
    for s, n in enumerate(cfg['sample_sizes'], start=1):
        for j in range(1, cfg['n_draws'] + 1):
            rng = np.random.default_rng(cfg['seed_base'] + 7919 * s + 3 * j + l)
            i = rng.integers(0, cfg['population_size'], n)
            rows.append(dict(pop=l, draw=j, n=n, **fit_one(pop['y'][i], pop['x'][i], pop['w'][i], cfg)))
    return rows


def run_mc(cfg=CFG, verbose=True):
    if verbose:
        print('=' * 78); print('VT-IIV MONTE CARLO (port of DT_IIV_montecarlo_v21.R)'); print('=' * 78)
        print('  DGP %s (delta_0 = %g, eps_x = %g); %d populations of N = %s; %d draws each; n = %s' % (
            cfg['dgp_variant'], cfg['delta0_target'], cfg['eps_x'], cfg['n_populations'], format(cfg['population_size'], ','),
            cfg['n_draws'], ', '.join(format(v, ',') for v in cfg['sample_sizes'])))
        print('  true beta %g (plim OLS %g); search domain [%g, %g], scale %s\n' % (
            cfg['beta_true'], cfg['beta_true'] + cfg['delta0_target'], cfg['delta_lower'], cfg['delta_upper'], cfg['scale']))
    _G['cfg'] = cfg; t0 = time.time(); pops = range(1, cfg['n_populations'] + 1)
    if cfg['n_cores'] > 1:
        with get_context('fork').Pool(cfg['n_cores']) as pool:
            res = pool.map(_population, pops, chunksize=1)
    else:
        res = [_population(l) for l in pops]
    df = pd.DataFrame([r for rr in res for r in rr]); df.attrs['elapsed'] = time.time() - t0
    return df


def table_mc(df, cfg=CFG):
    b0 = cfg['beta_true']; zq = norm.ppf(1 - cfg['alpha'] / 2); rows = []
    for n, d in df.groupby('n', sort=True):
        for meth, col in (('OLS', 'beta_ols'), ('IIV', 'beta')):
            v = d[col].values.astype(float); v = v[np.isfinite(v)]; m, s = v.mean(), v.std(ddof=1)
            rows.append(dict(n=n, method=meth, n_used=len(v), mean=m, sd=s, ci_lower=m - zq * s, ci_upper=m + zq * s,
                             bias=m - b0, rmse=np.sqrt(np.mean((v - b0) ** 2))))
    return pd.DataFrame(rows)


def table_mc_notes(df, k_true=-1):
    rows = []
    for n, d in df.groupby('n', sort=True):
        keep = np.isfinite(d.beta.values.astype(float)); dl = d.delta.values[keep]
        rows.append(dict(n=n, n_samples=len(d), n_estimates=int(keep.sum()),
                         n_boundary=int(d.status.fillna('').str.startswith('boundary').sum()), delta_median=np.median(dl),
                         delta_mean=dl.mean(), delta_sd=dl.std(ddof=1), pct_weak=100 * np.mean(d.weak.fillna(False) == True),
                         sign_accuracy=np.mean(d.k.dropna() == k_true)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- heteroscedasticity channels
def _M_numeric(tau, x, r):
    s = x + tau * r; g = np.sum(x * s) / np.sum(s ** 2); e = x - g * s; return np.mean((e ** 2 - np.mean(e ** 2)) * s)


def _M_closed(tau, A, B, m30, m21, m12, m03):
    return (B ** 2 * tau ** 4 * m30 + (B ** 2 * tau ** 5 - 2 * A * B * tau ** 3) * m21 +
            (A ** 2 * tau ** 2 - 2 * A * B * tau ** 4) * m12 + A ** 2 * tau ** 3 * m03) / (A + B * tau ** 2) ** 2


def _P_poly(tau, A, B, m30, m21, m12, m03, q21=None, q12=None):
    q21 = m21 if q21 is None else q21; q12 = m12 if q12 is None else q12
    return B ** 2 * m30 * tau ** 2 + q21 * (B ** 2 * tau ** 3 - 2 * A * B * tau) + q12 * (A ** 2 - 2 * A * B * tau ** 2) + A ** 2 * m03 * tau


def _M_gamma(gamma, P0, P1, P2):
    return gamma ** 2 * P2 - 2 * gamma * P1 + P0


def _varfit(e2, Z, floor_frac=1e-6):
    D = np.column_stack([np.ones(len(e2)), Z]); v = D @ np.linalg.lstsq(D, e2, rcond=None)[0]
    return np.maximum(v, floor_frac * np.mean(e2))


def _root_near(f, lo, hi, target=-1, ng=600):
    g = np.linspace(lo, hi, ng); v = np.array([f(t) for t in g])
    s = np.where(np.isfinite(v[1:]) & np.isfinite(v[:-1]) & (v[1:] * v[:-1] < 0))[0]
    cand = []
    for i in s:
        try:
            cand.append(brentq(f, g[i], g[i + 1], xtol=1e-12, rtol=4 * np.finfo(float).eps))
        except ValueError:
            pass
    return np.nan if not cand else cand[int(np.argmin(np.abs(np.array(cand) - target)))]


def het_one(N, driver='none', theta=0.0, seed=7, cfg=CFG, b=1.0, eps_x=0.8, grid=(-3.5, -0.06)):
    hc = dict(cfg); hc.update(dgp_variant='aligned' if driver == 'none' or theta == 0 else 'aligned_hetero',
                             hetero_driver='x' if driver == 'none' else driver, hetero_theta=0 if driver == 'none' else theta,
                             delta0_target=b, eps_x=eps_x)
    p = dgp_siv(N, hc, seed)
    yv = _resid(p['y'], p['w']); x = _resid(p['x'], p['w']); uv = _resid(p['u'], p['w'])
    Zr = np.column_stack([np.ones(N), x]); r = yv - Zr @ np.linalg.lstsq(Zr, yv, rcond=None)[0]
    A, B = np.mean(x ** 2), np.mean(r ** 2)
    m30, m21, m12, m03 = np.mean(x ** 3), np.mean(x ** 2 * r), np.mean(x * r ** 2), np.mean(r ** 3)
    bols = np.mean(x * yv) / A; bet = lambda t: np.nan if not np.isfinite(t) else bols + t * B / A
    tau_star = -np.mean(x * uv) / B
    Wc = _std(p['w']); Zv = np.column_stack([Wc, Wc ** 2])[:, :cfg['het_degree']]

    def pieces(t):
        s = x + t * r; return s, np.mean(x ** 2 * s), np.mean(x * s ** 2), np.mean(s ** 3)

    def fgls(t, weighted_moment=False):
        s, P0, P1, P2 = pieces(t); gO = np.mean(x * s) / np.mean(s ** 2); v = _varfit((x - gO * s) ** 2, Zv); wt = 1 / v
        gW = np.sum(wt * x * s) / np.sum(wt * s ** 2)
        if not weighted_moment:
            return _M_gamma(gW, P0, P1, P2)
        es = (x - gW * s) / np.sqrt(v); return np.mean((es ** 2 - np.mean(es ** 2)) * s)
    M_ols = lambda t: _M_numeric(t, x, r)
    tO = _root_near(M_ols, *grid, -b); tF = _root_near(fgls, *grid, -b)
    tW = _root_near(lambda t: fgls(t, True), *grid, -b)
    tM = _root_near(lambda t: _P_poly(t, A, B, m30, m21, m12, m03, 0, 0), *grid, -b)
    DV = np.nan if not np.isfinite(tO) else M_ols(tO) - fgls(tO)
    cf_gap = max(abs(M_ols(t) - _M_closed(t, A, B, m30, m21, m12, m03)) for t in np.linspace(-3, -0.2, 20))
    return dict(mu30=m30, mu21=m21, mu12=m12, mu03=m03, t21=_tstat(x ** 2 * r), t12=_tstat(x * r ** 2), delta_star=-tau_star,
                closed_form_gap=cf_gap, delta_OLS=-tO, beta_OLS_fs=bet(tO), delta_FGLS=-tF, beta_FGLS_fs=bet(tF),
                delta_wGLS=-tW, beta_wGLS_fs=bet(tW), delta_mix0=-tM, beta_mix0=bet(tM), Delta_V=DV, beta_ols=bols)


def _het_task(args):
    drv, th, sd = args
    try:
        return dict(design='%s%s' % (drv, th), driver=drv, theta=th, seed=sd, **het_one(_G['cfg']['het_N'], drv, th, sd, _G['cfg']))
    except ValueError:
        return dict(design='%s%s' % (drv, th), driver=drv, theta=th, seed=sd)


def run_het(cfg=CFG, verbose=True):
    designs = [('none', 0)] + [('w', t) for t in cfg['het_theta']] + [('x', t) for t in cfg['het_theta_x']]
    seeds = [7 + 10 * i for i in range(cfg['het_draws'])]; tasks = [(d, t, s) for d, t in designs for s in seeds]
    _G['cfg'] = cfg
    if cfg['n_cores'] > 1:
        with get_context('fork').Pool(cfg['n_cores']) as pool:
            rows = pool.map(_het_task, tasks)
    else:
        rows = [_het_task(t) for t in tasks]
    df = pd.DataFrame(rows)
    if verbose:
        for (d, t), g in df.groupby(['driver', 'theta'], sort=False):
            print('  %-5s theta=%-4s delta=%.3f  beta=%.3f  t(mu12)=%7.1f' % (d, t, g.delta_OLS.mean(), g.beta_OLS_fs.mean(), g.t12.mean()))
    return df


_LAB = {'none': 'none (homoscedastic)', 'w': r'control \(w\)', 'x': r'regressor \(x\)'}


def _het_tables(df):
    a, b = [], []
    for (drv, th), R in df.groupby(['driver', 'theta'], sort=False):
        m = lambda k: R[k].mean(); s = lambda k: R[k].std(ddof=1)
        lab = dict(driver=_LAB[drv], theta=('%g' % th) if drv != 'none' else '0')
        a.append(dict(lab, mu12=m('mu12'), t_mu12=m('t12'), delta_DT=m('delta_OLS'), sd_delta=s('delta_OLS'), beta_DT=m('beta_OLS_fs'),
                      sd_beta=s('beta_OLS_fs'), delta_mix0=m('delta_mix0'), sd_delta_mix0=s('delta_mix0'), beta_mix0=m('beta_mix0'),
                      sd_beta_mix0=s('beta_mix0'), mu30=m('mu30'), mu21=m('mu21'), mu03=m('mu03')))
        b.append(dict(lab, delta_star=m('delta_star'), delta_OLS=m('delta_OLS'), sd_OLS=s('delta_OLS'), beta_OLS=m('beta_OLS_fs'),
                      delta_FGLS=m('delta_FGLS'), sd_FGLS=s('delta_FGLS'), beta_FGLS=m('beta_FGLS_fs'), delta_wGLS=m('delta_wGLS'),
                      beta_wGLS=m('beta_wGLS_fs'), Delta_V=m('Delta_V')))
    return pd.DataFrame(a), pd.DataFrame(b)


# ---------------------------------------------------------------- LaTeX emitters
def _f(v, d=3):
    return '---' if v is None or not np.isfinite(v) else ('%.' + str(d) + 'f') % v


def _fs(v, d=3):
    return '---' if not np.isfinite(v) else ('+' if v >= 0 else '') + ('%.' + str(d) + 'f') % v


def _fn(v):
    return '---' if not np.isfinite(v) else '{:,}'.format(int(v)).replace(',', '{,}')


def _sg(v, d=3):
    s = _f(v, d); return s if s.startswith('-') else r'\phantom{-}' + s


def _provisional(cfg, ref=CFG):
    bad = []
    if cfg['n_populations'] * cfg['n_draws'] < ref['n_populations'] * ref['n_draws']:
        bad.append('%d estimation samples per size against the paper\'s %d' % (cfg['n_populations'] * cfg['n_draws'],
                                                                              ref['n_populations'] * ref['n_draws']))
    if tuple(cfg['sample_sizes']) != tuple(ref['sample_sizes']):
        bad.append('sample sizes %s against the paper\'s %s' % (', '.join(map(str, cfg['sample_sizes'])), ', '.join(map(str, ref['sample_sizes']))))
    if cfg['population_size'] < ref['population_size']:
        bad.append('populations of %s against %s' % (format(cfg['population_size'], ','), format(ref['population_size'], ',')))
    return '; '.join(bad) or None


def _write_tex(lines, name, cfg, prov=None):
    os.makedirs(cfg['tab_dir'], exist_ok=True); path = os.path.join(cfg['tab_dir'], name + '.tex')
    with open(path, 'w') as fh:
        fh.write('%% generated by sim_montecarlo.py (port of DT_IIV_montecarlo_v21.R) on %s\n' % time.strftime('%Y-%m-%d %H:%M'))
        if prov:
            fh.write('%% PRELIMINARY -- %s. Regenerate before submission.\n' % prov)
        fh.write('\n'.join(lines) + '\n')
    print('  ->', path); return path


def emit_simulation_results_tex(df, cfg):
    tb = table_mc(df, cfg); nt = table_mc_notes(df); body = []
    for i, n in enumerate(sorted(tb.n.unique())):
        o = tb[(tb.n == n) & (tb.method == 'OLS')].iloc[0]; v = tb[(tb.n == n) & (tb.method == 'IIV')].iloc[0]
        if i:
            body.append(r'\midrule')
        body.append('%s & OLS  & %s & %s & %s & %s & %s & %s \\\\' % (_fn(n), _f(o['mean']), _f(o.sd), _f(o.ci_lower), _f(o.ci_upper), _fs(o.bias), _f(o.rmse)))
        body.append('        & IIV  & %s & %s & %s & %s & %s & %s \\\\' % (_f(v['mean']), _f(v.sd), _f(v.ci_lower), _f(v.ci_upper), _fs(v.bias), _f(v.rmse)))
    scale_note = '; '.join(r'\(n=%s\): median \(%s\), SD \(%s\), \(%s\)\%% flagged weakly identified' % (
        _fn(r.n), _f(r.delta_median), _f(r.delta_sd), _f(r.pct_weak, 0)) for r in nt.itertuples())
    acc = nt.sign_accuracy.round(4)
    sign_note = ('the correct sign\n\\(k=-1\\) is selected in \\(100\\%\\) of samples at every sample size' if (acc == 1).all()
                 else 'the correct sign \\(k=-1\\) is selected in ' + ', '.join(
                     r'\(%s\%%\) of samples at \(n=%s\)' % (_f(100 * a, 1), _fn(n)) for a, n in zip(nt.sign_accuracy, nt.n)))
    prov = _provisional(cfg)
    notes = (r"\item \textit{Notes:} Aligned design of Section~\ref{dgp} with" "\n"
             r"\(\beta=%g\), \(b=\delta_0=%g\), \(\epsilon_x=%g\) and hence" "\n"
             r"\(\operatorname{plim}\widehat\beta_{\mathrm{OLS}}=%g\). Results are based on" "\n"
             r"\(%d\) populations of \(N=%s\) with \(%d\) samples drawn from each" "\n"
             r"(\(%d\) estimation samples per sample size). ``Std Error'' is the standard" "\n"
             r"deviation of the estimates across the samples entering each row and the" "\n"
             r"interval is \(\text{Mean}\pm1.96\times\text{Std Error}\); Bias \(=\)" "\n"
             r"Mean \(\beta-%g\). IIV is the VT-GMM estimator of Corollary~\ref{comp}" "\n"
             r"with the sign selected by the search of Subsection~\ref{sign} over" "\n"
             r"\(\mathcal D=[%g,%g]\); %s. Scale estimates: %s. Samples in which the" "\n"
             r"gate returns a boundary status are excluded from the IIV row, which" "\n"
             r"therefore rests on %s estimates respectively.") % (
        cfg['beta_true'], cfg['delta0_target'], cfg['eps_x'], cfg['beta_true'] + cfg['delta0_target'], cfg['n_populations'],
        _fn(cfg['population_size']), cfg['n_draws'], cfg['n_populations'] * cfg['n_draws'], cfg['beta_true'], cfg['delta_lower'],
        cfg['delta_upper'], sign_note, scale_note, ', '.join(_fn(v) for v in nt.n_estimates))
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{%sMonte Carlo Simulation Results: Estimator Performance}' % ('' if prov is None else r'\textbf{[PRELIMINARY]} '),
             r'\label{tab:simulation_results}', r'\begin{threeparttable}', r'\begin{tabular}{llcccccc}', r'\toprule',
             r'\(n\) & Method & Mean \(\beta\) & Std Error & 95\% CI Lower & 95\% CI Upper & Bias & RMSE \\', r'\midrule'] + body + \
            [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\small'] + \
            ([r'\item \textbf{PRELIMINARY --- not the paper''s design.} This run used %s.' % prov] if prov else []) + \
            [notes, r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
    return _write_tex(lines, 'tab_simulation_results', cfg, prov)


def emit_dgp_params_tex(pm, cfg):
    m = pm['moments']; t = pm['t']
    lines = [r'\begin{table}[htbp]', r'\centering', r'\caption{Monte Carlo design parameters and realised population moments}',
             r'\label{tab:dgp_params}', r'\begin{threeparttable}', r'\begin{tabular}{llr}', r'\toprule',
             r'Object & Definition & Value \\', r'\midrule',
             r'\(\epsilon_x\)      & skewness of the regressor transform            & \(%s\) \\' % _f(pm['eps_x'], 1),
             r'\(\kappa_x\)        & tail parameter of the regressor transform      & \(%s\) \\' % _f(pm['kappa_x'], 1),
             r'\(b=\delta_0\)      & \(\operatorname{cov}(x,u)\); OLS bias           & \(%s\) \\' % _f(pm['target_delta0'], 1),
             r'\(\lambda\)         & mixture weight from \eqref{eq:alignment_choice} & \(%s\) \\' % _f(pm['lam']),
             r'\(\beta\)           & structural coefficient                          & \(%s\) \\' % _f(pm['beta_true'], 1),
             r'\midrule',
             r'\(\mu_{30}\)        & \(\mathbb E[x^{3}]\)                            & \(%s\) \\' % _sg(m['mu30']),
             r'\(\mu_{21}\)        & \(\mathbb E[x^{2}r]\)                           & \(%s\) \\' % _sg(m['mu21']),
             r'\(\mu_{12}\)        & \(\mathbb E[xr^{2}]\)                           & \(%s\) \\' % _sg(m['mu12']),
             r'\(\mu_{03}\)        & \(\mathbb E[r^{3}]=b\mu_{30}\)                  & \(%s\) \\' % _sg(m['mu03']),
             r'\(k\)               & \(-\operatorname{sign}\{\operatorname{cov}(x,u)\}\) & \(%+d\) \\' % pm['k'],
             r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\small',
             (r"\item \textit{Notes:} Population moments computed on a single draw of" "\n"
              r"\(N=%s\). The realised \(\mu_{03}=%s\) differs from its target" "\n"
              r"\(b\mu_{30}=%s\) only by the finite-sample error in solving" "\n"
              r"\eqref{eq:alignment_choice}. The \(t\)-statistics of the diagnostic in" "\n"
              r"Remark~\ref{rem:practical_diag} are \(%s\) for \(\mu_{30}\) and \(%s\) for" "\n"
              r"\(\mu_{03}\), against \(%s\) and \(%s\) for \(\mu_{21}\) and \(\mu_{12}\):" "\n"
              r"identification runs through the two marginal third moments, as" "\n"
              r"\eqref{eq:M_aligned} implies. Minimising the population VT moment over" "\n"
              r"\(\mathcal D=[0.05,5]\) locates the root at \(%s\) against the target" "\n"
              r"\(\delta_0=%s\).") % (_fn(pm['N']), _f(m['mu03']), _f(pm['target_delta0'] * m['mu30']), _f(t['mu30'], 1),
                                      _f(t['mu03'], 1), _f(t['mu21'], 2), _f(t['mu12'], 2), _f(pm['dt_root']), _f(pm['target_delta0'])),
             r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
    return _write_tex(lines, 'tab_dgp_params', cfg)


def emit_het_channels_tex(het, cfg):
    d, _ = _het_tables(het); body = []; prev = None
    for r in d.itertuples():
        if prev is not None and r.driver != prev:
            body.append(r'\midrule')
        prev = r.driver
        body.append('%s & \\(%s\\) & \\(%s\\) & \\(%s\\)\n  & \\(%s\\) & \\(%s\\) & \\(%s\\) & \\(%s\\) \\\\' % (
            r.driver, r.theta, _sg(r.mu12), _sg(r.t_mu12, 1), _f(r.delta_DT), _f(r.beta_DT), _f(r.delta_mix0), _f(r.beta_mix0)))
        body.append('  & & & & \\((%s)\\) & \\((%s)\\) & \\((%s)\\) & \\((%s)\\) \\\\' % (
            _f(r.sd_delta), _f(r.sd_beta), _f(r.sd_delta_mix0), _f(r.sd_beta_mix0)))
    notes = (r"\item \textit{Notes:} Means over \(%d\) independent draws of \(N=%s\) from" "\n"
             r"the aligned design of Section~\ref{dgp}, with across-draw standard deviations" "\n"
             r"in parentheses. The design has \(\beta=2\), \(b=\delta_0=1\), \(k=-1\) and" "\n"
             r"\(\operatorname{plim}\widehat\beta_{\mathrm{OLS}}=3\); the orthogonality point is" "\n"
             r"\(\delta^\star\) in Table~\ref{tab:fgls}. The error variance is scaled by" "\n"
             r"\(h(\cdot)=\{0.3+\exp(\theta\,\cdot)\}^{1/2}\), evaluated at the standardized" "\n"
             r"control \(w\) in the middle block and at the regressor \(x\) in the last;" "\n"
             r"\(\theta=0\) is the homoscedastic baseline. In every row the marginal alignment" "\n"
             r"restriction \(\mu_{03}=b\,\mu_{30}\) is re-imposed by resolving for the skewness" "\n"
             r"of the exogenous error, so the comparison isolates the mixed-moment channel." "\n"
             r"\(\widehat\delta_0\) is the root of the VT moment; \(\widetilde\delta_0\) is the" "\n"
             r"root of the same polynomial \eqref{eq:M_third_moments} with \(\widehat\mu_{21}\)" "\n"
             r"and \(\widehat\mu_{12}\) set to zero (Remark~\ref{rem:mixed_sensitivity}). The" "\n"
             r"marginal moments are constant across rows at \(\widehat\mu_{30}=%s\) and" "\n"
             r"\(\widehat\mu_{03}=%s\); the other mixed moment is \(\widehat\mu_{21}=%s\)" "\n"
             r"throughout.") % (cfg['het_draws'], _fn(cfg['het_N']), _f(d.mu30.mean()), _f(d.mu03.mean()), _f(d.mu21.mean()))
    lines = [r'\begin{table}[!h]', r'\centering',
             r'\caption{The two channels: heteroscedasticity in the controls versus in the regressor}',
             r'\label{tab:het_channels}', r'\begin{threeparttable}', r'\begin{tabular}{llrrcccc}', r'\toprule',
             r'& & \multicolumn{2}{c}{Mixed moment}', r'& \multicolumn{2}{c}{VT estimate}',
             r'& \multicolumn{2}{c}{Mixed moments set to zero} \\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Variance driven by & \(\theta\)', r'& \(\widehat\mu_{12}\) & \(t\)', r'& \(\widehat\delta_0\) & \(\widehat\beta\)',
             r'& \(\widetilde\delta_0\) & \(\widetilde\beta\) \\', r'\midrule'] + body + \
            [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\small', notes, r'\end{tablenotes}',
             r'\end{threeparttable}', r'\end{table}']
    return _write_tex(lines, 'tab_het_channels', cfg)


def emit_fgls_tex(het, cfg):
    _, d = _het_tables(het); body = []; prev = None
    for r in d.itertuples():
        if prev is not None and r.driver != prev:
            body.append(r'\midrule')
        prev = r.driver
        body.append('%s & \\(%s\\) & \\(%s\\) & \\(%s\\) & \\(%s\\) & \\(%s\\) & \\(%s\\) & \\(%s\\) \\\\' % (
            r.driver, r.theta, _f(r.delta_star), _f(r.delta_OLS), _f(r.beta_OLS), _f(r.delta_FGLS), _f(r.beta_FGLS), _f(r.Delta_V)))
        body.append(' & & & \\((%s)\\) & & \\((%s)\\) & & \\\\' % (_f(r.sd_OLS), _f(r.sd_FGLS)))
    notes = (r"\item \textit{Notes:} Same draws and design as Table~\ref{tab:het_channels};" "\n"
             r"means over \(%d\) draws of \(N=%s\), with across-draw standard deviations of" "\n"
             r"\(\widehat\delta_0\) in parentheses. \(\delta^\star\) is the orthogonality point" "\n"
             r"\(-\operatorname{cov}(x,u)/\operatorname{var}(r)\), and \(\beta=2\). The FGLS" "\n"
             r"columns weight the first-stage regression of \(x\) on \(\mathbf s(\boldsymbol\delta)\)" "\n"
             r"by \(1/\widehat v_i\), where \(\widehat v\) is the fitted value of the squared OLS" "\n"
             r"first-stage residual on a quadratic in the exogenous control --- a" "\n"
             r"\emph{correctly specified} variance model in the control-driven rows. Weighting" "\n"
             r"the moment as well as the first stage gives \(\widehat\delta_0=%s\)." "\n"
             r"\(\Delta_{\mathbf V}\) is \eqref{eq:ols_fgls_gap} evaluated at" "\n"
             r"\(\widehat\delta_0\) with the same variance model (Remark~\ref{rem:delta_W}).") % (
        cfg['het_draws'], _fn(cfg['het_N']), ', '.join(_f(v) for v in d.delta_wGLS))
    lines = [r'\begin{table}[!h]', r'\centering', r'\caption{Ordinary least squares versus feasible GLS in the first stage}',
             r'\label{tab:fgls}', r'\begin{threeparttable}', r'\begin{tabular}{llcccccc}', r'\toprule',
             r'& & & \multicolumn{2}{c}{OLS first stage}', r'& \multicolumn{2}{c}{FGLS first stage} & \\',
             r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}', r'Variance driven by & \(\theta\) & \(\delta^\star\)',
             r'& \(\widehat\delta_0\) & \(\widehat\beta\)', r'& \(\widehat\delta_0\) & \(\widehat\beta\)', r'& \(\Delta_{\mathbf V}\) \\',
             r'\midrule'] + body + [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\small', notes, r'\end{tablenotes}',
                                   r'\end{threeparttable}', r'\end{table}']
    return _write_tex(lines, 'tab_fgls', cfg)


def emit_sim_numbers_tex(df, cfg):
    """Macros the supplement's prose quotes (\\Sim...), in the formats of the archived app_numbers.tex."""
    tb = table_mc(df, cfg); nt = table_mc_notes(df); words = ['One', 'Two', 'Three', 'Four', 'Five', 'Six']; m = {}
    sizes = sorted(tb.n.unique())
    for i, w in enumerate(words):
        if i < len(sizes):
            n = sizes[i]; o = tb[(tb.n == n) & (tb.method == 'OLS')].iloc[0]; v = tb[(tb.n == n) & (tb.method == 'IIV')].iloc[0]
            r = nt[nt.n == n].iloc[0]
            m.update({'SimSize' + w: _fn(n), 'SimBias' + w: _f(v.bias), 'SimRmse' + w: _f(v.rmse), 'SimOlsMean' + w: _f(o['mean'], 2),
                      'SimOlsRmse' + w: _f(o.rmse, 2), 'SimDeltaMedian' + w: _f(r.delta_median), 'SimDeltaSD' + w: _f(r.delta_sd),
                      'SimWeak' + w: _f(r.pct_weak, 0), 'SimSignAcc' + w: _f(100 * r.sign_accuracy, 1), 'SimNEst' + w: _fn(r.n_estimates)})
        else:
            for q in ('Size', 'Bias', 'Rmse', 'OlsMean', 'OlsRmse', 'DeltaMedian', 'DeltaSD', 'Weak', 'SignAcc', 'NEst'):
                m['Sim' + q + w] = r'\textit{n/a}'
    rm = tb[tb.method == 'IIV'].set_index('n').rmse
    m.update(SimPopulations=str(cfg['n_populations']), SimDraws=str(cfg['n_draws']), SimPopSize=_fn(cfg['population_size']),
             SimSamples=_fn(cfg['n_populations'] * cfg['n_draws']), SimSamplesTotal=_fn(len(df)),
             SimEstimates=_fn(nt.n_estimates.sum()), SimSizeCount=str(len(sizes)), SimSizeList=r',\,'.join(_fn(n) for n in sizes),
             SimBetaTrue='%g' % cfg['beta_true'], SimDeltaTrue='%g' % cfg['delta0_target'],
             SimOlsPlim='%g' % (cfg['beta_true'] + cfg['delta0_target']), SimDomainLo=_f(cfg['delta_lower'], 1),
             SimDomainHi=_f(cfg['delta_upper'], 1), SimRmseRatio=_f(rm.iloc[0] / rm.iloc[-1], 1),
             SimSignAccuracy=_f(100 * nt.sign_accuracy.min(), 1), SimSignAllCorrect='1' if (nt.sign_accuracy == 1).all() else '0')
    prov = _provisional(cfg)
    return _write_tex(['\\providecommand{\\%s}{%s}' % kv for kv in m.items()], 'sim_numbers', cfg, prov)


def emit_all(cfg, tag=''):
    """Rebuild every fragment from the saved CSV files (python sim_montecarlo.py tables [quick])."""
    df = pd.read_csv(out('sim_montecarlo_draws%s.csv' % tag))
    emit_simulation_results_tex(df, cfg); emit_sim_numbers_tex(df, cfg); emit_dgp_params_tex(dgp_population_moments(cfg), cfg)
    hp = out('sim_montecarlo_het%s.csv' % tag)
    if os.path.exists(hp):
        het = pd.read_csv(hp); emit_het_channels_tex(het, cfg); emit_fgls_tex(het, cfg)


# ---------------------------------------------------------------- self-test
def mc_selftest(verbose=True):
    ok = {}; cfg = dict(CFG); cfg['population_size'] = 40000
    p = dgp_siv(40000, cfg, 11); m30, m03 = np.mean(p['x0'] ** 3), np.mean(p['R'] ** 3)
    ok['alignment mu03 = b mu30 holds in the DGP'] = abs(m03 - cfg['delta0_target'] * m30) < 0.02 * abs(m30)
    c0 = dict(cfg); c0['eps_x'] = 0; p0 = dgp_siv(20000, c0, 11)
    ok['eps_x = 0 leaves mu30 at zero'] = abs(np.mean(p0['x0'] ** 3)) < 0.02
    x, r = p['x0'], p['R']; A, B = np.mean(x ** 2), np.mean(r ** 2)
    gap = max(abs(_M_numeric(t, x, r) - _M_closed(t, A, B, np.mean(x ** 3), np.mean(x ** 2 * r), np.mean(x * r ** 2), np.mean(r ** 3)))
              for t in np.linspace(-3, -0.2, 15))
    ok['closed-form VT moment equals the numerical one'] = gap < 1e-12
    rng = np.random.default_rng(7); n = 1200
    xs = (rng.chisquare(4, n) - 4) / np.sqrt(8); es = -(rng.chisquare(8, n) - 8) / 4; ys = 2 * xs - xs / np.sqrt(2) + es
    try:
        f = run_dt_iiv(ys, xs[:, None], delta_lower=0.2, delta_upper=3, scale='sd', relevance_c=0.05, n_grid=60, n_starts=3,
                       max_cycles=2, id_grid=21, inference='analytic', on_fail='error')
        vr = dt_variance_routes(f)
    except (RuntimeError, ValueError):
        vr = None
    ok['variance routes computed'] = vr is not None and all(np.all(np.isfinite(v)) for v in vr['se'].values())
    ok['naive < corrected < uncorrected'] = vr is not None and vr['se']['naive'][0] < vr['se']['corrected'][0] < vr['se']['uncorrected'][0]
    fake = pd.DataFrame(dict(pop=[1, 2, 3, 4] * 2, draw=[1, 2, 3, 4] * 2, n=[1000] * 4 + [2000] * 4,
                             beta_ols=[3.0, 3.1, 2.9, 3.0, 3.0, 3.0, 3.0, 3.0], beta=[1.9, 2.1, np.nan, 2.0, 2, 2, 2, 2],
                             delta=[0.9, 1.1, np.nan, 1.0, 1, 1, 1, 1], k=[-1, -1, 1, -1, -1, -1, -1, -1],
                             status=['ok', 'weak', 'boundary_lower', 'ok'] + ['ok'] * 4, weak=[False, True, None, False] + [False] * 4))
    fc = dict(CFG); fc.update(n_populations=2, n_draws=2, sample_sizes=(1000, 2000), population_size=1000,
                              tab_dir=os.path.join(OUT, 'tables_selftest'))
    try:
        txt = open(emit_simulation_results_tex(fake, fc)).read(); ok['emitters survive sign accuracy below 1'] = '75.0' in txt
    except (ValueError, KeyError) as e:
        ok['emitters survive sign accuracy below 1'] = False
    ok['formatters accept NA'] = [_f(1.5), _f(np.nan), _f(-2)] == ['1.500', '---', '-2.000'] and _fs(1.5) == '+1.500' and \
        [_fn(1000), _fn(np.nan)] == ['1{,}000', '---']
    if verbose:
        print('\nVT-IIV Monte Carlo self-test')
        for k, v in ok.items():
            print('  %-48s %s' % (k, 'PASS' if v else 'FAIL'))
    return all(ok.values())


# ---------------------------------------------------------------- driver
if __name__ == '__main__':
    quick = 'quick' in sys.argv[1:]; do_het = '--no-het' not in sys.argv[1:]
    cfg = quick_cfg() if quick else dict(CFG)
    if quick:
        print('\n*** QUICK MODE: not the paper\'s settings. ***\n')
    if 'tables' in sys.argv[1:]:
        emit_all(cfg, '_preliminary' if quick else ''); sys.exit(0)
    if not mc_selftest():
        sys.exit('mc_selftest() failed')
    check_identification(cfg)
    tag = '_preliminary' if quick else ''
    df = run_mc(cfg); df.to_csv(out('sim_montecarlo_draws%s.csv' % tag), index=False)
    print(table_mc(df, cfg).round(4).to_string(index=False)); print(table_mc_notes(df).round(4).to_string(index=False))
    errs = df.error.dropna()
    if len(errs):
        print('\n***', len(errs), 'fits errored:', errs.value_counts().head(5).to_dict())
    print('\nElapsed: %.1f min' % (df.attrs['elapsed'] / 60))
    emit_simulation_results_tex(df, cfg); emit_sim_numbers_tex(df, cfg)
    pm = dgp_population_moments(cfg); emit_dgp_params_tex(pm, cfg)
    if do_het:
        het = run_het(cfg); het.to_csv(out('sim_montecarlo_het%s.csv' % tag), index=False)
        a, b = _het_tables(het); print(a.round(4).to_string(index=False)); print(b.round(4).to_string(index=False))
        emit_het_channels_tex(het, cfg); emit_fgls_tex(het, cfg)
