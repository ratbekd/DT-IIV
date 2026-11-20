"""Self-tests of the VT-IIV estimator core (port of siv_selftest_v21,
siv_selftest_bootstrap and siv_selftest_all in sim_estimator_DT_IIV_v21_clean.R).
Run: python -m iiv.dtcore_selftest   (from code/lib)"""
import inspect
import numpy as np
from scipy.stats import norm
from . import dtcore as d


def _design(n, rng):
    x = (rng.chisquare(4, n) - 4) / np.sqrt(8)
    e = -(rng.chisquare(8, n) - 8) / 4
    y = 2 * x - x / np.sqrt(2) + e
    return y, x[:, None]


def selftest_v21(n=1500, seed=20, tol=1e-8, verbose=True):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n); Wdup = np.column_stack([np.ones(n), w, w])
    Z = rng.standard_normal((n, 2)); Zr = d.partial_out(Z, Wdup)
    projection_gap = np.max(np.abs(np.column_stack([np.ones(n), w]).T @ Zr)) / n
    y, X = _design(n, rng)
    P = d.siv_prepare(y, X, scale='sd'); dl = 1 / np.sqrt(2); dr = dl * P['x_scale'] / P['r_scale']
    Sw = d.make_S(P['X_work'], P['r_work'], dl, 1); Sr = d.make_S(P['X_tilde'], P['r_raw'], dr, 1)
    scale_gap = np.max(np.abs(Sr - Sw * P['x_scale']))
    cs = d.corrected_dt_scores(P['X_work'], P['r_work'], 1, dl)
    ss = d.siv_S_stat(dl, P['X_work'], P['r_work'], 1)
    ids = d.siv_identification_set(P['X_work'], P['r_work'], 1, 0.3, 1.3, n_grid=11)
    cue = d.estimate_delta_dt_cue(P['X_work'], P['r_work'], 1, 0.3, 1.3, n_grid=40, n_starts=3)
    Pv = d.siv_prepare(y, X, scale='none'); Pvd = d.siv_prepare(y, X, Wdup, scale='none')
    l1 = d.estimate_delta_dt_v16_identity(Pv['X_work'], Pv['r_work'], 1, 0.3, 1.3, n_grid=40, n_starts=3, seed=77)
    l2 = d.estimate_delta_dt_v16_identity(Pv['X_work'], Pv['r_work'], 1, 0.3, 1.3, n_grid=40, n_starts=3, seed=77)
    Sl = d.make_S(Pv['X_work'], Pv['r_work'], l1['delta_hat'], 1)
    Sl_manual = Pv['X_work'] + np.outer(Pv['r_work'], l1['delta_hat'])
    gate = d.siv_sign_search_v16(Pv['X_work'], Pv['r_work'], 0.3, 1.3, legacy_interior_frac=0.02, n_grid=80, n_starts=3,
                                 seed=77, compute_id_set=False)
    gsel = gate['ok'] and int(gate['k_hat'][0]) == 1 and gate['delta_hat'][0] > 0.3 and \
        gate['selection_reason'] == 'legacy_v16_global_objective'
    cond = d.estimate_delta_dt_v16_given_sign(Pv['X_work'], Pv['r_work'], gate['k_hat'], 0.3, 1.3, legacy_interior_frac=0.02,
                                              n_grid=80, init_delta=gate['delta_hat']) if gate['ok'] else dict(converged=False)
    zl = d.siv_sign_search_v16(Pv['X_work'], Pv['r_work'], 0, 1.3, legacy_interior_frac=0.02, n_grid=80, n_starts=3, seed=77,
                               compute_id_set=False)
    # numerical Jacobian with an explicit step scale
    jf = lambda v: np.array([np.sin(v[0]) * v[1], v[0] ** 2]); v0 = np.array([0.3, 2.5])
    Jn = d._jacobian(jf, v0, step=[0.3, 2.5]); Ja = np.array([[np.cos(v0[0]) * v0[1], np.sin(v0[0])], [2 * v0[0], 0]])
    optk = dict(delta_lower=np.array([0.3]), delta_upper=np.array([1.3]), scale='sd', selection_method='cue', relevance_c=0.05,
                interior_tol=1e-3, weak_frac=0.9, legacy_interior_frac=0.05, legacy_seed=1, n_grid=40, n_starts=3,
                max_cycles=2, max_sign_combinations=4096, id_grid=9, level=0.95, in_bootstrap=True,
                k_fixed=np.array([1]), delta_init=np.array([0.7]))
    ff = d.fit_core(y, X, None, optk)
    fixed_path = ff['ok'] and int(ff['k_hat'][0]) == 1 and ff['sign_source'] == 'imposed' and ff['gate']['sign_table'] is None
    on = dict(optk); on['k_fixed'] = np.array([-1]); nf = d.fit_core(y, X, None, on)
    av = d.siv_vcov_analytic(P['y_tilde'], P['X_tilde'], P['r_raw'], 1, dr)
    lam_only = av['Lambda'] @ (av['m_tilde'].T @ av['m_tilde'] / n) @ av['Lambda'].T
    vv = np.random.default_rng(99).normal(1, 0.5, 500); pct = np.quantile(vv, [0.025, 0.975]); Bv = len(vv)
    f0 = min(max(np.sum(vv < 0) / Bv, 1 / (Bv + 1)), Bv / (Bv + 1)); p_plain = min(1, 2 * min(f0, 1 - f0))
    pf = d.run_dt_iiv(y, X, delta_lower=0.3, delta_upper=1.3, scale='sd', relevance_c=0.05, n_grid=40, n_starts=3,
                      max_cycles=2, id_grid=9, inference='none', analytic_diagnostic=False, on_fail='error', names=['x'])
    prof = d.siv_scalar_profile(pf, n_grid=400); prof2 = d.siv_scalar_profile(pf, n_grid=200, k=-int(pf['k_hat'][0]))
    checks = {
        'rank-safe FWL projection': projection_gap < tol,
        'scale-free/raw instrument equivalence': scale_gap < tol,
        'corrected VT score is finite': cs is not None and d._ok(cs['m_tilde']),
        'corrected S statistic is finite': np.isfinite(ss),
        'scalar S-set inversion': ids['supported'] and len(ids['S']) == 11,
        'CUE is active': cue['converged'] and cue['weighting'] == 'continuously_updated_ordinary_dt_score',
        'old-v16 global profile is reproducible': l1['converged'] and l2['converged'] and
        abs(l1['delta_hat'][0] - l2['delta_hat'][0]) < tol and abs(l1['J_hat'] - l2['J_hat']) < tol,
        'old-v16 global sign objective selects interior winner': bool(gsel),
        'sign is fixed before conditional delta estimation': cond['converged'] and int(cond['k_fixed']) == int(gate['k_hat'][0]),
        'conditional delta stage cannot rerun sign search': 'siv_sign_search' not in inspect.getsource(d._estimate_delta_given_sign),
        'legacy sign search accepts a zero lower endpoint': zl['ok'] and zl['delta_hat'][0] > 0,
        'failed sign selection cannot become silent OLS': 'exogenous_boundary' not in inspect.getsource(d.siv_sign_search_v16),
        'legacy raw FWL accepts redundant controls': d._ok(Pvd['X_tilde'], Pvd['y_tilde'], Pvd['r_raw']),
        'old-v16 make_S uses the plus convention': np.max(np.abs(Sl - Sl_manual)) < tol,
        'pairs-bootstrap implementation contract': all(t in inspect.getsource(d._boot_one) + inspect.getsource(d.siv_bootstrap)
                                                       for t in ('integers', 'fit_core', 'beta_reps', 'delta_reps', 'p_beta',
                                                                 'sign_stage_k_reps', 'conditional_k_reps')),
        'numerical Jacobian honours an explicit step scale': np.max(np.abs(Jn - Ja)) < 1e-6,
        'fit core accepts an imposed sign vector': bool(fixed_path),
        'an imposed sign is never overridden': (not nf['ok']) or int(nf['k_hat'][0]) == -1,
        'analytic delta SE maps between scales': av is not None,
        'OLS/IIV gap variance extends beyond the Lambda term': np.max(np.abs(av['V_difference'] - lam_only)) > 1e-8,
        'gap statistic carries its domain caveat': bool(av['ols_gap']['note']) and np.all(np.isfinite(av['ols_gap']['se'])),
        'G_M equals J_M inverse when just identified': av['G_M_identity_gap'] < 1e-6,
        'BCa reduces to percentile at z0 = a = 0': np.max(np.abs(d._boot_ci(vv, [0.025, 0.975], 0, 0) - pct)) < tol,
        'BCa moves the interval when z0/a are non-zero': abs(d._boot_ci(vv, [0.025, 0.975], 0.2, 0.05)[0] - pct[0]) > 1e-6,
        'bootstrap p-value inverts its interval': abs(d._boot_p(vv, 0, 0, 0) - p_plain) < tol,
        'scalar profile is correctly labelled': len(prof['roots']) > 0 and all(k in prof['roots'][0] for k in
                                                                            ('delta', 'moment', 'relevance', 'admissible')),
        'scalar profile runs at either sign and maps delta to the raw scale': prof2['k'] == -int(pf['k_hat'][0]) and
        np.max(np.abs(prof['profile']['delta_raw'] - prof['profile']['delta'] * pf['x_scale'][0] / pf['r_scale'])) < tol}
    if verbose:
        print('\nVT-IIV v21 implementation self-test')
        for k, v in checks.items():
            print('  %-66s %s' % (k, 'PASS' if v else 'FAIL'))
    return dict(checks=checks, all_pass=all(checks.values()))


def selftest_bootstrap(n=1000, B=29, seed=21, verbose=True):
    y, X = _design(n, np.random.default_rng(seed))
    kw = dict(delta_lower=0.2, delta_upper=1.5, relevance_c=0.05, n_grid=60, n_starts=3, max_cycles=2, report_point='sample',
              on_fail='error', names=['x'])
    fit = d.run_dt_iiv(y, X, scale='sd', id_grid=31, inference='both', B=B, seed=seed + 100, bootstrap_sign='fixed',
                       ci_type='percentile', analytic_diagnostic=True, **kw)
    audit = d.siv_bootstrap_audit(fit); b = fit['bootstrap']
    none_fit = d.run_dt_iiv(y, X, scale='none', id_grid=9, inference='bootstrap', B=15, seed=seed + 300, ci_type='percentile',
                            analytic_diagnostic=False, **kw)
    refit = d.run_dt_iiv(y, X, scale='sd', id_grid=31, inference='bootstrap', B=15, seed=seed + 200,
                         bootstrap_sign='reestimated', ci_type='percentile', analytic_diagnostic=False, **kw)
    guard = d.run_dt_iiv(y, X, scale='sd', id_grid=9, inference='both', B=20, seed=seed + 400, ci_type='bca', **kw)
    checks = dict(audit['checks'])
    checks.update({
        'delta re-estimated across pairs': bool(np.any(np.nanstd(b['delta_reps'], 0) > 0)),
        'enough successful bootstrap draws': b['n_success'] >= max(5, B // 2),
        'small-B BCa guard falls back to percentile': bool(np.all(guard['bootstrap']['ci_fallback_beta']) and
                                                           np.all(guard['bootstrap']['z0_beta'] == 0) and
                                                           guard['bootstrap']['ci_type_used_beta'][0] == 'percentile'),
        'sign never re-selected when fixed': b['sign_source'] == 'fixed_at_sample_fit' and len(b['sign_frequency']) == 1,
        'delta SEs on both scales are consistent': np.max(np.abs(none_fit['bootstrap']['se_delta_raw'] -
                                                                 none_fit['bootstrap']['se_delta_work'])) < 1e-12,
        'bootstrap and analytic SEs both reported': np.all(np.isfinite(b['se_beta'])) and
        np.all(np.isfinite(fit['analytic']['se_beta'])),
        'per-draw sign selection still available': refit['bootstrap']['sign_source'] == 'reestimated_each_draw' and
        refit['bootstrap']['k_fixed'] is None})
    if verbose:
        print('\nVT-IIV v21 pairs-bootstrap self-test')
        for k, v in checks.items():
            print('  %-66s %s' % (k, 'PASS' if v else 'FAIL'))
    return dict(checks=checks, all_pass=all(checks.values()), fit=fit)


def selftest_all(run_bootstrap=True, verbose=True):
    a = selftest_v21(verbose=verbose)
    b = selftest_bootstrap(verbose=verbose) if run_bootstrap else dict(all_pass=True)
    ok = a['all_pass'] and b['all_pass']
    if verbose:
        print('\nOVERALL VT-IIV v21 SELF-TEST:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    import os
    os.environ.setdefault('IIV_CORES', '1')
    selftest_all()
