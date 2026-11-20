"""Sets between the one-sided and the CH sets in the seven applications
(supplement, Subsection S5.7, Table S18): the 95% set as the bound c on
corr(u^2, s) and corr(u^2, s^2) varies from 0 (CH set) to infinity (one-sided
set), with the sample correlations at the one-sided CUE.

Usage: python power_apps.py
Writes output/applications/power_apps_bnd.csv (sets) and power_apps_breakdown.csv
(breakdown values for zero, OLS and 2SLS) and prints both.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
import _setup
from iiv.data import load
from iiv.core import Context, f3, comps
from iiv import power

NG = 1441          # grid in theta for the chi2 and F(nu) sets, as in apps_sets.py
NB = 241           # coarser grid for the simulated and bootstrap critical values; crossings refined
OUTDIR = _setup.OUT


def components(C, P, fun, n=NG):
    """Components in beta of {theta: fun(theta) <= 0} on |theta| <= pi/4, crossings refined."""
    th = np.linspace(-np.pi / 4, np.pi / 4, n)
    return [(C.beta_at(lo, P), C.beta_at(hi, P), e) for lo, hi, e in comps(fun, th)]


def show(cm):
    if not cm:
        return 'empty'
    return ' U '.join('[%s, %s]%s' % (f3(min(a, b)), f3(max(a, b)), 'b' if e else '') for a, b, e in cm)


def hull(cm):
    return (np.nan, np.nan) if not cm else (min(min(a, b) for a, b, _ in cm), max(max(a, b) for a, b, _ in cm))


def row(app, method, cm, **kw):
    lo, hi = hull(cm)
    return dict(app=app, method=method, set=show(cm), lo=lo, hi=hi, pieces=len(cm), **kw)


# ---------------------------------------------------------------- the curve
C_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, np.inf)


def part_bnd():
    rows = []; Z = np.random.default_rng(20261026).standard_normal((10000, 4))
    for ds in ('mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh'):
        C = Context(load(ds)); P = C.prep()
        fit = C.fit_cue(P, power.ONE_SIDED); t = fit['theta']
        s = np.cos(t) * P.xw + np.sin(t) * P.rw; u = np.cos(t) * P.rw - np.sin(t) * P.xw
        rho = (np.corrcoef(u ** 2, s)[0, 1], np.corrcoef(u ** 2, s ** 2)[0, 1])
        print('\n== %s : OLS %s ; one-sided CUE %s ; corr(u^2, s) %.3f, corr(u^2, s^2) %.3f at the CUE ==' % (
            C.lab, f3(P.b), f3(fit['beta']), *rho))
        for c in C_GRID:
            cm = components(C, P, lambda t: (lambda r: r[0] - r[1])(power.bounded_test(C, P, C.beta_at(t, P), c, Z)), NB)
            print('  c = %-5g %s' % (c, show(cm)))
            rows.append(row(C.lab, 'bounded', cm, c=c, rho_s=rho[0], rho_s2=rho[1], ols=P.b, cue=fit['beta']))
    return pd.DataFrame(rows)


def part_breakdown():
    """Smallest bound c with a nonempty set, and breakdown values: the smallest c at
    which zero, OLS and 2SLS enter the set."""
    rows = []; Z = np.random.default_rng(20261026).standard_normal((10000, 4))
    print('\n== Breakdown values: smallest c at which the value enters the set ==')
    for ds in ('mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh'):
        C = Context(load(ds)); P = C.prep(); b2sls = C.tsls(P)[0]
        r = dict(app=C.lab, c_min=power.c_nonempty(C, P, Z))
        for nm, b in (('zero', 0.0), ('ols', P.b), ('tsls', b2sls)):
            r[nm] = b; r['c_' + nm] = power.breakdown(C, P, b, Z)
        print('  %-45s nonempty from %s | zero %s | OLS %s (%s) | 2SLS %s (%s)' % (C.lab[:45], f3(r['c_min']), f3(r['c_zero']), f3(r['c_ols']), f3(P.b),
                                                              f3(r['c_tsls']), f3(b2sls)))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    part_bnd().to_csv(_setup.os.path.join(OUTDIR, 'power_apps_bnd.csv'), index=False)
    part_breakdown().to_csv(_setup.os.path.join(OUTDIR, 'power_apps_breakdown.csv'), index=False)
