"""Full-refit jackknife sets beside the analytic sets in the seven applications.

For each sample and family (P one-sided powers, R median/MAD bounded, B fixed bounded,
L localized) the fixed-slope test is inverted on a grid of angles in |theta| <= pi/4
under five calibrations:
  an_chi   analytic corrected covariance, chi2_2 cutoff
  an_adj   analytic corrected covariance, adjusted cutoff of eq. (30)
  jk_chi   full-refit delete-cluster jackknife, chi2_2 cutoff
  jk_F     full-refit delete-cluster jackknife, Hotelling F cutoff
  wild     linearized wild calibration of the jackknife statistic (clustered samples)

Usage: python tail_apps.py dataset [ngrid]      writes output/r61/tail_apps_<dataset>.json
"""
import sys, json
import numpy as np
from scipy.stats import chi2
import r61lib as L
from iiv.data import load
from iiv.core import Context

FAMS = ['P', 'R', 'B', 'L']


def run(ds, ngrid=241, B=9999):
    C = Context(load(ds)); P = C.prep()
    th = np.linspace(-np.pi / 4, np.pi / 4, ngrid); bg = C.beta_at(th, P)
    b_iv = float(C.tsls(P)[0])
    pts = np.r_[bg, P.b, b_iv, 0.0]                      # grid, then OLS, 2SLS and zero
    m, M = L.jk_refit(C.W1, C.y, C.x, C.cl_codes, C.G, pts, FAMS)
    rng = np.random.default_rng(20261110)
    c_chi = chi2.ppf(0.95, 2); c_F = L.crit_F(C.G)
    res = dict(dataset=ds, n=int(C.n), G=int(C.G), clustered=bool(C.clustered), ols=float(P.b), tsls=b_iv,
               ratio=float(P.sr / P.sx), fams={})
    for jf, f in enumerate(FAMS):
        S = np.empty(len(pts)); nu = np.empty(len(pts)); T = np.empty(len(pts)); pw = np.full(len(pts), np.nan)
        for i, b in enumerate(pts):
            S[i], nu[i], _ = L.analytic(C, P, b, L.FAMS[f])
            T[i], V, d = L.jk_stat(m[i, jf], M[:, i, jf])
            if C.clustered:
                pw[i] = L.wild_p(T[i], V, d, B, rng)
        cadj = np.array([L.crit_adj(C.G, 2, v) for v in nu])
        acc = dict(an_chi=S <= c_chi, an_adj=S <= cadj, jk_chi=T <= c_chi, jk_F=T <= c_F)
        if C.clustered:
            acc['wild'] = pw > 0.05
        fr = {}
        for k, a in acc.items():
            fr[k] = dict(comps=L.comps(a[:ngrid], bg), ols_in=bool(a[ngrid]), tsls_in=bool(a[ngrid + 1]),
                         zero_in=bool(a[ngrid + 2]))
        fr['at'] = dict(S=S[ngrid:].tolist(), nu=nu[ngrid:].tolist(), T=T[ngrid:].tolist(), wild=pw[ngrid:].tolist())
        res['fams'][f] = fr
    res['theta_tsls'] = float(C.theta_of(b_iv, P)); res['theta_zero'] = float(C.theta_of(0.0, P))
    return res


if __name__ == '__main__':
    ds = sys.argv[1]; ngrid = int(sys.argv[2]) if len(sys.argv) > 2 else 241
    r = run(ds, ngrid)
    json.dump(r, open(L.out('tail_apps_%s.json' % ds), 'w'), indent=1)
    fmt = lambda cs: ' U '.join('[%.3f, %.3f]%s' % (a, b, '+' if e else '') for a, b, e in cs) or 'empty'
    print('== %s: n %d, G %d, OLS %.3f, 2SLS %.3f' % (ds, r['n'], r['G'], r['ols'], r['tsls']))
    for f in FAMS:
        for k, v in r['fams'][f].items():
            if k != 'at':
                print('  %s %-7s %s' % (f, k, fmt(v['comps'])))
