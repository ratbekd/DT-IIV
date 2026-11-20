"""Approximate mean independence in the seven applications (main text, Subsection 5.3).

Fixed basis h = (tanh s, exp(-s^2/2)), K = 2, d = 5. For each slope on a grid of
angles in |theta| <= pi/4 the script computes t_b = (mu_b, vech Q_b), its corrected
(cluster) covariance, and the projection distance D_n(b; eps). It reports
  the joint-region set C_n(eps) on a tolerance grid (chi2_5 cutoff),
  the simpler box set (chi2_2 cutoff),
  the exact-MI fixed-slope set of the same two moments (chi2_2 cutoff), and
  the entry tolerances of zero, OLS and the external-instrument estimate.

Usage: python ami_apps.py dataset [ngrid]      writes output/r61/ami_apps_<dataset>.json
"""
import sys, json
import numpy as np
from scipy.stats import chi2
import r61lib as L
from iiv.data import load
from iiv.core import Context

EPS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30]


def run(ds, ngrid=241):
    C = Context(load(ds)); P = C.prep()
    th = np.linspace(-np.pi / 4, np.pi / 4, ngrid); bg = C.beta_at(th, P)
    b_iv = float(C.tsls(P)[0])
    c5 = chi2.ppf(0.95, 5); c2 = chi2.ppf(0.95, 2)
    tv = [L.ami_tV(C, P, b) for b in bg]
    S_exact = np.array([L.analytic(C, P, b, L.fam_B)[0] for b in bg])
    r2 = np.array([float(t[:2] @ np.linalg.solve(np.array([[t[2], t[3]], [t[3], t[4]]]), t[:2])) for t, _ in tv])
    res = dict(dataset=ds, n=int(C.n), G=int(C.G), ols=float(P.b), tsls=b_iv, ratio=float(P.sr / P.sx),
               eps=EPS, joint={}, box={}, exact=L.comps(S_exact <= c2, bg),
               theta_tsls=float(C.theta_of(b_iv, P)), theta_zero=float(C.theta_of(0.0, P)),
               r_hat_min=float(np.sqrt(r2.min())), r_hat_ols=float(np.sqrt(r2[np.argmin(np.abs(bg - P.b))])))
    for e in EPS:
        D = np.array([L.ami_D(t, V, C.n, e, c5) for t, V in tv])
        Bx = np.array([L.ami_box(t, V, C.n, e) for t, V in tv])
        res['joint']['%g' % e] = L.comps(D <= c5, bg)
        res['box']['%g' % e] = L.comps(Bx <= c2, bg)
    ent = {}
    for nm, b in (('zero', 0.0), ('ols', float(P.b)), ('tsls', b_iv)):
        inside = abs(C.theta_of(b, P)) <= np.pi / 4
        if not inside:
            ent[nm] = dict(in_domain=False, eps_enter=None, r_hat=None)
            continue
        t, V = L.ami_tV(C, P, b)
        Q = np.array([[t[2], t[3]], [t[3], t[4]]])
        e = L.eps_enter(t, V, C.n, c5)
        ent[nm] = dict(in_domain=True, eps_enter=None if np.isnan(e) else float(e),
                       r_hat=float(np.sqrt(t[:2] @ np.linalg.solve(Q, t[:2]))))
    res['entry'] = ent
    return res


if __name__ == '__main__':
    ds = sys.argv[1]; ngrid = int(sys.argv[2]) if len(sys.argv) > 2 else 241
    r = run(ds, ngrid)
    json.dump(r, open(L.out('ami_apps_%s.json' % ds), 'w'), indent=1)
    fmt = lambda cs: ' U '.join('[%.3f, %.3f]%s' % (a, b, '+' if e else '') for a, b, e in cs) or 'empty'
    print('== %s: n %d, G %d, OLS %.3f, 2SLS %.3f' % (ds, r['n'], r['G'], r['ols'], r['tsls']))
    print('  exact MI (chi2_2):', fmt(r['exact']))
    for e in r['eps']:
        print('  eps %-5g joint: %-40s box: %s' % (e, fmt(r['joint']['%g' % e]), fmt(r['box']['%g' % e])))
    print('  entry:', r['entry'])
