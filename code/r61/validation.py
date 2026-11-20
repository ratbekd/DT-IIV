"""Validation against known benchmarks (R61 supplement, subsection on validation).

Data from the R package causaldata (Huntington-Klein and Barrett, version 0.1.4, MIT licence):
  thornton_hiv.csv     Thornton (2008), Malawi: whether a respondent collected the HIV test result
                       (got), the randomized cash incentive (tinc, in kwacha 100s), any incentive
                       (any), distance to the randomly located test centre (distvct, km), age, village.
  nhefs_complete.csv   NHEFS, Hernan and Robins (2020): weight change 1971-82 (wt82_71), quitting
                       smoking (qsmk), covariates, cigarette price in 1982 (price82).

Specifications
  dist     got on distvct; controls age, tinc, any; clustered by village. Exogenous by design.
  tinc     got on tinc; controls age, distvct, distvct^2; clustered by village. Exogenous by design.
  tinc_any as tinc, adding the any-incentive indicator.
  qsmk     wt82_71 on qsmk; the covariates of Hernan and Robins (2020, ch. 12); price82 as the
           instrument of their ch. 16; no clusters (the jackknife uses 50 random groups).

For each: shape of the residualized regressor; OLS; 2SLS and first-stage F where there is an
instrument; admissible VT roots with the fourth-order screen (Bonferroni over roots, adjusted
critical value); CUE J test of the two-sided pair; fixed-slope sets of the one-sided pair and the
median/MAD bounded pair (adjusted critical value); the four-moment delete-group jackknife test of
the linear model (quad_feedback.py) and its set, and the feedback coefficient eta.

Usage: python validation.py [cores]       writes output/r61/validation.json
"""
import sys, os, json
import numpy as np, pandas as pd
import r61lib as L
from iiv.core import Context
from iiv.data import model_matrix
import quad_feedback as QF

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def specs():
    t = pd.read_csv(os.path.join(DATA, 'thornton_hiv.csv')); t['dist2'] = t.distvct ** 2
    h = pd.read_csv(os.path.join(DATA, 'nhefs_complete.csv'))
    for v in ('age', 'smokeintensity', 'smokeyrs', 'wt71'):
        h[v + '2'] = h[v] ** 2
    for v in ('education', 'exercise', 'active'):
        h[v + '_c'] = h[v].astype(str)
    h['qsmk_f'] = h.qsmk.astype(float)
    hw = ['sex', 'race', 'age', 'age2', 'education_c', 'smokeintensity', 'smokeintensity2', 'smokeyrs', 'smokeyrs2',
          'exercise_c', 'active_c', 'wt71', 'wt712']
    return {'dist': (t, 'got', 'distvct', ['age', 'tinc', 'any'], [], 'villnum'),
            'tinc': (t, 'got', 'tinc', ['age', 'distvct', 'dist2'], [], 'villnum'),
            'tinc_any': (t, 'got', 'tinc', ['age', 'distvct', 'dist2', 'any'], [], 'villnum'),
            'qsmk': (h, 'wt82_71', 'qsmk_f', hw, ['price82'], None)}


def one(df, y, x, w, z, cl, cores):
    d = df.dropna(subset=[y, x] + w + z + ([cl] if cl else [])).reset_index(drop=True)
    n = len(d)
    W1, _ = model_matrix(d, w)
    Z = d[z].values.astype(float) if z else np.random.default_rng(1).standard_normal((n, 1))
    C = Context(dict(y=d[y].values.astype(float), x=d[x].values.astype(float), W1=W1, Z=Z,
                     cl=None if cl is None else d[cl].astype(str).values, lab=x))
    P = C.prep(); s = P.xw
    u = P.rw                                             # OLS residual, standardized
    se_ols = float(np.sqrt(C.meat((s * u)[:, None])[0, 0] / n / np.mean(s ** 2) ** 2) * P.sr / P.sx)
    out = dict(n=n, G=int(C.G) if cl else None, values=int(d[x].nunique()), skew=float(np.mean(s ** 3)),
               kurt=float(np.mean(s ** 4)), ols=float(P.b), ols_se=se_ols)
    if z:
        b_iv, se_iv = C.tsls(P)
        xh = C.Zt @ np.linalg.lstsq(C.Zt, P.xt, rcond=None)[0]
        out.update(tsls=float(b_iv), tsls_se=float(se_iv),
                   F=float((np.sum(xh ** 2) / len(z)) / (np.sum((P.xt - xh) ** 2) / (n - W1.shape[1] - len(z)))))
    roots = [r for r in C.vt_roots(P) if abs(r) <= np.pi / 4]
    out['vt'] = []
    for r in roots:
        J, nu = C.Q_c(r, P, [1, 3, 4, 5], return_nu=True)
        out['vt'].append(dict(beta=float(C.beta_at(r, P)), screen_p=float(min(1, C.p_S(J, 3, nu) * len(roots)))))
    fr = C.fit_cue(P, [1, 2]); out['two_sided'] = dict(beta=float(fr['beta']), J_p=float(fr['J_p']))
    th = np.linspace(-np.pi / 4, np.pi / 4, 241); bg = C.beta_at(th, P)
    for fam, key in (('P', 'one_sided'), ('R', 'bounded')):
        acc = []
        for bb in bg:
            S, nu, _ = L.analytic(C, P, bb, L.FAMS[fam]); acc.append(S <= L.crit_adj(C.G, 2, nu))
        out[key] = L.comps(acc, bg)
        out[key + '_ols_in'] = bool(L.analytic(C, P, P.b, L.FAMS[fam])[0] <= L.crit_adj(C.G, 2, L.analytic(C, P, P.b, L.FAMS[fam])[1]))
    yy, xx = d[y].values.astype(float), d[x].values.astype(float)
    cj = d[cl].astype('category').cat.codes.values if cl else np.random.default_rng(20261114).permutation(np.arange(n) % 50)
    k = P.sr / P.sx
    grid = P.b + np.tan(np.linspace(-np.pi / 4, np.pi / 4, 61)) * k
    jk = QF.run(yy, xx, W1, cj, grid, cores, x)
    out['jk'] = dict(G=int(len(np.unique(cj))), J_lin=jk['J_lin'], p_lin=jk['p_lin'], beta_lin=jk['beta_lin'],
                     set_lin=jk['set_lin']['F'], eta=jk['gamma_quad'], p_eta=jk['p_quad'])
    return out


def main():
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    res = {}
    for nm, sp in specs().items():
        print('==== %s' % nm, flush=True)
        res[nm] = one(*sp, cores)
        r = res[nm]
        print('  n %d, OLS %.4f (%.4f), VT %s, two-sided J p %.3f, one-sided %s, bounded %s, jackknife J p %.4f set %s, eta %.3f'
              % (r['n'], r['ols'], r['ols_se'], r['vt'], r['two_sided']['J_p'], r['one_sided'], r['bounded'],
                 r['jk']['p_lin'], r['jk']['set_lin'], r['jk']['eta']), flush=True)
    json.dump(res, open(L.out('validation.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
