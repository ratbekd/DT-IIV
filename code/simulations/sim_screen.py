"""Size and power of the kurtosis screen for VT roots in the semi-synthetic STAR
designs (port of sim_screen.R). Designs D0, D1, D1t, D2, D3.
Usage: python sim_screen.py [replications]   (writes sim_screen.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, f3
from iiv.sim import StarDesign, replicate, vt_roots, screen_J, cue_J, cdist, nanmean, nanmedian, ALPHA

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
C = Context(load('star')); S = StarDesign(C); beta0 = S.beta0; tol = 0.12
aug = {'two_sided': [1, 2], 'kurt': [3, 4, 5], 'five': [1, 2, 3, 4, 5], 'var123': [1, 5, 6], 'two_sided_var': [1, 2, 5, 6]}


def one_rep(design):
    def f(rng):
        y, x = S.data(design, rng); P = C.prep(y, x); th0 = C.theta_of(beta0, P)
        rts = vt_roots(P); nr = len(rts); Jr = []
        for r in rts:
            try:
                Jr.append(screen_J(C, P, r))
            except (np.linalg.LinAlgError, ValueError):
                Jr.append(np.nan)
        pv = 1 - chi2.cdf(np.array(Jr), 3) if nr else np.array([])
        dist = cdist(rts, th0) if nr else np.array([])
        is_true = (dist < tol) & (dist == dist.min()) if nr else np.zeros(0, bool)
        rej = pv < ALPHA; rejB = pv < ALPHA / max(nr, 1)

        def pick(keep):
            c = np.where(keep & (np.abs(rts) <= np.pi / 4))[0]
            return c[np.argmin(np.abs(rts[c]))] if len(c) else None
        out_ = dict(n_roots=nr, true_found=float(is_true.any()),
                    rej_true=float(rej[is_true][0]) if is_true.any() else np.nan,
                    rejB_true=float(rejB[is_true][0]) if is_true.any() else np.nan,
                    n_other=int(np.sum(~is_true)), other_rej=int(np.sum(rej[~is_true])), other_rejB=int(np.sum(rejB[~is_true])),
                    all_rej=float(nr > 0 and bool(np.all(rej[~np.isnan(pv)]))))
        for nm, keep in (('label', np.ones(nr, bool)), ('screen', ~rej), ('screenB', ~rejB)):
            i = pick(keep)
            out_[nm + '.hit'] = 0.0 if i is None else float(is_true[i]); out_[nm + '.none'] = float(i is None)
            out_[nm + '.est'] = np.nan if i is None else C.beta_at(rts[i], P)
        for nm, cols in aug.items():
            try:
                _, J, _ = cue_J(C, P, cols); out_['J.' + nm] = float(J > chi2.ppf(1 - ALPHA, len(cols) - 1))
            except (np.linalg.LinAlgError, ValueError):
                out_['J.' + nm] = np.nan
        return out_
    return f


res = {}
for i, d in enumerate(('D0', 'D1', 'D1t', 'D2', 'D3')):
    res[d] = pd.DataFrame(replicate(one_rep(d), R, 20261005 + i)); print(d, 'done')
print('\nbeta0 =', beta0, '; replications =', R, '; n =', C.n)
rows = []
for d, M in res.items():
    no = M.n_other.sum()
    v = dict(design=d, mean_roots=M.n_roots.mean(), true_found=M.true_found.mean(), rej_true=nanmean(M.rej_true),
             rejB_true=nanmean(M.rejB_true), other_rej=M.other_rej.sum() / no if no else np.nan,
             other_rejB=M.other_rejB.sum() / no if no else np.nan, all_rej=M.all_rej.mean())
    print('\n====', d, '====')
    print('mean number of DT roots', f3(v['mean_roots']), '; root near the truth in', f3(v['true_found']), 'of replications')
    print('screen rejects the true root:', f3(v['rej_true']), ' (Bonferroni', f3(v['rejB_true']), ')')
    print('screen rejects other roots:  ', f3(v['other_rej']), ' (Bonferroni', f3(v['other_rejB']), ')')
    print('all roots rejected:', f3(v['all_rej']))
    for r in ('label', 'screen', 'screenB'):
        v[r + '_hit'] = M[r + '.hit'].mean(); v[r + '_none'] = M[r + '.none'].mean(); v[r + '_median'] = nanmedian(M[r + '.est'])
        print('rule %-8s: true root selected %s, nothing selected %s, median beta %s' % (r, f3(v[r + '_hit']), f3(v[r + '_none']),
                                                                                      f3(v[r + '_median'])))
    for nm in aug:
        v['J_' + nm] = nanmean(M['J.' + nm])
    print('J-test rejection:', '; '.join('%s %s' % (nm, f3(v['J_' + nm])) for nm in aug))
    rows.append(v)
pd.DataFrame(rows).to_csv(out('sim_screen.csv'), index=False)
