"""Group-covariance (heteroscedasticity) moments on STAR (port of sim_star_het.R).
h(W): pupil, teacher and school-type indicators, centred and scaled, and a
variance index: the out-of-fold fit of rw^2 - xw^2 on W with five folds drawn
by class. Part 1: the real sample (2SLS benchmark). Part 2: designs D0, D1, D1w, D2.
Usage: python sim_star_het.py [replications]   (writes sim_star_het.csv)"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, optimize, f3, Prep
from iiv.sim import StarDesign, replicate, safe, mad, nanmedian, nanmean, ALPHA

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D = load('star'); C = Context(D); S = StarDesign(C); beta0 = S.beta0; dd = D['dd']; E = 'hetx'
Hn = ['girl', 'white', 'freelunch', 'tblack', 'exp', 'masters', 'suburb', 'rural', 'urban']
Hf = np.column_stack([dd[v].values.astype(float) for v in Hn[:6]] + [(dd.schtype == t).values.astype(float) for t in Hn[6:]])
H_fix = (Hf - Hf.mean(0)) / Hf.std(0, ddof=1); q = H_fix.shape[1]
h_cols = list(range(7, 7 + q)); idx_cols = [7 + q]
rng0 = np.random.default_rng(20261009)
uc, cl_i = np.unique(C.cl, return_inverse=True)
fold = rng0.permutation(np.resize(np.arange(1, 6), len(uc)))[cl_i]


def h_index(xw, rw):
    tg = rw ** 2 - xw ** 2; out_ = tg.copy()
    for k in range(1, 6):
        tr = fold != k; co = np.linalg.lstsq(C.W1[tr], tg[tr], rcond=None)[0]; out_[~tr] = C.W1[~tr] @ co
    return (out_ - out_.mean()) / out_.std(ddof=1)


def prep(y=None, x=None):
    P = C.prep(y, x); P['H'] = np.column_stack([H_fix, h_index(P.xw, P.rw)]); return P


def fit_any(P, cols, b_true):
    m = len(cols); th = np.linspace(-np.pi / 4, np.pi / 4, 91)
    Qv = [C.Q_u(a, P, cols, E) for a in th]; i = int(np.argmin(Qv))
    t, _ = optimize(lambda a: C.Q_u(a, P, cols, E), th[max(0, i - 1)], th[min(90, i + 1)])
    J = optimize(lambda a: C.Q_c(a, P, cols, ext=E), t - 0.05, t + 0.05)[1] if m > 1 else np.nan
    crit = chi2.ppf(1 - ALPHA, m)
    return dict(est=C.beta_at(t, P), cover=float(C.Q_c(None, P, cols, beta=b_true, ext=E) <= crit),
                excl_ols=float(C.Q_c(None, P, cols, beta=P.b, ext=E) > crit),
                J_rej=np.nan if np.isnan(J) else float(J > chi2.ppf(1 - ALPHA, m - 1)))


sets_h = {'one_sided': [2, 4], 'het_controls': h_cols, 'het_index': idx_cols, 'het_index_one': [2, 4] + idx_cols}
P0 = prep(); b_iv = C.tsls(P0)[0]; th_iv = C.theta_of(b_iv, P0)
print('\n==== STAR, real data: n =', C.n, '; clusters =', C.G, '====')
print('OLS', f3(P0.b), '; 2SLS (random assignment)', f3(b_iv), '; implied rho_xu at 2SLS', f3(-np.sin(th_iv)), '\n')
s_b = np.cos(th_iv) * P0.xw + np.sin(th_iv) * P0.rw; u_b = np.cos(th_iv) * P0.rw - np.sin(th_iv) * P0.xw
tcl = lambda v: np.sqrt(np.sum(C.rowsum(v - v.mean()) ** 2)) / C.n
print("Per control h (closed form). 'slope' = cov(h,u^2) - cov(h,s^2) and 'validity' = cov(h, s u),\n "
      "both evaluated at the 2SLS benchmark, with cluster-robust t-statistics.")
print('%-10s %8s %8s %8s %8s %9s %8s' % ('h', 'beta', 'rho_xu', 'slope', 't', 'validity', 't'))
for j, nm in enumerate(Hn + ['idx_var']):
    h = P0.H[:, j]; a_ = np.mean(h * P0.xw * P0.rw); d_ = np.mean(h * (P0.xw ** 2 - P0.rw ** 2)); th = 0.5 * np.arctan(2 * a_ / d_)
    sl = h * (u_b ** 2 - s_b ** 2); va = h * s_b * u_b
    print('%-10s %8s %8s %8s %8s %9s %8s' % (nm, f3(C.beta_at(th, P0)), f3(-np.sin(th)), f3(sl.mean()), f3(sl.mean() / tcl(sl)),
                                            f3(va.mean()), f3(va.mean() / tcl(va))))
print('\nMoment sets (CUE; corrected, clustered S-set and J-test):')
print('%-13s %3s %8s %19s %7s %8s %9s %9s' % ('set', 'm', 'beta', '95% S-set', 'pieces', 'J p', 'OLS in', '2SLS in'))
thg = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 721); bg = C.beta_at(thg, P0)
for nm, cols in sets_h.items():
    m = len(cols); th = np.linspace(-np.pi / 4, np.pi / 4, 181)
    Qv = [C.Q_u(a, P0, cols, E) for a in th]; i = int(np.argmin(Qv))
    t, _ = optimize(lambda a: C.Q_u(a, P0, cols, E), th[max(0, i - 1)], th[min(180, i + 1)])
    J = optimize(lambda a: C.Q_c(a, P0, cols, ext=E), t - 0.05, t + 0.05)[1] if m > 1 else np.nan
    crit = chi2.ppf(0.95, m)
    acc = np.array([C.Q_c(None, P0, cols, beta=b, ext=E) <= crit for b in bg]); inn = acc & (np.abs(thg) <= np.pi / 4)
    print('%-13s %3d %8s %19s %7d %8s %9s %9s' % (nm, m, f3(C.beta_at(t, P0)),
          '[%s, %s]' % (f3(bg[inn].min()), f3(bg[inn].max())) if inn.any() else 'empty',
          int(np.sum(np.diff(np.r_[0, inn.astype(int), 0]) == 1)), f3(1 - chi2.cdf(J, m - 1)) if m > 1 else '--',
          str(C.Q_c(None, P0, cols, beta=P0.b, ext=E) <= crit).upper(), str(C.Q_c(None, P0, cols, beta=b_iv, ext=E) <= crit).upper()))

KEYS = ['est', 'cover', 'excl_ols', 'J_rej']


def one_rep(design):
    def f(rng):
        y, x = S.data(design, rng); P = prep(y, x); r = dict(ols=P.b, tsls=C.tsls(P)[0])
        for nm, cols in sets_h.items():
            r.update({nm + '.' + k: v for k, v in safe(lambda: fit_any(P, cols, beta0), KEYS).items()})
        return r
    return f


print('\n==== Semi-synthetic STAR: beta0 =', beta0, '; replications =', R, '====')
rows = []
for i, d in enumerate(('D0', 'D1', 'D1w', 'D2')):
    M = pd.DataFrame(replicate(one_rep(d), R, 20261009 + i))
    print('\n----', d, ': median OLS', f3(np.median(M.ols)), '; median 2SLS', f3(np.median(M.tsls)), '----')
    print('%-13s %8s %8s %8s %9s %9s' % ('set', 'median', 'MAD', 'cover', 'excl OLS', 'J reject'))
    for s in sets_h:
        v = dict(design=d, set=s, median=nanmedian(M[s + '.est']), mad=mad(M[s + '.est']), cover=nanmean(M[s + '.cover']),
                 excl_ols=nanmean(M[s + '.excl_ols']), J_reject=nanmean(M[s + '.J_rej']))
        print('%-13s %8s %8s %8s %9s %9s' % (s, f3(v['median']), f3(v['mad']), f3(v['cover']), f3(v['excl_ols']),
                                            '--' if np.isnan(v['J_reject']) else f3(v['J_reject'])))
        rows.append(v)
pd.DataFrame(rows).to_csv(out('sim_star_het.csv'), index=False)
