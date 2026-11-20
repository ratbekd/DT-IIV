"""DT-IIV on the 401(k) data (port of dtiiv_k401.R). pira on p401k; controls inc,
incsq, age, agesq, marr, fsize; comparison IV e401k. Writes the \\AppPension...
macros to pension_numbers.tex (in tables/ under the output folder).
Usage: python dtiiv_k401.py"""
import os, numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import OUT
from iiv.data import _p

from iiv.dtk import M_n, moments_raw, root_table, select_root, D_SD as D_sd, REL_FLOOR as rel_floor
n_grid = 300; alpha = 0.05
w_vars = ['inc', 'incsq', 'age', 'agesq', 'marr', 'fsize']
dd = pd.read_csv(_p('k401.csv')).dropna(subset=['pira', 'p401k', 'e401k'] + w_vars).reset_index(drop=True)
Xm = dd.p401k.values.astype(float); Wm = dd[w_vars].values.astype(float); ym = dd.pira.values.astype(float)
Zm = dd[['e401k']].values.astype(float); n = len(dd)
W1 = np.column_stack([np.ones(n), Wm])
resid_on = lambda v: v - W1 @ np.linalg.lstsq(W1, v, rcond=None)[0]


x = resid_on(Xm); yt = resid_on(ym); bols = np.sum(x * yt) / np.sum(x * x); r = yt - x * bols
rt = root_table(x, r, yt); sel = select_root(rt)
zx = x / np.sqrt(np.mean(x ** 2)); zr = r / np.sqrt(np.mean(r ** 2))
k_ctrl = Wm.shape[1] + 1
se_ols = np.sqrt(np.sum(x ** 2 * (yt - x * bols) ** 2)) / np.sum(x ** 2) * np.sqrt(n / (n - k_ctrl - 1))
Zt = resid_on(Zm); pi_h = np.linalg.lstsq(Zt, x, rcond=None)[0]; xh = Zt @ pi_h
b_iv = np.sum(xh * yt) / np.sum(xh * x); u_iv = yt - x * b_iv
se_iv = np.sqrt(np.sum(xh ** 2 * u_iv ** 2)) / abs(np.sum(xh * x)) * np.sqrt(n / (n - k_ctrl - 1))
v1 = x - xh; q = Zm.shape[1]; df1 = n - k_ctrl - q
F_cl = (np.sum(xh ** 2) / q) / (np.sum(v1 ** 2) / df1)
mo = moments_raw(x, r); sdx, sdr = np.sqrt(mo['A']), np.sqrt(mo['B'])

dgrid = np.exp(np.linspace(np.log(D_sd[0]), np.log(D_sd[1]), n_grid))
n_cross = {k: int(np.sum(np.diff(np.sign([M_n(k * d * sdx / sdr, x, r) for d in dgrid])) != 0)) for k in (-1, 1)}
k_hat = int(sel.k); tau_h = sel.tau
s_hat = x + tau_h * r
b_iiv_id = bols + tau_h * mo['B'] / mo['A']
tau_til = -mo['A'] ** 2 * mo['m03'] / (mo['B'] ** 2 * mo['m30'])
beta_til = bols + tau_til * mo['B'] / mo['A']; dsd_til = abs(tau_til) * sdr / sdx


def corrected_scores(tau, beta_iiv):
    A, B = mo['A'], mo['B']; s = x + tau * r; D = np.mean(s ** 2)
    g = np.sum(x * s) / np.sum(s * s); e = x - g * s; m = (e ** 2 - np.mean(e ** 2)) * s; h = 1e-6
    J_tau = (M_n(tau + h, x, r) - M_n(tau - h, x, r)) / (2 * h)
    J_b = (M_n(tau, x, r - x * h) - M_n(tau, x, r + x * h)) / (2 * h)
    J_G = -2 * np.mean(e * s ** 2)
    mt = m + J_b * (x * r / A) + J_G * (s * e / D)
    gt = s * (yt - x * beta_iiv) - tau * (x * (bols - beta_iiv)) * r
    return dict(mt=mt, psi=gt / A - B / (A * J_tau) * mt, J_tau=J_tau, m=m)


cs = corrected_scores(tau_h, b_iiv_id); se_corr = np.sqrt(np.mean(cs['psi'] ** 2) / n)
S_stat = []
for d in dgrid:
    tau = k_hat * d * sdx / sdr; cc = corrected_scores(tau, bols + tau * mo['B'] / mo['A'])
    S_stat.append(n * np.mean(cc['m']) ** 2 / np.var(cc['mt'], ddof=1))
S_stat = np.array(S_stat); keep = S_stat <= chi2.ppf(1 - alpha, 1)
beta_grid = bols + k_hat * dgrid * sdx / sdr * mo['B'] / mo['A']
S_share = keep.mean(); S_beta = (beta_grid[keep].min(), beta_grid[keep].max())
S_runs = int(np.sum(np.diff(np.r_[0, keep.astype(int), 0]) == 1))
raw_t = lambda v: (np.mean(v), np.mean(v) / (np.std(v, ddof=1) / np.sqrt(n)))
tm = {'mu30': raw_t(x ** 3), 'mu21': raw_t(x ** 2 * r), 'mu12': raw_t(x * r ** 2), 'mu03': raw_t(r ** 3)}

f = lambda v, d=3: ('%.' + str(d) + 'f') % v
print('\n==== DT-IIV, 401(k): pira on p401k, n =', n, '====')
show = rt.copy(); show['tau'] = show.tau.round(4)
for c in ('delta_sd', 'cor_sx', 'beta'):
    show[c] = show[c].round(3)
print(show.to_string(index=False))
print('grid sign changes, k = -1:', n_cross[-1], '; k = +1:', n_cross[1])
print('OLS', f(bols), '(HC1', f(se_ols), '); 2SLS', f(b_iv), '(HC1', f(se_iv), '), first-stage F', f(F_cl, 0))
print('DT-IIV', f(b_iiv_id), '; k =', k_hat, '; delta raw', f(abs(tau_h)), '; delta SD', f(sel.delta_sd), '; cor(s,x)',
      f(sel.cor_sx), '; corrected SE', f(se_corr))
for kname, (e, t) in tm.items():
    print('%-5s est %9.4f  t %9.4f' % (kname, e, t))
print('b-hat = mu03/mu30 =', f(tm['mu03'][0] / tm['mu30'][0]))
print('S-set: share of grid', f(S_share), '; pieces', S_runs, '; beta in [', f(S_beta[0]), ',', f(S_beta[1]), ']')
print('mean pira', f(dd.pira.mean()), '; mean p401k', f(dd.p401k.mean()))

oth = rt[rt.tau != tau_h].sort_values('cor_sx', ascending=False).reset_index(drop=True)
pieces = []; i = 0
while i < len(keep):
    if keep[i]:
        j = i
        while j + 1 < len(keep) and keep[j + 1]:
            j += 1
        seg = beta_grid[i:j + 1]; pieces.append((seg.min(), seg.max())); i = j + 1
    else:
        i += 1
fq = lambda v, d=3: '\\textbf{??}' if not np.isfinite(v) else f(v, d)
f_root = lambda j: '\\textbf{??}' if j >= len(oth) else fq(oth.beta[j], 1 if abs(oth.beta[j]) >= 10 else 3)
f_piece = lambda j, e: '\\textbf{??}' if j >= len(pieces) else fq(pieces[j][e])
defs = dict(N='{:,}'.format(n).replace(',', '{,}'), OLS=f(bols), IV=f(b_iv), IIV=f(b_iiv_id), Delta=f(abs(tau_h)),
            MuThreeZero=f(tm['mu30'][0], 4), MuTwoOne=f(tm['mu21'][0], 4), MuTwoOneT=f(tm['mu21'][1], 2),
            MuOneTwo=f(tm['mu12'][0], 4), MuOneTwoT=f(tm['mu12'][1], 2), MuZeroThree=f(tm['mu03'][0], 4),
            SSet=f(100 * S_share, 0), IdLo=f(S_beta[0]), IdHi=f(S_beta[1]), IIVSE=fq(se_corr),
            BRatio=fq(tm['mu03'][0] / tm['mu30'][0], 2), RootTwo=f_root(0), RootThree=f_root(1),
            PieceOneLo=f_piece(0, 0), PieceOneHi=f_piece(0, 1), PieceTwoLo=f_piece(1, 0), PieceTwoHi=f_piece(1, 1),
            ShareIRA=fq(100 * dd.pira.mean(), 1))
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
path = os.path.join(OUT, 'tables', 'pension_numbers.tex')
with open(path, 'w') as fh:
    fh.write('% generated by dtiiv_k401.py\n% 401(k) numbers quoted in the running text. Do not edit by hand.\n')
    fh.writelines('\\newcommand{\\AppPension%s}{%s}\n' % kv for kv in defs.items())
print('  -> %s (%d macros)' % (path, len(defs)))
if len(oth) != 2 or len(pieces) != 2:
    print('NOTE: found', len(oth) + 1, 'DT roots and', len(pieces), 'S-set pieces; the text assumes 3 and 2, so check that subsection.')
print('alignment-restriction root (mixed moments set to zero): tau', f(tau_til), '; delta SD', f(dsd_til), '; beta', f(beta_til))
for p in pieces:
    print('S-set piece: beta in [', f(p[0]), ',', f(p[1]), ']')
print('S statistic at the 2SLS value:', f(S_stat[np.argmin(np.abs(beta_grid - b_iv))]), '; at OLS (delta -> 0):',
      f(S_stat[0]), '; critical value', f(chi2.ppf(0.95, 1)))
