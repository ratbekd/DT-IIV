"""The bias floor under heteroscedasticity (Remark 5.1 of the main text, general form).

Baseline: E[u0 | s0] = 0 and E[u0^2 | s0] = sigma^2(s0) = (1 + s0^2)/2, the HET design of
Table 3. Departure: u0 = eps q(s0) + sqrt(1 - eps^2) sigma(s0) e, e ~ N(0, 1).
With mu = sigma^2 psi - (sigma^2)' and lambda = mu - s, the claim is that the worst-case
displacement of the angle per unit of eps, ||H|| / |D(h)|, is at least 1 / ||lambda||,
with equality when H is proportional to lambda.

Part A  population check by quadrature.   Part B  estimation pipeline, Gaussian s0, h = s^3.

Usage: python bias_floor_het.py [folder for bias_floor_het.tex]
"""
import sys, os, json
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import r61lib as L

_x, _w = np.polynomial.legendre.leggauss(3000)
P_GRID, W_S = (_x + 1) / 2, _w / 2
_z, _v = np.polynomial.hermite_e.hermegauss(80)
E_GRID, W_E = _z, _v / np.sqrt(2 * np.pi)
A_LOG = np.sqrt(3) / np.pi
sig2 = lambda s: (1 + s ** 2) / 2
dsig2 = lambda s: s

LAWS = {
    'gauss': dict(q=lambda p: stats.norm.ppf(p), psi=lambda s: s),
    't5': dict(q=lambda p: stats.t.ppf(p, 5) / np.sqrt(5 / 3), psi=lambda s: 6 * s / (3 + s ** 2)),
    'logistic': dict(q=lambda p: A_LOG * np.log(p / (1 - p)), psi=lambda s: np.tanh(s / (2 * A_LOG)) / A_LOG),
    'gamma6': dict(q=lambda p: (stats.gamma.ppf(p, 6) - 6) / np.sqrt(6),
                   psi=lambda s: np.sqrt(6) * (1 - 5 / (np.sqrt(6) * s + 6))),
}
FUN = {'tanh': np.tanh, 'gauss': lambda s: np.exp(-s ** 2 / 2), 's2': lambda s: s ** 2, 's3': lambda s: s ** 3}
Es = lambda v: float(np.sum(W_S * v))


def moment(phi, s0, h, eps=0.0, Hn=None):
    u0 = np.sqrt(1 - eps ** 2) * np.sqrt(sig2(s0))[:, None] * E_GRID[None, :]
    if eps:
        u0 = u0 + eps * Hn[:, None]
    s = np.cos(phi) * s0[:, None] + np.sin(phi) * u0
    u = np.cos(phi) * u0 - np.sin(phi) * s0[:, None]
    return float(W_S @ (u * h(s)) @ W_E)


def analyse(s0, h, eps_list=(0.01, 0.05)):
    hv = h(s0); H = hv - Es(hv) - Es(s0 * hv) * s0; nH = np.sqrt(Es(H ** 2))
    d = 1e-4; D = (moment(d, s0, h) - moment(-d, s0, h)) / (2 * d)
    ex = {}
    grid = np.linspace(-0.7, 0.7, 141)
    for eps in eps_list:
        f = lambda ph: moment(ph, s0, h, eps, H / nH)
        v = np.array([f(g) for g in grid]); idx = np.where(np.diff(np.sign(v)) != 0)[0]
        ex['%g' % eps] = abs(min((brentq(f, grid[i], grid[i + 1], xtol=1e-10) for i in idx), key=abs)) / eps if len(idx) else None
    return dict(normH=nH, D=D, predicted=nH / abs(D), exact=ex), H, D


def best_combo(s0, names):
    Hs, Ds = zip(*[analyse(s0, FUN[n], ())[1:] for n in names])
    Hm = np.column_stack(Hs); Dv = np.array(Ds)
    c = np.linalg.solve((Hm * W_S[:, None]).T @ Hm, Dv)
    return lambda s: sum(cj * FUN[n](s) for cj, n in zip(c, names))


res = dict(A=[], B=[])
print('PART A. sigma^2(s) = (1 + s^2)/2. Worst-case displacement of the angle per unit of eps')
print('%-9s %-22s %8s %8s | %9s %9s %9s | %7s' % ('law', 'tendency h', '||H||', 'D(h)', 'predicted', 'eps=0.01', 'eps=0.05', 'floor'))
for lw, spec in LAWS.items():
    s0 = spec['q'](P_GRID)
    lamf = lambda s, psi=spec['psi']: sig2(s) * psi(s) - dsig2(s) - s
    lam = lamf(s0)
    nl = np.sqrt(Es(lam ** 2)); floor = 1 / nl
    print('%s: ||lambda||^2 = %.4f, E[lambda] = %.4f, E[lambda s0] = %.4f, floor = %.4f' % (lw, nl ** 2, Es(lam), Es(lam * s0), floor))
    todo = []
    if lw != 'gamma6':                     # the score of the gamma law is not defined off its support
        todo.append(('lambda', lamf))
    todo += [('tanh s', FUN['tanh']), ('exp(-s^2/2)', FUN['gauss']), ('bounded pair, best', best_combo(s0, ['tanh', 'gauss']))]
    if lw != 't5':                         # E s0^6 is infinite for t5
        todo += [('s^2', FUN['s2']), ('s^3', FUN['s3']), ('powers, best', best_combo(s0, ['s2', 's3']))]
    for nm, h in todo:
        o, H, D = analyse(s0, h)
        e1, e5 = o['exact']['0.01'], o['exact']['0.05']
        print('%-9s %-22s %8.4f %8.4f | %9.4f %9s %9s | %7.4f  E[lambda H] = %.4f' % (
            '', nm, o['normH'], D, o['predicted'], '%.4f' % e1 if e1 else 'none', '%.4f' % e5 if e5 else 'none', floor, Es(lam * H)))
        res['A'].append(dict(law=lw, h=nm, normH=o['normH'], D=D, predicted=o['predicted'], exact_001=e1, exact_005=e5,
                             floor=floor, lam2=nl ** 2, ElamH=Es(lam * H)))

print('\nPART B. Estimation pipeline: Gaussian s0, sigma^2 = (1 + s^2)/2, tendency s^3, worst-case departure')
n, R = 5000, 400
rng = np.random.default_rng(20261114)
FL = 1 / np.sqrt(1.5)
print('n = %d, %d replications; predicted slope bias = -%.4f eps' % (n, R, FL))
for eps in (0.0, 0.02, 0.05):
    est = []
    for r in range(R):
        s0 = rng.standard_normal(n)
        q = (s0 ** 3 - 3 * s0) / np.sqrt(6)
        u0 = eps * q + np.sqrt(1 - eps ** 2) * np.sqrt(sig2(s0)) * rng.standard_normal(n)
        w = rng.standard_normal(n)
        x = 2 + 0.5 * w + s0 + 0.5 * u0; y = 1 + x + 0.3 * w + u0
        C = L.make_context(y, x, np.column_stack([np.ones(n), w])); P = C.prep()

        def g(th):
            s = np.cos(th) * P.xw + np.sin(th) * P.rw; u = np.cos(th) * P.rw - np.sin(th) * P.xw
            return np.mean(u * s ** 3)
        grid = np.linspace(-np.pi / 4, np.pi / 4, 161)
        v = np.array([g(t) for t in grid]); idx = np.where(np.diff(np.sign(v)) != 0)[0]
        if len(idx):
            t0 = C.theta_of(1.0, P)
            est.append(C.beta_at(min((brentq(g, grid[i], grid[i + 1]) for i in idx), key=lambda t: abs(t - t0)), P) - 1.0)
    est = np.array(est)
    print('%6.2f | mean bias %8.4f  s.e. %.4f  sd %.4f | predicted %8.4f  (%d roots)' % (
        eps, est.mean(), est.std(ddof=1) / np.sqrt(len(est)), est.std(ddof=1), -FL * eps, len(est)))
    res['B'].append(dict(eps=eps, mean_bias=float(est.mean()), se=float(est.std(ddof=1) / np.sqrt(len(est))), predicted=-FL * eps))
json.dump(res, open(L.out('bias_floor_het.json'), 'w'), indent=1)

if len(sys.argv) > 1:
    LAW = {'gauss': 'Gaussian', 't5': r'\(t_5\)', 'logistic': 'Logistic', 'gamma6': 'Gamma, shape 6'}
    HN = {'lambda': r'\(\lambda\)', 'tanh s': r'\(\tanh s\)', 'exp(-s^2/2)': r'\(\exp(-s^2/2)\)', 's^2': r'\(s^2\)',
          's^3': r'\(s^3\)', 'bounded pair, best': 'bounded pair, best', 'powers, best': 'powers, best'}
    rows = []; last = None
    for a in res['A']:
        if abs(a['D']) < 1e-3:             # zero slope: the tendency does not identify the angle
            continue
        if last is not None and a['law'] != last:
            rows.append(r'\addlinespace')
        lead = '%s (%.2f)' % (LAW[a['law']], a['lam2']) if a['law'] != last else ''
        last = a['law']
        rows.append('%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (
            lead, HN[a['h']], a['normH'], abs(a['D']), a['predicted'], a['exact_001'], a['exact_005'], a['floor']))
    tex = r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption{The bias floor with a heteroscedastic error}
\label{tab:bias_floor_het}
\begin{threeparttable}
\begin{tabular}{llcccccc}
\toprule
 & & & & First-order & \multicolumn{2}{c}{Exact root} & \\
\cmidrule(lr){6-7}
Law of \(s_0\) (\(\|\lambda\|_2^2\)) & Tendency \(h\) & \(\|H\|_2\) & \(|D(h)|\) & prediction & \(\epsilon=0.01\) & \(\epsilon=0.05\) & Floor \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} As Panel~A of Table~\ref{tab:bias_floor}, with
\(E[u_0^2\mid s_0]=\sigma^2(s_0)=(1+s_0^2)/2\) at \(\epsilon=0\), the heteroscedastic
design of Table~\ref{M-tab:mc_optimal}. Here \(\lambda=\sigma^2\psi-(\sigma^2)'-s\), and the
floor is \(1/\|\lambda\|_2\). The row \(\lambda\) uses \(\lambda\) itself as the tendency.
It is omitted for the gamma law, whose score is not defined below the support, and the
powers are omitted for \(t_5\). Tendencies with a zero slope in this design are omitted:
\(s^2\) for every law, and \(\exp(-s^2/2)\) for the symmetric laws. The quadrature is
accurate to about \(0.001\), which accounts for predictions that fall below the floor
in the third decimal.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    open(os.path.join(sys.argv[1], 'bias_floor_het.tex'), 'w', newline='\n').write(tex)
    print('wrote bias_floor_het.tex')
