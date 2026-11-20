"""The fixed-slope tests of the optimal-tendency simulation at matched size.

Reads the statistics saved by main_paper/01_mc_optimal.py (one file per design:
replications x slopes x [S, K, J for the powers, the sieve and the bounded basis]) and
reports, for the S test with the powers and the KJ tests with the three bases,
  nominal size and power (rates at the chi-squared critical values), and
  matched-size power: the cutoff of each test is the 95th percentile of its statistic
  at the true slope in one half of the replications, and rejection is evaluated in the
  other half; each half supplies the cutoff for the other, and the two rates are
  averaged.
The KJ test rejects when min(p_K / alpha_K, p_J / alpha_J) < 1, with alpha_K = 0.04 and
alpha_J = 0.01. Its matched-size version replaces the threshold 1 by the 5th percentile
of that ratio at the true slope, which keeps the split of the size between K and J.

Usage: python mc_optimal_matched.py [folder for mc_optimal.tex]
       writes output/r61/mc_optimal_matched.json and, with a folder, the LaTeX table.
"""
import os, sys, json
import numpy as np
from scipy.stats import chi2
import r61lib as L

SRC = os.path.join(L.HERE, '..', '..', 'output', 'main_paper')
DELTAS = [-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
BASES = [('powers', 2), ('sieve', 5), ('bounded', 4)]
DESIGNS = [(k, e) for k in ['chi2', 't5', 'mix', 'gauss'] for e in ['CH', 'HET']]
AK, AJ = 0.04, 0.01
I0, IP, IM = DELTAS.index(0.0), DELTAS.index(0.1), DELTAS.index(-0.1)
TESTS = ['S powers', 'KJ powers', 'KJ sieve', 'KJ bounded']
SNAME = {'chi2': r'\(\chi^2_4\)', 't5': r'\(t_5\)', 'mix': 'bimodal', 'gauss': 'Gaussian'}


def scores(res, di, bi, m):
    """Scores for which a larger value is more evidence against the null: S, and the
    KJ score -min(p_K/aK, p_J/aJ), whose nominal cutoff is -1."""
    S, K, J = res[:, di, 3 * bi], res[:, di, 3 * bi + 1], res[:, di, 3 * bi + 2]
    kj = -np.minimum(chi2.sf(K, 1) / AK, chi2.sf(np.maximum(J, 0), m - 1) / AJ)
    return S, kj


def matched(sc0, sca):
    """Two-fold matched-size rejection: (held-out size, power)."""
    R = len(sc0); h = R // 2
    folds = [(slice(0, h), slice(h, R)), (slice(h, R), slice(0, h))]
    size = np.mean([np.mean(sc0[ev] > np.quantile(sc0[tr], 0.95)) for tr, ev in folds])
    power = np.mean([np.mean(sca[ev] > np.quantile(sc0[tr], 0.95)) for tr, ev in folds])
    return float(size), float(power)


out = dict(designs=[], tests=TESTS)
print('%5s %-6s %-4s | %s' % ('n', 's0', 'err', ' | '.join('%-26s' % t for t in TESTS)))
print('%18s | %s' % ('', ' | '.join('size  power  m-size m-pow ' for _ in TESTS)))
for n in (500, 1000):
    for kind, err in DESIGNS:
        res = np.load(os.path.join(SRC, 'mc_optimal_%s_%s_%d.npy' % (kind, err, n)))
        row = dict(n=n, kind=kind, err=err, R=int(res.shape[0]), tests={})
        for bi, (nm, m) in enumerate(BASES):
            S0, kj0 = scores(res, I0, bi, m)
            Sp, kjp = scores(res, IP, bi, m); Sm, kjm = scores(res, IM, bi, m)
            todo = [('KJ ' + nm, kj0, kjp, kjm, -1.0)]
            if nm == 'powers':
                todo.insert(0, ('S powers', S0, Sp, Sm, chi2.ppf(0.95, m)))
            for name, s0, sp, sm, cut in todo:
                msz_p, mp = matched(s0, sp); msz_m, mm = matched(s0, sm)
                row['tests'][name] = dict(size=float(np.mean(s0 > cut)),
                                          power=float((np.mean(sp > cut) + np.mean(sm > cut)) / 2),
                                          matched_size=msz_p, matched_power=(mp + mm) / 2)
        out['designs'].append(row)
        print('%5d %-6s %-4s | %s' % (n, kind, err, ' | '.join(
            '%.3f %.3f  %.3f  %.3f ' % (row['tests'][t]['size'], row['tests'][t]['power'],
                                        row['tests'][t]['matched_size'], row['tests'][t]['matched_power'])
            for t in TESTS)))
json.dump(out, open(L.out('mc_optimal_matched.json'), 'w'), indent=1)

if len(sys.argv) > 1:
    R = out['designs'][0]['R']
    lines = []
    for n in (500, 1000):
        lines.append(r'\multicolumn{14}{l}{\textit{\(n=%s\)}} \\' % ('{:,}'.format(n).replace(',', '{,}')))
        for d in out['designs']:
            if d['n'] != n:
                continue
            cells = ' & '.join('%.3f & %.3f & %.3f' % (d['tests'][t]['size'], d['tests'][t]['power'],
                                                      d['tests'][t]['matched_power']) for t in TESTS)
            lines.append('%s & %s & %s \\\\' % (SNAME[d['kind']], d['err'], cells))
        if n == 500:
            lines.append(r'\midrule')
    tex = r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\caption{Size and power of the fixed-slope tests: rejection rates at nominal and at matched size}
\label{tab:mc_optimal}
\begin{threeparttable}
\begin{tabular}{llcccccccccccc}
\toprule
 & & \multicolumn{3}{c}{\(S\), powers} & \multicolumn{3}{c}{\(KJ\), powers} & \multicolumn{3}{c}{\(KJ\), sieve} & \multicolumn{3}{c}{\(KJ\), bounded} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}
\(s_0\) & Error & size & power & match. & size & power & match. & size & power & match. & size & power & match. \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(%s\) replications per design. The data are
\(x=2+0.5w+s_0+0.5u_0\) and \(y=1+x+0.3w+u_0\), with \(w\sim N(0,1)\) and
\(s_0\) standardized. Under CH, \(u_0\sim N(0,1)\) independent of \(s_0\);
under HET, \(u_0=e\{(1+s_0^2)/2\}^{1/2}\) with \(e\sim N(0,1)\), so MI holds
and CH fails. The bimodal \(s_0\) is an equal mixture of \(N(\pm1,0.36)\).
Size is the rejection rate at the true slope and power the average rejection
rate at \(\beta_0\pm0.1\), both with \(\chi^2\) critical values. ``Match.'' is
the power at matched size: the cutoff of each test is the \(95\)th percentile
of its statistic at the true slope in one half of the replications, rejection
at \(\beta_0\pm0.1\) is evaluated in the other half, and each half supplies the
cutoff for the other. For the \(KJ\) tests the statistic is
\(\min(p_K/\alpha_K,\,p_J/\alpha_J)\), with \(p_K\) and \(p_J\) the \(\chi^2\)
\(p\)-values of \(K_n\) and \(J_n\). These cutoffs require the truth, so the
matched rates are a diagnostic; their simulation error is about \(0.02\).
\(S\), powers is the one-sided set of Table~\ref{tab:apps_hierarchy}. The
\(KJ\) tests use \(\alpha_K=0.04\) and \(\alpha_J=0.01\) and the bases of
Subsection~\ref{subsec:optimal}. All tests use the first-step corrections.
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % ('{:,}'.format(R).replace(',', '{,}'))
    with open(os.path.join(sys.argv[1], 'mc_optimal.tex'), 'w', newline='\n') as f:
        f.write(tex)
    print('wrote mc_optimal.tex')
