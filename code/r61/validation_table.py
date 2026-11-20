"""LaTeX table of the validation cases (reads output/r61/validation.json).
Usage: python validation_table.py <folder>/tables      writes validation.tex"""
import sys, os, json
import r61lib as L

R = json.load(open(L.out('validation.json')))
COLS = ['dist', 'tinc', 'tinc_any', 'qsmk']


def f(v, d=3):
    return r'\(%.*f\)' % (d, v)


def pv(p):
    return r'\(<0.001\)' if p < 0.001 else r'\(%.3f\)' % p


def sset(cs, d=3):
    if not cs:
        return r'\(\emptyset\)'
    return r'\(\cup\)'.join(r'\([%.*f,\,%.*f]\)%s' % (d, a, d, b, r'\(^{\dagger}\)' if e else '') for a, b, e in cs)


def th(v):
    return r'\(%s\)' % '{:,}'.format(v).replace(',', '{,}')


def stack(cs, d=3):
    """One interval per line inside a cell."""
    if not cs:
        return r'\(\emptyset\)'
    parts = [r'\([%.*f,\,%.*f]\)%s' % (d, a, d, b, r'\(^{\dagger}\)' if e else '') for a, b, e in cs]
    return parts[0] if len(parts) == 1 else r'\begin{tabular}[t]{@{}c@{}}' + r' \(\cup\)\\'.join(parts) + r'\end{tabular}'


rows = []
rows.append(('Observations (clusters)', [r'%s (%s)' % (th(R[c]['n']), th(R[c]['G'])) if R[c]['G'] else th(R[c]['n']) for c in COLS]))
rows.append(('Distinct values of the regressor', [th(R[c]['values']) for c in COLS]))
rows.append(('Skewness, kurtosis of \\(\\tilde x\\)', [r'\(%.2f\), \(%.2f\)' % (R[c]['skew'], R[c]['kurt']) for c in COLS]))
rows.append(('OLS (s.e.)', [r'%s (%s)' % (f(R[c]['ols']), f(R[c]['ols_se'])) for c in COLS]))
rows.append(('Benchmark', [r'exogenous' if c != 'qsmk' else r'%s (%s)' % (f(R[c]['tsls'], 1), f(R[c]['tsls_se'], 1)) for c in COLS]))
rows.append(('\\quad first-stage \\(F\\)', ['---' if c != 'qsmk' else f(R[c]['F'], 1) for c in COLS]))
rows.append(('VT root (screen \\(p\\))', [r'\begin{tabular}[t]{@{}c@{}}' + r'\\'.join(r'%s (%s)' % (f(v['beta']), pv(v['screen_p'])) for v in R[c]['vt']) + r'\end{tabular}'
                                         if R[c]['vt'] else 'none' for c in COLS]))
rows.append(('Two-sided pair: J \\(p\\)', [pv(R[c]['two_sided']['J_p']) for c in COLS]))
rows.append(('One-sided set', [stack(R[c]['one_sided']) for c in COLS]))
rows.append(('Bounded set', [stack(R[c]['bounded']) for c in COLS]))
rows.append(('\\quad OLS in one-sided, bounded set', [r'%s, %s' % ('yes' if R[c]['one_sided_ols_in'] else 'no', 'yes' if R[c]['bounded_ols_in'] else 'no') for c in COLS]))
rows.append(('Four moments, jackknife: J \\(p\\)', [pv(R[c]['jk']['p_lin']) for c in COLS]))
rows.append(('\\quad set', [stack(R[c]['jk']['set_lin']) for c in COLS]))
rows.append(('\\quad feedback \\(\\hat\\eta\\) (J \\(p\\))', [r'%s (%s)' % (f(R[c]['jk']['eta']), pv(R[c]['jk']['p_eta'])) for c in COLS]))

T = [r'\begin{table}[!h]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
     r'\caption{Validation against known benchmarks}', r'\label{tab:validation}',
     r'\begin{threeparttable}', r'\begin{tabular}{lcccc}', r'\toprule',
     r' & \multicolumn{3}{c}{Thornton (2008): collected HIV result} & NHEFS: weight gain \\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-5}',
     r' & Distance & Incentive & Incentive & Quit smoking \\',
     r' & (km) & (100 kwacha) & \(+\) any incentive & \\', r'\midrule']
for nm, vals in rows:
    T.append(nm + ' & ' + ' & '.join(vals) + r' \\')
T += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} Data from the \texttt{causaldata} package \citep{causaldata}. '
      r'Thornton: outcome, whether the respondent collected the HIV test result; distance to the test centre '
      r'and the cash incentive were both randomized. Controls: age, the incentive and the any-incentive '
      r'indicator (distance column); age, distance and its square (incentive columns). Clustered by village. '
      r'NHEFS: weight change 1971--1982 (kg) on quitting smoking, with the covariates of '
      r'\citet[ch.~12]{hernanrobins2020}; the benchmark is 2SLS with the 1982 cigarette price as instrument '
      r'\citep[ch.~16]{hernanrobins2020}; heteroscedasticity-robust inference. '
      r'Sets use the adjusted critical value, on \(241\) slopes with \(|\theta|\le\pi/4\) around OLS. '
      r'The jackknife rows use the four mean tendencies of Table~\ref{tab:ak_misspec}, '
      r'deleting villages (Thornton) or each of \(50\) random groups (NHEFS), on \(61\) slopes; '
      r'\(\hat\eta\) is the feedback coefficient of that table. \(^{\dagger}\)The interval reaches the edge of '
      r'the slope grid. Code: \texttt{validation.py}, \texttt{validation\_table.py}.',
      r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
dest = sys.argv[1] if len(sys.argv) > 1 else L.out('')
open(os.path.join(dest, 'validation.tex'), 'w', newline='\r\n').write('\n'.join(T) + '\n')
print('\n'.join(T))
