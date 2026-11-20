"""LaTeX tables for the R61 additions, from the JSON output of the r61 scripts.

Usage: python make_tables.py <folder for the .tex files>
Writes tail_coverage.tex, tail_power.tex, tail_full.tex, tail_adh.tex, tail_apps.tex,
ami_apps.tex, ami_curve.tex and ami_sim.tex. A table is skipped when its input is missing.
"""
import sys, os, json
import numpy as np
import r61lib as L

DEST = sys.argv[1]
APPS = [('mroz', 'Mroz'), ('card', 'Card'), ('star', 'STAR'), ('maim5v', 'Maimonides, grade 5'),
        ('maim4v', 'Maimonides, grade 4'), ('banerjee', 'Banerjee--Iyer'), ('adh', 'ADH')]
DES = {'tail-const-48': ('Tail', 'constant', 48), 'tail-vary-48': ('Tail', 'varying', 48),
       'ideal-const-48': ('Laplace', 'constant', 48), 'ideal-vary-48': ('Laplace', 'varying', 48),
       'tail-const-192': ('Tail', 'constant', 192), 'tail-vary-192': ('Tail', 'varying', 192)}
FAMNAME = {'P': r'Powers \(h_P\)', 'B': r'Bounded \(h_B\)', 'R': r'Bounded, robust scale', 'L': r'Localized \(h_L\)'}


def load(name):
    p = L.out(name)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, text):
    with open(os.path.join(DEST, name), 'w', newline='\n') as f:
        f.write(text)
    print('wrote', name)


def fset(cs, nd=3):
    """Set as LaTeX: components joined by a union, dagger when it reaches the domain edge."""
    if not cs:
        return 'empty'
    f = '%.' + str(nd) + 'f'
    parts = ['[' + f % a + r',\,' + f % b + ']' for a, b, _ in cs]
    edge = any(e for _, _, e in cs)
    if len(parts) == 1:
        s = r'\(' + parts[0] + r'\)'
    else:
        s = r'\(\substack{' + r'\\{}\cup'.join(parts) + r'}\)'
    return s + (r'\(^{\dagger}\)' if edge else '')


def whole(cs):
    return len(cs) == 1 and cs[0][2] and abs(cs[0][0]) > 0 and False


# ---------------------------------------------------------------------- tail simulation
rep = load('tail_report.json')
if rep:
    cov, pw = rep['cover'], rep['power']
    rows = []
    for d in rep['designs']:
        sh, var, G = DES[d]
        for k, f in enumerate(('P', 'B', 'R')):
            lead = '%s & %s & %d' % (sh, var, G) if k == 0 else ' & & '
            rows.append('%s & %s & %s \\\\' % (lead, FAMNAME[f], ' & '.join('%.3f' % cov['%s|%s|%s' % (d, f, c)]
                                                                           for c in ('an_chi', 'an_adj', 'jk_chi', 'jk_F'))))
        rows.append(r'\addlinespace')
    write('tail_coverage.tex', r"""\begin{table}[!h]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Extreme-tail cluster designs: coverage of the true slope}
\label{tab:tail_coverage}
\begin{threeparttable}
\begin{tabular}{lllccccc}
\toprule
 & & & & \multicolumn{2}{c}{Analytic covariance} & \multicolumn{2}{c}{Full-refit jackknife} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Regressor & Variance & \(G\) & Moments & \(\chi^2_2\) & adjusted & \(\chi^2_2\) & \(F\) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(%d\) replications per design; nominal coverage \(0.95\). ``Tail''
designs resample whole state blocks of the standardized residualized ADH regressor;
``Laplace'' designs draw the regressor from a standardized Laplace distribution at the
same cluster sizes. Mean independence holds in every design. The average number of
observations is about \(1{,}440\) with \(48\) clusters and \(5{,}770\) with \(192\).
``Adjusted'' is the critical value \eqref{eq:crit_nu}; \(F\) is \eqref{eq:tail_f}.
``Bounded, robust scale'' is the median- and MAD-scaled pair \eqref{eq:bounded_moments}.
The simulation error of an entry near \(0.95\) is about \(0.009\), and near \(0.70\)
about \(0.019\).
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % rep['R'])

    rows = []
    for d in rep['designs'][:4]:
        sh, var, G = DES[d]
        first = True
        for f, c, lab in (('P', 'an_chi', r'analytic, \(\chi^2_2\)'), ('P', 'jk_F', r'jackknife, \(F\)'),
                          ('B', 'an_adj', 'analytic, adjusted'), ('B', 'jk_F', r'jackknife, \(F\)'),
                          ('R', 'an_adj', 'analytic, adjusted'), ('R', 'jk_F', r'jackknife, \(F\)')):
            r = pw['%s|%s|%s' % (d, f, c)]
            lead = '%s, %s' % (sh, var) if first else ''
            first = False
            rows.append('%s & %s & %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (
                lead, FAMNAME[f], lab, r['nominal_size'], r['nominal_0.6'], r['nominal_0.8'],
                r['heldout_size'], r['matched_0.6'], r['matched_0.8']))
        rows.append(r'\addlinespace')
    write('tail_power.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Extreme-tail cluster designs with 48 clusters: rejection rates at nominal and at matched size}
\label{tab:tail_power}
\begin{threeparttable}
\begin{tabular}{lllcccccc}
\toprule
 & & & \multicolumn{3}{c}{Nominal critical value} & \multicolumn{3}{c}{Matched size (held out)} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Design & Moments & Calibration & \(\beta_0\) & \(0.6\) & \(0.8\) & \(\beta_0\) & \(0.6\) & \(0.8\) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} Rejection rates of \(H_0\colon\beta=b\) for \(b=\beta_0=0.7\), the
true slope, and for \(b=0.6\) and \(0.8\). The first three columns use the nominal
\(5\%%\) critical value and all \(%d\) replications. For the last three, the critical
value of each procedure is the \(95\)th percentile of its statistic at the true slope
in the first \(%d\) replications, and the rates are computed on the remaining \(%d\).
These cutoffs require the truth and are a diagnostic, not a feasible correction.
The simulation error of an entry in the last three columns is at most \(0.029\).
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % (rep['R'], rep['R'] // 2, rep['R'] - rep['R'] // 2))

    rows = []
    for d in rep['designs']:
        sh, var, G = DES[d]
        for k, f in enumerate(('P', 'B', 'L', 'R')):
            lead = '%s & %s & %d' % (sh, var, G) if k == 0 else ' & & '
            r = pw['%s|%s|jk_F' % (d, f)]
            rows.append('%s & %s & %s & %.3f & %.3f \\\\' % (
                lead, FAMNAME[f], ' & '.join('%.3f' % cov['%s|%s|%s' % (d, f, c)] for c in rep['cal']),
                r['matched_0.6'], r['matched_0.8']))
        rows.append(r'\addlinespace')
    write('tail_full.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Extreme-tail cluster designs: all families and calibrations}
\label{tab:tail_full}
\begin{threeparttable}
\begin{tabular}{llllcccccccc}
\toprule
 & & & & \multicolumn{6}{c}{Coverage of the true slope} & \multicolumn{2}{c}{Matched-size rejection} \\
\cmidrule(lr){5-10}\cmidrule(lr){11-12}
 & & & & \multicolumn{2}{c}{Analytic} & \multicolumn{2}{c}{Jackknife} & & & \multicolumn{2}{c}{jackknife, \(F\)} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){11-12}
Regressor & Variance & \(G\) & Moments & \(\chi^2_2\) & adj. & \(\chi^2_2\) & \(F\) & Wild & Frozen & \(0.6\) & \(0.8\) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(%d\) replications per design; designs as in Table~\ref{M-tab:tail_coverage}.
``Wild'' is the linearized wild calibration of the jackknife statistic with \(999\) draws.
``Frozen'' deletes clusters from the raw moment averages and holds the first step at its
full-sample value; it is a diagnostic and not a valid procedure. The last two columns
are rejection rates of \(\beta=0.6\) and \(0.8\) with cutoffs estimated at the true slope
in the first half of the replications and evaluated in the second half.
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % rep['R'])

# ---------------------------------------------------------------------- tail: applications
ta = {k: load('tail_apps_%s.json' % k) for k, _ in APPS}
if ta.get('adh'):
    r = ta['adh']; rows = []
    for f in ('P', 'B', 'L', 'R'):
        fr = r['fams'][f]
        for k, (c, lab) in enumerate((('an_chi', r'analytic, \(\chi^2_2\)'), ('an_adj', 'analytic, adjusted'),
                                      ('jk_chi', r'jackknife, \(\chi^2_2\)'), ('jk_F', r'jackknife, \(F\)'),
                                      ('wild', 'jackknife, wild'))):
            yn = lambda v: 'yes' if v else 'no'
            rows.append('%s & %s & %s & %s & %s & %s \\\\' % (FAMNAME[f] if k == 0 else '', lab, fset(fr[c]['comps']),
                                                              yn(fr[c]['zero_in']), yn(fr[c]['ols_in']), yn(fr[c]['tsls_in'])))
        rows.append(r'\addlinespace')
    write('tail_adh.tex', r"""\begin{table}[!h]
\centering
\small
\caption{ADH: fixed-slope sets of the fixed pairs under each calibration}
\label{tab:tail_adh}
\begin{threeparttable}
\begin{tabular}{llcccc}
\toprule
 & & & \multicolumn{3}{c}{Set contains} \\
\cmidrule(lr){4-6}
Moments & Calibration & \(95\%%\) set & zero & OLS & 2SLS \\
\midrule
""" % () + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(1{,}444\) observations, \(48\) state clusters, all controls of
Table~\ref{M-tab:apps_hierarchy}. OLS is \(%.3f\) and 2SLS is \(%.3f\). The sets are
computed on a grid of \(241\) angles in \(|\theta|\le\pi/4\) and their endpoints are grid
points. The wild calibration uses \(9{,}999\) draws. With \(48\) clusters the stress
designs of Table~\ref{M-tab:tail_coverage} do not support reading these as \(95\%%\)
confidence sets.
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % (r['ols'], r['tsls']))

if all(ta.values()):
    rows = []
    for k, nm in APPS:
        r = ta[k]
        for j, f in enumerate(('P', 'R')):
            fr = r['fams'][f]
            lead = '%s & %s' % (nm, ('{:,}'.format(r['G']).replace(',', '{,}'))) if j == 0 else ' & '
            rows.append('%s & %s & %s & %s & %s \\\\' % (lead, 'powers' if f == 'P' else 'bounded',
                                                         fset(fr['an_chi']['comps']), fset(fr['an_adj']['comps']),
                                                         fset(fr['jk_F']['comps'])))
        rows.append(r'\addlinespace')
    write('tail_apps.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{The one-sided and bounded sets under the analytic and the full-refit calibrations}
\label{tab:tail_apps}
\begin{threeparttable}
\begin{tabular}{lrlccc}
\toprule
 & & & \multicolumn{2}{c}{Analytic covariance} & Full-refit jackknife \\
\cmidrule(lr){4-5}\cmidrule(lr){6-6}
Application & \(G\) & Moments & \(\chi^2_2\) & adjusted & \(F\) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(G\) is the number of clusters, or of observations when the data
are not clustered. ``Powers'' is the one-sided set \((s^2,s^3)\); ``bounded'' is the
median- and MAD-scaled pair (\ref{M-eq:bounded_moments}). ``Adjusted'' uses the critical value
(\ref{M-eq:crit_nu}); the jackknife refits the first step after deleting each cluster and uses
the critical value (\ref{M-eq:tail_f}). Grid of \(241\) angles in \(|\theta|\le\pi/4\); endpoints
are grid points, so they differ slightly from the polished endpoints of
Tables~\ref{M-tab:apps_hierarchy} and~\ref{tab:bounded_apps}. Clustering and controls as in
Table~\ref{M-tab:apps_hierarchy}. \(^{\dagger}\)The set reaches the labelling boundary.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------------- approximate MI
am = {k: load('ami_apps_%s.json' % k) for k, _ in APPS}
if all(am.values()):
    def ent(r, nm):
        e = r['entry'][nm]
        if not e['in_domain']:
            return 'outside'
        if e['eps_enter'] is None:
            return r'\(>1\)'
        return r'\(0\)' if e['eps_enter'] == 0 else r'\(%.3f\)' % e['eps_enter']
    rows = []
    for k, nm in APPS:
        r = am[k]
        rows.append('%s & %s & %s & %s & %s & %s & %s \\\\' % (nm, fset(r['exact']), fset(r['joint']['0']),
                                                               fset(r['joint']['0.05']), fset(r['joint']['0.1']),
                                                               ent(r, 'zero'), ent(r, 'tsls')))
    write('ami_apps.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Approximate mean independence in seven applications}
\label{tab:ami_apps}
\begin{threeparttable}
\begin{tabular}{lcccccc}
\toprule
 & Exact MI & \multicolumn{3}{c}{Joint-region set \(\mathcal C_n(\epsilon)\)} & \multicolumn{2}{c}{Entry tolerance} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Application & \(\chi^2_2\) set & \(\epsilon=0\) & \(\epsilon=0.05\) & \(\epsilon=0.10\) & zero & 2SLS \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} Fixed basis \(h(s)=(\tanh s,\exp(-s^2/2))'\), so \(K=2\) and
\(d=5\). ``Exact MI'' inverts the corrected fixed-slope test of the two moments
\(E[u\,h(s)]=0\) with the \(\chi^2_2\) critical value. \(\mathcal C_n(\epsilon)\) is the
set \eqref{eq:ami_confidence_r60} with the \(\chi^2_5\) critical value and the covariance
of \eqref{eq:ami_if_r60}. The entry tolerance of a slope is the smallest \(\epsilon\) at
which it enters \(\mathcal C_n(\epsilon)\); \(0\) means that it is already in
\(\mathcal C_n(0)\), and ``outside'' that it lies outside the labelling domain. Grid of
\(241\) angles in \(|\theta|\le\pi/4\); endpoints are grid points. Clustering and
controls as in Table~\ref{tab:apps_hierarchy}. \(^{\dagger}\)The set reaches the
labelling boundary.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")
    rows = []
    for k, nm in APPS:
        r = am[k]
        for j, (key, lab) in enumerate((('joint', 'joint'), ('box', 'box'))):
            rows.append('%s & %s & %s \\\\' % (nm if j == 0 else '', lab,
                                               ' & '.join(fset(r[key]['%g' % e], 2) for e in (0, 0.025, 0.05, 0.1, 0.2))))
        rows.append(r'\addlinespace')
    write('ami_curve.tex', r"""\begin{table}[!h]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{Approximate mean independence: joint-region and box sets on a grid of tolerances}
\label{tab:ami_curve}
\begin{threeparttable}
\begin{tabular}{llccccc}
\toprule
Application & Set & \(\epsilon=0\) & \(0.025\) & \(0.05\) & \(0.10\) & \(0.20\) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\scriptsize
\item \textit{Notes:} ``Joint'' is the set (\ref{M-eq:ami_confidence_r60}) of the main text with
the \(\chi^2_5\) critical value. ``Box'' projects the Wald region of the two moments on
the box \(|\mu_j|\le\epsilon\), with the \(\chi^2_2\) critical value; both functions of the
basis are bounded by one. Basis, grid, clustering and controls as in
Table~\ref{M-tab:ami_apps}. \(^{\dagger}\)The set reaches the labelling boundary.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

sm = load('ami_sim.json')
if sm:
    E = sm['eps']; nE = len(E); rows = []
    last = None
    for c in sm['cells']:
        lead = '%s & %s' % ('{:,}'.format(c['n']).replace(',', '{,}'), 'quadratic' if c['shape'] == 'quad' else r'\(\tanh\)')
        show = lead if lead != last else ' & '
        last = lead
        cov, pwr = c['cover'], c['power']
        rows.append('%s & %.1f & %.3f & %s & %.3f & %s \\\\' % (
            show, c['eps0'], cov[0], ' & '.join('%.3f' % v for v in cov[1:1 + nE]),
            pwr[0], ' & '.join('%.3f' % v for v in pwr[1:1 + nE])))
    write('ami_sim.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Approximate mean independence: coverage and rejection in a design with a known conditional-mean norm}
\label{tab:ami_sim}
\begin{threeparttable}
\begin{tabular}{llccccccccccccc}
\toprule
 & & & \multicolumn{6}{c}{Coverage of \(\beta_0\)} & \multicolumn{6}{c}{Rejection at \(\beta_0\pm0.1\)} \\
\cmidrule(lr){4-9}\cmidrule(lr){10-15}
 & & & Exact & \multicolumn{5}{c}{\(\mathcal C_n(\epsilon)\), \(\epsilon=\)} & Exact & \multicolumn{5}{c}{\(\mathcal C_n(\epsilon)\), \(\epsilon=\)} \\
\cmidrule(lr){5-9}\cmidrule(lr){11-15}
\(n\) & \(q\) & \(\epsilon_0\) & MI & """ + ' & '.join('\\(%g\\)' % e for e in E) + ' & MI & ' + ' & '.join('\\(%g\\)' % e for e in E) + r""" \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(%d\) replications per design. The data are those of
Table~\ref{tab:mc_optimal} with \(s_0\) a standardized \(\chi^2_4\) variable and
\(u_0=\epsilon_0q(s_0)+(1-\epsilon_0^2)^{1/2}e\), \(e\sim N(0,1)\), where \(q\) has mean
zero and unit variance and is uncorrelated with \(s_0\). Hence
\(\|E[u_0\mid s_0]\|_2=\epsilon_0\) exactly. The quadratic \(q\) is proportional to
\(s_0^2-1-\sqrt2\,s_0\), and the \(\tanh\) one to the part of \(\tanh s_0\) orthogonal to
\((1,s_0)\). ``Exact MI'' is the fixed-slope test of \(E[u\,h(s)]=0\) with the
\(\chi^2_2\) critical value; \(\mathcal C_n(\epsilon)\) uses the \(\chi^2_5\) critical value.
Nominal coverage is \(0.95\). The theory guarantees coverage of at least \(0.95\) when
\(\epsilon\ge\epsilon_0\).
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % sm['R'])

fc = load('ami_factor.json')
if fc:
    E = fc['eps']; rows = []
    for d, p in enumerate(fc['pop']):
        cells = [c for c in fc['cells'] if c['design'] == d]
        for j, c in enumerate(cells):
            lead = ('%s & %.2f & %.3f & %.3f' % (p['label'].capitalize(), p['rho_xu'], p['eps0'], p['eps_bounded'])) if j == 0 else ' & & & '
            cv = c['cover']
            rows.append('%s & %s & %.3f & %.3f & %s \\\\' % (lead, '{:,}'.format(c['n']).replace(',', '{,}'), cv[0], cv[1],
                                                             ' & '.join('%.3f' % v for v in cv[2:])))
        rows.append(r'\addlinespace')
    write('ami_factor.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{The common-factor model: size of the departure from mean independence and coverage of the true slope}
\label{tab:ami_factor}
\begin{threeparttable}
\begin{tabular}{lcccrccccccc}
\toprule
 & & & & & \multicolumn{2}{c}{Exact MI} & \multicolumn{5}{c}{\(\mathcal C_n(\epsilon)\), \(\epsilon=\)} \\
\cmidrule(lr){6-7}\cmidrule(lr){8-12}
Components & \(\rho_{xu}\) & \(\epsilon_0\) & \(\epsilon_{0,h}\) & \(n\) & powers & bounded & """ + ' & '.join('\\(%g\\)' % e for e in E) + r""" \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} \(\varepsilon=f+e\), \(\widetilde x=af+z\) and
\(\widetilde y=\widetilde x+\varepsilon\), with \(f\), \(e\) and \(z\) independent with
unit variance. In the first two rows \(f\) is a standardized \(\chi^2_4\) variable and
\(e\), \(z\) are Gaussian, with \(a=1\) and \(a=2\); in the third all three are
standardized \(\chi^2_4\); in the fourth \(f\) is a standardized \(t_5\); in the fifth all
are Gaussian. \(\epsilon_0=\|E[u_0\mid s_0]\|_2\) for the orthogonal remainder, computed
in a sample of four million draws; the value \(0.001\) in the last row is the sampling
error of that computation. \(\epsilon_{0,h}\) is the part of the norm captured by the
basis \(h=(\tanh s,\exp(-s^2/2))'\). The remaining columns are coverage rates of the true
slope in \(%d\) replications at nominal \(0.95\): the exact-MI fixed-slope sets of the
powers \((s^2,s^3)\) and of the bounded pair with the \(\chi^2_2\) critical value, and
the sets (\ref{M-eq:ami_confidence_r60}) of the main text.
\end{tablenotes}
\end{threeparttable}
\end{table}
""" % fc['R'])

bf = load('bias_floor.json')
if bf:
    LAW = {'t5': r'\(t_5\)', 'logistic': 'Logistic', 'gamma6': 'Gamma, shape 6'}
    HN = {'location score': r'location score \(\psi\)', 'tanh s': r'\(\tanh s\)', 'exp(-s^2/2)': r'\(\exp(-s^2/2)\)',
          'bounded pair, best': 'bounded pair, best', 's^2': r'\(s^2\)', 's^3': r'\(s^3\)',
          'powers, best': 'powers, best'}
    rows = []; last = None
    for a in bf['A']:
        if last is not None and a['law'] != last:
            rows.append(r'\addlinespace')
        lead = LAW[a['law']] if a['law'] != last else ''
        last = a['law']
        rows.append('%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (
            lead, HN[a['h']], a['normH'], abs(a['D']), a['predicted'], a['exact_001'], a['exact_005'], a['floor']))
    rowsB = []
    for (k, nm), c in zip(APPS, bf['C']):
        assert c['dataset'] == k
        rowsB.append('%s & %.3f & %.3f & %.2f & %.2f \\\\' % (nm, c['ratio'], c['S2'], 1 / np.sqrt(c['S2']), c['rate']))
    write('bias_floor.tex', r"""\begin{table}[!h]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption{The bias floor under approximate mean independence}
\label{tab:bias_floor}
\begin{threeparttable}
\begin{tabular}{llcccccc}
\toprule
\multicolumn{8}{l}{\textit{Panel A. Displacement of the angle per unit of \(\epsilon\) under the worst-case departure}} \\
 & & & & First-order & \multicolumn{2}{c}{Exact root} & \\
\cmidrule(lr){6-7}
Law of \(s_0\) & Tendency \(h\) & \(\|H\|_2\) & \(|D(h)|\) & prediction & \(\epsilon=0.01\) & \(\epsilon=0.05\) & Floor \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{lcccc}
\toprule
\multicolumn{5}{l}{\textit{Panel B. Sensitivity of the bounded pair in the applications, at the OLS slope}} \\
 & & & \multicolumn{2}{c}{Displacement per unit of \(\epsilon\)} \\
\cmidrule(lr){4-5}
Application & \(\widehat\sigma_r/\widehat\sigma_x\) & \(\widehat{\mathbf D}'\widehat{\mathbf Q}^{-1}\widehat{\mathbf D}\) & angle & slope \\
\midrule
""" + '\n'.join(rowsB) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} Panel A: population values by quadrature under conditional
homoscedasticity. \(H\) is the part of \(h\) orthogonal to \((1,s_0)\) and \(D(h)\) the
slope of \(E[u\,h(s)]\) at the truth. The departure is \(E[u_0\mid s_0]=\epsilon H/\|H\|_2\),
the worst case for the tendency. The first-order prediction is \(\|H\|_2/|D(h)|\); the
exact root is that of the population moment, divided by \(\epsilon\). The floor is
\(\{\mathcal J(f)-1\}^{-1/2}\), with \(\mathcal J(f)=1.25\), \(\pi^2/9\) and \(1.5\) for the
three laws, which are standardized. ``Best'' is the combination of a pair that minimizes
the first-order prediction. For \(t_5\) the powers are omitted because \(E[s_0^6]\) is
infinite. Panel B: \(\widehat{\mathbf D}\) is the slope of the two sample moments of
\(h=(\tanh s,\exp(-s^2/2))'\) with respect to the angle and \(\widehat{\mathbf Q}\) the
matrix of (\ref{M-eq:ami_projection_r60}), both at the OLS slope. The displacement of the
angle is \((\widehat{\mathbf D}'\widehat{\mathbf Q}^{-1}\widehat{\mathbf D})^{-1/2}\), and
that of the slope multiplies it by \(\widehat\sigma_r/\widehat\sigma_x\).
\end{tablenotes}
\end{threeparttable}
\end{table}
""")
