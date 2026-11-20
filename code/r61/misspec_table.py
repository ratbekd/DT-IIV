"""LaTeX table of the Angrist-Krueger misspecification checks (supplement, curvature subsection):
linear, quadratic schooling term, nonlinear feedback, degree step 1{educ = 16}.
Reads quad_ak.json, curved_ak.json, curved_ak_deg16.json from output/r61/.

Usage: python misspec_table.py <folder>/tables      writes ak_misspec.tex
"""
import sys, os, json
from scipy.stats import chi2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
ld = lambda f: json.load(open(os.path.join(OUT, f)))
qf, qc, qd = ld('quad_ak.json'), ld('curved_ak.json'), ld('curved_ak_deg16.json')


def sset(cs):
    if not cs:
        return r'\(\emptyset\)'
    return r' \(\cup\) '.join(r'\([%.3f,\,%.3f]\)' % (a, b) for a, b, e in cs)


def pv(p):
    return r'\(<0.001\)' if p < 0.001 else r'\(%.3f\)' % p


rows = []
rows.append(('Linear', '---', qf['J_lin'], qf['beta_lin'], qf['p_lin'], 3, qf['set_lin']['chi'], qf['set_lin']['F'], '---'))
rows.append((r'Quadratic, \(\kappa(x-\bar x)^2\)', r'\(\hat\kappa=%.4f\)' % qc['kappa_curv'], qc['J_curv'], qc['beta_curv'],
             qc['p_curv'], 2, qc['set_proj']['chi'], qc['set_proj']['F'], sset(qc['set_prof']['F'])))
rows.append((r'Feedback, \(\eta(u_w^2-1)\)', r'\(\hat\eta=%.3f\)' % qf['gamma_quad'], qf['J_quad'], qf['beta_quad'],
             qf['p_quad'], 2, qf['set_proj']['chi'], qf['set_proj']['F'], sset(qf['set_prof']['F'])))
rows.append((r'Degree step, \(\gamma\,1\{x=16\}\)', r'\(\hat\gamma=%.3f\)' % qd['kappa_curv'], qd['J_curv'], qd['beta_curv'],
             qd['p_curv'], 2, qd['set_proj']['chi'], qd['set_proj']['F'], sset(qd['set_prof']['F'])))

L = [r'\begin{table}[!h]', r'\centering', r'\small',
     r'\caption{Quarter of birth and schooling: smooth and step departures from the linear model}',
     r'\label{tab:ak_misspec}', r'\begin{threeparttable}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r' & Extra & Min.\ \(T\) & & J \(p\) & \(\beta\) set & \(\beta\) set \\',
     r'Specification & parameter & (at \(\beta\)) & df & (\(\chi^2\)) & projection, F & profiled, F \\', r'\midrule']
for nm, par, J, b, p, df, cchi, cF, prof in rows:
    L.append(r'%s & %s & \(%.1f\) (\(%.3f\)) & \(%d\) & %s & %s & %s \\' % (nm, par, J, b, df, pv(p), sset(cF), prof))
c = qf['cuts']
L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} Sample and controls as in Table~\ref{tab:ak_curvature}. Moments: '
      r'\(\mathbb E[u_wh(s)]\) with \(h(s)=(s^2,s^3,\tanh s,e^{-s^2/2})\), where \(s\) is the standardized '
      r'\(v=x_w-\lambda u_w-\eta(u_w^2-1)\) in the feedback model and \(v=x_w-\lambda u_w\) otherwise. '
      r'\(T\) is the Wald statistic with the delete-cluster jackknife covariance that refits the projections on '
      r'the controls (\(51\) state-of-birth clusters). The extra parameter is profiled out; the J test of the '
      r'joint minimum has \(4-2\) degrees of freedom (\(4-1\) in the linear model). Projection sets use the '
      r'Hotelling cutoff \(%.2f\) for four moments (\(\chi^2_{4,0.95}=%.2f\)); profiled sets use \(%.2f\) '
      r'(\(\chi^2_{3,0.95}=%.2f\)). Slope grid: \(61\) points with \(|\theta|\le\pi/4\) for the first three rows '
      r'and \(21\) points on \([0.048,\,0.068]\) for the degree step. Code: \texttt{quad\_feedback.py}, '
      r'\texttt{curved\_outcome.py}.' % (c['F_proj'], c['chi_proj'], c['F_prof'], c['chi_prof']),
      r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
dest = sys.argv[1] if len(sys.argv) > 1 else OUT
open(os.path.join(dest, 'ak_misspec.tex'), 'w', newline='\r\n').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ---- seven applications: apps_misspec.tex --------------------------------------------------
NAMES = [('mroz', 'Mroz'), ('card', 'Card'), ('star', 'STAR'), ('maim5v', 'Maimonides, grade 5'),
         ('maim4v', 'Maimonides, grade 4'), ('banerjee', 'Banerjee--Iyer'), ('adh', 'ADH')]
A = {k: ld('misspec_%s.json' % k) for k, _ in NAMES}


def aset(cs):
    if not cs:
        return r'\(\emptyset\)'
    return r' \(\cup\) '.join(r'\([%.2f,\,%.2f]\)%s' % (a, b, r'\(^{\dagger}\)' if e else '') for a, b, e in cs)


def num(v, d=3):
    return r'\(%.*f\)' % (d, v)


T = [r'\begin{table}[!h]', r'\centering', r'\small',
     r'\caption{Nonlinear feedback and a quadratic outcome term in the seven applications}',
     r'\label{tab:apps_misspec}', r'\begin{threeparttable}', r'\begin{tabular}{llcccl}', r'\toprule',
     r'Application & \(n\) (groups) & \(\hat\beta\) & Extra & J \(p\) & \(\beta\) set, projection, F \\', r'\midrule',
     r'\multicolumn{6}{l}{\emph{A. Linear model}} \\']
for k, nm in NAMES:
    a = A[k]; f = a['feedback']
    grp = r'\(%s\) (\(%s\))%s' % ('{:,}'.format(a['n']).replace(',', '{,}'), '{:,}'.format(a['G']).replace(',', '{,}'), r'\(^{a}\)' if a['cltype'] == 'random groups' else '')
    T.append(r'%s & %s & %s & --- & %s & %s \\' % (nm, grp, num(f['beta_lin']), pv(f['p_lin']), aset(f['set_lin']['F'])))
for title, key, par, bk, pk in ((r'B. Nonlinear feedback, \(\eta(u_w^2-1)\)', 'feedback', r'\hat\eta', 'beta_quad', 'gamma_quad'),
                                (r'C. Quadratic outcome term, \(\kappa(x-\bar x)^2\)', 'curved', r'\hat\kappa', 'beta_curv', 'kappa_curv')):
    T += [r'\midrule', r'\multicolumn{6}{l}{\emph{%s}} \\' % title]
    for k, nm in NAMES:
        r = A[k][key]
        jk = 'p_quad' if key == 'feedback' else 'p_curv'
        flag = r'\(^{b}\)' if key == 'curved' and abs(r[pk]) > 0.98 * r['kbound_raw'] else ''
        T.append(r'%s & & %s & \(%s=%.3f\)%s & %s & %s \\' % (nm, num(r[bk]), par, r[pk], flag, pv(r[jk]), aset(r['set_proj']['F'])))
T += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} Moments and covariance as in Table~\ref{tab:ak_misspec}: four mean tendencies '
      r'\(\mathbb E[u_wh(s)]\), \(h(s)=(s^2,s^3,\tanh s,e^{-s^2/2})\), with the delete-group jackknife '
      r'covariance that refits the projections on the controls. The extra parameter is profiled out; '
      r'\(\hat\beta\) and the extra parameter minimise \(T\) jointly, and the J test has \(3\) degrees of freedom in '
      r'Panel~A and \(2\) in Panels~B and~C. Sets: slopes with \(\min T\) below the Hotelling cutoff for four '
      r'moments, on \(61\) slopes with \(|\theta|\le\pi/4\) around OLS. \(^{\dagger}\)The interval reaches the '
      r'edge of the slope grid. \(^{a}\)No clusters; the jackknife deletes each of \(50\) random groups. '
      r'\(^{b}\)At the bound of the search over \(\kappa\). Code: \texttt{apps\_misspec.py}, \texttt{misspec\_table.py}.',
      r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
open(os.path.join(dest, 'apps_misspec.tex'), 'w', newline='\r\n').write('\n'.join(T) + '\n')
print('\n'.join(T))
