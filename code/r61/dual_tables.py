"""LaTeX tables of the u-side tendencies.
  dual_apps.tex   main text: sets of the mean, u-side and combined tendencies in the applications
  dual_info.tex   supplement: information by error distribution, and Angrist-Krueger
  dual_sim.tex    supplement: rejection rates in the four synthetic designs
Reads output/r61/dual_apps.json, dual_ak.json, dual_info.json, dual_check.json.
Usage: python dual_tables.py <folder>/tables
"""
import sys, os, json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')
ld = lambda f: json.load(open(os.path.join(OUT, f)))
dest = sys.argv[1] if len(sys.argv) > 1 else OUT


def pv(p):
    return '---' if p is None else (r'\(<0.001\)' if p < 0.001 else r'\(%.3f\)' % p)


def sset(cs, d):
    if not cs:
        return r'\(\emptyset\)'
    return r' \(\cup\) '.join(r'\([%.*f,\,%.*f]\)%s' % (d, a, d, b, r'\(^{\dagger}\)' if e else '') for a, b, e in cs)


def stack(cs, d):
    if not cs:
        return r'\(\emptyset\)'
    parts = [r'\([%.*f,\,%.*f]\)%s' % (d, a, d, b, r'\(^{\dagger}\)' if e else '') for a, b, e in cs]
    return parts[0] if len(parts) == 1 else r'\begin{tabular}[t]{@{}c@{}}' + (r' \(\cup\)' + '\\\\').join(parts) + r'\end{tabular}'


def write(name, L):
    open(os.path.join(dest, name), 'w', newline='\r\n').write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---- applications ---------------------------------------------------------------------------
A = ld('dual_apps.json'); K = ld('dual_ak.json')
NAMES = [('mroz', 'Mroz', 2), ('card', 'Card', 3), ('star', 'STAR', 2), ('maim5v', 'Maimonides, grade 5', 2),
         ('maim4v', 'Maimonides, grade 4', 2), ('banerjee', 'Banerjee--Iyer', 2), ('adh', 'ADH', 2)]
rows = [(nm, A[k], d) for k, nm, d in NAMES] + [('Angrist--Krueger', K, 3)]
L = [r'\begin{table}[!h]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
     r'\caption{Mean, error-side and combined tendencies in the applications}', r'\label{tab:dual_apps}',
     r'\begin{threeparttable}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r' & \multicolumn{2}{c}{Mean tendencies} & \multicolumn{2}{c}{Error-side tendencies} & \multicolumn{2}{c}{Both} \\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
     r'Application & Set & J \(p\) & Set & J \(p\) & Set & J \(p\) \\', r'\midrule']
for nm, r, d in rows:
    R = r['res']
    L.append(r'%s & %s & %s & %s & %s & %s & %s \\' % (nm, stack(R['s_four']['set_F'], d), pv(R['s_four']['p_J']),
                                                     stack(R['u_bounded']['set_F'], d), pv(R['u_bounded']['p_J']),
                                                     stack(R['both']['set_F'], d), pv(R['both']['p_J'])))
L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} Mean tendencies: \(E[u\,h(s)]\) with \(h(s)=(s^2,s^3,\tanh s,e^{-s^2/2})\). '
      r'Error-side tendencies: \(E[s\,a(u)]\) with \(a(u)=(\tanh u,\tanh(u/2),\phi(u),u\phi(u))\), '
      r'\(\phi(u)=e^{-u^2/2}\). Both: the eight moments together. Each set collects the slopes at which the '
      r'Wald statistic with the delete-cluster jackknife covariance lies below the Hotelling cutoff for '
      r'its number of moments; the jackknife refits the projections on the controls, the scales and the '
      r'projection of \(\widetilde x\) on \(u\). Mroz and Card have no clusters; their jackknife deletes each '
      r'of \(50\) random groups. Slopes: \(121\) points (\(241\) for Angrist--Krueger) with '
      r'\(|\theta|\le\pi/4\) around OLS. J \(p\): the minimum of the statistic against '
      r'\(\chi^2_{m-1}\). \(^{\dagger}\)The interval reaches the edge of the slope grid. '
      r'Code: \texttt{dual\_tendency.py}.', r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
write('dual_apps.tex', L)

# ---- information ----------------------------------------------------------------------------
I = ld('dual_info.json'); Q = I['quadrature']
L = [r'\begin{table}[!h]', r'\centering', r'\small',
     r'\caption{Information of the error-side tendencies}', r'\label{tab:dual_info}',
     r'\begin{threeparttable}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r'Error \(u_0\) & Skewness & Kurtosis & \(\mathcal J(f_u)-1\) & VT & Bounded & Sieve \\', r'\midrule']
for nm, r in Q.items():
    if nm == 'skew-normal(-4)':
        continue
    lab = nm.replace('t(', r'Student \(t_{').replace(')', r'}\)') if nm.startswith('t(') else nm
    L.append(r'%s & \(%.2f\) & \(%.2f\) & \(%.3f\) & \(%.3f\) & \(%.3f\) & \(%.3f\) \\' % (
        lab, r['mu3'], r['mu4'], r['J_minus_1'], r['I_VT'], r['I_bounded'], r['I_sieve']))
ak = I['ak']
L.append(r'\midrule')
for k, r in ak['kde'].items():
    L.append(r'Angrist--Krueger, bandwidth \(\times%s\) & \(%.2f\) & \(%.2f\) & \(%.3f\) & \(%.3f\) & \(%.3f\) & \(%.3f\) \\' % (
        k, ak['mu3'], ak['mu4'], r['J_minus_1'], ak['I_VT'], r['I_bounded'], r['I_sieve']))
mc = I['montecarlo']
L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} Information \(\mathcal I(a)\) of Proposition~\ref{M-prop:dual}, per observation, '
      r'under \(E[s_0\mid u_0]=0\) and \(\operatorname{var}(s_0\mid u_0)=1\), for standardized errors. '
      r'\(\mathcal J(f_u)-1\) is the bound; VT is \(a(u)=u^2\), with information '
      r'\(\mu_3^2/(\mu_4-1-\mu_3^2)\); Bounded and Sieve are the best combinations within '
      r'\((\tanh u,\tanh(u/2),\phi(u),u\phi(u))\) and \((u^2,u^3,\tanh u,\phi(u),u\phi(u))\). '
      r'Grid quadrature with \(400{,}001\) points. Angrist--Krueger: the standardized residual at the VT root '
      r'(\(0.0599\)); the score is a Gaussian kernel estimate from \(40{,}000\) draws, with the rule-of-thumb '
      r'bandwidth multiplied by the factor shown, and the moments are sample averages over the \(329{,}509\) '
      r'observations. A Monte Carlo check with a Gaussian \(s_0\), a logistic \(u_0\), \(n=%d\) and '
      r'\(%d\) replications gives \(n\operatorname{var}(\widehat\theta)=%.2f\) for the optimal tendency, against '
      r'the bound \(1/\{\mathcal J(f_u)-1\}=%.2f\). Code: \texttt{dual\_info.py}.' % (mc['n'], mc['reps'], mc['n_var'], mc['bound']),
      r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
write('dual_info.tex', L)

# ---- simulations ----------------------------------------------------------------------------
S = ld('dual_check.json')
LAB = {'VT': 'VT', 'u_bounded': 'Error-side, bounded', 'u_sieve': 'Error-side, sieve',
       's_four': 'Mean tendencies', 'both': 'Both (eight moments)'}
DES = {'A: s Gaussian, u lognormal': r'A: Gaussian \(s_0\), lognormal \(u_0\)',
       'B: s Gaussian, u t5': r'B: Gaussian \(s_0\), \(t_5\) \(u_0\)',
       'C: s chi2(4), u Gaussian': r'C: \(\chi^2_4\) \(s_0\), Gaussian \(u_0\)',
       'D: s chi2(4), u t5': r'D: \(\chi^2_4\) \(s_0\), \(t_5\) \(u_0\)'}
L = [r'\begin{table}[!h]', r'\centering', r'\small',
     r'\caption{Error-side tendencies in simulated designs: rejection rates}', r'\label{tab:dual_sim}',
     r'\begin{threeparttable}', r'\begin{tabular}{llccc}', r'\toprule',
     r'Design & Moments & \(b=1\) (truth) & \(b=0.9\) & \(b=1.1\) \\', r'\midrule']
for i, (dk, dl) in enumerate(DES.items()):
    if i:
        L.append(r'\midrule')
    for j, k in enumerate(['VT', 'u_bounded', 'u_sieve', 's_four', 'both']):
        v = S[dk][k]
        L.append(r'%s & %s & \(%.3f\) & \(%.3f\) & \(%.3f\) \\' % (dl if j == 0 else '', LAB[k], *v))
L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}', r'\footnotesize',
      r'\item \textit{Notes:} \(y=2+x+0.3w+u_0\), \(x=1+0.5w+s_0+0.4u_0\), with \(s_0\), \(u_0\) and \(w\) '
      r'independent and standardized; \(n=5{,}000\) in \(100\) clusters of \(50\); \(200\) replications. '
      r'Rejection at the \(5\%\) level with the Hotelling cutoff and the delete-cluster jackknife covariance. '
      r'Code: \texttt{dual\_tendency.py check}.', r'\end{tablenotes}', r'\end{threeparttable}', r'\end{table}']
write('dual_sim.tex', L)
