"""Summaries of tail_sim.npz: coverage at the true slope under each calibration, and
rejection at matched size with held-out cutoffs (first half estimates the 95th
percentile of each statistic at the truth, second half evaluates).

Writes output/r61/tail_report.json and prints the tables."""
import json
import numpy as np
from scipy.stats import chi2
import r61lib as L

Z = np.load(L.out('tail_sim.npz'))
st = Z['stats']; fams = list(Z['fams']); designs = list(Z['designs']); betas = Z['betas']
D, R = st.shape[:2]
half = R // 2
CAL = ['an_chi', 'an_adj', 'jk_chi', 'jk_F', 'wild', 'frozen']
c_chi = chi2.ppf(0.95, 2)


def scores(d, jf, ib):
    """Statistic of each calibration, scaled so that rejection is score > 1."""
    G = int(designs[d].split('-')[2])
    S, nu, T, pw, Tfr = (st[d, :, jf, ib, k] for k in range(5))
    cadj = np.array([L.crit_adj(G, 2, v) for v in nu]); cF = L.crit_F(G)
    return dict(an_chi=S / c_chi, an_adj=S / cadj, jk_chi=T / c_chi, jk_F=T / cF,
                wild=(1 - pw) / 0.95 + 1e-12 * (pw <= 0.05), frozen=Tfr / cF)


rep = dict(designs=designs, fams=fams, cal=CAL, R=int(R), nobs=Z['nobs'].mean(1).tolist(), cover={}, power={})
for d in range(D):
    for jf, f in enumerate(fams):
        s0 = scores(d, jf, 0)
        for c in CAL:
            rej0 = (s0[c] > 1) if c != 'wild' else (st[d, :, jf, 0, 3] <= 0.05)
            key = '%s|%s|%s' % (designs[d], f, c)
            rep['cover'][key] = float(1 - rej0.mean())
            # matched size: cutoff from the first half at the truth, evaluated on the second half
            sc0 = s0[c] if c != 'wild' else -st[d, :, jf, 0, 3]
            cut = np.quantile(sc0[:half], 0.95)
            row = dict(nominal_size=float(rej0.mean()), heldout_size=float((sc0[half:] > cut).mean()))
            for ib in (1, 2):
                sa = scores(d, jf, ib)
                sca = sa[c] if c != 'wild' else -st[d, :, jf, ib, 3]
                reja = (sa[c] > 1) if c != 'wild' else (st[d, :, jf, ib, 3] <= 0.05)
                row['nominal_%.1f' % betas[ib]] = float(reja.mean())
                row['matched_%.1f' % betas[ib]] = float((sca[half:] > cut).mean())
            rep['power'][key] = row
json.dump(rep, open(L.out('tail_report.json'), 'w'), indent=1)

print('Coverage at the true slope (nominal 0.95), %d replications; mean n per design: %s'
      % (R, ', '.join('%d' % v for v in rep['nobs'])))
print('%-16s %-3s ' % ('design', 'fam') + ' '.join('%7s' % c for c in CAL))
for d in range(D):
    for f in fams:
        print('%-16s %-3s ' % (designs[d], f) + ' '.join('%7.3f' % rep['cover']['%s|%s|%s' % (designs[d], f, c)] for c in CAL))
print('\nRejection of beta = 0.6 and 0.8 at matched size (held-out half), and held-out size')
print('%-16s %-3s %-7s %6s %6s %6s %6s %6s %6s' % ('design', 'fam', 'cal', 'size', 'h-size', 'n 0.6', 'm 0.6', 'n 0.8', 'm 0.8'))
for d in range(D):
    for f in fams:
        for c in ('an_chi', 'an_adj', 'jk_F'):
            r = rep['power']['%s|%s|%s' % (designs[d], f, c)]
            print('%-16s %-3s %-7s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' % (designs[d], f, c, r['nominal_size'], r['heldout_size'],
                  r['nominal_0.6'], r['matched_0.6'], r['nominal_0.8'], r['matched_0.8']))
