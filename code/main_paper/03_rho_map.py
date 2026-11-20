"""Table 6: sets on the scale of the endogeneity correlation rho_xu = -sin(theta).
Inverts the one-sided test over |rho_xu| < 1 (beyond the labelling domain) and maps
benchmarks and published sets onto the rho scale."""
import os
import numpy as np
from scipy.stats import chi2
from _init import outdir, Tee
from data import APPS, iv_estimate
from oneside import OneSided

out = outdir('main_paper'); log = Tee(os.path.join(out, 'table6_rho_map.txt'))


def comps(r, b, acc):
    res, i = [], 0
    while i < len(r):
        if acc[i]:
            j = i
            while j + 1 < len(r) and acc[j + 1]:
                j += 1
            res.append((r[i], r[j], b[i], b[j])); i = j + 1
        else:
            i += 1
    return res


for name, f in APPS.items():
    D = f(); m = OneSided(D['y'], D['x'], D['W'], D['cl'])
    rho = np.linspace(-0.995, 0.995, 2001)
    b = m.bols - np.tan(np.arcsin(rho)) * m.sr / m.sx
    S = np.array([m.S(bb) for bb in b])
    np.save(os.path.join(out, f'rho_scan_{name}.npy'), np.column_stack([rho, b, S]))
    biv = iv_estimate(D['y'], D['x'], D['W'], D['Z'])
    r = lambda bb: -np.sin(np.arctan((bb - m.bols) * m.sx / m.sr))
    log(f"== {name}: OLS {m.bols:.4f}, sigma_r/sigma_x {m.sr / m.sx:.4f}, 2SLS {biv:.3f} -> rho {r(biv):+.3f}")
    acc = S <= chi2.ppf(.95, 2)
    for r0, r1, b0, b1 in comps(rho, b, acc):
        tag = 'inside' if max(abs(r0), abs(r1)) <= 1 / np.sqrt(2) + 1e-9 else 'beyond labelling domain'
        log(f"   one-sided accepted: rho [{min(r0, r1):+.3f}, {max(r0, r1):+.3f}]  beta [{min(b0, b1):.3f}, {max(b0, b1):.3f}]  ({tag})")
    i = int(np.argmin(S)); log(f"   min S = {S[i]:.2f} at rho {rho[i]:+.3f}; S at OLS = {m.S(m.bols):.2f}")
