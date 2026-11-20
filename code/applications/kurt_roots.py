"""Kurtosis as a root-selection device (port of kurt_roots.R). For every VT root
on the circle: the corrected J of {VT, three kurtosis moments} minimised within
0.1 rad of the root (df = 3). Usage: python kurt_roots.py   (writes kurt_roots.csv)"""
import numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context, optimize, uniroot

rows = []
for ds in ['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']:
    C = Context(load(ds)); P = C.prep(); f = lambda a: C.gbar(a, P, [1])[0]
    th = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 2001); m1 = np.array([f(a) for a in th])
    for i in np.where(np.diff(np.sign(m1)) != 0)[0]:
        r = uniroot(f, th[i], th[i + 1])
        a, J = optimize(lambda a: C.Q_c(a, P, [1, 3, 4, 5]), max(r - 0.1, -np.pi / 2 + 1e-3), min(r + 0.1, np.pi / 2 - 1e-3))
        rows.append(dict(dataset=ds, t_root=np.tan(r), rho_xu=-np.sin(r), cor_sx=np.cos(r), beta_root=C.beta_at(r, P),
                         J=J, p=1 - chi2.cdf(J, 3), beta_local=C.beta_at(a, P)))
tab = pd.DataFrame(rows); tab.to_csv(out('kurt_roots.csv'), index=False)
print(tab.round(3).to_string(index=False))
