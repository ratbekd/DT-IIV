"""Endogeneity correlation rho_xu = -sin(theta) (port of iiv_rho_xu.R): estimate and
robust set for rho_xu, IIV sets under prior domains for rho_xu, and the smallest
|rho_xu| the IIV restriction admits. Usage: python iiv_rho_xu.py   (writes iiv_rho_xu.csv)"""
import numpy as np, pandas as pd
from scipy.stats import chi2
from _setup import out
from iiv.data import load
from iiv.core import Context

priors = {'labelling': (-0.707, 0.707), 'abs_le_0.3': (-0.3, 0.3), 'positive': (0, 0.707), 'negative': (-0.707, 0)}
thg = np.linspace(-np.pi / 4, np.pi / 4, 401); rows = []
for ds in ['card', 'mroz', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']:
    C = Context(load(ds)); P = C.prep(); rxu = -np.sin(thg); bet = C.beta_at(thg, P)
    for nm, cols in {'one_sided': [2, 4], 'two_sided': [1, 2]}.items():
        th_hat = thg[int(np.argmin([C.Q_u(a, P, cols) for a in thg]))]
        acc = np.array([C.Q_b(b, P, cols) for b in bet]) <= chi2.ppf(0.95, len(cols))
        for pn, (lo, hi) in priors.items():
            inD = (rxu >= lo) & (rxu <= hi); a = acc & inD
            rows.append(dict(dataset=ds, set=nm, prior=pn, ols=P.b, rho_xu_hat=-np.sin(th_hat), beta_hat=C.beta_at(th_hat, P),
                             rho_lo=rxu[a].min() if a.any() else np.nan, rho_hi=rxu[a].max() if a.any() else np.nan,
                             beta_lo=bet[a].min() if a.any() else np.nan, beta_hi=bet[a].max() if a.any() else np.nan,
                             min_abs_rho=np.abs(rxu[a]).min() if a.any() else np.nan,
                             prior_only_lo=bet[inD].min(), prior_only_hi=bet[inD].max()))
    print('done', ds)
pd.DataFrame(rows).to_csv(out('iiv_rho_xu.csv'), index=False)
