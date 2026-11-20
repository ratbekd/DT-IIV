"""Nonlinear feedback (eta) and quadratic outcome (kappa) checks in the seven applications.

For each dataset: the linear model, min over eta (quad_feedback.py) and min over kappa
(curved_outcome.py) of the jackknife Wald statistic of the four mean tendencies, on 61
slopes with |theta| <= pi/4 around OLS. Samples without clusters use the delete-group
jackknife over NGRP random groups (valid under independence).

Usage: python apps_misspec.py [cores] [datasets...]    writes output/r61/misspec_<dataset>.json
"""
import sys, os, json, time
import numpy as np
from iiv.data import load
import quad_feedback as QF
import curved_outcome as CO

NGRP = 50
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'r61')


def main():
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dss = sys.argv[2:] or ['mroz', 'card', 'star', 'maim5v', 'maim4v', 'banerjee', 'adh']
    for ds in dss:
        t0 = time.time()
        D = load(ds)
        y, x, W = D['y'], D['x'], D['W1']
        if D['cl'] is None:
            cl = np.random.default_rng(20261114).permutation(np.arange(len(y)) % NGRP); cltype = 'random groups'
        else:
            cl = np.unique(D['cl'], return_inverse=True)[1]; cltype = 'clusters'
        xt = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]; yt = y - W @ np.linalg.lstsq(W, y, rcond=None)[0]
        bols = xt @ yt / (xt @ xt); k = np.sqrt(np.mean((yt - bols * xt) ** 2) / np.mean(xt ** 2))
        grid = bols + np.tan(np.linspace(-np.pi / 4, np.pi / 4, 61)) * k
        G = len(np.unique(cl))
        print('==== %s: n %d, %d %s, OLS %.4f, distinct x %d ====' % (ds, len(y), G, cltype, bols, len(np.unique(x))), flush=True)
        print(' nonlinear feedback (eta):', flush=True)
        fb = QF.run(y, x, W, cl, grid, cores, ds)
        print(' quadratic outcome (kappa):', flush=True)
        cv = CO.run(y, x, W, cl, grid, cores, ds)
        json.dump(dict(dataset=ds, n=len(y), G=G, cltype=cltype, ols=float(bols), feedback=fb, curved=cv),
                  open(os.path.join(OUT, 'misspec_%s.json' % ds), 'w'), indent=1)
        print(' (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
