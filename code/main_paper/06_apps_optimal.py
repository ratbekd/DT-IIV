"""Table 9: score tests with optimal tendencies in seven applications.
For each sample: the one-sided set (S, powers), the KJ sets with the powers,
sieve and bounded bases, and the corrected information index and Delta_80
at the OLS slope."""
import os
import numpy as np
from _init import outdir, Tee
from data import APPS
from ktest import KTest, B_poly, B_sieve, B_bounded, accept, intervals, info_index
from oneside import fmt

log = Tee(os.path.join(outdir('main_paper'), 'table9_apps_optimal.txt'))
for name, f in APPS.items():
    D = f()
    G = len(np.unique(D['cl'])) if D['cl'] is not None else len(D['y'])
    grid = None
    log(f"== {name}: n={len(D['y'])} G={G}")
    for nm, B in (('powers', B_poly), ('sieve', B_sieve), ('bounded', B_bounded)):
        mo = KTest(D['y'], D['x'], D['W'], D['cl'], basis=B)
        if grid is None:
            grid = mo.bgrid(1441); log(f"   OLS {mo.bols:.4f}")
        vals, aS, aK, aKJ = accept(mo, grid)
        I, d80 = info_index(mo, mo.bols)
        log(f"   {nm:8s} I={I:+.4f} Delta80={d80:.3f} | S: {fmt(intervals(grid, aS))} | "
            f"K: {fmt(intervals(grid, aK))} | KJ: {fmt(intervals(grid, aKJ))}")
