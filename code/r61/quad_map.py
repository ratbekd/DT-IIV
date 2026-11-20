"""Profile of the nonlinear-feedback statistic over the slope in synthetic data, to see
whether the slope remains identified once gamma is free."""
import sys, numpy as np
from multiprocessing import get_context
import quad_feedback as Q

if __name__ == '__main__':
    grid = np.linspace(0.4, 1.6, 25)
    for n, gamma0 in ((20000, 0.0), (20000, 0.25), (100000, 0.0), (100000, 0.25)):
        rng = np.random.default_rng(7)
        y, x, W, cl = Q.synth(n, 100, gamma0, rng)
        with get_context('spawn').Pool(10, initializer=Q._init, initargs=((y, x, W, cl),)) as pool:
            res = pool.map(Q._prof, list(grid))
        c = Q.cuts(100)
        Tmin = np.array([r[0] for r in res]); T0 = np.array([r[2] for r in res]); g = np.array([r[1] for r in res])
        print('n %6d gamma0 %.2f | linear set %-22s | quadratic projection set %-22s profiled %s' % (
            n, gamma0, Q.fmt(Q.comps(T0 <= c['chi_proj'], grid)), Q.fmt(Q.comps(Tmin <= c['chi_proj'], grid)),
            Q.fmt(Q.comps(Tmin <= c['chi_prof'], grid))), flush=True)
        print('      beta : ' + ' '.join('%6.2f' % b for b in grid[::2]))
        print('      minT : ' + ' '.join('%6.1f' % v for v in Tmin[::2]))
        print('      g-hat: ' + ' '.join('%6.2f' % v for v in g[::2]), flush=True)
