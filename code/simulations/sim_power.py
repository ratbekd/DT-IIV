"""Monte Carlo evidence for the sets between the one-sided and the CH sets
(supplement, Subsection S5.7, Table S17): coverage and power as the bound c on
corr(u^2, s) and corr(u^2, s^2) varies.

Usage: python sim_power.py [replications]
Writes output/simulations/sim_power_bnd.csv and prints the table.
"""
import sys, numpy as np, pandas as pd
from _setup import out
from iiv.core import Context
from iiv.sim import replicate
from iiv import power

BETA = 1.0


def chi4(rng, n):
    return (rng.chisquare(4, n) - 4) / np.sqrt(8)


def make(y, x, W, cl=None):
    return dict(y=y, x=x, W1=W, Z=W[:, 1:2], cl=cl, lab='sim', W1_names=None)


# ---------------------------------------------------------------- bounded variance tendencies
C_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, np.inf)


def dgp_bnd(rng, n, het):
    w = rng.standard_normal(n); s0 = chi4(rng, n); e = rng.standard_normal(n)
    u0 = e * np.sqrt((1 + het * (s0 ** 2 - 1)) / 1) if het else e
    u0 = u0 / np.sqrt(np.mean(u0 ** 2)) if het else u0
    x = 2 + 0.5 * w + s0 + 0.5 * u0; y = 1 + BETA * x + 0.3 * w + u0
    return w, s0, u0, x, y


def bnd_truth(het, N=2_000_000, seed=5):
    """Population corr(u0^2, s0) and corr(u0^2, s0^2) in the design."""
    _, s0, u0, _, _ = dgp_bnd(np.random.default_rng(seed), N, het)
    u2 = u0 ** 2
    return np.corrcoef(u2, s0)[0, 1], np.corrcoef(u2, s0 ** 2)[0, 1]


def rep_bnd(het, n, d_alt):
    def f(rng):
        w, s0, u0, x, y = dgp_bnd(rng, n, het)
        C = Context(make(y, x, np.column_stack([np.ones(n), w]))); P = C.prep()
        Z = rng.standard_normal((4000, 4)); o = {}
        for c in C_GRID:
            S, cr = power.bounded_test(C, P, BETA, c, Z); o['cover_%g' % c] = float(S <= cr)
            pw = []
            for sgn in (-1, 1):
                S, cr = power.bounded_test(C, P, BETA + sgn * d_alt, c, Z); pw.append(S > cr)
            o['power_%g' % c] = np.mean(pw)
        return o
    return f


def run_bnd(R):
    rows = []
    for het in (0.0, 0.25, 0.5):
        rho = bnd_truth(het)
        for n in (500, 1000):
            r = pd.DataFrame(replicate(rep_bnd(het, n, 0.1), R, 21 + n + int(100 * het))).mean()
            for c in C_GRID:
                rows.append(dict(het=het, rho_s=rho[0], rho_s2=rho[1], n=n, c=c, cover=r['cover_%g' % c], power=r['power_%g' % c]))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    pd.set_option('display.width', 200)
    t = run_bnd(R); t.to_csv(out('sim_power_bnd.csv'), index=False)
    print('\n== bounded variance tendencies, %d replications ==' % R); print(t.round(3).to_string(index=False))
