"""Distributions of the exogenous component s0 used in the simulations (standardized)."""
import numpy as np


def draw_s(kind, n, rng):
    if kind == 'chi2':
        z = rng.chisquare(4, n); return (z - 4) / np.sqrt(8)
    if kind == 't5':
        z = rng.standard_t(5, n); return z / np.sqrt(5 / 3)
    if kind == 'gauss':
        return rng.standard_normal(n)
    if kind == 'mix':      # symmetric bimodal: equal mixture of N(+-1, 0.36)
        z = rng.choice([-1, 1], n) * 1.0 + 0.6 * rng.standard_normal(n); return z / np.sqrt(1.36)
    if kind == 'logis':
        a = np.sqrt(3) / np.pi; p = rng.uniform(size=n); return a * np.log(p / (1 - p))
    raise ValueError(kind)
