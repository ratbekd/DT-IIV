"""Brute-force check of the projection distance ami_D on random problems, with a
random search that concentrates around the solver's neighbourhood and globally."""
import numpy as np
import r61lib as L

rng = np.random.default_rng(7)
bad = 0; worst = 0.0; N = 150
for r in range(N):
    A = rng.standard_normal((5, 5)); V = (A @ A.T / 5 + 0.2 * np.eye(5)) * rng.choice([1.0, 0.1, 0.01])
    t = np.r_[rng.normal(0, rng.choice([0.05, 0.3]), 2), 1.0 + 0.2 * rng.standard_normal(),
              0.2 * rng.standard_normal(), 1.0 + 0.2 * rng.standard_normal()]
    eps = rng.choice([0.0, 0.05, 0.1, 0.3]); n = 500
    D = L.ami_D(t, V, n, eps); Vi = np.linalg.inv(V)
    K = 40000
    c11 = np.abs(rng.normal(1, 0.4, K)); c21 = rng.normal(0, 0.4, K); c22 = np.abs(rng.normal(1, 0.4, K))
    ang = rng.uniform(0, 2 * np.pi, K); rad = eps * rng.uniform(0, 1, K) ** 0.25
    a1, a2 = rad * np.cos(ang), rad * np.sin(ang)
    T = np.column_stack([c11 * a1, c21 * a1 + c22 * a2, c11 ** 2, c11 * c21, c21 ** 2 + c22 ** 2])
    E = t[None, :] - T
    lo = n * np.min(np.einsum('ki,ij,kj->k', E, Vi, E))
    if D > lo * (1 + 1e-6) + 1e-8:
        bad += 1; worst = max(worst, (D - lo) / max(lo, 1e-12))
        print('  miss: D %.5f random %.5f eps %.2f' % (D, lo, eps))
print('projection check: %d of %d problems where random search beat the solver (worst relative gap %.2e)'
      % (bad, N, worst))
