"""Bootstrap shares of resamples with 0 / 1 / 2+ VT roots in |theta| <= pi/4
(port of apps_roots_r47.R). Pairs or cluster resamples; seed 20261016.
Usage: python apps_roots.py <dataset> [B]"""
import sys, numpy as np
import _setup
from iiv.data import load
from iiv.core import Context, coef_P, polyroot_real, cluster_draws, boot_prep

ds = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 999
C = Context(load(ds))


def nroots(W, y, x):
    d = boot_prep(W, y, x); tau = polyroot_real(coef_P(d['xw'], d['rw'])); tau = tau[np.abs(tau) <= 1]
    return tau, d


tau, d = nroots(C.W1, C.y, C.x)
print('====', C.lab, '; full sample roots in |tau| <= 1:', len(tau), '; beta:',
      ', '.join('%.3f' % v for v in d['b'] + tau * d['k']))
res = np.array([(len(t), np.sum(t < 0), np.sum(t > 0)) for t in
                (nroots(C.W1[i], C.y[i], C.x[i])[0] for i in cluster_draws(C, B, 20261016))])
print('resamples %d: zero %.1f%%  one %.1f%%  two+ %.1f%% | draws with a root of each sign: neg %.1f%% pos %.1f%%' % (
    B, 100 * np.mean(res[:, 0] == 0), 100 * np.mean(res[:, 0] == 1), 100 * np.mean(res[:, 0] >= 2),
    100 * np.mean(res[:, 1] > 0), 100 * np.mean(res[:, 2] > 0)))
