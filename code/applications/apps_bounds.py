"""Banerjee-Iyer outcome bounds: shares of the outcome at 0 and 1, skewness of the
OLS residual in the full sample and inside the bounds, and the exchange rate
(port of apps_bounds_r47.R). Usage: python apps_bounds.py"""
import numpy as np
import _setup
from iiv.data import load
from iiv.core import Context

C = Context(load('banerjee')); P = C.prep(); y, x = C.y, C.x
inside = (y > 0) & (y < 1)
sk = lambda v: np.mean((v - v.mean()) ** 3) / np.std(v, ddof=1) ** 3
print('skewness of OLS residual: all %.3f | interior obs %.3f (n %d) | share at 0 %.3f at 1 %.3f' % (
    sk(P.r), sk(P.r[inside]), inside.sum(), np.mean(y == 0), np.mean(y == 1)))
Q = np.linalg.qr(C.W1[inside])[0]
a = x[inside] - Q @ (Q.T @ x[inside]); yy = y[inside] - Q @ (Q.T @ y[inside]); b = np.sum(a * yy) / np.sum(a ** 2)
print('interior subsample re-estimated: skewness of OLS residual %.3f, skewness of residualized regressor %.3f' % (
    sk(yy - b * a), sk(a)))
print('exchange rate sr/sx %.3f' % (P.sr / P.sx))
