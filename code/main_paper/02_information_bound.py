"""Section 5.2, text after Proposition 5.4: the information bound 1 - 1/J(f) under CH
for Student t5 and logistic s0, against the index of the powers, the sieve and the
bounded basis, and of the location score itself, in samples of 400,000."""
import os
import numpy as np
from _init import outdir, Tee
from opttend import Tend, B_poly, B_sieve, B_bounded
from draws import draw_s

n = 400000
log = Tee(os.path.join(outdir('main_paper'), 'sec52_information_bound.txt'))
nu = 5; c = np.sqrt((nu - 2) / nu); a = np.sqrt(3) / np.pi
cases = {
    't5': (lambda s: ((nu + 1) * (s / c) / (nu + (s / c) ** 2) / c)[:, None], (nu + 1) / (nu + 3) / c ** 2),
    'logis': (lambda s: (np.tanh(s / (2 * a)) / a)[:, None], (1 / 3) / a ** 2),
}
rng = np.random.default_rng(5)
for kind, (psi, J) in cases.items():
    s0 = draw_s(kind, n, rng); u0 = rng.standard_normal(n); w = rng.standard_normal(n)
    x = 2 + .5 * w + s0 + .5 * u0; y = 1 + x + .3 * w + u0
    W = np.column_stack([np.ones(n), w])
    idx = {nm: Tend(y, x, W, basis=B).info(1.0)[0]
           for nm, B in (('powers', B_poly), ('sieve', B_sieve), ('bounded', B_bounded), ('score psi', psi))}
    log(f"{kind:6s} theory 1-1/J = {1 - 1 / J:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in idx.items()))
