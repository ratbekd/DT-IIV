"""Table 7: Card, correlation between each instrument and the structural error
y - b x (after the controls) implied by slopes at the ends of the IIV sets."""
import os
import numpy as np
from _init import outdir, Tee
from data import card

log = Tee(os.path.join(outdir('main_paper'), 'table7_card_zu.txt'))
D = card(('nearc2', 'nearc4')); W = D['W']
P = lambda v: v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
yt, xt = P(D['y']), P(D['x']); Z = np.column_stack([P(D['Z'][:, 0]), P(D['Z'][:, 1])])
xh = Z @ np.linalg.lstsq(Z, xt, rcond=None)[0]
c = lambda a, b: np.corrcoef(a, b)[0, 1]
log(f"2SLS both instruments {np.sum(xh * yt) / np.sum(xh * xt):.4f}; corr(fitted, educ) {c(xh, xt):.3f}")
for j, nm in enumerate(('nearc2', 'nearc4')):
    log(f"{nm}: IV {np.sum(Z[:, j] * yt) / np.sum(Z[:, j] * xt):.4f}, corr(z, educ) {c(Z[:, j], xt):.3f}")
log("b        both    nearc2  nearc4")
for b in (-0.100, -0.036, 0.016, 0.020, 0.072, 0.093, 0.111, 0.137, 0.157):
    u = yt - b * xt
    log(f"{b:+.3f}  {c(xh, u):+.4f} {c(Z[:, 0], u):+.4f} {c(Z[:, 1], u):+.4f}")
