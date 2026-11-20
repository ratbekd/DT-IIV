"""Finite-sample behaviour of the approximate-MI sensitivity sets (main text, Subsection 5.3).

Design of Table 2 with a controlled departure from mean independence:
  s0 = (chi2_4 - 4)/sqrt(8),  u0 = eps0 q(s0) + sqrt(1 - eps0^2) e,  e ~ N(0, 1),
  x = 2 + 0.5 w + s0 + 0.5 u0,  y = 1 + x + 0.3 w + u0,  w ~ N(0, 1),
where q has mean zero, unit variance and zero covariance with s0, so that s0 is the
orthogonal remainder, var(u0) = 1 and ||E[u0 | s0]||_2 = eps0 exactly.
  shape 'quad'  q proportional to s0^2 - 1 - kappa3 s0
  shape 'tanh'  q proportional to tanh(s0) - a - c s0   (close to the span of the basis)

For each replication the script evaluates, at the true slope and at beta0 +/- 0.1,
the exact-MI fixed-slope statistic of the pair (tanh s, exp(-s^2/2)), the joint
statistic D_n(b; eps) and the box statistic on a grid of tolerances.

Usage: python ami_sim.py [R] [cores]      writes output/r61/ami_sim.json
"""
import sys, os, json
import numpy as np
from multiprocessing import get_context
from scipy.stats import chi2
from scipy.integrate import quad
import r61lib as L

EPS0 = [0.0, 0.1, 0.2]
EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
SHAPES = ['quad', 'tanh']
NS = [500, 2000]
BETA0 = 1.0
SLOPES = [1.0, 0.9, 1.1]
C5 = chi2.ppf(0.95, 5); C2 = chi2.ppf(0.95, 2)


def _tanh_consts():
    pdf = lambda v: chi2.pdf(v, 4)
    sf = lambda v: (v - 4) / np.sqrt(8)
    E = lambda fn: quad(lambda v: fn(sf(v)) * pdf(v), 0, np.inf, limit=400)[0]
    a = E(np.tanh); c = E(lambda s: s * np.tanh(s))
    var = E(lambda s: (np.tanh(s) - a - c * s) ** 2)
    return a, c, np.sqrt(var)


TA, TC, TN = _tanh_consts()


def qfun(s, shape):
    if shape == 'quad':
        return (s ** 2 - 1 - np.sqrt(2) * s) / np.sqrt(3)
    return (np.tanh(s) - TA - TC * s) / TN


def one(args):
    n, shape, e0, seed = args
    rng = np.random.default_rng(seed)
    s0 = (rng.chisquare(4, n) - 4) / np.sqrt(8)
    u0 = e0 * qfun(s0, shape) + np.sqrt(1 - e0 ** 2) * rng.standard_normal(n)
    w = rng.standard_normal(n)
    x = 2 + 0.5 * w + s0 + 0.5 * u0
    y = 1 + x + 0.3 * w + u0
    C = L.make_context(y, x, np.column_stack([np.ones(n), w])); P = C.prep()
    acc = np.zeros((len(SLOPES), 1 + 2 * len(EPS)), bool)
    for i, b in enumerate(SLOPES):
        acc[i, 0] = L.analytic(C, P, b, L.fam_B)[0] <= C2
        t, V = L.ami_tV(C, P, b)
        for j, e in enumerate(EPS):
            acc[i, 1 + j] = L.ami_D(t, V, n, e, C5) <= C5
            acc[i, 1 + len(EPS) + j] = L.ami_box(t, V, n, e) <= C2
    return acc


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
    cells = [(n, sh, e0) for n in NS for sh in SHAPES for e0 in EPS0 if not (sh == 'tanh' and e0 == 0.0)]
    ss = np.random.SeedSequence(20261111).spawn(len(cells) * R)
    jobs = [(n, sh, e0, ss[k * R + r]) for k, (n, sh, e0) in enumerate(cells) for r in range(R)]
    with get_context('spawn').Pool(cores) as pool:
        res = pool.map(one, jobs, chunksize=8)
    res = np.array(res).reshape(len(cells), R, len(SLOPES), 1 + 2 * len(EPS))
    outp = dict(R=R, eps=EPS, slopes=SLOPES, consts=dict(tanh=[TA, TC, TN]), cells=[])
    print('norm of E[u0|s0] = eps0; coverage of beta0 and rejection at beta0 +/- 0.1; %d replications' % R)
    print('%5s %5s %5s | %6s | %s | %s' % ('n', 'shape', 'eps0', 'exact', 'joint: ' + ' '.join('%5g' % e for e in EPS),
                                           'box: ' + ' '.join('%5g' % e for e in EPS)))
    for k, (n, sh, e0) in enumerate(cells):
        cov = res[k, :, 0].mean(0); pw = 1 - res[k, :, 1:].mean((0, 1))
        outp['cells'].append(dict(n=n, shape=sh, eps0=e0, cover=cov.tolist(), power=pw.tolist()))
        print('%5d %5s %5.2f | %6.3f | %s | %s   coverage' % (n, sh, e0, cov[0], ' '.join('%5.3f' % v for v in cov[1:1 + len(EPS)]),
                                                             ' '.join('%5.3f' % v for v in cov[1 + len(EPS):])))
        print('%5s %5s %5s | %6.3f | %s | %s   rejection away' % ('', '', '', pw[0], ' '.join('%5.3f' % v for v in pw[1:1 + len(EPS)]),
                                                                 ' '.join('%5.3f' % v for v in pw[1 + len(EPS):])))
    json.dump(outp, open(L.out('ami_sim.json'), 'w'), indent=1)
