"""The common-factor failure case of the primitive model (main text, Subsection 3.4).

  eps = f + e,   x = a f + z,   y = beta0 x + eps,   f, e, z independent and centred.

The orthogonal remainder is v = x - lambda eps with lambda = cov(x, eps)/var(eps), and
E[eps | v] is not zero unless the components are Gaussian. For each design the script
  (1) computes the norm eps0 = ||E[u0 | s0]||_2 of the standardized pair by a binned
      regression in a sample of 4 million draws, with the part of that norm captured by
      the basis h = (tanh s, exp(-s^2/2)) and by the powers (s^2, s^3);
  (2) reports, in R samples of size n, the coverage of the true slope by the exact-MI
      sets (powers and bounded pair, chi2_2) and by the joint-region sets C_n(eps).

Usage: python ami_factor.py [R] [cores]      writes output/r61/ami_factor.json
"""
import sys, os, json
import numpy as np
from multiprocessing import get_context
from scipy.stats import chi2
import r61lib as L

BETA0 = 1.0
EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
NS = [500, 2000]
C5 = chi2.ppf(0.95, 5); C2 = chi2.ppf(0.95, 2)
# design: (label, law of f, law of e, law of z, loading a)
DESIGNS = [('skewed factor', 'chi', 'norm', 'norm', 1.0),
           ('skewed factor, strong loading', 'chi', 'norm', 'norm', 2.0),
           ('all components skewed', 'chi', 'chi', 'chi', 1.0),
           ('heavy-tailed factor', 't5', 'norm', 'norm', 1.0),
           ('Gaussian components', 'norm', 'norm', 'norm', 1.0)]


def rdraw(law, n, rng):
    if law == 'chi':
        return (rng.chisquare(4, n) - 4) / np.sqrt(8)
    if law == 't5':
        return rng.standard_t(5, n) / np.sqrt(5 / 3)
    return rng.standard_normal(n)


def gen(d, n, rng):
    _, lf, le, lz, a = DESIGNS[d]
    f, e, z = rdraw(lf, n, rng), rdraw(le, n, rng), rdraw(lz, n, rng)
    eps = f + e
    x = a * f + z
    return x, eps


def population(d, N=4_000_000, bins=400, seed=99):
    rng = np.random.default_rng(seed)
    x, eps = gen(d, N, rng)
    lam = np.mean(x * eps) / np.mean(eps ** 2)
    v = x - lam * eps
    s0 = v / v.std(); u0 = eps / eps.std()
    q = np.quantile(s0, np.linspace(0, 1, bins + 1)); q[0] -= 1; q[-1] += 1
    ib = np.clip(np.searchsorted(q, s0) - 1, 0, bins - 1)
    cnt = np.bincount(ib, minlength=bins); mu = np.bincount(ib, weights=u0, minlength=bins) / cnt
    # the squared norm of the binned mean is biased up by (number of bins)/N
    norm2 = max(np.sum(cnt * mu ** 2) / N - bins / N, 0.0)
    # variation of the conditional mean within bins (it matters in the tail bins): project
    # the within-bin residual on the within-bin parts of the powers and the bounded pair
    Hx = np.column_stack([L.fam_P(s0, deriv=False), L.fam_B(s0, deriv=False)])
    for j in range(Hx.shape[1]):
        Hx[:, j] -= (np.bincount(ib, weights=Hx[:, j], minlength=bins) / cnt)[ib]
    rres = u0 - mu[ib]
    mw = (rres[:, None] * Hx).mean(0)
    norm2 += float(mw @ np.linalg.solve(Hx.T @ Hx / N, mw))

    def captured(h):
        H = h - h.mean(0) - np.outer(s0, (s0[:, None] * h).mean(0))
        m = (u0[:, None] * H).mean(0); Q = H.T @ H / N
        return float(np.sqrt(max(m @ np.linalg.solve(Q, m), 0.0)))
    return dict(label=DESIGNS[d][0], lam=float(lam), rho_xu=float(np.corrcoef(x, eps)[0, 1]),
                eps0=float(np.sqrt(norm2)), eps_bounded=captured(L.fam_B(s0, deriv=False)),
                eps_powers=captured(L.fam_P(s0, deriv=False)))


def one(args):
    d, n, seed = args
    rng = np.random.default_rng(seed)
    x, eps = gen(d, n, rng)
    y = BETA0 * x + eps
    C = L.make_context(y, x, np.ones((n, 1))); P = C.prep()
    acc = np.zeros(2 + len(EPS), bool)
    acc[0] = L.analytic(C, P, BETA0, L.fam_P)[0] <= C2
    acc[1] = L.analytic(C, P, BETA0, L.fam_B)[0] <= C2
    t, V = L.ami_tV(C, P, BETA0)
    for j, e in enumerate(EPS):
        acc[2 + j] = L.ami_D(t, V, n, e, C5) <= C5
    return acc


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
    pop = [population(d) for d in range(len(DESIGNS))]
    cells = [(d, n) for d in range(len(DESIGNS)) for n in NS]
    ss = np.random.SeedSequence(20261112).spawn(len(cells) * R)
    jobs = [(d, n, ss[k * R + r]) for k, (d, n) in enumerate(cells) for r in range(R)]
    with get_context('spawn').Pool(cores) as pool:
        res = np.array(pool.map(one, jobs, chunksize=8)).reshape(len(cells), R, 2 + len(EPS))
    outp = dict(R=R, eps=EPS, pop=pop, cells=[])
    for p in pop:
        print('%-32s lambda %.3f  corr(x,eps) %.3f  norm %.3f  (bounded basis %.3f, powers %.3f)'
              % (p['label'], p['lam'], p['rho_xu'], p['eps0'], p['eps_bounded'], p['eps_powers']))
    print('coverage of beta0: exact powers, exact bounded | joint ' + ' '.join('%5g' % e for e in EPS))
    for k, (d, n) in enumerate(cells):
        c = res[k].mean(0)
        outp['cells'].append(dict(design=d, n=n, cover=c.tolist()))
        print('%-32s n %5d | %.3f %.3f | %s' % (DESIGNS[d][0], n, c[0], c[1], ' '.join('%.3f' % v for v in c[2:])))
    json.dump(outp, open(L.out('ami_factor.json'), 'w'), indent=1)
