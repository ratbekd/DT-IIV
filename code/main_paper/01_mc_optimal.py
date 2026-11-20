"""Table 2: size and power of the fixed-slope tests (S with powers; KJ with powers,
sieve and bounded bases). Design of Subsection 5.2.

Usage: python 01_mc_optimal.py [reps=1000]
"""
import sys, os
from multiprocessing import Pool
import numpy as np
from scipy.stats import chi2
from _init import outdir, Tee
from ktest import KTest, B_poly, B_sieve, B_bounded
from draws import draw_s

BETA = 1.0
DELTAS = [-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
BASES = [('powers', B_poly, 2), ('sieve', B_sieve, 5), ('bounded', B_bounded, 4)]
DESIGNS = [(k, e) for k in ['chi2', 't5', 'mix', 'gauss'] for e in ['CH', 'HET']]


def gen(kind, err, n, rng):
    s0 = draw_s(kind, n, rng); e = rng.standard_normal(n)
    u0 = e if err == 'CH' else e * np.sqrt((1 + s0 ** 2) / 2)
    w = rng.standard_normal(n)
    x = 2 + 0.5 * w + s0 + 0.5 * u0
    y = 1 + BETA * x + 0.3 * w + u0
    return y, x, np.column_stack([np.ones(n), w])


def one(args):
    kind, err, n, rep = args
    y, x, W = gen(kind, err, n, np.random.default_rng(7919 * rep + 3))
    mods = [KTest(y, x, W, basis=B) for _, B, _ in BASES]
    return np.array([[v for mo in mods for v in mo.SKJ(BETA + d)] for d in DELTAS])


def rates(res, di, bi, m):
    S, K, J = res[:, di, 3 * bi], res[:, di, 3 * bi + 1], res[:, di, 3 * bi + 2]
    return (np.mean(S > chi2.ppf(.95, m)), np.mean(K > chi2.ppf(.95, 1)),
            np.mean((K > chi2.ppf(.96, 1)) | (J > chi2.ppf(.99, m - 1))))


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = outdir('main_paper'); log = Tee(os.path.join(out, 'table2_mc_optimal.txt'))
    log(f"Table 2. Rejection rates, {R} replications. Columns: S (powers), KJ (powers, sieve, bounded).")
    log("size = rate at delta=0; power = mean rate at delta=+-0.1")
    with Pool(os.cpu_count()) as p:
        for n in (500, 1000):
            for kind, err in DESIGNS:
                res = np.array(p.map(one, [(kind, err, n, r) for r in range(R)]))
                np.save(os.path.join(out, f'mc_optimal_{kind}_{err}_{n}.npy'), res)
                i0, ip, im = DELTAS.index(0.0), DELTAS.index(0.1), DELTAS.index(-0.1)
                cols = []
                for bi, (nm, _, m) in enumerate(BASES):
                    s0, p_, m_ = rates(res, i0, bi, m), rates(res, ip, bi, m), rates(res, im, bi, m)
                    if nm == 'powers':
                        cols.append(('S powers', s0[0], (p_[0] + m_[0]) / 2))
                    cols.append((f'KJ {nm}', s0[2], (p_[2] + m_[2]) / 2))
                log(f"n={n:5d} {kind:5s} {err:3s} | " + " | ".join(f"{c}: size {a:.3f} power {b:.3f}" for c, a, b in cols))
                for di, d in enumerate(DELTAS):
                    log(f"      delta={d:+.2f} " + " | ".join(
                        f"{nm}: S {rates(res, di, bi, m)[0]:.3f} K {rates(res, di, bi, m)[1]:.3f} KJ {rates(res, di, bi, m)[2]:.3f}"
                        for bi, (nm, _, m) in enumerate(BASES)))
