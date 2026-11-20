import numpy as np
from scipy.stats import chi2
from oneside import OneSided, invert, fmt

def design(d, ctrl):
    cols = [c for c in ctrl if d[c].std() > 0]
    W = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    # drop collinear columns
    keep = [0]
    for j in range(1, W.shape[1]):
        if np.linalg.matrix_rank(W[:, keep + [j]]) == len(keep) + 1: keep.append(j)
    return W[:, keep]

def minkowski(setlists, w):
    out = [(0.0, 0.0)]
    for sets, wc in zip(setlists, w):
        out = [(a + wc*min(l, h), b + wc*max(l, h)) for a, b in out for (l, h, *_) in sets]
    out.sort(); merged = []
    for a, b in out:
        if merged and a <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else: merged.append((a, b))
    return merged

def run_cells(d, yv, xv, ctrl, cellvar, labels, cl=None, alpha=.05, ngrid=1441):
    C = len(labels); m = 2
    mods, rows, wts = [], [], []
    for val, lab in labels.items():
        dc = d[d[cellvar] == val]
        W = design(dc, ctrl)
        mod = OneSided(dc[yv].values, dc[xv].values, W, None if cl is None else dc[cl].values)
        est, J = mod.cue()
        s95, _ = invert(mod.S, mod.bgrid(ngrid), chi2.ppf(1-alpha, m))
        lev = (1-alpha)**(1/C)
        sC, _ = invert(mod.S, mod.bgrid(ngrid), chi2.ppf(lev, m))
        mods.append(mod); wts.append(len(dc)/len(d))
        rows.append(dict(cell=lab, n=len(dc), ols=mod.bols, cue=est, Jp=1-chi2.cdf(J, m-1),
                         set95=s95, setC=sC))
    lo = max(md.bols - md.sr/md.sx for md in mods); hi = min(md.bols + md.sr/md.sx for md in mods)
    grid = np.linspace(lo, hi, ngrid)
    Sp = lambda b: sum(md.S(b) for md in mods)
    pooled, vals = invert(Sp, grid, chi2.ppf(1-alpha, m*C))
    i = int(np.argmin(vals)); H = vals[i]; bH = grid[i]
    return dict(rows=rows, pooled=pooled, H=H, bH=bH,
                p_robust=1-chi2.cdf(H, m*C), p_strong=1-chi2.cdf(H, m*C-1),
                avg=minkowski([r['setC'] for r in rows], wts), wts=wts, domain=(lo, hi))

def report(title, res):
    print("==", title, "== pooled domain [%.3f, %.3f]" % res['domain'])
    for r in res['rows']:
        print(f"  {r['cell']:12s} n={r['n']:5d} OLS {r['ols']:.3f} CUE {r['cue']:.3f} Jp {r['Jp']:.3f}"
              f"  C95: {fmt(r['set95'])}   C(adj): {fmt(r['setC'])}")
    print(f"  pooled set: {fmt(res['pooled'])}")
    print(f"  H = {res['H']:.2f} at b={res['bH']:.3f}; p robust (chi2_{2*len(res['rows'])}) = {res['p_robust']:.3f}; p strong = {res['p_strong']:.3f}")
    print("  weighted avg set:", " U ".join("[%.3f, %.3f]" % ab for ab in res['avg']), "weights", np.round(res['wts'], 3))
