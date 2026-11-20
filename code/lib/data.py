"""Data loaders for all applications.

Each loader returns a dict with
    y, x   outcome and endogenous regressor (numpy arrays)
    W      control matrix with a constant, collinear columns removed
    cl     cluster identifiers, or None for heteroscedasticity-robust inference
    Z      external instrument(s) used as the comparison estimate (n x q)
    df     the estimation sample (pandas DataFrame)
and, where relevant, the cell variable used in the cell analysis.

The specifications follow Table 3 of the main paper.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def _path(name):
    return os.path.join(DATA, name)


def design(d, ctrl):
    """Constant plus the listed controls, dropping constant and collinear columns."""
    cols = [c for c in ctrl if d[c].std() > 0]
    W = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in cols])
    keep = [0]
    for j in range(1, W.shape[1]):
        if np.linalg.matrix_rank(W[:, keep + [j]]) == len(keep) + 1:
            keep.append(j)
    return W[:, keep]


def _out(d, y, x, ctrl, cl, Z):
    Zm = None if Z is None else np.asarray(d[Z].values, float)
    return dict(y=d[y].values.astype(float), x=d[x].values.astype(float), W=design(d, ctrl),
                cl=None if cl is None else d[cl].values, Z=Zm, df=d, ctrl=ctrl)


# ---------------------------------------------------------------- Card (1995)
CARD_CTRL = ['exper', 'expersq', 'black', 'smsa', 'south', 'smsa66', 'married1'] + \
            ['reg66%d' % i for i in range(2, 10)]


def card(instruments=('nearc2', 'nearc4')):
    """Log wage on schooling; 3,003 men with nonmissing marital status.
    Instruments: college proximity (nearc2, nearc4) or parental education."""
    d = pd.read_csv(_path('card.csv'))
    d = d.dropna(subset=['married'] + list(instruments)).reset_index(drop=True)
    d['married1'] = (d.married == 1).astype(float)      # married, living with spouse
    return _out(d, 'lwage', 'educ', CARD_CTRL, None, list(instruments))


# ---------------------------------------------------------------- Mroz (1987)
MROZ_CTRL = ['educ', 'age', 'kidslt6', 'kidsge6', 'nwifeinc']


def mroz():
    """Annual hours (thousands) of 428 working married women on the log wage.
    Instruments: experience and its square."""
    d = pd.read_csv(_path('mroz.csv'))
    d = d[d.inlf == 1].dropna(subset=['lwage']).reset_index(drop=True)
    d['hrs'] = d.hours / 1000
    return _out(d, 'hrs', 'lwage', MROZ_CTRL, None, ['exper', 'expersq'])


# ---------------------------------------------------------------- STAR (Krueger 1999)
def star():
    """Kindergarten STAR: mean of reading and math scores on class size.
    Class size is counted from teacher identifiers, keeping classes of 8-30 pupils.
    Controls: female, free lunch, white or Asian, teacher with a master's degree,
    teacher experience, white teacher, and school fixed effects. Pupils missing a
    control other than pupil ethnicity are dropped (n = 5,657 in 319 classes).
    Instruments: assignment to a small class and to a regular class with an aide."""
    s = pd.read_csv(_path('star.csv'))
    k = s[s.gr == 'K'].copy()
    k['csize'] = k.groupby('tch').id.transform('count').astype(float)
    k = k[(k.csize >= 8) & (k.csize <= 30)]
    k['yv'] = (k.read + k.math) / 2
    k = k.dropna(subset=['yv', 'sx', 'ses', 'hdeg', 'exp', 'trace']).reset_index(drop=True)
    X = pd.DataFrame({'female': k.sx == 'F', 'freelunch': k.ses == 'F',
                      'whiteasian': k.eth.isin(['W', 'A']),
                      'master': ~k.hdeg.isin(['BS/BA']), 'texp': k.exp,
                      'twhite': k.trace == 'W'}).astype(float)
    S = pd.get_dummies(k.sch.astype(str), prefix='sch', drop_first=True).astype(float)
    D = pd.concat([k[['yv', 'csize', 'tch', 'cltype', 'schtype']], X, S], axis=1)
    D['small'] = (D.cltype == 'small').astype(float)
    D['regaide'] = (D.cltype == 'reg+A').astype(float)
    D['rural'] = (D.schtype == 'rural').astype(int)
    return _out(D, 'yv', 'csize', list(X.columns) + list(S.columns), 'tch', ['small', 'regaide'])


# ---------------------------------------------------------------- Maimonides (Angrist and Lavy 1999)
def maimonides(grade):
    """Class-average verbal score on class size, grades 4 and 5, clustered by school.
    Controls: percent disadvantaged and enrollment. Instrument: Maimonides' rule.
    One grade-5 class average in the public file exceeds 100 (187.6); 100 is
    subtracted, as in the published replications."""
    f = {4: 'final4.dta', 5: 'final5.dta'}[grade]
    d = pd.read_stata(_path(f))
    d['avgverb'] = np.where(d.avgverb > 100, d.avgverb - 100, d.avgverb)
    d = d[(d.classize > 1) & (d.classize < 45) & (d.c_size > 5)].copy()
    d = d.dropna(subset=['avgverb', 'tipuach', 'c_size', 'classize'])
    d = d[(d.c_leom == 1) & (d.c_pik < 3)]
    d['fsc'] = d.c_size / (np.floor((d.c_size - 1) / 40) + 1)
    d['classize'] = d.classize.astype(float)
    d = d.reset_index(drop=True)
    st = d.groupby('schlcode').tipuach.mean()
    d['hi_dis'] = (d.schlcode.map(st) > st.median()).astype(int)
    return _out(d, 'avgverb', 'classize', ['tipuach', 'c_size'], 'schlcode', ['fsc'])


# ---------------------------------------------------------------- Banerjee and Iyer (2005)
BI_CTRL = ['alt', 'lat', 'totrain', 'coastal', 'so_black', 'so_red', 'so_all', 'year']


def banerjee_iyer():
    """Share of wheat area under high-yielding varieties on the non-landlord share,
    district-years from 1965 with a share of at most one, clustered by `district`.
    Instrument: the timing of British conquest (`instru`)."""
    b = pd.read_csv(_path('banerjee_iyer.csv'), low_memory=False)
    d = b[b.year >= 1965].dropna(subset=['instru', 'p_nland'] + BI_CTRL).copy()
    d['yv'] = np.where(d.awheat == 0, 0.0, d.hyvwheat / d.awheat.replace(0, np.nan))
    d = d[d.yv <= 1].reset_index(drop=True)
    return _out(d, 'yv', 'p_nland', BI_CTRL, 'district', ['instru'])


# ---------------------------------------------------------------- ADH (2013)
ADH_CTRL = ['t2', 'l_shind_manuf_cbp', 'l_sh_popedu_c', 'l_sh_popfborn', 'l_sh_empl_f',
            'l_sh_routine33', 'l_task_outsource', 'reg_midatl', 'reg_encen', 'reg_wncen',
            'reg_satl', 'reg_escen', 'reg_wscen', 'reg_mount', 'reg_pacif']


def adh():
    """Change in the manufacturing employment share on Chinese import exposure,
    controls of ADH Table 3 column 6 without population weights, clustered by state.
    Instrument: exposure in other high-income countries."""
    d = pd.read_stata(_path('workfile_china.dta')).reset_index(drop=True)
    return _out(d, 'd_sh_empl_mfg', 'd_tradeusch_pw', ADH_CTRL, 'statefip', ['d_tradeotch_pw_lag'])


APPS = {
    'Mroz': mroz,
    'Card': lambda: card(('nearc2', 'nearc4')),
    'STAR': star,
    'Maimonides4': lambda: maimonides(4),
    'Maimonides5': lambda: maimonides(5),
    'Banerjee-Iyer': banerjee_iyer,
    'ADH': adh,
}


def iv_estimate(y, x, W, Z):
    """2SLS slope with the controls partialled out."""
    P = lambda v: v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
    yt, xt = P(y), P(x)
    Zt = P(Z if Z.ndim > 1 else Z[:, None])
    xh = Zt @ np.linalg.lstsq(Zt, xt, rcond=None)[0]
    return float((xh @ yt) / (xh @ xt))
