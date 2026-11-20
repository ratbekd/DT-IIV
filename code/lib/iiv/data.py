"""Data blocks of the applications (port of the data section of indep_iiv.R).

load(ds) returns a dict with y, x, W1, Z, cl, lab, dd, z_vars, B_boot.

W1 follows indep_iiv.R: model.matrix(~ controls) with treatment-coded factors,
aliased columns dropped, and the non-constant columns standardized. Every
statistic in the package is invariant to that rescaling of the controls.
"""
import os
import numpy as np
import pandas as pd

DATA = os.environ.get('IIV_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data'))
# default: <package>/data, with this file in <package>/code/lib/iiv/


def _p(name):
    return os.path.join(DATA, name)


def _first(*names):
    """First of several accepted file names that exists in the data folder."""
    for nm in names:
        if os.path.exists(_p(nm)):
            return _p(nm)
    return _p(names[0])


def _read_dta(name):
    return pd.read_stata(_p(name), convert_categoricals=False)


def model_matrix(dd, w_vars):
    """Constant plus controls; object and category columns become treatment dummies
    (first level in sorted order dropped), as R's model.matrix does. Aliased
    columns are dropped in order, then non-constant columns are standardized."""
    cols = [np.ones(len(dd))]; names = ['(Intercept)']
    for v in w_vars:
        c = dd[v]
        if not pd.api.types.is_numeric_dtype(c):
            levels = sorted(pd.unique(c.astype(str)))
            for lv in levels[1:]:
                cols.append((c.astype(str) == lv).values.astype(float)); names.append(v + lv)
        else:
            cols.append(c.values.astype(float)); names.append(v)
    W = np.column_stack(cols)
    keep = []
    for j in range(W.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(W[:, cand]) == len(cand):
            keep.append(j)
    W = W[:, keep]
    rest = W[:, 1:]
    rest = (rest - rest.mean(0)) / rest.std(0, ddof=1)
    return np.column_stack([np.ones(len(dd)), rest]), [names[j] for j in keep]


def _complete(dd, cols):
    return dd.dropna(subset=cols).reset_index(drop=True)


def load(ds):
    cl = None
    B_boot = 999
    if ds == 'card':
        w = ['exper', 'expersq', 'black', 'smsa', 'south', 'smsa66'] + ['reg66%d' % i for i in range(2, 10)] + ['married1']
        dd = pd.read_csv(_p('card.csv'))
        dd = dd[dd.married.notna()].reset_index(drop=True)
        dd['married1'] = (dd.married == 1).astype(float)
        y, x = dd.lwage.values, dd.educ.values
        z = ['nearc2', 'nearc4']; lab = 'Card: lwage on educ'
    elif ds == 'mroz':
        w = ['educ', 'age', 'kidslt6', 'kidsge6', 'nwifeinc']
        dd = pd.read_csv(_first('mroz.csv', 'mroz_clean.csv'))
        dd['lwage'] = pd.to_numeric(dd.lwage, errors='coerce')
        dd = dd[(dd.inlf == 1) & dd.lwage.notna()].reset_index(drop=True)
        y, x = dd.hours.values / 1000, dd.lwage.values
        z = ['exper', 'expersq']; lab = 'Mroz: hours (thousands) on lwage'
    elif ds in ('banerjee', 'banerjee_int'):
        w = ['alt', 'lat', 'totrain', 'coastal', 'so_black', 'so_red', 'so_all', 'year']
        dd = pd.read_csv(_first('banerjee_iyer.csv', 'banerjee.csv'), low_memory=False)
        dd = dd[dd.p_nland.notna() & (dd.year >= 1965) & (dd.phwht <= 1)]
        dd = _complete(dd, ['phwht', 'instru'] + w)
        if ds == 'banerjee_int':                     # outcome strictly inside its bounds
            dd = dd[(dd.phwht > 0) & (dd.phwht < 1)]
        y, x, cl = dd.phwht.values, dd.p_nland.values, dd.district.values
        z = ['instru']; lab = 'Banerjee-Iyer: phwht on p_nland' + (' (0 < y < 1)' if ds == 'banerjee_int' else '')
    elif ds in ('ditella', 'ditella_cl'):
        w = ['mostSeriousCrime', 'age', 'ageSquared', 'argentine', 'numberPreviousImprisonments',
             'judicialDistrict', 'yearOfImprisonment']
        z = ['judgeAlreadyUsedEM', 'percJudgeSentToEM']
        dd = _read_dta('ditella.dta')
        dd = dd[dd.offendersPerJudge > 9]
        dd = _complete(dd, ['recidivism', 'electronicMonitoring'] + w + z)
        y, x = dd.recidivism.values, dd.electronicMonitoring.values
        if ds == 'ditella_cl':
            cl = dd.court.values
        lab = 'Di Tella-Schargrodsky: recidivism on electronicMonitoring'
    elif ds in ('burde_dist', 'burde_enr'):
        w = ['headchild', 'age', 'yrsvill', 'farsi', 'tajik', 'farmers', 'agehead', 'educhead', 'nhh',
             'land', 'sheep', 'chagcharan']
        z = ['buildschool']
        dd = _read_dta('burde_linden.dta').reset_index(drop=True)
        y = dd.testscore.values
        x = (dd.distschool if ds == 'burde_dist' else dd.enrolled).values
        lab = 'Burde-Linden: testscore on ' + ('distschool' if ds == 'burde_dist' else 'enrolled')
    elif ds == 'star':
        s = pd.read_csv(_first('star.csv', 'star_mlmrev.csv'))
        k = s[s.gr == 'K'].copy()
        k['csize'] = k.groupby('tch').id.transform('count').astype(float)
        k = k[(k.csize >= 8) & (k.csize <= 30) & k.read.notna() & k.math.notna()].copy()
        k['score'] = (k.read + k.math) / 2
        k['small'] = (k.cltype == 'small').astype(float)
        k['aide'] = (k.cltype == 'reg+A').astype(float)
        k['girl'] = np.where(k.sx.isna(), np.nan, (k.sx == 'F').astype(float))
        k['white'] = k.eth.isin(['W', 'A']).astype(float)            # NA ethnicity -> 0, as %in% does
        k['freelunch'] = np.where(k.ses.isna(), np.nan, (k.ses == 'F').astype(float))
        k['tblack'] = np.where(k.trace.isna(), np.nan, (k.trace == 'B').astype(float))
        k['masters'] = np.where(k.hdeg.isna(), np.nan, (k.hdeg != 'BS/BA').astype(float))
        k['school'] = k.sch.astype(str)
        dd = _complete(k, ['exp', 'girl', 'white', 'freelunch', 'tblack', 'masters'])
        w = ['school', 'girl', 'white', 'freelunch', 'tblack', 'exp', 'masters']
        z = ['small', 'aide']
        y, x, cl = dd.score.values, dd.csize.values, dd.tch.astype(str).values
        lab = 'STAR kindergarten: score on class size'
    elif ds in ('maim5v', 'maim5m', 'maim4v', 'maim4m'):
        f = 'final5.dta' if ds[4] == '5' else 'final4.dta'
        yv = 'avgverb' if ds[5] == 'v' else 'avgmath'
        dd = _read_dta(f)
        dd['avgverb'] = np.where(dd.avgverb > 100, dd.avgverb - 100, dd.avgverb)
        dd['avgmath'] = np.where(dd.avgmath > 100, dd.avgmath - 100, dd.avgmath)
        dd.loc[dd.verbsize == 0, 'avgverb'] = np.nan
        dd.loc[dd.mathsize == 0, 'avgmath'] = np.nan
        dd['func1'] = dd.c_size / (np.floor((dd.c_size - 1) / 40) + 1)
        dd = dd[(dd.classize > 1) & (dd.classize < 45) & (dd.c_size > 5) & (dd.c_leom == 1) & (dd.c_pik < 3)
                & dd[yv].notna() & dd.tipuach.notna()].reset_index(drop=True)
        w = ['tipuach', 'c_size']; z = ['func1']
        y, x, cl = dd[yv].values, dd.classize.values.astype(float), dd.schlcode.astype(str).values
        lab = 'Maimonides grade %s %s on class size' % (ds[4], yv)
    elif ds in ('adh', 'adh_t99', 'adh_t95'):
        w = ['l_shind_manuf_cbp', 'l_sh_popedu_c', 'l_sh_popfborn', 'l_sh_empl_f', 'l_sh_routine33',
             'l_task_outsource', 'reg_midatl', 'reg_encen', 'reg_wncen', 'reg_satl', 'reg_escen', 'reg_wscen',
             'reg_mount', 'reg_pacif', 't2']
        z = ['d_tradeotch_pw_lag']
        dd = _read_dta('workfile_china.dta')
        if ds != 'adh':
            q = 0.99 if ds == 'adh_t99' else 0.95
            dd = dd[dd.d_tradeusch_pw <= dd.d_tradeusch_pw.quantile(q)]   # R type-7 quantile
        dd = dd.reset_index(drop=True)
        y, x, cl = dd.d_sh_empl_mfg.values, dd.d_tradeusch_pw.values, dd.statefip.astype(str).values
        lab = 'ADH: change in manufacturing share on import exposure (unweighted)'
    elif ds == 'ae':
        w = ['agem1', 'black', 'hispan', 'othrace']; z = ['samesex']
        dd = pd.read_csv(_p('ae.csv'))
        y, x = dd.weeksm1.values, dd.morekids.values
        B_boot = 199; lab = 'Angrist-Evans: weeksm1 on morekids'
    else:
        raise ValueError('unknown dataset ' + ds)
    W1, W1_names = model_matrix(dd, w)
    Z = dd[z].values.astype(float)
    return dict(y=np.asarray(y, float), x=np.asarray(x, float), W1=W1, Z=Z,
                cl=None if cl is None else np.asarray(cl).astype(str), lab=lab, dd=dd,
                z_vars=z, w_vars=w, B_boot=B_boot, W1_names=W1_names)
