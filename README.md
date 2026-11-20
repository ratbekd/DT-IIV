# Replication package: An Intrinsic Instrumental Variable Method

Dzhumashev, R. and Tursunalieva, A. "An Intrinsic Instrumental Variable Method:
Using the Dual Tendency Condition for Coplanar Instruments" (MSIV_R59) and its
supplement.

All code is in Python. The scripts port the authors' earlier R code line by line.
Deterministic results (estimates, S-sets, J-tests, roots, standard errors) agree
with the R code to the printed precision. Bootstrap and Monte Carlo results use
NumPy random streams, so they differ draw by draw from the R runs and agree
within simulation error.

## Requirements

Python 3.10 or later with NumPy, SciPy and pandas (`requirements.txt`).
`pip install -r requirements.txt`.

## Running

From the package root:

    bash run_all.sh          # everything
    bash run_all.sh main     # main-paper Tables 2, 6-9 and Section 5.2
    bash run_all.sh apps     # application scripts
    bash run_all.sh sims     # semi-synthetic simulations

Each script can also run on its own from its folder, for example
`cd code/applications && python3 apps_sets.py card`. Printed results go to
`output/<folder>/`, with CSV tables beside them. On two cores, `main` takes
about 1.5 hours, `apps` about 1.5 hours and `sims` about 10 hours. Set
`IIV_CORES` to change the number of worker processes.

## Layout

    code/lib/iiv/        estimation library
      data.py            data blocks of the seven applications (load(dataset))
      core.py            rotation, moment sets, first-step correction, clustered
                         S statistic, F(nu) critical values, CUE, S-sets,
                         leave-one-cluster-out variance index, bounded moments
      sim.py             calibration of the semi-synthetic designs, error draws,
                         parallel replications
      dtk.py             closed-form VT roots used for the 401(k) and
                         Angrist-Evans samples
      dtcore.py          VT-IIV estimator core: sign search, CUE scale estimation,
                         corrected analytic variance, S-set, pairs bootstrap (BCa)
      dtcore_selftest.py self-tests of the core (python -m iiv.dtcore_selftest)
      power.py           sets between the one-sided and the CH sets: bounded
                         variance tendencies (supplement S5.7)
      texmacros.py       writer for the macros the prose quotes
    code/lib/            optimal tendencies, K test, cells (main paper, Section 5)
    code/main_paper/     Tables 2, 6, 7, 8, 9 and Section 5.2
    code/applications/   application results (main paper and supplement)
    code/simulations/    semi-synthetic simulations (main paper and supplement)
    data/                input data
    output/              results produced by run_all.sh

Dataset names used by the scripts: `mroz`, `card`, `star`, `maim5v`, `maim4v`,
`banerjee`, `adh` (and `maim5m`, `maim4m`, `adh_t99`, `adh_t95`).

## Where each result comes from

| Result | Script |
|---|---|
| Table 2 (optimal tendencies, Monte Carlo) | main_paper/01_mc_optimal.py |
| Section 5.2 (information bound) | main_paper/02_information_bound.py |
| Table 6 (one-sided set on the rho_xu scale) | main_paper/03_rho_map.py |
| Table 7 (Card, external instruments as z-u checks) | main_paper/04_card_zu.py |
| Table 8 (cell homogeneity) | main_paper/05_cells.py |
| Table 9 (applications, optimal tendency and K test) | main_paper/06_apps_optimal.py |
| Table 4 and the hierarchy of moment sets | applications/apps_sets.py, apps_moment_sets.py, apps_crit.py |
| Root screen, VT roots, standardized slopes | applications/apps_sets.py, apps_numbers.py, kurt_roots.py |
| Root stability under resampling | applications/apps_roots.py |
| Banerjee-Iyer outcome bounds | applications/apps_bounds.py |
| Banerjee-Iyer on the outcomes strictly inside the bounds | applications/bi_interior.py |
| Endogeneity correlation and prior domains | applications/apps_numbers.py, iiv_rho_xu.py |
| One-sided identification check | applications/onesided_ident.py |
| Group-covariance (variance index) moments | applications/apps_het.py |
| Bounded mean tendencies | applications/apps_rob.py |
| Cross-observation moment | applications/dep_check.py |
| Full estimator with bootstrap, per application | applications/indep_iiv.py |
| 401(k) application (pension macros) | applications/dtiiv_k401.py |
| Mroz: VT moment, roots, standard errors, bootstrap (supplement, Mroz table) | applications/mroz_dtiiv.py |
| Naive, corrected and bootstrap SEs (supplement, inference table: Mroz and Card columns); Card two-regressor table | applications/dtcore_apps.py |
| Monte Carlo of the VT-IIV estimator (supplement: simulation results, design parameters, two heteroscedasticity channels, OLS versus FGLS first stage) | simulations/sim_montecarlo.py |
| Sets between the one-sided and the CH sets, breakdown values (supplement S5.7, Proposition S5.1, Tables S17-S18) | simulations/sim_power.py, applications/power_apps.py, applications/power_tables.py |
| Semi-synthetic STAR designs (main paper) | simulations/sim_star.py |
| Designs calibrated to five applications (main paper) | simulations/sim_semi.py |
| Chi-squared and adjusted critical values (supplement) | simulations/sim_semi_crit.py |
| Bounded sets in calibrated designs (supplement) | simulations/sim_semi_rob.py |
| Variance index in calibrated designs (supplement) | simulations/sim_semi_het.py |
| Root screen in the STAR designs (supplement) | simulations/sim_screen.py |
| Group-covariance moments on STAR (supplement) | simulations/sim_star_het.py |
| 401(k) binary-outcome check (supplement) | simulations/sim_k401.py |
| Angrist-Evans checks (supplement) | simulations/sim_ae.py (needs ae.csv) |
| Extreme-tail cluster designs and full-refit jackknife calibration: dropped from the paper (the adjusted critical value covers as well for bounded moments); kept for the record | r61/tail_sim.py, r61/tail_report.py, r61/tail_apps.py |
| Approximate mean independence: sensitivity sets and entry tolerances in the applications (R61 main paper, Subsection 6.4; supplement S5.13) | r61/ami_apps.py |
| Approximate mean independence: coverage and power with a known conditional-mean norm (R61 main paper, Subsection 5.2) | r61/ami_sim.py |
| Common-factor model: size of the departure from mean independence and coverage (R61 supplement S5.13) | r61/ami_factor.py |
| Bias floor under approximate mean independence (R61 main paper, Remark 5.1; supplement S5.13, Tables S28-S29) | r61/bias_floor.py, r61/bias_floor_het.py |
| Score tests at matched size (R61 main paper, Table 3, columns "match.") | r61/mc_optimal_matched.py (reads the statistics saved by main_paper/01_mc_optimal.py) |
| Angrist-Krueger: quadratic schooling term, nonlinear feedback and degree step under the same mean tendencies (R61 supplement S5.10, Table S23) | r61/quad_feedback.py, r61/curved_outcome.py (modes `ak`, `akdeg`), r61/misspec_table.py |
| Nonlinear feedback and quadratic outcome term in the seven applications (R61 supplement S5.10, Table S24) | r61/apps_misspec.py, r61/misspec_table.py |
| Validation against known benchmarks: Thornton (2008) randomized distance and incentive, NHEFS with a weak price instrument (R61 supplement S5.11, Table S25) | r61/validation.py, r61/validation_table.py |
| Error-side tendencies E[s a(u)]: information bound, quadrature, Monte Carlo and the Angrist-Krueger residual (R61 main paper, Proposition prop:dual; supplement Table dual_info) | r61/dual_info.py |
| Error-side tendencies: simulated designs, Angrist-Krueger and the seven applications (R61 main paper, Table dual_apps; supplement Table dual_sim) | r61/dual_tendency.py (modes `check`, `ak`, `apps`), r61/dual_tables.py |
| Error-side tendencies in Angrist-Krueger with the degree step 1{educ = 16} profiled out (R61 supplement S5.12) | r61/dual_step.py |
| Angrist-Krueger: moment t statistics, error spread by schooling and the scale-adjusted error-side test, collinearity of the error-side moments (R61 supplement S5.12) | r61/dual_diag.py (modes `quick`, `collin`, `scaled`) |
| Angrist-Krueger: full VT diagnostics (not in the paper) | r61/ak_vt.py |
| LaTeX tables of the R61 additions | r61/make_tables.py, r61/mc_optimal_matched.py |

## R61 additions

The folder `code/r61/` holds the computations added in MSIV_R61. They use the
estimation library in `code/lib/iiv/` and write to `output/r61/`.

    bash run_all.sh r61      # about 1.5 hours on ten cores

or, from `code/r61/`:

    python3 selftest.py; python3 selftest_proj.py     # checks of r61lib.py
    python3 tail_sim.py 600; python3 tail_report.py   # six designs, 600 replications
    for d in mroz card star maim5v maim4v banerjee adh; do
      python3 tail_apps.py $d; python3 ami_apps.py $d; done
    python3 ami_sim.py 500; python3 ami_factor.py 500
    python3 bias_floor.py                             # quadrature, simulation, applications
    python3 make_tables.py <folder>/tables            # tail_*.tex, ami_*.tex, bias_floor.tex
    AK=<path>/NEW7080.dta                             # Angrist-Krueger checks, about 1 hour each
    python3 quad_feedback.py ak $AK; python3 curved_outcome.py ak $AK
    python3 curved_outcome.py akdeg $AK
    PYTHONPATH=../lib python3 apps_misspec.py 10      # seven applications, about 1.3 hours
    python3 misspec_table.py <folder>/tables          # ak_misspec.tex, apps_misspec.tex
    PYTHONPATH=../lib python3 validation.py 10        # Thornton and NHEFS, a few minutes
    python3 validation_table.py <folder>/tables       # validation.tex
    python3 dual_info.py $AK                          # information bound, about 15 minutes
    python3 dual_tendency.py check 200                # simulated designs, about 20 minutes
    python3 dual_tendency.py ak $AK 8; PYTHONPATH=../lib python3 dual_tendency.py apps 8
    python3 dual_step.py $AK 8                        # degree step, about 1 hour
    python3 dual_diag.py quick $AK; python3 dual_diag.py collin $AK; python3 dual_diag.py scaled $AK 8
    python3 dual_tables.py <folder>/tables            # dual_apps.tex, dual_info.tex, dual_sim.tex

`r61lib.py` contains the fixed pairs of mean tendencies (powers, bounded,
localized, and the median/MAD-scaled bounded pair), the analytic statistic with
the adjusted critical value, the full-refit delete-cluster jackknife with the
chi-squared and Hotelling F critical values, the linearized wild calibration,
the frozen-first-step diagnostic, and the joint statistic, covariance and
projection distance of the approximate mean-independence sets. `tail_apps.py`
for Card deletes each of the 3,003 observations in turn and takes about half an
hour. The random streams are NumPy `SeedSequence` streams with fixed seeds, one
per replication, so the results do not depend on the number of cores.

`MSIV_R61.tex` and `MSIV_R61_supplement.tex` read each other's `.aux` file
through `xr-hyper` for cross-document references: compile the main text, then
the supplement, then both again.

## Data

| File | Source | Used for |
|---|---|---|
| card.csv | Card (1995), NLS Young Men | Card |
| mroz.csv | Mroz (1987), PSID 1975 | Mroz |
| star.csv | Tennessee STAR, `mlmRev::star` | STAR |
| final4.dta, final5.dta | Angrist and Lavy (1999) | Maimonides |
| banerjee_iyer.csv | Banerjee and Iyer (2005) | Banerjee-Iyer |
| workfile_china.dta | Autor, Dorn and Hanson (2013) | ADH |
| thornton_hiv.csv | Thornton (2008), from the R package `causaldata` 0.1.4 (MIT licence) | validation |
| nhefs_complete.csv | NHEFS, Hernan and Robins (2020), from `causaldata` 0.1.4 | validation |
| k401.csv | Wooldridge `k401ksubs` | 401(k) |

Not included: `NEW7080.dta` (Angrist and Krueger 1991, from the Angrist data archive at MIT; pass its path to the Angrist-Krueger scripts), `ae.csv` (Angrist and Evans 1998), `ditella.dta` (Di Tella and
Schargrodsky 2013) and `burde_linden.dta` (Burde and Linden 2013). The loaders
for these samples are kept in `code/lib/iiv/data.py`; scripts that need them skip
them when the file is absent.

## Not included

`sim_montecarlo.py` writes the LaTeX fragments `tab_simulation_results`,
`tab_dgp_params`, `tab_het_channels`, `tab_fgls` and the prose macros
`sim_numbers.tex` to `output/simulations/tables/`. Copy them to a `tables/`
folder next to the .tex file to compile the supplement.
`python3 sim_montecarlo.py tables` rebuilds the fragments from the saved CSV
files without rerunning, and `python3 sim_montecarlo.py quick` runs a small
version for testing; its fragments are marked PRELIMINARY.

The coverage experiment (`tab_coverage`) comes from `IIV_coverage_mc.R`, which
was not available, so it is not ported. Its R output is archived, unchanged, in
`output/tables_R_core/`, together with the other R fragments the Python code
replaces. The R35 table emitters were not ported, because they format objects
from drivers that were not available. The Angrist-Evans tables (`tab_ae_main`,
`tab_ae_censoring`, `ae_numbers`), `tab_varshape` and `tab_outcome_scale` were
not supplied. The legacy estimator `sim_estimator_DT_IIV_v18.R` is superseded
by v21, which `dtcore.py` ports, and nothing in the paper depends on it.
