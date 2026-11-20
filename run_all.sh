#!/usr/bin/env bash
# Reproduces every result of the paper and its supplement. Run from the package root:
#   bash run_all.sh               everything (main-paper tables, applications, simulations)
#   bash run_all.sh main          Tables 2 and 6-9 and Section 5.2 of the main paper
#   bash run_all.sh apps          the application scripts (deterministic, plus bootstraps)
#   bash run_all.sh sims          the semi-synthetic simulations (several hours on 2 cores)
# Printed results go to output/<folder>/<script>[_<dataset>].txt; tables go to CSV files
# in the same folders. Set IIV_CORES to limit the number of worker processes.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; PART=${1:-all}
APPS="mroz card star maim5v maim4v banerjee adh"

main() {
  (cd "$ROOT/code/lib" && python3 -m iiv.dtcore_selftest)
  cd "$ROOT/code/main_paper"
  python3 02_information_bound.py; python3 04_card_zu.py; python3 03_rho_map.py
  python3 05_cells.py; python3 06_apps_optimal.py; python3 01_mc_optimal.py 1000
}

sims() {
  cd "$ROOT/code/simulations"; O="$ROOT/output/simulations"; mkdir -p "$O"
  python3 sim_star.py 500 > "$O/sim_star.txt"
  for d in maim4v maim5v adh; do python3 sim_semi.py $d 500 > "$O/sim_semi_$d.txt"; done
  for d in card banerjee;    do python3 sim_semi.py $d 300 > "$O/sim_semi_$d.txt"; done
  python3 sim_semi_crit.py star 400 > "$O/sim_semi_crit_star.txt"
  for d in card maim5v maim4v banerjee mroz adh; do python3 sim_semi_crit.py $d 1000 > "$O/sim_semi_crit_$d.txt"; done
  for d in adh card maim5v maim4v banerjee mroz; do python3 sim_semi_rob.py $d 500 > "$O/sim_semi_rob_$d.txt"; done
  python3 sim_semi_rob.py star 300 > "$O/sim_semi_rob_star.txt"
  for d in card banerjee maim5v maim4v; do python3 sim_semi_het.py $d 300 > "$O/sim_semi_het_$d.txt"; done
  python3 sim_semi_het.py adh 300 t2 > "$O/sim_semi_het_adh.txt"
  python3 sim_screen.py 200 > "$O/sim_screen.txt"
  python3 sim_star_het.py 200 > "$O/sim_star_het.txt"
  python3 sim_k401.py 200 > "$O/sim_k401.txt"
  python3 sim_montecarlo.py > "$O/sim_montecarlo.txt"   # supplement Monte Carlo tables, about 10 minutes
  python3 sim_power.py 1000 > "$O/sim_power.txt"         # sets between one-sided and CH, about 40 minutes
  if [ -f "$ROOT/data/ae.csv" ]; then python3 sim_ae.py 100 > "$O/sim_ae.txt"; fi
}

apps() {
  cd "$ROOT/code/applications"; O="$ROOT/output/applications"; mkdir -p "$O"
  python3 apps_moment_sets.py > "$O/apps_moment_sets.txt"
  for d in $APPS; do
    python3 apps_sets.py $d    > "$O/apps_sets_$d.txt"
    python3 apps_numbers.py $d > "$O/apps_numbers_$d.txt"
    python3 apps_roots.py $d   > "$O/apps_roots_$d.txt"
    python3 indep_iiv.py $d    > "$O/indep_iiv_$d.txt"
  done
  python3 indep_iiv.py card --legacy-card > "$O/indep_iiv_card_legacy.txt"
  python3 apps_bounds.py    > "$O/apps_bounds.txt"
  python3 apps_het.py       > "$O/apps_het.txt"
  python3 apps_rob.py       > "$O/apps_rob.txt"
  python3 apps_crit.py      > "$O/apps_crit.txt"     # calibrated p-values need the sim_semi_crit draws
  python3 kurt_roots.py     > "$O/kurt_roots.txt"
  python3 onesided_ident.py > "$O/onesided_ident.txt"
  python3 iiv_rho_xu.py     > "$O/iiv_rho_xu.txt"
  python3 dep_check.py      > "$O/dep_check.txt"
  python3 dtiiv_k401.py     > "$O/dtiiv_k401.txt"
  python3 bi_interior.py    > "$O/bi_interior.txt"     # Banerjee-Iyer on 0 < y < 1
  python3 mroz_dtiiv.py     > "$O/mroz_dtiiv.txt"
  python3 dtcore_apps.py    > "$O/dtcore_apps.txt"     # VT-IIV estimator core, B = 1999
  python3 power_apps.py     > "$O/power_apps.txt"      # sets between one-sided and CH, about 20 minutes
  python3 power_tables.py   > "$O/power_tables.txt"    # tables S17-S18 (needs sim_power.py output)
}

r61() {
  # MSIV_R61 additions: extreme-tail calibration and approximate mean independence
  cd "$ROOT/code/r61"; O="$ROOT/output/r61"; mkdir -p "$O"
  python3 selftest.py      > "$O/selftest.txt"
  python3 selftest_proj.py > "$O/selftest_proj.txt"
  python3 tail_sim.py 600  > "$O/tail_sim.log"
  python3 tail_report.py   > "$O/tail_report.txt"
  for d in $APPS; do
    python3 tail_apps.py $d > "$O/tail_apps_$d.txt"
    python3 ami_apps.py $d  > "$O/ami_apps_$d.txt"
  done
  python3 ami_sim.py 500    > "$O/ami_sim.txt"
  python3 ami_factor.py 500 > "$O/ami_factor.txt"
  python3 bias_floor.py     > "$O/bias_floor.txt"      # reads the ami_apps output
  mkdir -p "$O/tables"; python3 make_tables.py "$O/tables"
  python3 bias_floor_het.py "$O/tables" > "$O/bias_floor_het.txt"
  # Table 3 at matched size: needs the statistics saved by main_paper/01_mc_optimal.py
  if [ -f "$ROOT/output/main_paper/mc_optimal_chi2_CH_500.npy" ]; then
    python3 mc_optimal_matched.py "$O/tables" > "$O/mc_optimal_matched.txt"
  fi
}

case $PART in
  main) main ;;
  apps) apps ;;
  sims) sims ;;
  r61)  r61 ;;
  all)  main; sims; apps; r61 ;;
  *) echo "usage: bash run_all.sh [all|main|apps|sims|r61]"; exit 1 ;;
esac
