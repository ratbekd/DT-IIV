"""Table 8: one-sided sets within cells of the controls and the test of a common
slope (Proposition 5.6): Card by region and by race, STAR by school location,
Maimonides grades 4 and 5 by school disadvantage."""
import os
from _init import outdir, Tee
from data import card, star, maimonides, CARD_CTRL
from cells import run_cells, report
import cells

log = Tee(os.path.join(outdir('main_paper'), 'table8_cells.txt'))
cells.print = log      # route report() output to the results file

D = card(('nearc2', 'nearc4')); d = D['df']
report("Card by current region", run_cells(d, 'lwage', 'educ', CARD_CTRL, 'south', {0: 'non-South', 1: 'South'}))
report("Card by race", run_cells(d, 'lwage', 'educ', CARD_CTRL, 'black', {0: 'non-Black', 1: 'Black'}))
S = star()
report("STAR by school location", run_cells(S['df'], 'yv', 'csize', S['ctrl'], 'rural', {1: 'rural', 0: 'non-rural'}, cl='tch'))
for g in (4, 5):
    M = maimonides(g)
    report(f"Maimonides grade {g} by school disadvantage",
           run_cells(M['df'], 'avgverb', 'classize', M['ctrl'], 'hi_dis', {0: 'low disadv.', 1: 'high disadv.'}, cl='schlcode'))
