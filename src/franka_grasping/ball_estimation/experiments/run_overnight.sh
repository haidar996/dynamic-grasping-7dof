#!/bin/bash
# All heavy computation after the research audit + Phase 1.2, in dependency order (2026-11-01).  ~11 h, 3 workers.
# Run from anywhere:  bash experiments/run_overnight.sh [part1]   (part1: stop after Part 1, ~3.5 h)
#                         Logs: outputs/*.log, progress: outputs/overnight_progress.log.   Restart after a crash: same command
# (finished steps are skipped; step_0c rendering and step_1c learning also resume inside the step).
set -e
cd "$(dirname "$0")/.."
# a finished step leaves a marker in outputs/.overnight_done/ and is skipped on restart (delete the folder to redo all)
mkdir -p outputs/.overnight_done
run() {
  local m="outputs/.overnight_done/$(echo "$1" | tr ' ,+()' '_____')"
  if [ -e "$m" ]; then echo "$(date +%H:%M) skip  $1 (done)" >> outputs/overnight_progress.log; return; fi
  echo "$(date +%H:%M) start $1" >> outputs/overnight_progress.log; shift; "$@"; touch "$m"
}
run "fit c,E on train"      python3 -u experiments/step_0k_fit_constants_train.py 3   > outputs/fit_constants_train.log 2>&1
run "tune noise on val"     python3 -u experiments/step_0i_tune_ekf_noise.py 3        > outputs/ekf_tuning_val.log 2>&1
run "baselines val+test"    python3 -u experiments/step_0d_baselines_with_arm.py all_conds 3 > outputs/baselines_v3_val_test.log 2>&1
run "paired tests"          python3 -u experiments/step_0j_paired_tests.py            > outputs/paired_tests_v3.log 2>&1
run "mu vs E (train+val)"   python3 -u experiments/step_1a_mu_vs_e_gap.py 3           > outputs/mu_vs_e_gap_trainval.log 2>&1
run "mismatch physics check" python3 -u experiments/step_0l_check_mismatch_physics.py 3 > outputs/check_mismatch_physics.log 2>&1
run "render mismatch data"  python3 -u experiments/step_0c_generate_500_with_arm.py --physics mismatch > outputs/raw_mismatch_render.log 2>&1
if [ "$1" = "part1" ]; then echo "$(date +%H:%M) PART 1 DONE" >> outputs/overnight_progress.log; exit 0; fi
# ---- Part 2: ekf-mu ----
run "ekf-mu grid (val)"     python3 -u experiments/step_1d_tune_ekf_mu_grid.py        > outputs/ekf_mu_grid_val.log 2>&1
run "ekf-mu learn nll"      python3 -u experiments/step_1c_learn_ekf_mu.py nll all_conds 3 > outputs/ekf_mu_learned_nll.log 2>&1
run "ekf-mu learn h05"      python3 -u experiments/step_1c_learn_ekf_mu.py h05 all_conds 3 > outputs/ekf_mu_learned_h05.log 2>&1
run "ekf-mu evaluation"     python3 -u experiments/step_1e_eval_ekf_mu.py all_conds 3 > outputs/ekf_mu_eval.log 2>&1
echo "$(date +%H:%M) ALL DONE" >> outputs/overnight_progress.log
