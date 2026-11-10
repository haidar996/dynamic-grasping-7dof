#!/bin/bash
# One SECTION = one sensor condition of Part 2 (2026-11-08, user: run each section alone, on request):
#   learn ekf-mu by NLL (step_1c nll) -> learn by h0.5 (step_1c h05) -> evaluate this condition (step_1e, merged into
#   outputs/ekf_mu_eval.json / _errors.pkl.gz).  ~30-45 min.  Learned conditions are skipped by step_1c (resume).
# Usage (from anywhere):  bash experiments/run_section.sh <condition>      e.g.  bash experiments/run_section.sh ideal
# Conditions: ideal noise dropout_flagged dropout_silent delay all noise_real all_real plane all_plane dropout_silent_noisy
# Done marker: outputs/.sections_done/<condition>.  Progress: outputs/overnight_progress.log.
# To stop a running section: kill its python PIDs (ps -eo pid,args | grep step_), NOT `pkill -f` (kills the caller too).
#
# Mismatch physics (audit A1, docs/A1_ekfmu_mismatch_eval_design.md, 2026-11-10); every output gets a _mismatch suffix:
#   arm R (retrain), once first:  PHYSICS=mismatch bash experiments/run_section.sh baselines
#        = fit c, E on mismatch train (step_0k) -> re-tune sigma, q of ekf / ekf_fit on mismatch val (step_0i, R subset)
#   arm R, per condition:         PHYSICS=mismatch bash experiments/run_section.sh <condition>
#        = ekf-mu grid on mismatch val (step_1d) -> learn nll -> learn h05 (mismatch train) -> evaluate ARM=retrain
#   arm T (transfer), eval only:  PHYSICS=mismatch ARM=transfer bash experiments/run_section.sh <condition>
#        = matched-learned parameters evaluated on mismatch val + test (dev)
#   Markers: outputs/.sections_done_mismatch/ (R), outputs/.sections_done_mismatch_transfer/ (T).
set -e
cd "$(dirname "$0")/.."
C="$1"
PHYSICS="${PHYSICS:-matched}"; ARM="${ARM:-}"
export PHYSICS
python3 -c "import sys; sys.path.insert(0, '.'); import physics_cfg"          # validates PHYSICS
if [ "$C" != baselines ]; then
  python3 -c "import sys; sys.path.insert(0, '.'); from dataset import CONDITIONS; assert '$C' in CONDITIONS, 'unknown condition: $C'"
fi
if ps -eo args | grep -E "python3 -u experiments/step_(1[cde]|0[ik])" | grep -vq grep; then echo "another learning/eval run is active: stop it first"; exit 1; fi

if [ "$PHYSICS" = matched ]; then
  [ "$C" != baselines ] || { echo "baselines section exists only for PHYSICS=mismatch"; exit 1; }
  [ -z "$ARM" ] || { echo "ARM is only used with PHYSICS=mismatch"; exit 1; }
  mkdir -p outputs/.sections_done
  log() { echo "$(date +%H:%M) section $C: $1" >> outputs/overnight_progress.log; echo "$(date +%H:%M) $1"; }
  log "start"
  log "learn nll";  python3 -u experiments/step_1c_learn_ekf_mu.py nll "$C" 3 >> outputs/ekf_mu_learned_nll.log 2>&1
  log "learn h05";  python3 -u experiments/step_1c_learn_ekf_mu.py h05 "$C" 3 >> outputs/ekf_mu_learned_h05.log 2>&1
  log "evaluate";   python3 -u experiments/step_1e_eval_ekf_mu.py "$C" 3 >> outputs/ekf_mu_eval.log 2>&1
  touch "outputs/.sections_done/$C"
  log "DONE"
  exit 0
fi

# ---------------- mismatch ----------------
log() { echo "$(date +%H:%M) mismatch${ARM:+ $ARM} section $C: $1" >> outputs/overnight_progress.log; echo "$(date +%H:%M) $1"; }
if [ "$ARM" = transfer ]; then
  [ "$C" != baselines ] || { echo "arm T has no baselines section (it uses the matched tuning)"; exit 1; }
  mkdir -p outputs/.sections_done_mismatch_transfer
  log "evaluate (matched parameters)"
  ARM=transfer python3 -u experiments/step_1e_eval_ekf_mu.py "$C" 3 >> outputs/ekf_mu_eval_mismatch_transfer.log 2>&1
  touch "outputs/.sections_done_mismatch_transfer/$C"
  log "DONE"
  exit 0
fi
[ -z "$ARM" ] || [ "$ARM" = retrain ] || { echo "ARM must be transfer or retrain (default)"; exit 1; }
mkdir -p outputs/.sections_done_mismatch
if [ "$C" = baselines ]; then
  log "fit c, E on mismatch train";    python3 -u experiments/step_0k_fit_constants_train.py 3 >> outputs/fit_constants_train_mismatch.log 2>&1
  log "tune noise on mismatch val";    python3 -u experiments/step_0i_tune_ekf_noise.py 3 >> outputs/ekf_tuning_val_mismatch.log 2>&1
  touch outputs/.sections_done_mismatch/baselines
  log "DONE"
  exit 0
fi
[ -f outputs/.sections_done_mismatch/baselines ] || { echo "run the mismatch baselines section first"; exit 1; }
python3 -c "import json; assert '$C' in json.load(open('outputs/ekf_tuning_val_mismatch.json'))['best']['ekf'], \
'$C has no mismatch noise tuning (arm-R subset: ideal, noise, all_real, dropout_silent_noisy); add it with step_0i first'"
log "start"
log "ekf-mu grid (mismatch val)"; python3 -u experiments/step_1d_tune_ekf_mu_grid.py "$C" >> outputs/ekf_mu_grid_val_mismatch.log 2>&1
log "learn nll";  python3 -u experiments/step_1c_learn_ekf_mu.py nll "$C" 3 >> outputs/ekf_mu_learned_nll_mismatch.log 2>&1
log "learn h05";  python3 -u experiments/step_1c_learn_ekf_mu.py h05 "$C" 3 >> outputs/ekf_mu_learned_h05_mismatch.log 2>&1
log "evaluate (retrained)"; ARM=retrain python3 -u experiments/step_1e_eval_ekf_mu.py "$C" 3 >> outputs/ekf_mu_eval_mismatch_retrain.log 2>&1
touch "outputs/.sections_done_mismatch/$C"
log "DONE"
