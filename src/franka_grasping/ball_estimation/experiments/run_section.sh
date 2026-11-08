#!/bin/bash
# One SECTION = one sensor condition of Part 2 (2026-11-08, user: run each section alone, on request):
#   learn ekf-mu by NLL (step_1c nll) -> learn by h0.5 (step_1c h05) -> evaluate this condition (step_1e, merged into
#   outputs/ekf_mu_eval.json / _errors.pkl.gz).  ~30-45 min.  Learned conditions are skipped by step_1c (resume).
# Usage (from anywhere):  bash experiments/run_section.sh <condition>      e.g.  bash experiments/run_section.sh ideal
# Conditions: ideal noise dropout_flagged dropout_silent delay all noise_real all_real plane all_plane dropout_silent_noisy
# Done marker: outputs/.sections_done/<condition>.  Progress: outputs/overnight_progress.log.
# To stop a running section: kill its python PIDs (ps -eo pid,args | grep step_1), NOT `pkill -f` (kills the caller too).
set -e
cd "$(dirname "$0")/.."
C="$1"
python3 -c "import sys; sys.path.insert(0, '.'); from dataset import CONDITIONS; assert '$C' in CONDITIONS, 'unknown condition: $C'"
if ps -eo args | grep -E "python3 -u experiments/step_1[ce]" | grep -vq grep; then echo "another learning/eval run is active: stop it first"; exit 1; fi
mkdir -p outputs/.sections_done
log() { echo "$(date +%H:%M) section $C: $1" >> outputs/overnight_progress.log; echo "$(date +%H:%M) $1"; }
log "start"
log "learn nll";  python3 -u experiments/step_1c_learn_ekf_mu.py nll "$C" 3 >> outputs/ekf_mu_learned_nll.log 2>&1
log "learn h05";  python3 -u experiments/step_1c_learn_ekf_mu.py h05 "$C" 3 >> outputs/ekf_mu_learned_h05.log 2>&1
log "evaluate";   python3 -u experiments/step_1e_eval_ekf_mu.py "$C" 3 >> outputs/ekf_mu_eval.log 2>&1
touch "outputs/.sections_done/$C"
log "DONE"
