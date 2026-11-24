# Frozen Protocol — B6

**Written:** 2026-11-24  
**Rule:** this file is committed BEFORE the final test renders run. Nothing below may change after the first render completes.

---

## 1. Final test data

- **Seed:** 20261101 (written to `final_test_plan.json` on 2026-11-01, before any method existed)
- **Launches:** `sample_launches(np.random.default_rng(20261101), arena, k=10)` → 50 speeds × 10 = 500 launches
- **Camera:** `camera_settings_locked.yaml` (3 m, 49°, 480×480, 60 Hz, same scene XML)
- **Physics variants:** matched (standard friction), mismatch (A1 design, `--physics mismatch`)
- **Noise seeds:** `apply_condition` seed = 1000 + launch index (same rule as development data)
- **Why new data:** `split_500.json` test set was inspected during Phase 0 and used in the μ-vs-E decision — it is development test only

---

## 2. Frozen estimators

All hyperparameters are locked at the values in the JSON files below. No further tuning is permitted.

| estimator | parameter file | trained on |
|-----------|---------------|------------|
| ekf19b_fit | `outputs/fit_constants_train.json` | matched train |
| ekfmu_grid | `outputs/ekf_mu_grid_val.json` | matched val |
| ekfmu_h05 | `outputs/ekf_mu_learned_h05.json` | matched val |
| knet | `outputs/knet_train.json` (matched weights) | matched train |
| ekfimm | `outputs/ekf_imm_<cond>.json` per condition | matched val |
| ekfmu_h05 (mismatch-retrain) | `outputs/ekf_mu_learned_h05_mismatch.json` | mismatch val |
| knet (mismatch-retrain) | `outputs/knet_train_mismatch.json` | mismatch train |
| ekfimm (mismatch-retrain) | `outputs/ekf_imm_<cond>_mismatch.json` per condition | mismatch val |

---

## 3. Evaluation protocol

- **Conditions:** all 11 (ideal, noise, dropout_flagged, dropout_silent, delay, all, noise_real, all_real, plane, all_plane, dropout_silent_noisy)
- **Core 5:** ideal, noise, all_real, all_plane, dropout_silent_noisy
- **Primary metric:** h0.5 (RMS position error at 0.5 s prediction horizon, metres)
- **Secondary metrics:** now_pos, now_vel, h0.2, h1.0, NIS, NEES, conformal coverage at 90%/95%
- **Statistical test:** Wilcoxon signed-rank (per-launch RMS), Holm correction over the primary family
- **Bootstrap CI:** 1000 resamples, BCa, on per-launch RMS launch-mean

---

## 4. Primary endpoint (pre-registered)

> **ekfmu_h05 h0.5 mean RMS on matched physics, Wilcoxon signed-rank vs ekf19b_fit, Holm-corrected over all 5 core conditions × 2 estimators (ekfmu_h05 + ekfimm) = 10 tests, one-tailed p < 0.05.**

This corresponds to plan §8 item 4: "h0.5 test, per condition, ekf-μ-h05 and ekf-imm vs ekf19b_fit, both physics, Holm over the family."

**Expected direction:** ekfmu_h05 < ekf19b_fit (learned filter better than hand-tuned). We commit to this direction now. Any result in the opposite direction is reported as-is; we do not retest or select.

---

## 5. Render commands

```bash
# matched physics (run first, seed 20261101 baked into sample_launches call)
nohup python3 -u experiments/step_0c_generate_500_with_arm.py \
    --seed 20261101 --out final_test \
    >> outputs/final_test_matched.log 2>&1 &

# mismatch physics (run in parallel)
nohup python3 -u experiments/step_0c_generate_500_with_arm.py \
    --seed 20261101 --out final_test_mismatch --physics mismatch \
    >> outputs/final_test_mismatch.log 2>&1 &
```

Expected duration: ~90 min each (confirmed from development render: 90.5 min for 500 launches, 3 workers).

---

## 6. B7 rule

Run the final evaluation script **once**. Report every number it produces. Never re-run to pick a better result.
