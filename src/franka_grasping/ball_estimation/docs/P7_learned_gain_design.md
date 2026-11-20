# P7 — learned comparison: KalmanNet-style learned gain on the ekf19b model (design, 2026-11-13, before any code)

## Why this method (plan decision 2)
The chapter requires a learned estimator. Paper 04 (KalmanNet) keeps the physics model for the prediction step and replaces the
Kalman gain by a recurrent network; it is the closest learned method to our filters, reuses `estimators/model.py` unchanged, needs
only a small GRU on CPU, and its sim-to-sim transfer is exactly the C3 question. Pre-registered prediction (plan §5): the learned
gain trained on matched data loses MORE on mismatch than ekf-μ-h05 (paired matched → mismatch change), because it has no physical
prior for the new regime. After P5 this needs a sharper statement, because ekf-μ itself transferred catastrophically (+30…58 mm):
- **Prediction L1 (matched):** on matched test (dev) the learned gain beats ekf19b_fit (train-fitted fixed μ) and is within 10 mm of
  ekf-μ-h05 in ideal and noise. Refuted if it does not beat ekf19b_fit (paired, p < 0.05).
- **Prediction L2 (transfer):** its matched → mismatch degradation (paired, same launches) is larger than that of ekf19b_fit (+20 mm)
  and at least as large as that of ekf-μ-h05.
- **Prediction L3 (retrain):** retrained on mismatch train, it does not beat the mismatch-re-tuned ekf19b_fit by more than ekf-μ-grid
  does (≈ 3 mm). If it does (e.g. > 6 mm), the learned gain is capturing slide → roll that ekf-μ cannot — that would be a positive
  result for learning and must be reported as such.

## Filter
- Prediction: identical to ekf19b_fit (c, E from the train fit; walls, arm box; sub-steps 5 ms); predicts at every frame tick.
  No covariance is propagated: the network replaces P.
- Update at a valid (non-stale) measurement: innovation y = z − ŝ_pos; gain K = f_θ(features) ∈ R^{4×2}; ŝ += K y.
  Features (KalmanNet "F2/F3" set, all scaled): y, Δŝ = ŝ⁻_k − ŝ⁺_{k−1} (state change by prediction), ŝ⁺_{k−1} − ŝ⁺_{k−2}
  (previous update step), the last innovation, the time since the last valid measurement, and a bounce flag (velocity sign flip
  in the prediction). GRU hidden 32, two linear layers, output 8 numbers reshaped to K. ~4 k parameters.
- Missing / stale frames: predict only (as every other filter here; StaleGuard logic inline).
- Delay condition: same as the others — the metric forecasts from the capture-time state to the arrival time.

## Training
- Data: 350 train launches of the condition, batch numpy/torch in lockstep on the 60 Hz grid (same layout as
  `estimators/ekf_mu_batch.prepare`: Z (B,T,2), V (B,T), target (B,T,2)).
- Loss: the same forecast criterion as ekf-μ-h05 — launch-mean RMS of the 0.5 s forecast error on moving frames (every 6th frame,
  warm-up 0.1 s) — plus 0.1 × now_pos RMS so the filter state stays meaningful. Forecasts use the model (`forecast`), so the loss
  is differentiable through torch re-implementations of `propagate` and `forecast` (friction + walls + box, piecewise smooth).
- Optimiser Adam 1e-3, truncated BPTT over whole launches (600 steps, 1 GRU), batch 32 launches, 60 epochs, early stop on VAL
  launch-mean h0.5 (patience 10). Seed fixed (0); report the seed-to-seed spread with 3 seeds on ideal only.
- Budget: a forward pass over 350 launches × 600 steps with a 32-GRU is ≈ 1–2 s on 4 CPU threads → one epoch ≈ 1 min with
  backward; 60 epochs ≈ 1 h per condition. Core set matched (5) + mismatch retrain (5) ≈ 10 h. If that is too slow, cut to 3
  conditions (ideal, noise, all_real) and say so.
- Variants: none beyond the main model (ablations cost hours; the paper's ablation budget went into ekf-μ).

## Evaluation
Identical protocol to step_1e (per-launch RMS, launch mean + bootstrap CI, paired tests vs ekf19b_fit, matched → mismatch paired
deltas, gap closed on mismatch); label `knet`. Arms: matched (train/val/test), mismatch T (matched weights), mismatch R (retrained).
No NIS/NEES (the network has no covariance; stated as a limitation and as the reason a planner would still need ekf-μ-nll's covariance).

## Files
`estimators/knet.py` (torch filter + torch model), `experiments/step_2a_train_knet.py` (train, PHYSICS switch, outputs/knet_<cond>[_mismatch].pt
+ json), step_1e gets `knet` as an estimator label when a weight file exists. Smoke test: 3 launches, 2 epochs, both physics.

## Decisions taken by me (reversible, no result exists)
Features/hidden size as above; loss = h05 + 0.1 now_pos; 60 epochs / patience 10; 3 seeds on ideal only; core set, both physics.

## Result 1 — knet on ideal (trained matched train, 60 epochs, 145 min; evaluated 2026-11-13 19:05)
| h0.5 (mm), paired vs ekf19b_fit | matched test | matched val | mismatch-T test | mismatch-T val |
|---|---|---|---|---|
| ekf19b_fit | 55 | 47 | 65 | 63 |
| ekfmu_h05 | 30 (−24.8) | 27 (−20.2) | 114 (+49.5) | 102 (+38.7) |
| ekfmu_nll | 38 (−16.3) | 31 (−15.9) | 84 (+18.9) | 82 (+19.0) |
| **knet** | 56 (+1.1, n.s.) | 47 (−0.3) | 64 (−0.4, n.s.) | 64 (+0.4, n.s.) |
- **L1 refuted:** the learned gain does not beat the fixed-μ filter on matched data (it learns ≈ the fixed filter's gain: same model, no μ state).
- **L2 refuted:** it does not degrade more than ekf19b_fit under transfer — it degrades exactly like it (+9 mm), and far LESS than ekf-μ (+49 / +19).
- Reading: with the physics prediction fixed, a learned gain cannot recover the per-launch friction; ekf-μ can (matched) but pays for it under
  model mismatch. The learned gain inherits the robustness of the fixed model, not the adaptivity. Noise / all_real training waits for the user.

## Result 2 (2026-11-14): knet on noise (matched)
Training: early stop at epoch 22, best epoch 12, val h0.5 52.5 mm (K0 start 59.1), 52 min. Weights outputs/knet_noise.pt.
Eval (step_1e, 75 test launches), h0.5 vs ekf19b_fit: knet 60 vs 59 mm, paired +0.6 [-1.3, +2.8] mm, Wilcoxon p 0.27 (val +0.6, p 0.73).
knet ms/update 4.9 vs 0.3. Conclusion: same as ideal: knet = fixed-mu filter on noise.
all_real training stopped by the user at ~epoch 22 (val 76.5 mm, still improving); no weights saved — not evaluated.

## Result 3 (2026-11-16): knet on all_real (matched training)
Training (outputs/knet_train.log): retrained from scratch, seed 0, 350 train / 75 val launches; best epoch 56, val h0.5 72.8 mm (epoch 0 = fixed gain K0: 84.6 mm), 148.5 min. Weights outputs/knet_all_real.pt. (The earlier run stopped at epoch 22 was discarded.) Epochs run: 60 (cap; no early stop; epoch 60 val 73.0 mm, so the curve had flattened).
Eval (step_1e, outputs/ekf_mu_eval.json, run 04:43-04:50), h0.5 mm, launch mean [bootstrap 95% CI], paired vs ekf19b_fit:
| matched | test (73 launches) | val (72 launches) |
|---|---|---|
| ekf19b_fit | 76.1 [67.5, 85.1] | 70.8 [63.0, 78.0] |
| ekfmu_h05 | 51.4 (-24.6, p 9e-10) | 45.0 (-25.8, p 2e-11) |
| ekfmu_grid | 54.7 (-21.4) | 47.7 (-23.1) |
| ekfmu_nll | 57.4 (-18.7) | 52.6 (-18.2) |
| **knet** | 82.5 [70.6, 94.5], +6.4 [-1.3, +16.2], Wilcoxon p 0.095 | 72.8 [63.0, 82.8], +1.9 [-3.1, +9.3], Wilcoxon p 0.041 |
now_pos (mm, test): ekf19b_fit 8.6, ekfmu_h05 6.4, knet 11.5 (val: 8.7 / 6.8 / 10.7). knet ms/update 4.7 vs 0.3.
Mismatch transfer (matched-trained knet on mismatch physics, outputs/ekf_mu_eval_mismatch_transfer.json, run 04:50-04:57), h0.5 mm:
| mismatch-T | test (74 launches) | val (73 launches) |
|---|---|---|
| ekf19b_fit | 87.3 [77.0, 98.1] | 87.0 [76.5, 97.6] |
| ekfmu_h05 | 122.1 (+34.7, p 4e-7) | 123.0 (+35.9, p 6e-7) |
| **knet** | 88.2 [77.3, 99.3], +0.8 [-1.7, +4.0], p 0.55 | 91.3 [79.8, 104.1], +4.3 [-0.2, +10.6], p 0.59 |
now_pos (mm, test): ekf19b_fit 9.6, knet 12.0.
- **L1 refuted again:** knet does not beat ekf19b_fit on matched all_real; the sign is the opposite (test +6.4 mm worse, CI includes 0, p 0.095). On val the mean diff is +1.9 mm with CI including 0 but Wilcoxon p 0.041 (the two tests disagree; with the median-style test favouring fit slightly). So knet is not better and is, if anything, slightly worse; it is 21-31 mm behind ekf-mu-h05.
- **L2 refuted:** under transfer knet is indistinguishable from ekf19b_fit (+0.8 mm test, +4.3 val, n.s.) and far better than ekf-mu (+35 mm). Unpaired change in launch-mean h0.5 matched -> mismatch (different launch sets, so not a paired delta): knet +5.7 mm (82.5 -> 88.2), ekf19b_fit +11.2 mm (76.1 -> 87.3); I did not run the paired matched -> mismatch test.
- Verdict: knet ~ fixed-mu filter again (no significant difference on test), with a small non-significant worse tendency on all_real (training val 72.8 vs K0 84.6 improved over its own start, but still not below the fixed-mu filter on val 70.8). Same conclusion as Results 1-2 (ideal, noise), now with the most realistic condition: a learned gain on the fixed model does not recover per-launch friction; only ekf-mu does, and it fails on transfer.

## Result 4 (L3: retrain on mismatch, ideal + noise + all_real)

Setup: knet retrained from scratch on MISMATCH train data (350 train / 75 val launches), compared on the same launches with the mismatch-re-tuned baselines (ARM=retrain, outputs/ekf_mu_eval_mismatch_retrain.json / .pkl.gz). Paired numbers recomputed by outputs/l3_paired.py (reuses estimators.stats.paired: 2000-resample bootstrap, seed 0, Wilcoxon); it reproduces the step_1e knet vs ekf19b_fit numbers exactly.
- Training (outputs/knet_train_mismatch.log): ideal: K0 (epoch 0) val h0.5 53.7 mm -> best epoch 32, 49.0 mm (early stop at epoch 42, 90.6 min). noise: K0 val 57.7 mm -> best epoch 36, 52.5 mm (early stop at epoch 46, 101.2 min). all_real: K0 val 74.5 mm -> best epoch 59, 63.7 mm (ran to epoch 60, no early stop triggered, 136.8 min).

Launch-mean h0.5 (mm; test n=74, val n=73) and paired diff vs ekf19b_fit [95% CI] (negative = better):

| cond / split | ekf19b_fit | ekfmu_grid | ekfmu_h05 | knet |
|---|---|---|---|---|
| ideal test | 53.8 | 51.2 (-2.6 [-3.7,-1.5]) | 50.7 (-3.1 [-4.6,-1.7]) | 49.9 (-3.9 [-5.8,-2.0], p 2.6e-5) |
| ideal val | 51.6 | 48.6 (-2.9 [-4.2,-1.7]) | 53.5 (+1.9 [-3.8,+11.7]) | 49.0 (-2.6 [-6.2,+3.1], p 5.8e-8) |
| noise test | 56.4 | 54.6 (-1.8 [-3.0,-0.6]) | 55.3 (-1.1 [-2.4,+0.2], n.s.) | 52.8 (-3.6 [-6.0,-1.6], p 4.7e-4) |
| noise val | 57.1 | 54.7 (-2.4 [-3.8,-1.2]) | 56.3 (-0.8 [-2.0,+0.4], n.s.) | 52.5 (-4.6 [-6.2,-3.0], p 1.6e-7) |
| all_real test | 70 | 67 (-2.5 [-4.4,-0.5], p 0.039) | 68 (-2.2 [-5.3,+0.4], n.s.) | 66 (-4.0 [-8.5,+0.4], p 2.3e-4) |
| all_real val | 69 | 65 (-4.0 [-6.7,-1.6]) | 68 (-1.5 [-3.5,+0.6], n.s.) | 64 (-5.6 [-8.4,-2.8], p 1.4e-5) |

Direct paired knet - ekfmu_grid (the L3 test), h0.5 mm [95% CI], Wilcoxon p, knet better in:
- ideal test: -1.3 [-3.5,+0.7], p 0.96, 47%. ideal val: +0.3 [-3.7,+6.2], p 0.15, 52%.
- noise test: -1.8 [-4.5,+0.7], p 0.71, 51%. noise val: -2.1 [-4.3,+0.1], p 0.13, 59%.
- all_real test: -1.5 [-6.7,+3.5], p 0.40, 53%. all_real val: -1.6 [-5.3,+2.3], p 0.44, 52%.
- vs ekfmu_h05 (test): ideal -0.8 [-2.9,+1.2], p 0.85; noise -2.6 [-5.3,-0.2], p 0.18. (noise val: -3.8 [-5.9,-1.6], p 3.0e-3.) all_real test -1.8 [-6.7,+3.7], p 0.053; all_real val -4.1 [-7.2,-1.0], p 0.032 (knet better than ekfmu_h05 on val only, not on test).

Verdict L3: HOLDS in ideal, noise and all_real (3 of 5 conditions). knet's advantage over ekf19b_fit (3.6-4.0 mm on test; in all_real the test bootstrap CI [-8.5,+0.4] includes 0 although Wilcoxon p 2.3e-4 and val is clear at -5.6 [-8.4,-2.8]) is real, but its advantage over ekfmu_grid is not significant on test or val (CIs include 0; the Wilcoxon p values are also non-significant). Caveat: in several cells the bootstrap CI and the Wilcoxon p disagree (e.g. ideal val), which points to a few heavy-tailed launches; mean differences of 1-2 mm vs ekfmu_grid cannot be resolved with n=74.
Reading: on matched physics knet = fixed-mu (Results 1-3), but under mismatch retraining it gains about 4 mm over ekf19b_fit, the same order as ekf-mu's retrain gain (about 2-3 mm). It does not clearly exceed ekf-mu-grid, so there is no evidence that it captures slide -> roll beyond what ekf-mu does.
Scope: ideal, noise, all_real. all_plane and dsn are pending (runs 3-5, need user go).

### Result 4b: paired tests for now_pos, now_vel, h0.2, h1.0
knet - baseline on test launches, mean difference [95% CI] (mm; now_vel in m/s; negative = knet better); Holm p over the 6 now_pos/now_vel tests per set. Full output incl. val and all comparisons (also vs ekfmu_h05): outputs/paired_all_metrics.txt (script outputs/paired_all_metrics.py; h0.5 reproduces Result 4 logs).

| set | metric | vs ekf19b_fit | vs ekfmu_grid |
|---|---|---|---|
| retrain ideal | now_pos | +0.14 [-0.09,+0.47] (Holm 0.53) | +0.03 [-0.16,+0.29] (0.49) |
| | now_vel | +0.026 [+0.019,+0.033] (7e-11) | +0.025 [+0.019,+0.031] (4e-11) |
| | h0.2 / h1.0 | +0.10 [-0.62,+0.89] / -6.97 [-12.3,-1.6] | +0.43 [-0.30,+1.19] / +0.79 [-5.5,+7.0] |
| retrain noise | now_pos | +0.00 [-0.17,+0.25] (0.056) | -0.12 [-0.32,+0.12] (1.6e-3) |
| | now_vel | +0.037 [+0.021,+0.063] (2e-10) | +0.036 [+0.019,+0.063] (4e-10) |
| | h0.2 / h1.0 | +0.14 [-0.69,+0.97] / -8.09 [-13.1,-3.7] | +0.17 [-0.74,+1.09] / -1.80 [-8.4,+4.5] |
| retrain all_real | now_pos | +0.79 [-0.02,+1.71] (Holm 1.0) | +0.44 [-0.48,+1.43] (1.0) |
| | now_vel | +0.012 [-0.025,+0.049] (1.0) | +0.007 [-0.030,+0.044] (1.0) |
| | h0.2 / h1.0 | -0.55 [-2.95,+1.91] / -6.92 [-15.4,+1.6] | -0.18 [-2.89,+2.30] / -1.43 [-12.2,+8.7] |
| matched all_real | now_pos | +2.86 [+0.62,+6.31] (2.9e-3) | +4.89 [+2.62,+8.50] (1.5e-10) |
| | now_vel | +0.031 [+0.011,+0.056] (0.062) | +0.053 [+0.032,+0.079] (2.7e-9) |
| | h0.2 / h1.0 | +4.94 [+0.55,+10.4] / +4.34 [-11.1,+19.8] | +12.9 [+8.3,+18.6] / +65.7 [+41.8,+87.1] |
| transfer all_real | now_pos | +2.35 [+0.99,+4.63] (2e-5) | -6.87 [-11.5,-3.05] (4.7e-4) |
| | now_vel | +0.008 [-0.036,+0.049] (0.080) | -0.053 [-0.101,-0.006] (4e-3) |
| | h0.2 / h1.0 | +1.97 [+0.22,+4.30] / -1.10 [-4.9,+3.5] | -18.2 [-26.0,-11.0] / -77.1 [-109.6,-48.0] |

Retrain all_real: on test knet is statistically indistinguishable from the EKFs on now_pos and now_vel (all Holm p = 1.0; point estimates slightly worse). On val, now_vel vs ekfmu_grid is +0.015 m/s with Holm 0.041 (knet worse, CI [-0.003,+0.040] includes 0; single cell, not replicated on test). Unlike ideal/noise there is no significant now_vel deficit here.

Reading: in mismatch retrain (ideal, noise), knet's now_pos matches the EKFs to within a fraction of a mm, but its now_vel is significantly worse (+0.025 to +0.037 m/s, in about 85-90% of launches), so its gain over ekf19b_fit lives in the long horizons (h1.0 about -7 to -8 mm), not in the current-state estimate. On all_real, knet is not better than ekf19b_fit anywhere (now_pos worse by 2-3 mm, h0.5/h1.0 differences include 0); the large "wins" over ekfmu_grid in transfer all_real reflect ekfmu_grid failing there, not knet being good. Bootstrap CI and Wilcoxon p sometimes disagree (heavy tails, e.g. ekf19b_fit h0.2 matched all_real), so treat single borderline cells cautiously.

## B2 camera scale bias note (2026-11-20)
Question: where does the +0.18 % ideal-camera scale bias (PLAN_2026-10-29.md item 5; RESUME.md ~l.240 "pure 0.08 % scale") come from, how big, can any result change?
- Code: `sensing/camera.py` `Pinhole.back_project` (f = (R/2)/tan(fovy/2), cx=(W-1)/2, zc = z_surface + r/sqrt(1+a²+b²)); the old r-offset error was fixed 2026-11-01.
- Script `outputs/b2_scale_bias.py` (CPU, no MuJoCo): rasterises a sphere r=0.03 at H=3 m, FOV 49°, 480², takes centroid + depth at rounded pixel, back-projects. Result: radial error −0.29…+0.94 mm over 0.25–1.77 m, sign-varying (rel. −0.06…+0.15 %), i.e. no systematic scale term; it is pixel-quantisation noise (centroid/rounded depth pixel), < 1 mm = < 0.2 px (f ≈ 526 px; 1 px ≈ 5.7 mm at floor).
- So the quoted 0.08–0.18 % (≈ 1–3 mm at 1.5 m) is of the same order as that quantisation plus MuJoCo render pixel-centre convention (a 0.5 px offset would be ≈ 2.9 mm, constant, not scale). Not separable without a MuJoCo render; not worth one.
- Impact: ≤ 3 mm worst case vs h0.5 errors of tens of mm (knet val 46.7 mm) and ideal-vs-noise gaps ≫ 3 mm; it is common to all estimators (same measurements), so paired comparisons are unaffected. No reported result changes; nothing fixed. Mention as a < 3 mm systematic in the limitations.
