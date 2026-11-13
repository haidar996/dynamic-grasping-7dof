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
