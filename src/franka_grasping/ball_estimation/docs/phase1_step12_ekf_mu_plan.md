# Step 1.2 — ekf-μ (paper 03 method on our problem): implementation plan (2026-11-01, waiting for user decisions)

## Idea (paper 03 → us)
Paper 03 puts the unknown physics constants into the EKF state and learns all the filter's noise levels and starting
beliefs by maximising the likelihood of the measurements on training data. For us the unknown constant is the friction
μ. The μ-vs-E gap check showed μ explains the typical error: median h0.5 47 → 6 mm with the true μ.

## The filter
- **State:** z = (x, y, vx, vy, ℓ), with μ = exp(ℓ) (decision 1). μ stays positive, and the filter's uncertainty is
  relative (±20 % behaves the same for small and large μ).
- **Prediction** in sub-steps h ≤ 5 ms with the same motion model as ekf19b: friction μ, walls, arm box. E stays a fixed
  constant in 1.2; E moves into the state in 1.3 (paper 18).
- **Jacobian, the new parts:**
  - friction: ∂v'/∂ℓ = −h·μ·v̂, ∂p'/∂ℓ = h·∂v'/∂ℓ.
  - **bounce (audit B5):** our bounce maps the overshoot with slope −E (x' = L − E(x − L), vx' = −E·vx). On a hit, the
    Jacobian gets −E on (x, vx) for that axis (on the face axis for the arm box). This is exact for our discrete model
    and already contains the impact-timing effect of paper 03 S.4.
  - A unit test checks the Jacobian against finite differences, including steps with a bounce.
- **Noise:**
  - Q: white acceleration q on (x, y, vx, vy), as now.
  - q_μ: random walk on ℓ, so μ may drift a little.
  - R = σ²·I on the position.
- **Start:**
  - (x, y) = the first measurement, v = 0 with variance σ_v0².
  - ℓ0 = log μ̄ with variance s_ℓ² (the learned prior on μ).
- StaleGuard is kept, with policy "none", same as ekf19b+stale.

## Learning (paper 03 Eq. 7 on our data)
- **Learned numbers ψ (7, all through log so they stay positive except μ̄):**
  σ, q, q_μ, μ̄, s_ℓ, σ_v0, and the fixed E (decision 4).
- **Criterion:** negative log-likelihood of the measurements under the filter,
  Σ_k [log det S_k + y_kᵀ S_k⁻¹ y_k], with S = H·P⁻·Hᵀ + R and y the innovation.
  - Summed over the **350 train launches**.
  - Learned **per sensor condition**, like the baselines are tuned per condition.
- **Speed:** every launch has the same 600 frame times, so the filter runs **all 350 launches at once in numpy**, with
  masks for dropout and bounces. A test checks that this gives exactly the same numbers as the one-launch filter.
  This makes one likelihood evaluation take seconds, not minutes.
- **Optimiser (decision 2):** scipy L-BFGS-B with finite-difference gradients (7 numbers → 8 evaluations per gradient).
  No PyTorch needed. Starting point = the val-tuned ekf19b settings.

## Variants (ablation: what helps — μ in the state, or the learning?)
| Name | μ in state | Noise / priors from |
|---|---|---|
| ekf19b_fit (baseline) | no (c fitted on train) | val grid on h0.5 |
| ekf-μ hand | yes | σ, q from the baseline; q_μ, s_ℓ by a small val grid on h0.5 |
| **ekf-μ learned** | yes | maximum likelihood on train (paper 03) |
| ekf-μ learned-h05 (decision 3) | yes | same 7 numbers, but minimising h0.5 error on train |
| oracle | true μ, E | val grid |

## Evaluation (research protocol)
- VAL and TEST (dev) launches. The final test set (`final_test_plan.json`) is used once, at the end.
- **Matched AND mismatch physics data** (A1; the mismatch data is rendered when the laptop is free). On mismatch, the
  oracle is the per-launch hindsight best constant.
- All 11 sensor conditions.
- Per-launch RMS of now_pos, now_vel, h0.2, h0.5, h1.0: launch mean and median with 95 % bootstrap intervals, paired
  Wilcoxon tests against ekf19b_fit and the oracle.
- **μ recovery** (matched data): μ̂(t) vs the true μ. How fast does it converge, and does it depend on speed?
  Bounces? Noise?
- **Consistency:** NIS/NEES (is the predicted covariance honest?). A likelihood-trained filter should be calibrated.
  This is a check reviewers like.
- Run time per update (needed for real-time use in Phase 3).

## Order of work
1. `estimators/ekf_mu.py`: one-launch filter, same interface as EKF19, plus Jacobian unit tests. No heavy compute.
2. Batch (all-launches) version + equality test.
3. `experiments/step_1b_learn_ekf_mu.py`: ML learning per condition on train. Heavy compute → run when the laptop is free,
   together with the paused audit chain (`outputs/run_audit_chain.sh`).
4. Hand-tuned variant + evaluation script, then the results tables and the μ-recovery / NEES plots.

## Decisions for the user
1. μ parameterisation: **log μ (recommended)**, or the paper's a² + ε, or raw μ with clipping.
2. Optimiser: **scipy L-BFGS-B + batch numpy filter (recommended)**, or install PyTorch (autograd, but extra
   dependency and slower per step on CPU for this small model).
3. Criterion: **likelihood as the main method (paper 03) + the h0.5-trained variant as an ablation (recommended)**, or
   likelihood only.
4. Learn the fixed E together with the noise (**recommended**: E is a global constant, the same kind of number the
   baseline fits), or keep E from the train fit (step 0k).
