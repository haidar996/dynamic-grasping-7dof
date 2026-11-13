# A1 — ekf-μ on mismatch physics: results (started 2026-11-12)
Design and frozen predictions: docs/A1_ekfmu_mismatch_eval_design.md. Core set: ideal, noise, all_real, all_plane, dropout_silent_noisy.

## P1 — arm R (retrain) on ideal, 2026-11-12 01:19 → 02:27 (68 min)
Baselines re-tuned on mismatch (c 0.5, E 0.5; ekf σ 5 mm q 20; ekf_fit σ 20 mm q 5). Grid: q 0.5, q_μ 1.0, s_l 0.5 (both q and q_μ on the
TOP edge, still falling → rerun with a wider grid pending, run-1 file kept as ekf_mu_grid_val_mismatch_run1.json).
Learned (mismatch train): nll σ 0.44 mm, **q 0.065**, q_μ 1e-3, μ0 0.48, s_l 0.24, E 0.64 (start A won, −25.7028; start B −25.7005; no cap).
h05 σ 4.9 mm, **q 4.5**, q_μ 1e-3, μ0 0.45, s_l 2.2, E 0.51 (36 grads, early stop). q_μ did not move from its start in either fit.

| h0.5 (mm) | ekf19b | ekf19b_fit | hindsight | ekfmu_grid | ekfmu_nll | ekfmu_h05 |
|---|---|---|---|---|---|---|
| test (dev) | 67 | 54 [47,61] | 35 [29,41] | 54 | 54 | 51 [44,57] |
| val | 65 | 52 [45,58] | 36 [30,41] | 52 | 56 | 53 |

Paired vs ekf19b_fit, h0.5: test grid +0.2 (n.s.), nll +0.6 (n.s.), **h05 −3.1 [−4.6, −1.7] (p 3e-5, 76 %)**; val grid +0.7, nll +4.1, h05 +1.9 (CIs include 0).
Gap closed (fit → hindsight): test h05 +0.17, grid −0.01, nll −0.03; val all ≤ 0. NIS/NEES test: nll 2.07 / 20.3, h05 0.03 / 0.2, grid 0.03 / 0.3.
μ̂ vs true a0 at 2 s: r ≈ 0.6–0.8 (all variants) — μ̂ does track rolling resistance, but late.

### Predictions (ideal only so far)
- H-q (q ≥ 1e-3): **supported** on ideal (0.065 vs matched 0.016 old-rule). NEES < 20: **borderline fail** on test (20.3; val 15.5).
- Prediction 5 (ekf-μ − fit negative, significant, both arms): on arm R ideal **only h05 passes on test (−3 mm) and nothing on val → effectively REFUTED for ideal**:
  the matched gain of −16…−25 mm shrinks to ≤ 3 mm. 5b (gap closed > 1): no, ≤ 0.17. 5c (q_μ up): grid yes (1.0), learned fits no (flat).
### Reading (provisional, one condition)
1. On slide → roll physics a single drifting deceleration state does not help: the retrained h05 fit chooses q 4.5 (≈ 250× the matched value),
   i.e. it explains the motion by process noise, not by μ; all ekf-μ variants collapse onto ekf19b_fit (54 mm). Claim C1 does NOT transfer as stated.
2. The hindsight constant pair is still 18 mm better than every filter: the information exists per launch but needs a richer model
   (two-phase deceleration or speed-dependent a(v)) or a better learning signal, not a random walk on log μ.
3. The ML criterion on mismatch behaves as predicted: q rises 3 orders of magnitude above the matched noise fit and NIS is 2.07; NEES 20 instead of 662.
   Supports the inverse-crime reading of F5.
4. Transfer arm (T) and the other 4 core conditions are still needed before rewriting C1/C3.

### Grid rerun on ideal (16:40 → 16:55, wider grid q ≤ 5, q_μ ≤ 10)
Optimum q 5, q_μ 10, s_l 1.0 → val 48.6 mm (run 1: 52.3). Still on both top edges but flattening (q 2 → 5: 49.4 → 48.6; q_μ 1 → 10: 50.0 → 48.6);
widened twice already → accepted as saturation. Re-eval: ekfmu_grid test 49 mm, paired vs fit **−2.9 [−4.2, −1.7] (p 1e-5, 73 %)**, gap closed +0.18 —
the same ≈ 3 mm as h05. Reading unchanged: on mismatch ideal every ekf-μ variant ends 3 mm (test) / 0 mm (val) better than the fixed-μ filter, with
q and q_μ pushed to values where μ is effectively a free nuisance state, not a friction estimate.

## P4 — arm R on noise, 2026-11-13 00:50 → 02:02 (72 min)
Grid q 5, q_μ 10, s_l 1 (top edges again, accepted). nll σ 2.5 mm, q 0.043, q_μ 1e-4, μ0 0.85, s_l 0.47, E 0.65. h05 σ 16.5 mm, q 1.57, μ0 0.48, s_l 1.49, E 0.50.

| h0.5 (mm) | ekf19b_fit | hindsight | ekfmu_grid | ekfmu_nll | ekfmu_h05 |
|---|---|---|---|---|---|
| test (dev) | 56 [50,63] | 42 | 55 | **82** | 55 |
| val | 57 | 45 | 55 | 79 | 56 |

Paired vs fit (test): grid −1.8 [−3.0, −0.6] (p 0.007), h05 −1.1 (n.s.), **nll +25.2 [+17.8, +33.1] (p 2e-9)**. Gap closed: grid +0.12, h05 +0.07, nll −1.72.
NIS/NEES test: nll 1.93 / 6.9 (calibrated), h05 0.05 / 0.2, grid 0.02 / 0.1. μ̂ vs a0 at 2 s r 0.5–0.85.
Reading: as in ideal, adaptive μ ≈ fixed μ (≤ 2 mm). New: on mismatch + camera noise the ML-trained filter is 25 mm WORSE than the fixed-μ baseline while being
the only calibrated one (NIS 1.9, NEES 7): likelihood and forecast accuracy diverge strongly under model error. H-q holds (q 0.043 ≫ 8e-5 matched).
