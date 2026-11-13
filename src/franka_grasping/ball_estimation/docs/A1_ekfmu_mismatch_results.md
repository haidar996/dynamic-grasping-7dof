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

## P4 — arm R on all_real, 2026-11-13 04:21 → 05:31 (70 min)
Grid q 2, q_μ 10, s_l 0.05 (65.3 mm val). nll σ 8.1 mm, q 0.037, μ0 0.64, E 0.65. h05 σ 55 mm (!), q 1.45, σ_v0 7.7, E 0.47.

| h0.5 (mm) | ekf19b_fit | hindsight | ekfmu_grid | ekfmu_nll | ekfmu_h05 |
|---|---|---|---|---|---|
| test (dev) | 70 [62,78] | 50 | 67 | 79 | 68 |

Paired vs fit (test): grid −2.5 [−4.4, −0.5] (p 0.04), h05 −2.2 (n.s.), nll +9.0 [+4.3, +14.1] (p 8e-4). Gap closed: grid +0.13, h05 +0.11, nll −0.46.
Third core condition with the same picture: adaptive μ buys ≤ 3 mm of an 18–20 mm hindsight gap; ML-trained worse than fixed μ; h05 inflates σ (55 mm) to
discount a model it cannot trust (the same mechanism as ekf19b_fit's σ 80 mm on mismatch, F10).

## P4 — arm R on all_plane, 2026-11-13 06:03 → 07:21 (78 min)
Grid q 5, q_μ 10, s_l 0.5 (60.0 mm val). nll σ 1.7 mm, q 0.028, μ0 0.53, s_l 0.04, E 0.64. h05 σ 1.0 mm, q 0.013, μ0 0.52, s_l 0.10, E 0.50 (NIS 5.4: under-estimated R this time).

| h0.5 (mm) | ekf19b_fit | hindsight | ekfmu_grid | ekfmu_nll | ekfmu_h05 |
|---|---|---|---|---|---|
| test (dev) | 62 [55,69] | 43 | 59 | 66 | 62 |

Paired vs fit (test): grid −2.5 [−3.6, −1.3] (p 3e-4), h05 −0.2 (n.s.), nll +4.3 [+0.2, +8.4] (p 0.008). Gap closed: grid +0.13, nll −0.23, h05 ≈ 0.
Fourth core condition, same result. Across ideal / noise / all_real / all_plane: grid −2.9 / −1.8 / −2.5 / −2.5 (all p < 0.05), h05 −3.1 / −1.1 / −2.2 / −0.2
(only ideal significant), nll +0.6 / +25 / +9 / +4. Hindsight −18 / −15 / −20 / −19. Prediction 5 (C1 on mismatch) is REFUTED for h05 and nll; grid passes
with a ≈ 2–3 mm effect (gap closed 0.12–0.18).

## P4 — arm R on dropout_silent_noisy, 2026-11-13 07:21 → 08:40 (79 min) — arm R COMPLETE (5/5 core)
Grid q 0.5, q_μ 1, s_l 0.05 (83.1 mm val). nll degenerate again: q 4.5, μ0 1.99 (upper bound), s_l 0.018. h05 σ 3 mm, q 0.012, μ0 0.47, E 0.48.

| h0.5 (mm) | ekf19b_fit | hindsight | ekfmu_grid | ekfmu_nll | ekfmu_h05 |
|---|---|---|---|---|---|
| test (dev) | 77 [66,88] | 58 | 76 | **146** | 80 |

Paired vs fit (test): grid −0.4 (n.s.), h05 +3.4 (n.s.), nll +69.5 [+58, +81] (p 4e-13). Same failure mode as matched: ML explains stale frames by motion.

### Arm R summary (test dev, paired h0.5 vs ekf19b_fit re-tuned on mismatch; mm)
| cond | fit | hindsight | grid | h05 | nll |
|---|---|---|---|---|---|
| ideal | 54 | 35 (−18) | −2.9* | −3.1* | +0.6 |
| noise | 56 | 42 (−15) | −1.8* | −1.1 | +25* |
| all_real | 70 | 50 (−20) | −2.5* | −2.2 | +9* |
| all_plane | 62 | 43 (−19) | −2.5* | −0.2 | +4* |
| dsn | 77 | 58 (−19) | −0.4 | +3.4 | +70* |
(* p < 0.05 raw; Holm over the 5-condition family: grid stays significant in ideal, all_real, all_plane; h05 only in ideal.)
Verdict on the frozen predictions: H-q supported (nll q 0.03–0.07 in ideal/noise/all_real; the 1e-3 threshold cleared by > 10×). NEES < 20: nll 20 / 7 / — /
— / 7 → mostly yes. NLL–h05 gap narrower: NO — it widened (nll is now worse than fixed μ). Prediction 5: refuted for h05 and nll; grid ≈ 2–3 mm. 5b (gap > 1): no.
5c (q_μ up): grid yes (1–10), learned no.

## P5 — arm T (transfer: every parameter as learned / tuned on MATCHED data), 2026-11-13 08:42 → 08:58
Test (dev) h0.5 launch-mean (mm); fit = ekf19b_fit with matched tuning; in brackets the paired difference vs that fit; "/R" = the retrained value from arm R.

| cond | fit | hindsight | grid T (/R) | h05 T (/R) | nll T (/R) |
|---|---|---|---|---|---|
| ideal | 65 | 35 | 105 (+40) / 51 | 114 (+50) / 51 | 84 (+19) / 54 |
| noise | 70 | 42 | 123 (+53) / 55 | 116 (+46) / 55 | 118 (+48) / 82 |
| all_real | 87 | 50 | 131 (+44) / 67 | 122 (+35) / 68 | 129 (+42) / 79 |
| all_plane | 76 | 43 | 124 (+48) / 59 | 125 (+49) / 62 | 114 (+37) / 66 |
| dsn | 97 | 58 | 104 (+7) / 76 | 119 (+22) / 80 | 128 (+31) / 146 |
| dropout_flagged | 65 | 36 | 97 (+32) | 115 (+51) | – |
| delay | 72 | 38 | 114 (+42) | 127 (+55) | – |
| noise_real | 79 | 48 | 137 (+58) | 110 (+31) | – |
| plane | 69 | 40 | 116 (+46) | 115 (+46) | – |

Findings (all paired CIs far from 0):
1. **Transfer is catastrophic for ekf-μ.** Every matched-learned variant is 30–58 mm WORSE than the matched-tuned fixed-μ filter on mismatch data, in every
   condition, and ~2× worse than its own retrained version. The matched fits use tiny process noise (q 1e-5…0.02) and a tight prior on μ, i.e. they trust a
   model that is now wrong; the fixed-μ filter with its larger tuned q degrades gracefully (45 → 65 mm, Step A).
2. **Prediction 4 (NLL degrades more than h05) is REFUTED:** nll transfers *less badly* than h05 in ideal, all_plane, all_real (84 vs 114, 114 vs 125, 129 vs 122)
   and about equally in noise. The h05 fits are more specialised to matched physics (their σ, s_l are tuned to exploit the exact model), not less.
3. **Retraining recovers only to the fixed-μ level** (arm R: ≤ 3 mm gain). So on this mismatch data the adaptive-μ structure is useful only if its noise
   parameters are learned on the deployment physics, and even then it adds almost nothing over a well-tuned fixed model.
4. The hindsight gap (15–20 mm) is untouched by either arm: the remaining error is model structure (slide → roll, speed-dependent restitution), not parameters.

## Status after P0–P5 (2026-11-13 09:00): the mismatch evaluation of ekf-μ is complete for the core set (arm R) and all 9 distinct conditions (arm T).
Claims to rewrite: C1 holds on matched physics only; C3 becomes "adaptive μ does not survive model mismatch: transfer −40 mm, retrain ≈ 0"; the paper's
positive message is the diagnosis (why: q → 0 under exact physics; hindsight bound shows the information is there; model structure is the bottleneck).
