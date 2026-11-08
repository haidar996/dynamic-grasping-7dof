# Part 2 — ekf-μ results per section (condition), matched physics

Each section: `bash experiments/run_section.sh <cond>` = learn nll → learn h05 (step_1c: 1 start "from ekf19b tuning",
early stop, train launches) → evaluate (step_1e, val + test (dev), merged into outputs/ekf_mu_eval.json).
Logs: outputs/ekf_mu_learned_nll.log, _h05.log, ekf_mu_eval.log. Numbers: per-launch RMS, launch-mean [95% CI] / median (mm).

## Section 1 — ideal (2026-11-08 06:31 → 06:59, 28 min)
Learning: nll 9.3 min (31 grads, early stop), train −25.14079 (first run 2 starts / 71.5 min: −25.14093 → early stop lost
nothing; val h0.5 31.1 both). h05 17.4 min (38 grads, early stop). No parameter at a bound.
Learned: nll σ 0.71 mm, q 0.016, μ0 0.81, s_l 0.30, E 0.72 | h05 σ 4.2 mm, q 0.018, μ0 0.70, s_l 1.55, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 45 [40,51] | 47 | 27 [21,33] | 31 [26,37] | 27 [22,33] | 4 |
| test (dev) | 54 [47,62] | 55 | 39 [26,53] | 38 [31,46] | 30 [23,37] | 7 |

Paired vs ekf19b_fit (h0.5): val grid −20.4, nll −15.9, h05 −20.2; test grid −16.0, nll −16.3, h05 −24.8 mm (all p ≤ 2e-9).
μ recovery median |μ̂−μ| at 1 s (test): grid 0.023, nll 0.070, h05 0.014 (prior error 0.38).
Consistency (expected NIS 2, NEES 4): test grid NIS 39 / NEES 400 (!), nll 1.3 / 20, h05 0.09 / 1.2; val grid 0.16 / 1.6.

Findings:
1. μ in the state works: every ekf-μ variant beats the fitted constant-μ filter by 16–25 mm (≈ −35 to −45 %).
2. Learning on the forecast criterion (h05, on train) is best on test (30 mm) and equals the val-grid on val (27).
   The paper-03 criterion (NLL) is worse for forecasting (38 / 31 mm) and recovers μ slowly (0.07 at 1 s vs 0.014).
3. ekfmu_grid is fragile on test: NIS 39, NEES 400, now_pos mean 4 vs median 1 mm → a few test launches diverge with the
   val-tuned settings (σ 2 mm, small q). TODO: find which launches (corner hit, e.g. launch 303?).
4. Still far from the oracle (4–7 mm): remaining gap = E and the noise of μ̂ early in the flight.

## Section 2 — noise (06:59 → 07:29, 30 min)
Learning: nll 6.8 min, h05 20.5 min, both early stop, no bound. nll σ 2.4 mm, q 0.0024, μ0 0.77, s_l 0.49, E 0.72 |
h05 σ 7.3 mm, q 0.0048, μ0 0.81, s_l 1.10, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 50 | 52 [46,58] | 31 | 31 | 30 [25,35] | 5 |
| test (dev) | 59 | 59 [52,67] | 36 | 35 | 34 [29,41] | 8 |

Paired vs ekf19b_fit (h0.5): val grid −20.6, nll −20.9, h05 −22.0; test −23.8 / −23.9 / −24.9 mm (all p ≤ 3e-11, 84–93 % of launches).
With noise the three ekf-μ variants are equal within ~1–2 mm (CIs overlap); h05 slightly best.
μ recovery at 1 s (test): grid 0.016, nll 0.027, h05 0.017. NIS (expected 2): nll 1.96 (calibrated!), grid 2.98, h05 0.23.
NEES test: grid 331, nll 79, h05 22 (val 8.5 / 4.4 / 0.8) → test outliers again.

## Diverging test launches (from ekf_mu_eval_errors.pkl.gz, now_pos per-launch RMS, mm)
- ideal: launch 461 (4.7 m/s, μ 1.49, 1 arm hit): grid 137.7 vs fit 2.4, nll 2.9, h05 3.3; launch 480 (4.9 m/s, μ 1.23,
  2 arm hits): grid 39.6 vs ~1.3 others → ekfmu_grid (σ 2 mm, s_l 3, q_mu 0.01 for ideal) diverges on fast high-μ arm-hit
  launches; learned variants do not.
- launch 235 (2.4 m/s, μ 0.94, no arm hit): bad for EVERY filter (fit 58 mm now_pos in ideal) → TODO inspect (corner / model).

## Section 3 — all_real (07:29 → 07:58, 29 min)
Learning: nll 9.3 min (q at LOWER BOUND 1e-4 = q → 0 limit; σ 8.0 mm, μ0 0.64, s_l 0.57, E 0.75), h05 17.3 min (no bound;
σ 14 mm, q 0.0007, μ0 0.88, s_l 0.75, σ_v0 7.0, E 0.73).

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 68 | 71 | 48 | 52 | 45 | 11 |
| test (dev) | 76 | 76 | 55 | 58 | 51 | 23 |

Paired vs fit: val grid −23.1, nll −18.6, h05 −25.8; test −21.3 / −18.5 / −24.6 mm (p ≤ 8e-8). NIS nll 2.0–2.1 (calibrated),
grid 0.9, h05 0.7. μ error at 1 s 0.03–0.05. → Same ranking as ideal/noise under all realistic effects: h05 > grid > nll.

## Section 4 — dropout_silent_noisy (07:58 → 08:13, 15 min) — ekf-μ does NOT help; NLL learning COLLAPSES
Learning: nll 4.4 min, AT BOUNDS σ 0.1 mm (lower), s_l 3.0, E 0.99 (upper); q 77.6, μ0 0.10 → degenerate. h05 8.6 min, no bound.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 93 | 94 | 95 | **286** | 122 | 45 |
| test (dev) | 95 | 96 | 99 | **312** | 99 | 38 |

Paired vs fit: grid +0.2 / +2.9, h05 +27.5 (val, p 0.3) / +2.7 (test), nll +192 / +215 mm (p 1e-13).
h05 val NIS 50 / test 17 → a few launches blow up (val mean 122 vs test 99).
Interpretation: in dsn, frozen frames carry FRESH noise, so the stale guard cannot spot them; they look like real
measurements of a stopped ball. Maximum likelihood then explains every jump by huge process noise (q 78) and near-zero
measurement noise (σ 0.1 mm) = the filter just follows the measurements, and its forecasts are useless. NLL learning is
unsafe when the measurement model is wrong in this way; the forecast criterion (h05) and the val grid are not fooled
(but gain nothing over a fixed μ either). Report as a failure mode of paper 03's criterion, not a bug.

## Section 5 — dropout_flagged (08:55 → 09:35, 40 min)
Learning: nll 15.3 min, h05 22.7 min, both early stop, no bound (values close to ideal: nll σ 0.8 mm, q 0.008, μ0 0.79;
h05 σ 4.3 mm, q 0.020, μ0 0.71, s_l 1.52).

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 46 | 48 [42,54] | 27 | 31 | 27 [22,33] | 4 |
| test (dev) | 55 | 55 [48,63] | 32 | 37 | 30 [23,37] | 11 |

Paired vs fit: val grid −20.8, nll −17.3, h05 −20.6; test −23.4 / −18.2 / −25.4 mm (p ≤ 1e-7, 84–92 % of launches).
μ error at 1 s (test): grid 0.030, nll 0.047, h05 0.014. Flagged dropouts cost almost nothing (≈ ideal). Grid stable here
(test NEES 3.4, unlike ideal).

## Summary (sections done so far) (h0.5 test (dev), mm)
| cond | fit | grid | nll | h05 |
|---|---|---|---|---|
| ideal | 55 | 39 | 38 | 30 |
| noise | 59 | 36 | 35 | 34 |
| all_real | 76 | 55 | 58 | 51 |
| dropout_silent_noisy | 96 | 99 | 312 | 99 |
| dropout_flagged | 55 | 32 | 37 | 30 |
