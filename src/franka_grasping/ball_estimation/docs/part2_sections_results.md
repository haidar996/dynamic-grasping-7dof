# Part 2 — ekf-μ results per section (condition), matched physics

Each section: `bash experiments/run_section.sh <cond>` = learn nll → learn h05 (step_1c: 1 start "from ekf19b tuning",
early stop, train launches) → evaluate (step_1e, val + test (dev), merged into outputs/ekf_mu_eval.json).
Logs: outputs/ekf_mu_learned_nll.log, _h05.log, ekf_mu_eval.log. Numbers: per-launch RMS, launch-mean [95% CI] / median (mm).

## NLL rerun with the safer stop rule (started 2026-11-09), READ THIS FIRST
Sections 1–11 below were run with NLL TOL 1e-3; their NLL numbers are SUPERSEDED as each condition is rerun with TOL 1e-4 (same starts, PATIENCE 10). Old results:
outputs/ekf_mu_learned_nll_tol1e-3.*, ekf_mu_eval_tol1e-3.*, .sections_done_tol1e-3. h05, grid, fixed-μ and oracle are unchanged (noise_real: per-launch h0.5 differs by
exactly 0 mm from the archive).

### noise_real, acceptance test (03:26 → 03:57): PASSED
Train NLL from the ORIGINAL start: −17.17240 (target ≈ −17.17; old −17.06935; continuation tests −17.173, −17.174). 86 gradient evals, 28.7 min (old 19, 5.7 min). No bound.
Parameters σ 8.0 mm, q 0.00030, μ0 0.59, s_l 0.38, σ_v0 2.08, E 0.74 (old q 0.093, μ0 0.85, E 0.83).

| h0.5 | ekf19b_fit | ekfmu_grid | ekfmu_nll (new) | ekfmu_nll (old) | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 61 [55,67] | 40 [35,45] | 47 [42,52] | 55 | 38 [33,43] | 9 |
| test (dev) | 68 [60,75] | 48 [41,56] | 51 [44,59] | 62 | 45 [39,53] | 18 |

Paired vs ekf19b_fit (new nll): val −13.8 [−17.8,−9.7], test −16.6 [−22.2,−9.7] (p ≤ 2e-7, 78 % of launches). Old: −5.4, −5.8.
Paired new − old nll: val −8.4 [−13.0,−3.8], test −10.8 [−16.2,−4.1]. h05 − new nll: val −9.0 [−12.2,−5.9], test −5.8 [−11.4,−1.3] (53 % of launches, p 0.03).
New nll − grid: val +6.9 [+4.4,+9.4], test +3.3 [+0.9,+5.9]. Raw p, not corrected.
μ error at 1 s (test): nll 0.041 (old 0.184), grid 0.039, h05 0.047.
Consistency got WORSE in one respect: NIS 2.00 (test), 2.02 (val), but NEES test 131.7 (old 7.1), val 8.4 (old 3.3). Observation (not tested): very large test NEES appears with small q: all nll q 0.00018, NEES 396. Candidate
explanation for the NEES TODO: small q makes the filter overconfident at a few events.
Three fits now agree on NLL within 0.001 but have μ0 0.68 / 0.58 / 0.59 and val h0.5 44.7 / 47.7 / 46.9 mm: the likelihood ridge is real.

### noise (04:09 → 04:19): the 1e-4 rule is ALSO not enough; rerun PAUSED after this
Train NLL −21.80748 after 23 gradient evals: same point as the old fit (22 evals), parameters and h0.5 unchanged. The last evals repeat −21.80744 … −21.80748, so the line search is stuck.
Check on the noise TRAIN launches (scratchpad, single evaluations; val only reported):
| parameters | train NLL | val h0.5 (mm) |
|---|---|---|
| noise, learned | −21.80748 | 31.0 |
| all, old fit | **−21.86183** | 35.9 |
| noise, only q taken from all | −21.68192 | 30.8 |
1. A better likelihood exists (0.054 better = 500 × TOL), so this is NOT the optimum. Tightening TOL does not fix a stuck optimiser. The noise_real acceptance test shows the rule worked there, not in general.
2. The better likelihood forecasts WORSE on val (35.9 vs 31.0 mm). So stopping early may have been HELPING NLL here, acting like a regulariser. Published NLL numbers may be better than a true NLL optimum would give, and
   "NLL ≈ h05 on noise" could be partly this accident. A true optimum is needed before comparing criteria.
3. q alone does not explain it: taking only q from all makes the NLL worse (−21.68). Parameters interact, so the best point needs a joint search.

### Pilot on noise (2026-11-09 04:32 → 05:18; train + val only, test (dev) NOT looked at): PASSED
Script experiments/step_1h_nll_restart_pilot.py, results outputs/pilot_nll_noise.json/.log. Two starts + restart on stall, cap 80 gradient evals per start. Pass mark −21.86, fixed before running.
| start | train NLL | val h0.5 (mm) | q | q_mu | μ0 | E | grads, rounds |
|---|---|---|---|---|---|---|---|
| published fit | −21.80748 | 31.0 | 0.0024 | 0.0010 | 0.77 | 0.72 | 23, 1 |
| A original, with restarts | −21.80783 | 31.0 | 0.0024 | 0.0011 | 0.82 | 0.71 | 60, 4 |
| B low process noise, with restarts | **−21.87164** | 32.8 | 0.0001 (AT BOUND) | 0.00011 | 0.65 | 0.70 | 80 (cap), 3 |
Reading:
1. Starts matter: A stays near −21.808 even with restarts, so it sits at a different optimum, not merely a stall.
2. Restarts matter too: B stalled at −21.83210 in round 1 and gained 0.0395 in round 2. Correction of the interim note "the second start helps, not the restarts": BOTH were needed to pass.
3. Forecast: the better likelihood is 1.8 mm WORSE on val than the published fit (unpaired means). That is the same direction as the all parameters (35.9), but small. Mild support for point 2 above, not proof.
4. B has q at its lower bound 1e-4, so the bound is active. Hypothesis (untested): on matched physics the model is exact, so maximum likelihood correctly wants q → 0. If so, this behaviour is partly the inverse crime and could change on mismatch data.
5. Cost: 46 min for this condition (old 7.6 min).
Proposed rerun, NOT run: 2 starts + restart on stall for NLL only, cap 80 per start, q lower bound decided first (see RESUME); 9 fits + 11 evals, roughly 7.5 h.

### noise with the final NLL rule (commit 29ce137: 2 starts + restart on stall, cap 80, q ≥ 1e-6), 2026-11-10 11:42 → 12:35, 53 min
Start A (original) −21.80783 (60 grads, 4 rounds) = pilot exactly. Start B (low process noise) **−21.87180** (80 grads = cap, 3 rounds; round 3 gained 0), chosen.
B: σ 2.46 mm, q 8.1e-5 (BELOW the old 1e-4 bound, so that bound was active in the pilot; no bound now), q_mu 1.1e-4, μ0 0.65, s_l 0.44, E 0.69.

| h0.5 | ekf19b_fit | ekfmu_grid | ekfmu_nll (final) | ekfmu_nll (TOL 1e-3) | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 52 | 31 | 33 [28,39] | 31 | 30 | 5 |
| test (dev) | 59 | 36 | 36 [30,42] | 35 | 34 | 8 |

Paired, new NLL − old NLL: val +2.2 [+0.7,+3.6] (p 0.006, new better in 34 %), test +0.7 [−0.7,+2.2] (n.s.). h05 − new NLL: val −3.2 [−5.4,−1.0], test −1.7 [−3.7,+0.3].
NIS 2.00 (test), 1.98 (val); NEES test 662 (old 79), val 8.6 (old 4.4). μ error at 1 s (test) 0.016 (old 0.027).
Reading:
1. The better likelihood forecasts slightly WORSE on val (significant) and no better on test. Confirms on a paired basis what the pilot suggested.
2. Large dev-test NEES grows as q falls, across the fits seen so far: q 0.093 → 7; 0.0048 (h05) → 22; 0.0024 → 79; 0.0003 → 132; 0.00018 → 396; 8e-5 → 662. Strong pattern (different fits and
   conditions, not a controlled test). NIS stays ≈ 2, so innovations look calibrated while the state covariance is overconfident at a few events.
3. Consistent with the inverse-crime hypothesis: on matched physics, maximum likelihood drives process noise towards 0, which is right between events but leaves no room for the rare
   big model errors (bounces, corner hits) that dominate NEES and forecast error. Untested; the mismatch data will tell.

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

## Section 6 — dropout_silent (nll 09:36 → 09:49, h05 + eval 12:55 → 13:22) — IDENTICAL to dropout_flagged, not an independent result
Numbers equal Section 5: test fit 55, grid 32, nll 37, h05 30; val 48, 27, 31, 27. Direct comparison with dropout_flagged:
learned parameters and train objectives (nll −25.061957269129547, h05 0.03164147147773303) are bit-identical, and all
4500 per-launch error arrays (75 test + 75 val launches × 6 estimators × 5 metrics) differ by exactly 0.
Cause = audit B4, not a new bug: silent_hold repeats the bit-identical last value (sensing/sensor.py) and StaleGuard drops
bit-identical values (estimators/filters.py), so silent loss becomes flagged loss by construction.
How to report: merge with dropout_flagged, do NOT count as a separate condition or a separate test in any multiple-comparison
count. Limitation to state: the guard relies on exact repeats. The hard, realistic version of silent loss is
dropout_silent_noisy (Section 4), where ekf-μ gains nothing.

## Section 7 — delay (13:28 → 14:05, 37 min)
Learning: nll 9.2 min, h05 25.6 min (53 grads), both early stop, no bound. nll is BIT-IDENTICAL to ideal (objective
−25.140794562392315, same parameters, same μ recovery at 1 s 0.070): a frame's data is unchanged, only its arrival time moves,
so the likelihood carries no delay information. h05 differs slightly: σ 4.3 mm, q 0.019, μ0 0.68, s_l 1.63, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 52 | 54 [47,60] | 30 [24,36] | 35 [29,41] | 30 [24,36] | 4 |
| test (dev) | 62 | 62 [54,71] | 41 [29,55] | 43 [35,51] | 33 [26,41] | 8 |

Paired vs ekf19b_fit: val grid −24.2, nll −18.9, h05 −23.9; test −20.7 / −19.1 / −28.6 mm (p ≤ 2e-9, 82–90 % of launches).
Cost of the 40 ms delay vs ideal (launch means, not a paired test): test fit +7, grid +2, nll +5, h05 +3; val fit +7, grid +3, nll +4, h05 +3 mm.
μ error at 1 s (test): grid 0.019, nll 0.070, h05 0.014. NIS (expected 2): test grid 0.29, nll 1.34, h05 0.08. NEES (expected 4):
test grid 16.9, nll 20.5, h05 1.2. Grid fragile on test again: mean 41 vs median 22 mm, wide CI, val NEES 1.6.
Limitation: the delay is exactly known and fixed (40 ms); a real, varying or unknown delay would cost more.

## Section 8 — all (14:06 → 14:57, 51 min, slowest so far)
Learning: nll 22.8 min (66 grads), h05 25.4 min (47 grads), both early stop, no bound. nll σ 2.5 mm, q 0.00018, μ0 0.40, s_l 0.88,
E 0.74 | h05 σ 6.6 mm, q 0.00069, μ0 0.84, s_l 1.00, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 59 | 61 [54,67] | 36 [30,42] | 42 [36,48] | 34 [29,40] | 6 |
| test (dev) | 67 | 67 [59,76] | 41 [34,47] | 45 [38,52] | 39 [33,46] | 10 |

Paired vs ekf19b_fit: val grid −24.6, nll −18.4, h05 −26.4; test −26.9 / −22.3 / −28.5 mm (p ≤ 2e-7, 75–89 % of launches).
Cost of realistic depth noise (all_real − all, launch means, not a paired test): test fit +9, grid +14, nll +13, h05 +12 mm.
μ error (test): at 0.1 s grid 0.31, nll 0.32, h05 0.33; at 1 s 0.018 / 0.020 / 0.016. On val nll starts worse (0.44 at 0.1 s,
μ0 = 0.40 vs prior 0.85) but catches up by 1 s (0.013). The low μ0 appears to cost only the early flight (not tested separately).
NIS (expected 2), test: grid 3.00, nll 2.01, h05 0.29. NEES (expected 4), test: grid 676, nll 396, h05 1.5; val: 40.8 / 25.3 / 4.8.
TODO: find what drives NEES 396–676 on test while now_pos stays 3–4 mm (per-launch NEES is not in the errors file).

## Section 9 — noise_real (22:18 → 22:47, 29 min) — weak NLL result is mostly an EARLY-STOP ARTEFACT (see "Continuation test" below)
Learning: nll 5.7 min (19 grads, fewest so far), h05 20.8 min (46 grads), both early stop, no bound. nll σ 7.9 mm, q 0.093,
μ0 0.85, s_l 0.48, E 0.83 | h05 σ 14 mm, q 0.0015, μ0 0.93, s_l 0.72, σ_v0 1.18, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 58 | 61 [55,67] | 40 [35,45] | 55 [50,62] | 38 [33,43] | 9 |
| test (dev) | 67 | 68 [60,75] | 48 [41,56] | 62 [55,69] | 45 [39,53] | 18 |

Paired vs ekf19b_fit: val grid −20.7, nll −5.4 (p 0.046, 56 %), h05 −22.8; test −19.8, nll −5.8 [−11.6, −0.5] (p 0.055, 59 %), h05 −22.3 mm.
μ error at 1 s (test): grid 0.039, nll 0.184, h05 0.047. NIS (expected 2), test: grid 0.90, nll 1.98, h05 0.68.
NEES (expected 4), test: grid 112, nll 7.1, h05 35.3; val: 27.8 / 3.3 / 2.9.

NLL question (all_real, same depth noise, gained −18.5 on test with q 1e-4; here q 0.093): objective at other parameters, noise_real
(scratchpad check, single evaluations, reproduces the eval table for learned nll 55 and h05 38):
| parameters | train NLL | val h0.5 (mm) |
|---|---|---|
| nll noise_real (learned) | −17.069 | 55.3 |
| nll all_real | −17.014 | 43.8 |
| h05 noise_real | −16.312 | 37.9 |
→ The hypothesis "a clearly better likelihood exists at the all_real parameters" is NOT supported: learned is better by 0.055. But the
NLL is nearly flat while forecast error changes by 11.5 mm: on this data NLL hardly distinguishes good from poor forecasting
parameters. STILL OPEN whether the optimiser stopped early: gain over the last 10 evals was 0.00098 (TOL 0.001) with E still moving
(0.73 → 0.83 in 3 evals).

### Continuation test (2026-11-09, val only; test (dev) NOT looked at)
Script experiments/step_1f_continue_nll_check.py, results outputs/continue_nll_noise_real.json/.log. Same procedure as step_1c on noise_real, but
stop at gain < 1e-4 over 10 gradient evals (or 40 evals). No published result was overwritten.
| | train NLL | val h0.5 (mm) | q | μ0 | s_l | E | grads |
|---|---|---|---|---|---|---|---|
| published fit (TOL 1e-3) | −17.06935 | 55.3 | 0.093 | 0.85 | 0.48 | 0.83 | 19 |
| A, continued from published fit | −17.17328 | 44.7 | 0.00021 | 0.68 | 0.44 | 0.77 | 40 (limit) |
| B, started from all_real parameters | −17.17365 | 47.7 | 0.00022 | 0.58 | 0.49 | 0.77 | 39 |
| for comparison: grid / h05 / fixed μ (fit) | | 40.0 / 37.9 / 60.7 | | | | | |
Findings:
1. Early stop DID cut this fit short: 0.104 of likelihood was left (100 × TOL), and q was wrong by > 400 ×. The published noise_real NLL row is not a valid
   picture of the NLL criterion. CORRECTION of the first reading above: "NLL gain nearly vanishes" was mostly this artefact.
2. The stop rule is unsafe: the published fit stopped with last-10 gain 0.00098, close to TOL, while 100 × TOL remained. A small last-10 gain does not
   prove convergence here. TOL 1e-3 was only validated on ideal (2 starts: −25.14079 vs −25.14093).
3. The corrected NLL still forecasts worse than grid and h05 on val (44.7–47.7 vs 40.0 and 37.9 mm), so h05 ≥ nll may survive, with a smaller gap.
4. A and B agree on NLL within 0.0004, yet differ by 3 mm in val h0.5, with μ0 0.68 vs 0.58: a flat ridge in the likelihood. The earlier "NLL is flat
   compared with forecast error" point survives only in this smaller form.
Other NLL fits may be affected (not tested). Same-noise pairs differ more than they should, given that delay adds no NLL information: noise q 0.0024 (22 grads,
last-10 gain 0.00063) vs all q 0.00018; all_plane q 0.0020 (0.00064) vs plane q 0.00015. dropout_flagged (0.00060) is also near TOL.

### h05 continuation test (2026-11-09, noise_real only, val only; test (dev) NOT looked at)
Script experiments/step_1g_continue_h05_check.py, results outputs/continue_h05_noise_real.json/.log. Stop at gain < 1e-6 m over 10 gradient evals (published 1e-5).
| | train h0.5 (mm) | val h0.5 (mm) | q | μ0 | σ_v0 | grads |
|---|---|---|---|---|---|---|
| published fit | 44.645 | 37.9 | 0.0015 | 0.93 | 1.18 | 46 |
| A, continued from published fit | 44.643 | 37.9 | 0.0015 | 0.93 | 1.19 | 14 |
| B, started from all_real h05 parameters | 44.686 | 37.8 | 0.00069 | 0.88 | 6.98 | 14 |
Reading: unlike NLL, h05 did not move: gain 0.002 mm on train and none on val. No evidence that early stop hurt h05 on noise_real. A and B differ in q (2 ×) and σ_v0 (6 ×),
yet their forecast error differs by only 0.04 mm on train and 0.1 mm on val, so h05 is flat in these two parameters.
Limits: one condition only. Both restarts barely moved, and this objective is non-smooth, so a stuck optimiser cannot be told apart from a converged one; B staying at its
different parameters fits either. The check shows that the h05 result is stable for forecasting, not that it found the global optimum.

## Section 10 — plane (nll 2026-11-08 22:58 → 23:13; h05 + eval 2026-11-09 00:15 → 00:35; first h05 attempt stopped by user)
Learning: nll 14.3 min (48 grads), h05 17.5 min (39 grads), both early stop, no bound. nll σ 1.75 mm, q 0.00015, μ0 0.65, s_l 0.40,
E 0.65 | h05 σ 6.5 mm, q 0.012, μ0 0.74, s_l 1.43, E 0.72.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 49 | 51 [45,56] | 30 [25,35] | 35 [29,42] | 29 [24,35] | 5 |
| test (dev) | 57 | 58 [50,65] | 33 [28,40] | 37 [31,44] | 33 [27,39] | 8 |

Paired vs ekf19b_fit: val grid −20.7, nll −15.3, h05 −21.1; test −24.1 / −20.3 / −24.9 mm (p ≤ 3e-5, 74–90 % of launches).
Paired plane − noise_real (same jitter draws; 73 of 75 launches have valid h0.5 in each set), mm:
| | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|
| test | −10.1 [−12.1,−7.8] | −14.4 [−20.9,−9.8] | −24.6 [−31.2,−17.8] | −12.7 [−20.1,−5.4] | −10.5 [−19.2,−4.5] |
| val | −10.2 [−11.6,−8.8] | −10.2 [−13.1,−7.7] | −20.0 [−28.0,−11.9] | −8.5 [−11.0,−6.0] | −4.2 [−4.7,−3.6] |
All p ≤ 2e-5 (Wilcoxon). The fixed-μ gain matches Part 1 (−10 mm). The NLL gain is inflated by the weak noise_real NLL fit (Section 9).
μ error at 1 s (test): grid 0.013, nll 0.015, h05 0.016 (noise_real: 0.039 / 0.184 / 0.047). NIS (expected 2), test: grid 0.68, nll 2.00,
h05 0.15. NEES (expected 4), test: grid 4.4, nll 14.8, h05 0.9; val: 3.0 / 10.4 / 0.6. Much calmer than all (396–676) and noise_real.
Plane assumes the floor is flat and at a known height; tilt or height error is not in this condition.

Early-stop evidence (open question from Section 9): nll in plane, all_real and noise_real all START at q = 0.1. plane (48 grads, q 1.5e-4)
and all_real (31 grads, q at the 1e-4 bound) travel far down; noise_real stops at q 0.093 after 19 grads. Same start, so noise_real looks like the outlier.
Confirmed by the continuation test (Section 9). The plane − noise_real NLL gains above use the too-early noise_real fit, so they are overstated.

## Section 11 — all_plane (2026-11-09 00:47 → 01:31, 44 min)
Learning: nll 9.2 min (31 grads), h05 31.9 min (67 grads, longest), both early stop, no bound. nll σ 1.7 mm, q 0.002, μ0 0.77, s_l 0.39,
E 0.69 | h05 σ 5.5 mm, q 0.0042, μ0 0.83, s_l 1.05, E 0.73.

| h0.5 | ekf19b | ekf19b_fit | ekfmu_grid | ekfmu_nll | ekfmu_h05 | oracle |
|---|---|---|---|---|---|---|
| val | 57 | 59 [52,66] | 34 [29,40] | 36 [30,42] | 33 [28,39] | 5 |
| test (dev) | 65 | 65 [57,74] | 38 [32,45] | 40 [33,47] | 37 [31,44] | 9 |

Paired vs ekf19b_fit: val grid −24.6, nll −22.8, h05 −25.8; test −27.6 / −25.7 / −28.6 mm (p ≤ 3e-8, 82–90 % of launches).
Paired all_plane − all_real (72–73 launches): test fit −10.7, grid −16.9, nll −17.9, h05 −14.7, oracle −13.1; val −11.8 / −13.3 / −16.0 /
−11.8 / −6.1 mm (all CIs exclude 0, p ≤ 1e-5).
μ error at 1 s (test): grid 0.016, nll 0.021, h05 0.014. NIS (expected 2), test: grid 0.68, nll 1.97, h05 0.20. NEES (expected 4), test: grid 4.3,
nll 9.6, h05 1.3; val: 3.0 / 6.3 / 1.0.

## Overall reading after 11/11 sections (matched physics; 10 distinct conditions, dropout_silent = dropout_flagged)
Paired h05 comparisons (bootstrap CI + Wilcoxon, raw p, NOT corrected for the 44 comparisons; test (dev) has been inspected repeatedly):
| cond | h05 − grid, test | h05 − grid, val | h05 − nll, test | h05 − nll, val |
|---|---|---|---|---|
| ideal | −8.8 [−21.5,−0.3] (p 0.53) | +0.1 [−1.2,+1.6] | −8.5 [−11.9,−5.7] | −4.3 [−7.1,−1.8] |
| noise | −1.1 [−2.3,+0.1] | −1.4 [−3.1,+0.3] | −1.0 [−2.1,+0.1] | −1.0 [−2.4,+0.2] |
| all_real | −3.3 [−8.6,+0.1] | −2.7 [−4.1,−1.4] | −6.1 [−12.3,−1.0] | −7.2 [−9.9,−4.5] |
| dropout_silent_noisy | −0.1 [−21.5,+25.3] | +27.2 [−2.0,+61.7] | −213 [−282,−164] | −164 [−199,−124] |
| dropout_flagged | −1.9 [−3.3,−0.7] | +0.2 [−0.9,+1.2] | −7.1 [−10.6,−4.0] | −3.4 [−6.3,−0.7] |
| delay | −7.9 [−18.8,−0.2] (p 0.36) | +0.3 [−1.1,+1.7] | −9.5 [−13.1,−6.3] | −5.0 [−8.2,−1.9] |
| all | −1.6 [−3.0,−0.1] | −1.8 [−3.5,−0.2] | −6.2 [−9.3,−3.2] | −8.0 [−11.4,−4.8] |
| noise_real | −2.5 [−7.5,+0.7] | −2.1 [−3.5,−0.8] | −16.6 [−24.4,−8.7] | −17.4 [−21.9,−12.6] |
| plane | −0.8 [−1.6,+0.0] | −0.4 [−1.2,+0.4] | −4.6 [−8.0,−1.2] | −5.8 [−9.5,−2.1] |
| all_plane | −1.0 [−1.7,−0.3] | −1.2 [−2.1,−0.3] | −2.9 [−5.4,−0.5] | −3.0 [−5.6,−0.5] |

1. ekf-μ beats the fixed-μ filter by about 20–29 mm for h05 and grid in 9 of 10 conditions. In dropout_silent_noisy it gains nothing.
2. h05 vs grid: usually ≤ 3 mm. The two larger test gaps (ideal, delay) are means pulled by grid outliers (Wilcoxon not significant, val tied).
   Fair summary: tied on typical launches, h05 more robust on the test (dev) launches; h05 worse on val in dropout_silent_noisy.
3. h05 vs nll: h05 is better or level everywhere in this table, but the whole NLL column is PROVISIONAL: the noise_real fit was stopped too early (Section 9,
   continuation test) and other fits may be too. Do not quote NLL numbers or the h05 − nll gaps until NLL is re-learned with a safer stop.
4. Consistency: nll has NIS near 2; h05 NIS 0.08–0.7 (too cautious), so h05 covariances are not honest uncertainties.
Still required before any claim: ekf-μ on mismatch physics (A1), NLL re-learning with a safer stop rule, NEES on all (grid 676, nll 396), grid outliers
461/480, launch 235, final test set (A2, render only when methods are frozen).

## Summary (all 11 sections) (h0.5 test (dev), mm)
| cond | fit | grid | nll | h05 |
|---|---|---|---|---|
| ideal | 55 | 39 | 38 | 30 |
| noise | 59 | 36 | 35 | 34 |
| all_real | 76 | 55 | 58 | 51 |
| dropout_silent_noisy | 96 | 99 | 312 | 99 |
| dropout_flagged | 55 | 32 | 37 | 30 |
| dropout_silent (= flagged, audit B4) | 55 | 32 | 37 | 30 |
| delay | 62 | 41 | 43 | 33 |
| all | 67 | 41 | 45 | 39 |
| noise_real | 68 | 48 | 62 | 45 |
| plane | 58 | 33 | 37 | 33 |
| all_plane | 65 | 38 | 40 | 37 |
