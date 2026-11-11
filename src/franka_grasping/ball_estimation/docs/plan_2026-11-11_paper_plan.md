# Paper plan after the rethink (2026-11-11)

Written after re-reading every results document (Part 1, Step A, Part 2 sections, the audit, the A1 designs) with the
paper as the goal. It replaces the day-to-day queue in RESUME.md as the reference for *what* we do and *why*; RESUME.md
keeps the live state. Compute is on a 4-core laptop that is often on battery, so every step has an hour estimate and a
go/no-go gate, and nothing heavy starts without an explicit go.

## 1. What the evidence supports today

| # | Finding | Evidence | Status |
|---|---|---|---|
| F1 | A fixed-μ physics EKF is ~8× better than model-free filters at 0.5 s, but far from the oracle | Part 1: fit 55–76 mm, cvkf 410–510, oracle 7–23 mm (test dev) | solid, matched only |
| F2 | The oracle gap is mostly the unknown per-launch friction ("μ first") | μ-only −21 mm vs E-only −11 mm (train+val, paired) | solid; weaker on mismatch (−17 vs −10) |
| F3 | μ in the EKF state recovers ~45 % of the fixed-μ error | ekf-μ −20…−29 mm vs ekf19b_fit, p ≤ 1e-7, 9 of 10 distinct conditions | solid, **matched only** |
| F4 | Training criterion matters: forecast loss (h05) ≥ val grid ≥ likelihood (NLL) for forecasting | all 10 conditions; h05 − nll −3…−17 mm | h05/grid solid; NLL column provisional (only `noise` under the final optimiser) |
| F5 | Maximum likelihood on exact physics drives q → 0, makes NEES explode (7 → 662) while NIS stays 2, and forecasts slightly worse | noise: −21.872 vs −21.807 NLL, +2.2 mm val (p 0.006), NEES 662 | pattern across 6 fits, mechanism untested → A1 arm R tests it |
| F6 | NLL learning collapses when stale frames carry fresh noise (q 78, σ 0.1 mm, 312 mm); forecast training does not, but no estimator gains anything | dropout_silent_noisy | solid; a detection problem, not an estimation one |
| F7 | Plane-ray measurement beats depth back-projection by ~10 mm for every filter | plane − noise_real, all_plane − all_real, p ≤ 1e-5 | solid (B1 answered) |
| F8 | Flagged dropout and a known 40 ms delay cost little (0–7 mm); undetected stale frames cost +36 mm | Part 1 + Part 2 | solid |
| F9 | On mismatch physics the fixed-μ EKF loses 20 mm and the true-constant oracle loses 30 mm; the best constant pair per launch still leaves 35 mm | Step A | solid, baselines only |
| F10 | Fixed-model filters compensate model error by inflating R: ekf19b_fit on mismatch wants σ 20–30 mm (true noise a few mm) | mismatch re-tune run 1 (grid edge, rerun pending) | provisional |
| F11 | h05-trained covariances are dishonest (NIS 0.1–0.7); NLL-trained are calibrated (NIS ≈ 2) but overconfident in the state (NEES) | all conditions | solid |

Not yet shown, and required before any claim: ekf-μ on mismatch physics (A1); NLL under the final optimiser on the
reported conditions; the untouched final test set (A2); the required learned-estimator comparison.

## 2. The paper's claims (write the claim, then test it)

Working title: *Adaptive friction estimation for fast planar ball forecasting: when does learning the filter help, and
what survives model mismatch?*

- **C1 (method):** putting the friction deceleration in the EKF state with a bounce-aware Jacobian cuts 0.5 s forecast
  error by about 45 % against a train-fitted fixed-μ EKF, across sensor conditions. (F3; must hold on mismatch, F9 →
  A1 arm T/R, and on the final test set.)
- **C2 (criterion):** training the filter by forecast loss beats maximum likelihood (paper 03's criterion) for forecasting;
  ML is also fragile: it over-trusts exact physics (q → 0, NEES ≫ 4) and collapses under undetected stale frames.
  Forecast-trained covariances are not calibrated, which a one-parameter post-hoc inflation tuned on val repairs
  (new, cheap; see §4 step P3). (F4, F5, F6, F11.)
- **C3 (mismatch):** under unmodelled physics (slide → roll, speed-dependent restitution, tilt) the fixed-μ EKF loses
  ~20 mm and compensates by inflating R; adaptive μ keeps part of its gain, and retraining on mismatch data moves q and
  q_μ up (if predictions H-q, 5b, 5c hold). (F9, F10 → A1.)
- **C4 (sensor):** ray–plane intersection should replace depth back-projection for a planar ball (−10 mm for every
  filter); flagged loss and known delay are cheap; undetected stale frames are the dominant sensor failure and must be
  solved at detection, not in the filter. (F7, F8, F6.)
- **C5 (learned comparison, required by the chapter scope):** one learned estimator trained on the same 350 launches,
  evaluated on both physics, against C1. Minimal choice: KalmanNet-style learned gain on top of the ekf19b model
  (paper 04), because it reuses our model, needs only a small GRU, and its sim-to-sim transfer is exactly the C3
  question. Needs PyTorch (CPU). Decision needed (§6).

Novelty framing (audit C8): paper 03's idea transferred to a planar rolling ball with an obstacle arm; the
criterion study (ML vs forecast loss) with the calibration repair; the matched-vs-mismatch protocol with the hindsight
bound; the sensor ablation. The paper does **not** claim a new filter.

## 3. What the rethink changes

1. **Prediction 5 was mis-reasoned** (design doc, amendment 6): the hindsight constant bound is not a ceiling for ekf-μ,
   whose μ drifts. Gap closed may exceed 1. Fixed before any result.
2. **Core condition set.** Eleven conditions carry redundancy (silent = flagged; delay ≈ ideal + 5 mm; noise_real/plane
   and all_real/all_plane are the same pair). For every remaining heavy step the **core set** is
   `ideal, noise, all_real, all_plane, dropout_silent_noisy` (clean; camera jitter; realistic depth; realistic plane;
   the failure case). The other five stay in a matched-only appendix with grid/h05/fit (sound) and *no NLL numbers*
   (their NLL came from the unsafe stop rule). This cuts the matched NLL reruns from 8 to 4 and the arm-R set from 4
   to 5 conditions, and shrinks the test family for Holm correction.
3. **NLL reruns are not optional** for the core set: C2 compares criteria, and a reviewer will ask whether NLL was
   optimised. Four fits (ideal, all_real, all_plane, dsn) ≈ 3.5 h.
4. **Covariance calibration (new, cheap).** For ekf-μ-h05, scale P by a factor α tuned on val so that NIS = 2 (one
   number per condition, no re-learning). Report NIS/NEES before and after. This turns F11 from a weakness into a
   recommendation and gives the planner (paper 08) an honest covariance. ~20 min compute.
5. **Diagnostics closed cheaply, not by more runs:** NEES blow-ups are explained by F5 (write it up; the mismatch arm R
   tests the mechanism); grid outliers 461/480 are fast arm-hit launches where val-grid σ = 2 mm diverges (report as
   fragility of grid tuning; learned variants do not diverge); launch 235 needs one 10-minute look (bad for every
   filter → probably a corner event; report, do not fix).
6. **dsn is a detection problem.** Stop trying to make estimators win there. Report it as the limit of filtering and
   point to a detection-side fix (frame-difference or timestamp check) as future work.
7. **Learned comparison is the largest open risk.** It is required by the chapter scope and has no code yet. Schedule
   it right after the mismatch arm so the gray-box story is frozen before it starts.

## 4. Work plan (order, hours, gates)

Compute estimates are from measured runs on this laptop (3 workers). "Gate" = what must be true to continue.

| Step | What | Compute | Gate / output |
|---|---|---|---|
| **P0** | Finish mismatch baselines: step_0i run 2 (widened grid) | 70 min | no optimum on an edge; else widen once more |
| **P1** | A1 arm R on `ideal` (pilot of H-q, predictions 1–5c) | 105 min | report; predictions marked confirmed/refuted |
| **P2** | Matched NLL reruns, core set: ideal, all_real, all_plane, dsn (noise done) | 4 × 50 min | each: train NLL ≤ old; start/round log saved |
| **P3** | Covariance calibration α on val for ekf-μ-h05 (and grid), core set, both physics when available | 20 min | NIS → 2 ± 0.2 on val; NEES reported on test |
| **P4** | A1 arm R: noise, all_real, all_plane, dsn | 4 × 100 min | gap-closed table; H-q verdict over 3 conditions |
| **P5** | A1 arm T (transfer), core set + the 5 appendix conditions (eval only) | 10 × 15 min | transfer table; prediction 4 verdict |
| **P6** | Launch 235 look; write F5/NEES, grid-outlier notes into the results doc | 10 min | notes in docs |
| **P7** | Learned comparison (C5): install torch (CPU); KalmanNet-style gain on the ekf19b model; train on matched train, select on val; evaluate both physics, core set | ~2 h code, 2–4 h training | same protocol as ekf-μ; decision §6 first |
| **P8** | Freeze: methods, hyper-parameters, protocol committed; render final test set in both physics | 2 × 90 min | `final_test_plan.json` rule followed |
| **P9** | Final evaluation once, all frozen methods, both physics, core set (+ appendix matched) | ~2 h | every number reported, Holm on the main family |
| **P10** | Paper: figures (μ̂(t), NEES vs q, matched→mismatch paired deltas, gap closed), tables, limitations (audit C list) | no compute | draft |

Total compute ≈ 22–25 h, in sections of ≤ 105 min, each on an explicit go and preferably on the charger.
Order rationale: P0–P1 first because the mismatch pilot decides how C2/C3 are written; P2 before P5 because arm T for
NLL needs the final matched parameters; P7 before P8 because the final set is rendered only once, after freezing.

## 5. Pre-registered predictions still open (frozen; the design doc has the exact thresholds)

- H-q: mismatch NLL q ≥ 1e-3 in ideal, noise, all_real. NEES < 20 in ≥ 3 of 4. NLL–h05 gap narrower on mismatch.
- Transfer: NLL degrades more than h05 from matched to mismatch (paired).
- C1 on mismatch: ekf-μ − fit negative and significant (Holm) in ideal, noise, all_real, both arms; gap closed may exceed 1.
- Calibration (new): after α-inflation, h05 NIS within 2 ± 0.2 on test (dev) in ≥ 4 of 5 core conditions, NEES < 20.
- Learned (new, before any code): a learned gain trained on matched data loses more on mismatch than ekf-μ-h05 (paired
  matched → mismatch change), because it has no physical prior for the new regime. Refuted if its paired loss is smaller.

## 6. Decisions for the user (my recommendation first)

1. **Core set** = ideal, noise, all_real, all_plane, dropout_silent_noisy for all remaining heavy steps; other five
   conditions matched-only appendix without NLL. (Alternative: keep all 11 everywhere, +~8 h.)
2. **Learned comparison (C5):** KalmanNet-style learned gain (paper 04) on CPU PyTorch, one method only. (Alternatives:
   differentiable EKF with learned heteroscedastic Q/R, paper 06, closer to what we have; or a GRU sequence predictor,
   paper 07, no filter at all.) Also: OK to install torch?
3. **Calibration step P3:** add it (recommended; 20 min). It changes C2 from "h05 covariances are dishonest" to "…and
   here is the fix".
4. **Order:** P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 as above. (Alternative: P7 earlier if torch
   installation or learning curve is the bigger risk for the deadline.)
5. **Deadline:** unknown to me. The plan above is ~25 h compute plus writing; if the deadline is near, the cut is P5's
   appendix conditions and the alternatives in decision 1.

## 7. Paper skeleton → experiment

1. Introduction, scope, contribution (C1–C5).  2. Related work (papers 03, 06, 04, 19, 01, 18, 10–12).
3. Simulation and sensor: matched and mismatch physics (A1 designs), 11 conditions, split, final test rule.
4. Estimators: ekf19b(+fit), ekf-μ (grid / NLL / h05 / h05+α), hindsight bound, learned gain.
5. Protocol: per-launch RMS, bootstrap CIs, paired tests, Holm; NIS/NEES; μ recovery; gap closed.
6. Results: 6.1 matched (F1–F4), 6.2 criterion study (F5, F6, F11 + calibration), 6.3 mismatch transfer and retrain
   (C3), 6.4 learned comparison (C5), 6.5 sensor ablation (C4), 6.6 final test set.
7. Limitations (audit C1–C8: μ is a deceleration; sensor idealisations; static arm; sim-only; dropout_silent = flagged by
   construction; dev test inspected).  8. Conclusion.

## Decisions (2026-11-12, user: "go" to all five recommendations)
1. Core set = ideal, noise, all_real, all_plane, dropout_silent_noisy; other five matched-only appendix without NLL.
2. Learned comparison = KalmanNet-style learned gain on the ekf19b model (paper 04); PyTorch (CPU) may be installed when P7 starts.
3. Calibration step P3 added.  4. Order P0 → P10 as listed.  5. Deadline not given; appendix conditions are the first cut if needed.
Arm-R subset becomes the core set (adds all_plane to the four of the A1 design; mismatch noise re-tune must cover it, see RESUME).
