# Strengths / weaknesses audit (2026-11-16)

Critical internal review (reviewer stance), written by a Sonnet audit agent from RESUME.md, docs/part2_sections_results.md,
docs/A1_ekfmu_mismatch_results.md, docs/P7_learned_gain_design.md, outputs/paired_all_metrics.txt. Numbers: launch-mean
h0.5 RMS (mm), dev test, n = 73–75. Saved by the lead session (subagent could not write files). Constraint: simulation only.

## Top 5 strengths
1. Pre-registration with refutations reported (A1 predictions 4, 5; P7 L1, L2; calibration fix withdrawn).
2. Inverse crime named on day 1 (research_audit A1); paired mismatch physics (same p0/v0, extra params seed 43) built and every
   key claim re-run on it; the −25 → −3 mm shrink is reported against ourselves.
3. Statistics: per-launch RMS, bootstrap CIs, paired Wilcoxon on identical launches/noise seeds, Holm, grid-edge rule enforced
   (P0 widened the mismatch grid twice).
4. Fair baselines: ekf19b_fit train-fitted; mismatch baselines re-tuned; hindsight bound makes "gap closed" interpretable.
5. Testable mechanism that held on mismatch: matched NLL drives q → 0, NEES 7 → 662 while NIS ≈ 2; on mismatch q rises ~1000×,
   NEES 7–20. NLL collapses on dsn (312 vs 96 mm).

## Top weaknesses (fixable in simulation?)
| # | Weakness | Evidence | Fixable in sim | Fix / cost |
|---|---|---|---|---|
| W1 | Simulation-only and still in-family: matched data come from the estimator's own equations (oracle 4–7 mm); mismatch physics is our own hand-written variant | research_audit A1 | partly | third physics on MuJoCo's own contact solver (eval-only) + mismatch dose-response sweep, ~20 h |
| W2 | ekf-μ gain collapses under mismatch: −16…−29 matched → −0.2…−3.1 retrain; transfer +30…+58 worse; q, q_μ at/near grid top | A1 results | partly | two-phase friction / IMM (slide→roll), ~14 h; hindsight gap 15–20 mm says the information exists |
| W3 | Power + test-set reuse: mismatch effects 1–3 mm vs CI ±1.5–4; val/test sign flips (ideal h05 val +1.9, test −3.1); dev test inspected many times, amendments followed results | P1, P4 | yes | fresh 500-launch test both physics, frozen methods, one primary endpoint + Holm, ~5 h (= P8/P9) |
| W7 | Learned baseline thin: knet keeps fixed model, only gain learned (cannot infer friction per launch); single seed; no direct sequence forecaster | P7 | yes | GRU forecaster + 3 seeds, ~15–18 h mostly unattended |
| W6 | Covariance unusable: h05/grid NIS 0.1–0.7; NLL NIS ≈ 2 but matched test NEES 79–662; scalar α did not transfer (NEES 1e3–1e5 when inflated) | P3 | partly | event-aware inflation at bounces/arm hits, 4–6 h (also: conformal radii, see literature doc) |
| W8/W5 | knet now_vel +0.026…+0.037 m/s worse (Holm < 1e-9, retrain ideal/noise; not in all_real); several optima on grid edges; NLL optimum start-dependent (noise start B hit the 80-eval cap); ekf_fit σ at 120 mm edge | Result 4b, P0, P2 | yes | now_vel loss term; widen grids once more + restart spread, ~10 h |
| W11/W9/W10 | No task metric (static arm, forecast RMS only); sensor realism (7 vs 33 mm noise covered by *_real conds, but calibration/blur/lens not modelled); camera scale bias +0.18 % (PLAN_2026-10-29.md item 5, cause untraced, ≈ 3 mm at 1.8 m = size of mismatch effects); dsn "solve at detection" untested | Phase 0 notes | partly | sim intercept/catch-point metric 6–8 h; check scale bias < 1 h; stale-frame detector ~3 h |

## Oracle-gap closed (audit agent's computation from the tables)
- Matched ideal / noise / all_real: h05 52 / 49 / 47 %; grid 33 / 45 / 40 %; nll 35 / 45 / 34 %; knet ≈ 0 % or negative.
- Mismatch retrain (vs hindsight bound): grid 18 / 12 / 13 %; knet 21 / 25 / 20 %. Transfer: negative for every ekf-μ variant.
- knet − ekfmu_grid on mismatch retrain: −1.3 [−3.5, +0.7], −1.8 [−4.5, +0.7], −1.5 [−6.7, +3.5] (n.s.).

## Top 5 actions (simulation only, best value per hour)
1. Fresh 500-launch final test, both physics, frozen methods, one pre-declared primary endpoint + Holm (~5 h) → closes test-reuse/power.
2. Direct sequence-model baseline + 3 seeds for knet (ideal, noise) (~15–18 h) → strong learned baseline + variance.
3. Mismatch-aware ekf-μ (two-phase friction or IMM) + third physics on MuJoCo contacts as eval-only arm (~26 h) → rescues C1 or gives a clean negative paper; answers inverse crime.
4. Interception / catch-point metric + per-launch least-squares (μ, E) baseline (~17 h) → numbers that matter + fair competitor.
5. Hygiene bundle: grids + restart spread, knet now_vel term, sensor calibration-error conds, verify 0.18 % bias, stale-frame detector, event-aware inflation (~12 h).

Fallback if time is short: claim C1 on matched only; lead with the criterion study, the mismatch diagnosis (transfer failure,
hindsight gap) and "a learned gain on a fixed model = the fixed-μ filter".

## Synthesis with the verified literature (lead, 2026-11-16)
Literature: docs/literature_solutions_2026-11-16.md (16 PDFs checked with pdftotext; Tobin 2017 = visual randomization only;
e(v) restitution not in literature; IMM for rolling ball not found → possible novelty; Gibbs & Candès ACI verified; Bayesian
KalmanNet mechanism verified; Achterhold 2023 = closest template, flying ball).

| Rank | Solution | Fixes | Evidence | Cost | Must be before freeze? |
|---|---|---|---|---|---|
| 1 | Fresh final test (P8/P9), one primary endpoint + Holm | W3 | standard practice | ~5 h | is the freeze |
| 2 | Adaptive/split conformal forecast radii per horizon (from val residuals) | W6 | Gibbs & Candès 2106.00170 (verified) | ~3–4 h, no retraining | yes |
| 3 | Two-regime IMM / two-phase friction EKF (slide ↔ roll) | W2 (+ novelty) | IMM > KalmanNet on radar (2411.16930), DIMMPF (2410.00620); none for rolling balls | ~14 h | yes |
| 4 | Third physics on MuJoCo native contacts, eval-only, + mismatch dose-response | W1 | held-out physics grids (2206.09314) | ~20 h | yes (render with final test) |
| 5 | Learned baseline: knet 3 seeds + GRU forecaster (+ Bayesian KalmanNet for uncertainty) | W7, W6 | 2309.03058 (verified mechanism) | ~15–18 h | yes |
| 6 | Hygiene: 0.18 % scale-bias check, grid widen + restarts, knet now_vel term, dsn stale-frame detector, intercept metric | W5, W8, W9–W11 | — | ~1 h (bias) … ~12 h all | mostly |

Packages: A minimal (1+2+bias+remaining P7 runs) ≈ 25 h; B strong-in-sim (A + 3 + 4) ≈ 60 h; C full (B + 5 + 6) ≈ 80 h.
