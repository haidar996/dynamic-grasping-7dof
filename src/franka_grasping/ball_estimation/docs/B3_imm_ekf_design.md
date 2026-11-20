# B3 - IMM EKF for a slide-to-roll ball ("ekf-imm"): design (2026-11-20, before any code, no run)

User decision (plan section 8(f), item 2): variant (ii), a true 2-mode IMM (slide, roll, Markov switching), label `ekf-imm`.
Variant (i), deterministic two-phase friction, is kept as an ablation inside this document, label `ekf-2ph`.

## 0. Frozen predictions (copied verbatim from docs/plan_2026-11-11_paper_plan.md section 8(c); not to be edited after this point)
- H-IMM1: on mismatch retrain, ekf-imm beats ekfmu_grid on h0.5 in >=2 of ideal/noise/all_real (Holm).
- H-IMM2: on matched physics, ekf-imm is not worse than ekf-μ by more than 3 mm at h0.5.
- H-IMM3: transfer loss of ekf-imm is smaller than that of ekf-μ.

Operational reading (fixed now, so the test cannot drift). Scope: conditions ideal, noise, all_real; dev test split; metric = launch-mean
h0.5 (per-launch RMS, as step_1e). "ekf-μ" in H-IMM2/3 = `ekfmu_grid` (the val-grid-tuned variant that H-IMM1 names); ekfmu_h05 is reported beside it
but does not decide. H-IMM1: for each condition paired test ekf-imm minus ekfmu_grid, mismatch-retrain arm, one-sided on "imm better"; Holm over the 3 tests;
H-IMM1 holds if >=2 are significant at 0.05 after Holm with negative mean difference. H-IMM2: per condition matched, paired mean difference
imm minus ekfmu_grid; holds if the upper end of the 95 % bootstrap CI is <= +3 mm in all 3 conditions (a point estimate <= 3 mm with a CI that crosses 3 mm
is reported as "not resolved", not as holds). H-IMM3: transfer loss = (mismatch-transfer h0.5) minus (matched h0.5), per method, on the same launches
where possible (launch ids are shared across physics); holds if paired loss(imm) < loss(ekfmu_grid) in >=2 of 3 conditions (Holm). The primary
endpoint of the paper (plan 8(f) item 4) is decided at B6, not here.

## 1. Why this model, and what is cited (only from docs/literature_solutions_2026-11-16.md, PDF-VERIFIED items)
- Mehrfard et al., arXiv:2411.16930 (PDF-VERIFIED): "KalmanNet is outperformed by the IMM filter" on automotive radar; supports an IMM baseline next to knet.
  Not ball tracking.
- Brandenburger, Hoffmann, Charlish, arXiv:2307.06618 (PDF-VERIFIED): IMM parameters optimised "solely using measurements"; justifies tuning
  the IMM parameters on a held-out set by an accuracy criterion without a ground-truth model of the switch.
- Brady et al., arXiv:2410.00620 (differentiable IMM particle filter) and Stamatelis and Alexandropoulos, arXiv:2511.09569 (JMFNet), both PDF-VERIFIED: jump-Markov
  filtering with learned parameters exists; neither is an EKF-IMM for a rolling ball.
- Murphy 1998, switching Kalman filters: V-meta only (not opened). Bar-Shalom IMM text: not opened; the mixing equations below are the standard IMM
  recursion written from general knowledge, not cited to a verified source.
- Novelty statement: the doc states "no paper on IMM for sliding vs rolling ball tracking was found" (search result, not proof). Wording in the paper:
  "we did not find".

## 2. State model
Per-mode state x = (x, y, vx, vy) in R^4 (mu is NOT in the state; this is the difference to ekf-μ, whose 5th state l = log mu caused the unobservability
and the transfer failure, P5). Both modes share dimension, so the IMM mixing is plain (no state-augmentation tricks).
Measurement: z = (x, y) + noise, R = sigma^2 I, identical to EKF19 / EKFMu. Prediction sub-step 5 ms, identical to `ekf_mu.py` (max_h = 0.005); predicts at every frame tick.
Shared with ekf19b_fit: walls (L), arm box (obs), restitution E (from the train fit, `ekf_fit` entry), extra velocity variance at a hit sigma_E = 0.13,
StaleGuard (inline or by `+stale`), sigma_v0, the first-frame initialisation (position = first measurement, velocity 0, P0 = diag(sigma^2, sigma^2, sigma_v0^2, sigma_v0^2)).

## 3. Per-mode dynamics
Same Coulomb-type law as `estimators/model.py` (constant deceleration c opposite to velocity, stopping inside the sub-step; wall mirror with slope -E; arm box), with
a mode-dependent constant:
- mode 1 (slide): c = mu_s.
- mode 2 (roll): c = mu_r (expected mu_r < mu_s, the rolling resistance is smaller than sliding friction).
Jacobian: `propagate_jac` of `ekf_mu.py` with the l-column removed (fixed c); reuse the existing function by a thin wrapper that fixes l = log(c) and drops row/column 5.
Known mismatch (stated, not fixed): the mismatch simulator has rolling resistance a0(1+k|v|) (docs/A1_ekfmu_mismatch_eval_design.md), a speed-dependent deceleration; a constant mu_r
approximates it. A speed-dependent roll mode would be one more parameter (k) and is NOT in the frozen scope.
Open interpretation (decision for the user, section 11 Q1): in the matched data mu is random per launch (that is why ekf-μ beats ekf19b_fit by ~45 %), whereas mu_s, mu_r here are
global constants fitted on train. Adaptivity then comes only from the mode probabilities. Expect ekf-imm to be BETWEEN ekf19b_fit and ekf-μ on matched; H-IMM2 is therefore at real risk and is
a fair test of the frozen claim.

## 4. IMM recursion (per update, standard)
Modes i, j in {1,2}; mode probabilities m_i; transition matrix Pi (Pi_ij = P(mode j at k | mode i at k-1)); per-mode state x_i, covariance P_i.
1. Mixing probabilities: cbar_j = sum_i Pi_ij m_i ; mu_{i|j} = Pi_ij m_i / cbar_j.
2. Mixed initial conditions: x0_j = sum_i mu_{i|j} x_i ; P0_j = sum_i mu_{i|j} [ P_i + (x_i - x0_j)(x_i - x0_j)^T ].
3. Per-mode EKF predict from (x0_j, P0_j) over dt with mode-j dynamics and Q_j (section 6), including the bounce Jacobian and the sigma_E term, exactly as `EKFMu._predict`.
4. Per-mode update (Joseph form, as `EKFMu.update`) with innovation y_j = z - H x_j^-, S_j = H P_j^- H^T + R; Gaussian likelihood Lambda_j = N(y_j; 0, S_j).
5. Mode probability: m_j = Lambda_j cbar_j / sum_l Lambda_l cbar_l, floored at m_min = 1e-4 and renormalised (prevents mode lock-out).
6. Combined output: x = sum_j m_j x_j ; P = sum_j m_j [ P_j + (x_j - x)(x_j - x)^T ] (used for NIS and NEES of the combined estimate).
7. Forecast: two options evaluated, both frozen: (F-mix) forecast the combined state x with the m-weighted deceleration c_eff = m_1 mu_s + m_2 mu_r (cheap, one rollout, interface `forecast(dt)`); (F-ml)
   forecast with the most probable mode. Primary = F-mix. F-ml is an ablation. (Forecasting each mode separately and averaging gives a mean that is the better estimator under the quadratic
   loss but costs 2 rollouts per forecast; kept as ablation F-avg, evaluated only if F-mix is close to the threshold.)
Missing/stale frames: predict-only for both modes, mode probabilities propagate only through Pi (no likelihood). Time-step note: Pi is specified per second and converted per tick, Pi(dt) = expm(G dt) with the
2x2 rate matrix G (so a variable dt, e.g. after dropped frames, is handled correctly).

## 5. Transition matrix
Two-state generator G = [[-a, a], [b, -b]], rates a = slide->roll (1/s) and b = roll->slide (1/s). Initial probabilities pi0 = (p_s0, 1 - p_s0).
Prior expectation: a launch starts sliding (launch slip in the mismatch simulator), so p_s0 high; roll->slide is rare (b small, possible after wall/arm-box bounces) but is kept
> 0 so the filter can recover from a wrong early switch. Parameters: a, b, p_s0. Bounds for tuning: a in [0.2, 10] 1/s, b in [0.005, 1] 1/s, p_s0 in [0.5, 0.99].
Bounce coupling (optional, NOT in the frozen model): a bounce could raise b; omitted to keep the parameter count low.

## 6. Per-mode process noise q
Same continuous white-noise-acceleration block as `EKFMu.Q` (q h^3/3, q h^2/2, q h) per axis, with a mode-specific level q_s, q_r (no q_mu, since there is no mu state). Rationale: sliding
and rolling have different unmodelled acceleration (the slip phase with speed-dependent restitution and tilt is rougher). Ranges (widened 2026-11-20 after the range audit; see step_3a BOUNDS comment): q in [0.001, 100] m^2/s^3 (log; chosen q spanned 0.003 to 20).

## 7. Parameters to tune and on what
Fixed from existing files, not re-tuned: sigma (sensor level, "ekf_fit" entry per condition, `ekf_tuning_val.json`), E (fit_constants_train), sigma_E = 0.13, sigma_v0, L, obs.
Tuned (8 numbers): mu_s, mu_r, q_s, q_r, a, b, p_s0, and a mixing floor m_min fixed at 1e-4 (not tuned). That is 7 tuned numbers.
Initialisation: mu_s and mu_r start from the train fit of c (fit_constants_train): mu_s = 1.3 c_fit, mu_r = 0.5 c_fit as a starting guess only (guess, flagged; the search range
decides).
Criterion: launch-mean h0.5 RMS on moving frames (every 6th frame, warm-up 0.1 s), the same criterion as step_1d / step_1c h05, computed on the VAL split only (never on test).
Scope: conditions ideal, noise, all_real; both physics (matched; mismatch retrain re-tunes on mismatch val, as every `*_mismatch.json` arm). The mismatch-transfer arm applies the matched
parameters unchanged. Constraint mu_r < mu_s enforced by parameterising mu_r = r mu_s, r in [0.02, 1] (linear). Search ranges (2026-11-20 audit): mu_s [0.2, 5] log, r [0.02, 1], q_s,q_r [0.001, 100] log, a [0.5, 300] log, b [0.005, 1] log, p_s0 [0.5, 0.99].
Search: gradient-free (the filter has switches and bounces; docs/literature note [20] warns about unreliable gradients through bounces). Stage 1: random Latin hypercube, 120 points; stage 2:
coordinate refinement around the best 5 for 40 evaluations; pick best by val h0.5. One seedless deterministic procedure with the random generator seed 0 recorded. 7 parameters ->
120 + 40 = 160 evaluations per (condition, physics) = 6 tuning runs.
Overfitting guard: report val h0.5 of the best and the 5th best; if their spread < 1 mm the optimum is flat and is declared stable. Number of tuned parameters (7) vs ekf-μ grid (a handful) is
reported; a parameter-count-matched comparison is not claimed.
Tuning on ideal could be transferred to noise as a start (warm start), not as a result.

## 8. Ablations (all use the same tuned filter unless stated; val and dev test, matched + retrain)
- ekf-2ph (variant (i)): deterministic switch. Mode is slide while speed > v_sw OR time since first frame < t_sw (one rule, chosen on val from {speed, time}), then roll; no mode probabilities, a single
  EKF whose c switches (mu_s -> mu_r), covariance propagated through the same Jacobian. Parameters: mu_s, mu_r, q, v_sw (or t_sw). Hard switch: the filter is not told when it happens other than by
  the rule. Tuned by the same search on val (4-5 parameters, 100 evaluations).
- imm-nomix: a = b = 0 (no switching, two parallel filters weighted by likelihood: a static multiple-model estimator). Tests whether the switching matters.
- imm-shared-q: q_s = q_r. Tests per-mode q.
- forecast F-ml and F-avg (section 4 item 7).
Ablations are descriptive; none decides H-IMM1/2/3, and none enters the Holm family of the frozen predictions (a separate Holm family over the ablations vs ekf-imm, reported as such).

## 9. Evaluation protocol (identical to step_1e, `experiments/step_1e_eval_ekf_mu.py`)
- Same launch ids and split (`split_500.json`), same sensor seeds (`apply_condition(cam, raw, CONDITIONS[cond], 1000 + i)`), same raw sets (matched; mismatch), so every comparison is paired by launch.
- Estimator label `ekfimm` (and `ekf2ph`, ablation labels) registered like `knet`: `estimators(cond)` loads `outputs/ekf_imm_<cond>[_mismatch].json` when present; `make()` gets a branch `ekfimm` -> `EKFIMM(**kw)` in
  `estimators/ekf_imm.py`; the interface is the one of `EKFMu`: `reset()`, `update(t, xy, valid)`, `ready`, `t`, `s` (combined 4-vector, padded so `s[:4]` works), `P`, `forecast(dt)`, `nis`,
  `nll`, `n_upd`; plus `mode_prob` and `mu` (= c_eff, so the mu-recovery code path works; descriptive).
- Arms: matched (val and dev test); mismatch ARM=transfer (matched parameters unchanged); mismatch ARM=retrain (re-tuned on mismatch val). `step_1e` is extended, not forked; output merges into
  the existing `outputs/ekf_mu_eval*.json/.pkl.gz` (per condition), so the earlier rows are untouched.
- Metrics: per-launch RMS now_pos, now_vel, h0.2, h0.5, h1.0; launch mean + 95 % bootstrap CI (2000 resamples, seed 0) and median; paired difference vs ekf19b_fit, vs ekfmu_grid, vs
  ekfmu_h05 (estimators.stats.paired); consistency NIS (expected 2) and NEES (expected 4) of the combined estimate; ms per update; on mismatch "gap closed" against the hindsight reference.
  Mode diagnostics (descriptive): mean m_roll vs time since first frame, and correlation of the switch time (m_roll crossing 0.5) with the true slide->roll time from the mismatch generator, if that time is stored in the raw files
  (to be checked at coding time; if not stored, skipped and said so).
- Holm: family 1 = H-IMM1 (3 tests); family 2 = H-IMM2 (3 CIs, no p); family 3 = H-IMM3 (3 tests); secondary now_pos / now_vel / h0.2 / h1.0 tests Holm-corrected as in P7 Result 4b.
- Order and honesty: tune on val, freeze the 6 parameter files (commit), then run dev test ONCE per arm. The dev test is the "dev" split of step_1e; the final sealed test is B6/B7, and ekf-imm's parameters
  go into the freeze unchanged. Dev-test numbers resolve H-IMM1/2/3 but are not the final claim.
- Mismatch-family caveat (audit A1): the filter model (constant c per mode) and both simulators are mine; mismatch physics is the one generator, so B3 results on mismatch inherit the single-family limitation,
  which B4 (physics C, MuJoCo native contacts) addresses; ekf-imm will be re-evaluated there.

## 10. Failure criteria (decided now)
- Smoke failure (before any tuning): on 5 val launches of ideal matched, per-mode filters with a = b = 0 and mu_s = mu_r = c_fit must reproduce ekf19b_fit h0.5 to < 1 % (same model, same sigma_E); the mixing recursion with identical
  modes must return the single EKF exactly (unit test). If not, a code bug, not a result.
- Mode degeneracy: if on val the mean m_roll after 0.5 s is within [0.45, 0.55] for > 90 % of launches in every condition (no information in the switch) or the tuned mu_s, mu_r differ by < 5 %, ekf-imm is declared degenerate
  (reduces to a fixed-mu EKF) and is reported as such.
- Prediction outcomes: H-IMM1 refuted if fewer than 2 of 3 conditions are Holm-significant in favour of ekf-imm on mismatch retrain. H-IMM2 refuted if any matched condition has the upper CI end > +3 mm AND
  the point estimate > +3 mm; "not resolved" if only the CI crosses. H-IMM3 refuted as in section 0.
- Consequences are pre-declared: H-IMM1 refuted -> C6 reported as a negative result (plan 8(e)), no redesign beyond the 2-day slip buffer; the slip buffer may fund exactly one pre-registered redesign
  (speed-dependent roll mode, k parameter) which then counts as a new, labelled exploratory variant, not as the original test.
- Consistency check: mean NIS outside [1.5, 2.5] or mean NEES outside [3, 6] is reported (a model-structure problem, B5 conformal radii cover it); it does not decide the predictions.
- Non-event: a negative result does not trigger re-tuning on dev test.

## 11. Compute and effort estimate
Per update: 2 EKF predict+update (4x4 or in the 5x5 code ~ same cost) + mixing; ekf-μ costs ~0.3 ms/update (step_1e), so estimate ~0.7-0.9 ms/update (knet 4.7 ms for reference).
One launch 600 ticks ~0.5 s single core; a 75-launch val pass ~40 s per parameter vector (rough; includes the python sub-step loop). Tuning: 160 evaluations x 6 (condition, physics) x ~40 s ~ 11 core-hours,
~3 h on 4 workers (the plan says tuning waits until B1 ends; <= 4 workers). Dev-test eval: 150 launches (val + test as step_1e) x 3 cond x 3 arms x 4 estimators (imm, 2ph, ablations) ~ 1 h. Ablation tuning ~ +2 h.
Total ~6 h compute (matches plan B3 ~6 h) and ~8 h agent/analysis: module + unit tests 3 h, step_1e extension and tuning script 2 h, smoke test 0.5 h, analysis and write-up 2.5 h. Estimates, not measured; the
smoke test measures ms/update and the real tuning cost before the full run.

## 12. Files (planned, none exist)
`estimators/ekf_imm.py` (EKFIMM, EKF2PH), `estimators/filters.py` (make branch `ekfimm`, `ekf2ph`), `experiments/step_3a_tune_ekf_imm.py` (val search, PHYSICS switch, writes `outputs/ekf_imm_<cond>[_mismatch].json`
with parameters, seed, evaluation log), `experiments/step_1e_eval_ekf_mu.py` (add labels), `tests/` unit tests (identical modes = single EKF; mixing probabilities sum to 1; a = b = 0 weights static).
Smoke test: 5 val + 5 test launches, `--quick`, both physics.

## 13. Questions for the user (my recommendation first)
1. Per-launch mu: the matched data has a random mu per launch, constants mu_s, mu_r cannot follow it. Recommend: keep the frozen model (global constants), accept the H-IMM2 risk and report it. Alternative (a
   later, labelled exploratory variant, not H-IMM): modes with a small mu random walk (q_mu per mode), which blurs the line to variant (iii).
2. Operational readings in section 0: "ekf-μ" = ekfmu_grid; one-sided Holm tests; H-IMM2 judged by the CI upper end <= 3 mm. OK?
3. Forecast rule: primary F-mix (m-weighted deceleration). OK, or prefer F-avg (more accurate in principle, 2x forecast cost)?
4. Tuning budget 160 evaluations x 6 sets and a 7-parameter model versus the few parameters of the ekf-μ grid: accept, or cut to a smaller model (q_s = q_r, 6 parameters: mu_s, r, q, a, b, p_s0)? Recommend: accept 7, report q-shared as ablation.
5. Does the mismatch raw data store the true slide->roll time per launch? Needed for the switch-time diagnostic; to be checked at coding time, no decision needed unless it is missing.
