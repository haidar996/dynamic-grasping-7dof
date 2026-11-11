# A1 for ekf-μ: evaluation on mismatch physics — design (2026-11-10, waiting for user decisions)

## Why
Every ekf-μ number so far (docs/part2_sections_results.md) is on **matched physics**: the simulator uses exactly the
filter's model (constant deceleration μ, one restitution E). Step A (docs/stepA_mismatch_results_2026-11-07.md) showed
that this inflates model-based results: the true-constants oracle went from 4 mm to ~35 mm (hindsight bound) on
mismatch data, and ekf19b from 45 to 67 mm (val ideal). No ekf-μ claim is publishable until it is repeated on the
mismatch data (audit A1).

A second reason came from the NLL reruns: on matched data maximum likelihood drives the process noise q towards 0
(0.093 → 8e-5), dev-test NEES rises as q falls (7 → 662), and the better likelihood forecasts slightly *worse* (noise:
val +2.2 mm, p 0.006). **Hypothesis H-q:** this is itself an inverse-crime artefact: when the model is exact, likelihood
rewards "no process noise". On mismatch data the model is wrong, so the learned q should stay well above 0. This design
tests that hypothesis directly.

## Data (already rendered, 2026-11-03)
`outputs/raw_60hz_h3_f49_sq480_with_arm_mismatch/`, 500 launches. They are **paired** with the matched data: same p0, v0,
the same camera and sensor-noise seeds (1000 + i), and the same split `split_500.json` (350 train / 75 val / 75 test-dev).
Physics: slide→roll, rolling resistance a0(1+k|v|), launch slip, speed-dependent restitution, separate e for walls and
arm, wall friction, floor tilt. There is no true μ or E, so per launch the "oracle" is the **hindsight bound** from
Step A (the best constant (μ, E) for that launch, with ekf-family noise: val ideal 35.5 mm, all_real 55.1 mm).

## Two questions, two arms

| Arm | Parameters | Question | Real-world analogue |
|---|---|---|---|
| **T, transfer** | learned on matched train/val, applied unchanged to mismatch val/test | Does what was learned in a simulator with the "right" model survive model error? | train in simulation, deploy on the real robot |
| **R, retrain** | everything re-learned on mismatch train (NLL, h05) or mismatch val (grids), same code and rules | Does learning adapt to model error, and does H-q hold? | learn from logs of the real system |

I recommend **both**: T is cheap (evaluation only), and R is the scientific test of H-q and of ekf-μ under model error.

## Fairness rule (reviewer point)
In arm R **every** method that has tuned or fitted parameters is re-tuned on the mismatch data with the same budget as
on matched data. Otherwise a retrained ekf-μ is compared against baselines tuned for a different world.
- ekf19b: σ, q re-tuned on mismatch val (step_0i grid, "ekf" family).
- ekf19b_fit: c, E re-fitted on mismatch train (step_0k), then σ, q re-tuned on mismatch val.
- cvkf: σ, q re-tuned on mismatch val.
- ekfmu_grid: step_1d grid on mismatch val.
- ekfmu_nll / ekfmu_h05: step_1c on mismatch train, with the **final rules** (NLL: 2 starts + restart on stall, cap 80;
  h05: one start, TOL 1e-5).
- Hindsight bound: unchanged (it is already per launch and in hindsight).

In arm T nothing is re-tuned. All matched parameters are used as they are, including the baselines. That is the
setting Step A already used for the baselines.

## Conditions (fixed before any result is seen)
- **Arm T:** all 9 distinct conditions (delay NLL = ideal, dropout_silent = flagged; both reported as merged). This is
  evaluation only.
- **Arm R:** retraining costs ~90 min per condition, so use a **subset of 4**, the same as Step A:
  **ideal, noise, all_real, dropout_silent_noisy**.
  - These cover clean, noisy, the realistic sensor, and the case where ekf-μ already fails on matched data.
  - **Optional:** extend R to the other 5 conditions only if the 4-condition result is mixed. The rule for that is
    written down now: extend if the sign of (ekf-μ − ekf19b_fit) differs between ideal and all_real.

## Measurements
Same as the matched evaluation (step_1e), on mismatch val and test (dev):
- Per-launch RMS of now_pos, now_vel, h0.2, h0.5, h1.0. Report the launch mean with a 95 % bootstrap CI.
- Paired bootstrap and Wilcoxon tests against ekf19b_fit.
- Paired matched → mismatch test, the same launch under both physics (as in Step A).
- NIS (expected 2) and NEES (expected 4).
- **μ recovery is replaced:** there is no true μ, so I report the correlation of μ̂ (at 1 s) with the true rolling
  resistance a0 (Step A: hindsight best μ vs a0 r = 0.74). This is descriptive only, not a pass/fail test.
- **Gap closed:** (ekf19b_fit − method) / (ekf19b_fit − hindsight bound). This is the fraction of the possible
  constant-model gain that each method gets.

## Pre-registered predictions (written before running; each is reported as confirmed or refuted)
1. **H-q (arm R):** the NLL-learned q on mismatch is ≥ 1e-3 in all three of ideal, noise and all_real. dsn is excluded
   because NLL already collapses there on matched data (q = 78).
   - Matched values: noise 8.1e-5 under the final rule. ideal 0.016 and all_real 1e-4 are under the old TOL 1e-3 rule,
     which is unsafe; their final-rule values are still pending.
   - **Secondary:** the ratio of mismatch q to matched final-rule q, once those are known.
   - **Refuted if:** q < 1e-3 in noise or all_real on mismatch.
2. **NEES (arm R):** dev-test NEES of ekfmu_nll is < 20 in at least 3 of 4 conditions (matched: hundreds).
3. **NLL vs h05 gap (arm R):** narrower than on matched data. On matched data, h05 is better than NLL in every
   condition.
4. **Transfer (arm T):** ekfmu_nll degrades more than ekfmu_h05 from matched to mismatch, because a near-zero q
   trusts a model that is now wrong. Measured by the paired difference of the matched → mismatch change.
5. **Main claim at risk:** on matched data, ekf-μ beats the fitted fixed-μ filter by 20–29 mm. On mismatch data the
   whole constant-model gain is bounded by about (63 − 35.5) = 28 mm (val ideal), so ekf-μ's gain must be smaller than
   that. Prediction: positive and significant (paired p < 0.05) in ideal, noise and all_real, in both arms. **If it
   fails, the thesis claim must be restated.**

## Multiple comparisons
A2/A3 audit: 9 + 4 conditions × 3 ekf-μ variants × 2 splits is a large family of tests. For this section the main
claims (predictions 1, 2 and 5) get Holm correction over their condition family. The other comparisons are reported as
descriptive with uncorrected p values and are labelled that way.

## Order relative to the matched NLL reruns
- **Arm T needs the final-rule matched NLL parameters.** Only noise exists so far, so T for NLL waits for each matched
  rerun. T for grid and h05 can run now, since those parameters are final.
- **Arm R does not depend on the matched reruns.**
- **Proposal:**
  1. R on **ideal** first, as a ~90-min pilot of H-q. Its result tells us whether the matched NLL story (q → 0) is a
     finding about the method or an artefact of matched data. That changes how the matched NLL column is written up,
     but not the need to rerun it.
  2. Then the matched NLL reruns, section by section.
  3. Then the rest of R and the whole of T.

## Code changes (small; nothing that exists is overwritten)
1. One shared switch `PHYSICS=matched|mismatch`, as an environment variable read by step_1c, step_1d, step_1e,
   step_0i and step_0k. It selects `RAW`. Every output name gets the suffix `_mismatch`
   (`ekf_mu_learned_nll_mismatch.json`, …). Matched files are never touched. The default stays `matched`, so the
   results that exist reproduce bit for bit. I'll check that with the noise eval on matched data.
2. step_1e:
   - A `--params matched|own` option for arm T vs arm R.
   - The oracle row becomes the hindsight bound, read from `outputs/mismatch_hindsight.json` with ekf-family noise.
   - μ recovery becomes corr(μ̂, a0), and a gap-closed column is added.
   - It must not crash on `rec["mu"] = NaN`.
3. `run_section.sh` gets the physics switch. The markers move to `outputs/.sections_done_mismatch/`.
4. **Smoke test:** `--quick` on 3 launches in both physics. Matched output must be identical to the current code.

## Cost (from measured matched runtimes; mismatch launches move ~10 % longer)

| Step | Estimate |
|---|---|
| Coding + smoke test | ~1.5 h of my work, ~5 min compute |
| Baseline re-tune, arm R (step_0k on ideal + step_0i for the 4 conditions, 4 families) | ~45 min |
| R per condition: grid 6 + NLL ~55 + h05 ~22 + eval ~15 | ~100 min |
| R, 4 conditions | ~6.5 h + ~45 min baseline re-tune ≈ 7.3 h |
| T, eval only, 9 conditions × ~15 min | ~2.3 h (after the matched NLL reruns) |

Each run is started only on an explicit go, section by section, as before.

## Decisions needed (my recommendation first)
1. **Arms:** both T and R, or only one of them.
2. **R subset:** ideal, noise, all_real, dsn (same as Step A), or all 9.
3. **Baseline re-tune in R:** yes (fair), or skip it (cheaper, but a reviewer will object).
4. **Order:** R-ideal pilot first, then the matched NLL reruns, or the reruns first.
5. **Predictions 1–5 and their thresholds:** accept, or change them now, before any result exists.

## Decisions (2026-11-10, user: "ok" to all recommendations)
1. Arms T and R. 2. R subset ideal, noise, all_real, dropout_silent_noisy (extend rule as written). 3. Baselines re-tuned in R.
4. Order: R-ideal pilot → matched NLL reruns → rest of R and T. 5. Predictions 1–5 accepted as written (frozen from here).

## Implementation notes (2026-11-10, before any result; amendments to the plan above)
- Switch: `physics_cfg.py` (PHYSICS env). step_0i, step_0k, step_1c, step_1d, step_1e read it; step_1e also reads
  ARM=transfer|retrain. Runner: `experiments/run_section.sh` (header lists the mismatch commands).
- **Amendment 1, cvkf:** cvkf is not re-tuned in arm R. step_1e does not report cvkf, and the cvkf mismatch numbers
  already exist in Step A.
- **Amendment 2, step_0k on mismatch:**
  - noise σ, q = the current matched "ekf" val tuning, the best available before the re-tune;
  - fitted on "ideal" only (the fit is taken from ideal in both physics);
  - wider grid: c 0.2–1.6 in steps of 0.1, E 0.30–0.95 in steps of 0.05. Step A's per-launch best constants spread far
    beyond the generator ranges.
- **Amendment 3, hindsight row:** the oracle row is replaced by Step A's "best μ & E" bound, the same filter, per-launch
  pair and matched "ekf" noise, in both arms, so it is a fixed reference.
- **Smoke tests:**
  1. Matched: step_1e on noise, 10 launches. All 300 per-launch error arrays are identical to the stored results.
  2. Mismatch arm T: on ideal and noise, ekf19b, ekf19b_fit and hindsight are identical to Step A's
     `mismatch_errors.pkl.gz`. That is 250 arrays, 0 different; hindsight is checked on ideal only, the only one of the
     two conditions where Step A stored it.
  3. Arm R chain on tiny subsets: baselines → grid → NLL → h05 → retrain eval. It runs end to end, and the
     temporary files were removed.
  4. run_section.sh refuses 7 wrong commands before starting anything.
- **Amendment 4, q_mu grid on mismatch:** in step_1d, q_mu 1e-6…1e-2 → 1e-6…1.0 (adds 0.1 and 1.0).
  - Reason: in smoke test 3, the grid optimum sat on the upper edge 1e-2 and the score was still falling (53.5 → 51.9 mm).
    On mismatch data μ drifts within a launch (slide → roll).
  - That smoke test used quick noise tuning from 5 launches, so its numbers are not results.
  - The grid now takes about 10 instead of 6 min per condition. The matched grid is unchanged.

## Arm R baselines, run 1 (2026-11-10) and amendment 5
- step_0k on mismatch train (36 min): best constants c 0.5, E 0.5, train h0.5 51.5 mm, interior of the grid
  (`outputs/fit_constants_train_mismatch.json`).
- step_0i on mismatch val (46 min): 4 of 8 optima on the TOP grid edge: ekf ideal q = 20; ekf_fit σ = 30 mm in noise,
  all_real and dropout_silent_noisy. With the wrong model the filter asks for far more measurement noise than the sensor
  has (true: a few mm), to discount a model it cannot trust. Kept as `outputs/ekf_tuning_val_mismatch_edge.json`.
- **Amendment 5 (before any ekf-μ result on mismatch):** grid widened upwards for mismatch only, σ to 120 mm (adds 50, 80,
  120) and q to 100 (adds 100); rerun of step_0i. Reason: a baseline whose best setting is outside the grid is tuned too
  weakly, which would flatter ekf-μ. The matched grids and files are unchanged.
- val h0.5 (mm) in run 1, mismatch (matched): ekf ideal 64.9 (45.3), noise 71.7 (50.2), all_real 89.9 (67.6),
  dsn 100.0 (92.9); ekf_fit 51.6 (46.9), 57.1 (51.9), 70.2 (70.8), 82.3 (94.5). Run-1 numbers are superseded by run 2
  wherever run 2 differs.

## Amendment 6 (2026-11-11, rethink, before any ekf-μ result on mismatch): prediction 5 was mis-reasoned
Prediction 5 said ekf-μ's gain on mismatch "must be smaller than about 28 mm" because the hindsight best CONSTANT pair
reaches only 35.5 mm (val ideal). That bound holds for constant-parameter filters. ekf-μ is not one: q_μ lets μ drift
within a launch, so it can in principle follow the slide → roll change and beat the constant bound. Corrected:
- Prediction 5 (kept): ekf-μ − ekf19b_fit is negative and significant (paired, Holm) in ideal, noise, all_real, both arms.
- Prediction 5b (new): "gap closed" = (fit − X)/(fit − hindsight) may EXCEED 1 for the retrained variants; if it does, it
  is evidence that adaptive μ captures slide → roll, and the hindsight row is then a reference, not a ceiling.
- Prediction 5c (new, arm R): the retrained h05 q_μ is larger than the matched one (matched h05 q_μ = 1e-3 in all five core conditions); this is
  why the mismatch q_μ grid was widened to 1.0 (amendment 4).

## Arm R baselines, run 2 (2026-11-12 00:11 → 01:18, 68 min, widened grid)
Changes vs run 1: ekf_fit all_real σ 30 → 80 mm (70.2 → 69.3 mm), ekf_fit dsn σ 30 → 120 mm (82.3 → 81.8 mm, still the top edge).
All other 6 optima unchanged and interior. The dsn one is accepted as saturation: 4× more σ bought 0.5 mm (flat surface; the
widen-once fallback was applied). Final mismatch baselines: `outputs/ekf_tuning_val_mismatch.json` (run 1 kept as *_edge.json).
