# Research audit — threats to publication (2026-11-01)

Asked by the user before step 1.2: "think deeply as a researcher about any bug in this research; anything that will
prevent it later from publishing". Every core file was re-read (arena_sim, launches, dataset, sensing/*, estimators/*,
step_0c/0d/0h/0i/1a) and four checks were run on the data (scripts in the session scratchpad, numbers below).
**No code was changed.** Each item: what, evidence, why a reviewer cares, proposed fix. Decisions belong to the user.

Severity: **A = would likely get the paper rejected**, B = reviewer will demand it, C = must be stated / small fix.

---

## A1. "Inverse crime": the simulator and the estimator use the same physics

- `arena_sim.py` does the ball physics itself: constant deceleration μ opposite to v, mirror bounce with restitution
  E on the normal velocity, tangential speed kept, the same E for walls and arm. MuJoCo contacts are **off**
  (`contype=0` on floor, walls, arm box). So MuJoCo only integrates and renders.
- `estimators/model.py` uses **exactly the same equations**, and μ and E are constant over a launch. The only
  differences are the step size and clamping vs mirroring at walls.
- **Evidence:** the oracle (true μ, E) forecasts 0.5 s ahead with a per-launch median of **9.5 mm** (test, ideal).
  A model that equals the truth always looks almost perfect.
- **Why it blocks publication:**
  - Data from your own model is a known reviewer objection ("inverse crime").
  - Every gray-box result in Phase 1 (μ in the state, E in the state) is guaranteed to look excellent, because the
    truth has exactly one constant μ and E per launch.
  - Phase 2 (learned filters) compares learned against gray-box in a world that **is** the gray-box model, so the
    outcome is decided by the data generator, not by the methods.
- **Real ball physics that is missing:**
  - A launched ball first slides (high deceleration, about μ_k·g) and then rolls (low deceleration). That gives two
    friction phases, not one.
  - Rolling resistance depends on speed.
  - At a bounce, spin changes the tangential speed (it is not kept).
  - Walls and the arm have different restitution, and E depends on speed.
  - Real floors are uneven and slightly tilted.
- **Fix (recommended):**
  - Make a second **"mismatch" dataset** whose physics the estimators do NOT know: slide-to-roll friction,
    speed-dependent E, tangential loss at bounces, different E for walls and arm, a small floor tilt.
  - Report every method on both the "matched" and the "mismatch" data. Phase 1 and Phase 2 conclusions then hold
    only if they survive the mismatch data.
  - Cost: change `arena_sim.step` behind a flag and re-render 500 launches (~90 min, 3 workers).

## A2. The test set has already been looked at and has shaped decisions

- We printed and studied TEST tables in Phase 0, and investigated launch 303 and the blind-spot launches.
- `step_1a_mu_vs_e_gap.py` and every "ALL 500" headline number include the test launches. The decision "μ first,
  then E" was made from all-500 results.
- **Why it matters:** a reviewer, or a careful examiner, will ask whether the test set was untouched.
  Right now it is not.
- **Fix (recommended):**
  - From now on, every design decision uses **train / val only**.
  - Before the final results, render a **fresh, larger test set** with a new seed (e.g. 500 launches, never looked
    at until the end) and report the final numbers on it. The current test set becomes a second validation set.
  - Cost: ~90 min rendering, once.

## A3. Results are not statistically supported yet (no confidence intervals, a few launches dominate)

- **Pooled RMS over frames weights launches unequally.** Fast launches have up to 600 moving frames, slow ones as
  few as 3, so 69 % of all evaluated frames come from launches at 2.5 m/s or faster.
- **Evidence (TEST, h0.5, 75 launches, bootstrap over launches):**

| Condition | Estimator | Pooled RMS | 95 % CI | Top-5 launches' share of squared error | Per-launch median |
|---|---|---|---|---|---|
| ideal | ekf19b | 77 mm | 58 – 96 mm | 54 % | 52 mm |
| ideal | oracle | 41 mm | 17 – 66 mm | **87 %** | 10 mm |
| all_real | ekf19b | 100 mm | 80 – 120 mm | — | — |
| all_real | oracle | 46 mm | 29 – 67 mm | — | — |

  - Two test launches have no h0.5 frame at all.
- **Why it matters:** improvements of 5–15 mm (the size Phase 1 will produce) cannot be claimed with ±20 mm
  intervals. RMS on 75 launches mostly measures 5 launches.
- **Fix:**
  - Main metric per launch: per-launch RMS, then the median and mean over launches.
  - Bootstrap 95 % CIs over launches.
  - **Paired** tests between methods: same launches and same noise seeds, e.g. a Wilcoxon signed-rank test on
    per-launch errors.
  - The larger fresh test set (A2) also shrinks the CIs.
  - Write one shared `stats.py` and use it everywhere.

## B1. Using noisy depth when the ball is on a known floor plane

- The ball centre always sits at z = r on a known floor. Intersecting the pixel ray with the plane z = r gives
  (x, y) **without any depth noise**: only pixel jitter is left, about 0.3 px ≈ 1.7 mm.
- We back-project with the measured depth (user decision in step 0.4). This is what makes `noise_real` / `all_real`
  hard: a lateral error of x·δz/z, about 13 mm at the wall with D435-like noise.
- **A reviewer will ask:** "why not intersect with the plane?" Without an answer, the realistic-noise conditions look
  artificially hard.
- **Fix, either:**
  - (a) Add the plane-intersection measurement as a baseline / condition and show both; or
  - (b) justify depth explicitly, e.g. the floor height or camera pose is only known roughly, or the method must
    extend to 3D / bouncing balls, and test with an extrinsic error.
- Cost of (a): ~10 lines in `sensing/camera.py`; it works offline on the existing raw data (u, v are stored).

## B2. "ideal" is not ideal: systematic bias and occlusion outliers in the measurements

- **Evidence (all 500, noise-free detections vs truth):**
  - Error: median 1.3 mm, p95 3.1 mm, **max 72 mm**.
  - The error grows with distance from the image centre (median 0.6 mm at 0.4–0.8 m, 2.3 mm beyond 1.2 m).
  - It is a **systematic outward radial bias**: +0.6 mm at 0.5–0.9 m, +1.6 mm at 0.9–1.3 m, +3.0 mm at 1.3–1.8 m.
    Cause: the depth is read on the sphere's surface but treated as if it were straight above the centre, plus
    lighting/threshold effects on the mask centroid.
  - Near the arm (partial occlusion → the visible-part centroid is shifted), p99 is about 20 mm and the max 72 mm.
- **Why it matters:**
  - Calling it "ideal / noise-free" is wrong.
  - The bias is not zero-mean noise, so an EKF cannot remove it.
  - The occlusion outliers are part of why ekf19b+stale "without gate" was chosen. The gate decision interacts
    with this.
- **Fix:**
  - Rename to "no added noise" in the thesis and document the bias and occlusion numbers.
  - Optionally correct the bias analytically: z_centre from the ray, not z_surface + r.

## B3. Baselines are not as strong or as fairly tuned as the proposed method will be

- ekf19b uses μ = 0.85, E = 0.725: the **midpoints of the uniform sampling ranges**, i.e. knowledge of the data
  generator. Meanwhile ekf-μ will be fitted by maximum likelihood on 350 train launches. Part of any gain would come
  from more tuning, not from the idea.
  - **Fix:** add "ekf19b with (μ, E, σ, q) fitted on train" as a baseline. That is the honest comparison for ekf-μ.
- **cvkf:** its tuned q is **on the grid edge (q = 20) in 5 of 8 conditions**, so it was not fully tuned. Gains are
  small, but a reviewer will see the flag.
  - **Fix:** extend its q grid (50, 200) or report it as tuned-to-saturation.
- **Oracle:** it uses the noise settings tuned for ekf19b, not its own, so the "upper bound" is not the best
  possible. Fine if stated; better to tune it too.
- **Missing standard baselines that reviewers expect:**
  - an online least-squares / sliding-window fit of μ (the simple competitor to ekf-μ)
  - an IMM (multiple-model filter)
  - for Phase 2, a plain learned sequence model
- `sigma_E = 0.13` equals the standard deviation of U(0.5, 0.95). This is again generator knowledge; it should be
  written down as such or fitted on train.

## B4. The silent-dropout result is circular

- `silent_hold` repeats the **bit-identical** last value, and StaleGuard drops bit-identical values.
- **Evidence:** tuned EKF val h0.5 is 61.5 mm for **both** dropout_flagged and dropout_silent, i.e. the guard turns
  silent loss into flagged loss perfectly.
- Good news from the checks: no moving ball ever gives two bit-identical ideal detections (0 of 78 044), so the guard
  never throws away good data here.
- **But:** a reviewer will say the stale problem is "solved by construction".
- **Fix:**
  - State it as a limitation, or
  - make a harder silent mode: a held frame re-measured with fresh pixel and depth noise, so it is not bit-identical.

## B5. The EKF covariance ignores the bounce (baseline approximation)

- In `EKF19._predict`, F has no reflection term. After a bounce the mean is reflected but P is not: the
  cross-covariances keep the wrong sign and are not scaled by E. ekf19b only adds variance.
- This copies Liu 2021, so as a baseline it is fine if described. Paper 03 S.4 shows the proper Jacobian through
  the bounce timing.
- **Fix:** keep it in the baseline (labelled), and do the correct bounce Jacobian in the proposed filters (ekf-μ / ekf-μE).

## C. Must be stated in the thesis (small or no code change)

1. **μ is a deceleration in m/s², not a friction coefficient.** Justify the range [0.2, 1.5] m/s². 1.5 m/s² is a
   high value for a *rolling* ball (≈ μ_k 0.15 sliding). Give a source, or re-derive it.
2. **Sensor idealisations:**
   - perfect intrinsics/extrinsics, no lens distortion
   - perfect timestamps (the filter knows t_capture and the delay exactly)
   - i.i.d. Gaussian noise (no correlated or flying-pixel depth error)
   - **no motion blur:** at 5 m/s and ~10 ms exposure the blur is 50 mm, about one ball diameter
   - perfect colour segmentation
   - dropout independent of the ball state
3. **Scope:**
   - static arm at Q_HOME, only the base box is an obstacle
   - planar ball, no spin
   - one arena size
   - μ and E constant within a launch and uniform over a range
4. **Simulator detail:** a wall bounce clamps the position to the wall (it loses up to v·dt = 10 mm at 5 m/s), while
   the estimator mirrors. This is a small model difference in the oracle; mention it, or switch the sim to mirroring
   when regenerating for A1.
5. Launch 303 (an exact-corner arm hit) and the blind-spot launches stay in the data; say so and show
   results with and without them.
6. "ALL 500" numbers mix train, val and test. Headline numbers must come from the held-out test set only (see A2).
7. **Reproducibility:**
   - Rendered detections may differ slightly on another GPU/driver, so **publish the raw dataset** (24 MB) together
     with the code and seeds.
   - `outputs/` is not in git, so the results need an archived copy.
8. **Novelty must be framed.** "Physics constants in the EKF state + maximum-likelihood learning of the noise" is
   paper 03's method.
   - Our contribution needs to be explicit, for example: a planar/rolling setting with an obstacle arm, a sensor
     ablation, the effect on catch success, and robustness under model mismatch (A1).
   - Write the claim before Phase 1 code so experiments test that claim.

## D. Data-safety risk (not scientific, but could lose the research)

- Branch `ball-estimation` was **never pushed** (no `origin/ball-estimation`). The laptop shuts down unexpectedly.
  One disk failure loses everything since the last push.
- `outputs/` (dataset, tuning, baselines) is not in git at all.
- **Fix:** push the branch to the private GitHub remote and back up `outputs/` (e.g. a zip on Google Drive / an
  external disk). Needs the user's OK.

---

## Checked and found OK (no action)

- MuJoCo contacts are off, so there is no double physics with the Python bounces.
- The simulator's arm bounce never logs a hit without flipping the normal velocity (0 events in 500 launches).
- No launch starts inside the arm.
- Camera geometry convention: cx = (W−1)/2, f from the vertical FOV. The measured bias is a few mm, not a
  convention error.
- Metrics are evaluated at the arrival time with forecast to t_arrival, so the delay is handled correctly. Ground
  truth is interpolated at 500 Hz.
- Same noise seeds (1000 + i) in tuning, baselines and the gap check. Tuning used only val launches.
- The split is stratified by speed. μ and E means are balanced, and launch 303 is noted.
- No bit-identical repeated detections while moving, so StaleGuard does not hurt the ideal condition.

## Proposed order (to discuss)

1. D (push + backup): 5 min.
2. A3 (shared `stats.py`: per-launch, bootstrap CI, paired test) and re-print the Phase-0 tables with CIs: ~1 h.
3. B1 (plane-intersection measurement) and B3 (train-fitted ekf19b baseline, cvkf grid): ~1–2 h.
4. A1 (mismatch physics dataset): design with the user first, then ~2 h code + 90 min rendering.
5. A2 (fresh, larger final test set, rendered once the methods are frozen; seed fixed now and written down).
6. Then step 1.2 (ekf-μ) with the B5 bounce Jacobian, evaluated on matched and mismatch data with CIs.

---

## Status of the fixes (user approved all, 2026-11-01)

| Item | Status |
|---|---|
| D | DONE: private repo `haidar996/ball-estimation-private` (remote `backup`), branch pushed; outputs zip as a private release + `~/backups/` |
| A3 | DONE: `estimators/stats.py`. step_0d / step_0i / step_1a now report per-launch statistics with CIs; paired tests in `step_0j` |
| B1 | DONE: `Pinhole.back_project_plane`, conditions `plane`, `all_plane` (paired with noise_real / all_real) |
| B2 | DONE: ray-corrected back-projection (bias +3.0 → +1.6 mm at the corners); the rest is a 0.08 % centroid scale, < 1/3 px, documented |
| B3 | DONE: `ekf19b_fit` (c, E fitted on train, step_0k); oracle and cvkf tuned as their own families (cvkf wider grid) |
| B4 | DONE: `silent_noisy` loss mode + condition `dropout_silent_noisy` |
| B5 | Planned for the proposed filters (ekf-μ): bounce Jacobian |
| A2 | Seed fixed: `final_test_plan.json` (seed 20261101). step_1a now uses train + val only; headline numbers on val + test (dev) |
| A1 | Designed (`docs/A1_mismatch_physics_design.md`, user decisions taken), implemented (`MismatchArena`), 12/12 closed-form checks pass; rendering pending |
| C | To be written into the thesis text (list above) |
