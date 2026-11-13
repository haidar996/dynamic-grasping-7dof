# RESUME — read this first in every new chat

## Message to paste in a new chat

> Continue the ball-estimation project. Read `src/franka_grasping/ball_estimation/RESUME.md` first
> (current state, decisions, next step), then tell me where we are before doing anything.

---

## Current state (updated 2026-11-01, end of turn)

- **2026-11-01: research audit done → `docs/research_audit_2026-11-01.md`. Waiting for user to choose which fixes
  (A1 mismatch physics, A2 fresh test set, A3 CIs, B1–B5, D push/backup) before step 1.2.**

- **Plan:** `PLAN_2026-10-29.md` (replaces FINAL_PLAN_2026-10-28.md). We restart Phase 0 from **step 0.2 = pick the FOV properly**.
- Step 0.1 (arena + static arm + bounces) done; quick test 10/10.
- **DECIDED 2026-10-29: image 480x480, FOV 49°** (rule: smallest FOV with arena + 5 cm margin in view).
- **Step 0.2 DONE 2026-10-30:** grid test confirms 49° → `camera_settings_locked.yaml`, results `outputs/fov_grid/` (table.json, fov_grid.png).
- **Step 0.3 DONE 2026-10-30:** 500 launches in `outputs/raw_60hz_h3_f49_sq480_with_arm/` (24 MB, 90.5 min), 11/11 checks pass.
- **Step 0.4 DONE 2026-10-30 (code + quick test):** user decided L from XML (1.22), new camera/data in step_0d,
  **EKF models the arm box**, **keep depth** back-projection. Quick baseline test (50 launches, all 6 conditions) OK.
- First full baseline run (untuned, 6 conditions) DONE 2026-10-30: renamed to
  `outputs/baselines_h3_f49_sq480_with_arm_stride1_v1_untuned.json` / `outputs/baselines_full_stride1_v1_untuned.log`.
- **Step 0.5 DONE 2026-10-30:** `split_500.json` (seed 2026; per speed 7 train, 2+1/1+2 val/test) via
  `experiments/step_0h_make_split.py` (refuses to overwrite). Balanced: speeds 50/50/50, mu 0.82/0.89/0.81, e 0.72/0.73/0.73,
  arm-hit launches 18.3%/12.0%/17.3%. Launch 303 is in test.
- **Step 0.6 DONE 2026-10-30 ~16:25:** tuning `outputs/ekf_tuning_val.json/.log`; final baselines
  `outputs/baselines_h3_f49_sq480_with_arm_stride1.json`, log `outputs/baselines_full_stride1.log` (8 conditions, ALL + TEST,
  RMS/median/p95). These are the official Phase-0 baselines (Phase 1 reports on TEST).
- **Step 0.7 DONE 2026-10-30:** README rewritten (current setup, scripts, legacy list), legacy GUI generator now exits,
  `ball_estimation/` committed on branch `ball-estimation` as **a711784** (155 files, 2 MB; outputs/ and PDFs excluded).
  Git has no user.name/email on this laptop: user chose author Haidar Saad, passed with `git -c` per commit (no config change). Not pushed.
- **PHASE 0 COMPLETE. Phase 1 started 2026-10-31.** 1.1 paper 03 read (pdftotext) + explained → `docs/phase1_paper03_takeaways.md`
  (old guide `papers/explained/03_achterhold_explained.tex` checked: accurate). Proposed step 1.2 = ekf-μ (μ in the state,
  learned noise by max likelihood on train). **Waiting for user decisions** (optimiser scipy vs PyTorch, μ only first, criterion).
  Papers 18, 01, 05 still to read.

## Open decisions (waiting for user)

(A2 L bug and A3 arm-in-EKF: FIXED in step 0.4.)

## Audit of earlier chats' work (2026-10-29)

- **A1 FOV 55° decision is not supported by its analysis.** `step_0a_camera_fov_test.py` called
  `sample_launches(rng, arena, 30)` = 30 launches *per speed*, then rendered only the first 30, so the FOV 55/60 sets are
  all 0.1 m/s launches (FOV 50: 0.1–0.3 m/s). Each FOV used different random launches (seed = fov). `step_0b_fov_analysis.py`
  looked at 5 launches; its "coverage efficiency" is capped at 100 for both 50° and 55°; the 94.3 vs 91.5 gap comes from the
  "centering" term, which depends on where the random launches were, not on the camera.
  Geometry: at 3 m the narrowest FOV that sees the whole arena is ~46–47°; 55° sees it with ~22% margin (fine);
  ~50° would give ~12% more pixels per metre (jitter error ~1.9 → ~1.7 mm). 55° is usable; only the reason was wrong.
- **A2 BUG: estimators assume the old 2x2 m arena.** `EKF19` default `L=0.97` (filters.py:36) = 1.0 − 0.03.
  New arena needs `L = 1.22`. `step_0d_baselines_with_arm.py` never passes L → every EKF (incl. oracle) would bounce
  the ball 25 cm early → Phase 0.4 numbers wrong. Does NOT affect Phase 0.3 data.
- **A3 Baseline EKFs know nothing about the arm base** (only walls). Arm bounces will look like model errors.
- **A4 FINAL_PLAN numbers are predictions, not results.** e.g. "EKF19b 7.2 mm", "learned-Q 5.2 mm", "learned-Q-E 2.8 mm"
  were never measured. Also it says the arm is at (1.25, 1.25); the arena is centred on the origin, so the arm is at (0, 0).
- **A5 Data split:** plan says 350/75/75 train/val/test; step_0d evaluates all 500 (fine for baselines with nothing
  learned, but Phase 1 must report on the test 75 only).
- **A6 Stale docs:** README (2x2 m, 3 m/60°), CHECKPOINT_2026-10-29.md (old state) are out of date.
- **A7 `experiments/step_0c_generate_with_gui.py`** writes into the real dataset folder with its own loop — do not run it.
- **A8 Nothing in `ball_estimation/` is committed to git** (whole folder untracked).
- OK: metrics.py, sensor.py, dataset conditions, camera geometry (ideal detection error median 1.64 mm).

## Log

### 2026-10-29 (this chat)
- Found Phase 0.3 bug: with the jointed arm included, the ball is qpos[9:11] but code used qpos[:2] → the launch pushed
  arm joints (damping 10000), the ball sat hidden in the arm base, uvz all NaN. Old 326 files → `outputs/_broken/`.
  Also: arm joints not actually locked; arm collision geoms contype=0 (ball would pass through).
- User decisions: arm frozen at **Q_HOME** (= [0,0,0,-0.1,0,0,0], nearly upright); ball bounces off arm base as a
  **Python box obstacle** with the same restitution E as the walls; box = link0 footprint (x −0.154..0.072, y ±0.095).
- Files: `scenes/make_static_panda.py` → `scenes/panda_static.xml` (no joints, nq=2); `scenes/arena.xml` includes it +
  invisible geom `arm_base`; `arena_sim.py` arm obstacle + assert ball is qpos[0:2]; bounce log now 4 columns
  (t, axis, side, source 0=wall 1=arm) in `dataset.py`/`launches.py`; launches never start in/near the base;
  `experiments/step_0c_generate_500_with_arm.py` rewritten (parallel, `--quick N`);
  `experiments/step_0c_check_with_arm.py` (10 realism checks + plot); `experiments/step_0c_view_with_arm.py` (GUI replay).
- Quick test 10 launches: 10/10 pass (static arm, walls + 2 arm bounces, camera sees 99.9%, misses only by the arm,
  detection median 1.64 mm). Plot: `outputs/test_with_arm/check.png`.
- Viewer froze once (on-screen text update while holding the viewer lock) → removed; now works. Run it yourself with
  `! cd .../ball_estimation && python3 experiments/step_0c_view_with_arm.py`.
- Allowed start points: x,y in [−1.12, 1.12] m, excluding x∈(−0.284, 0.202), y∈(−0.225, 0.225) around the arm.
- Agreed: update this file at the end of every turn.
- User chose to restart from the FOV pick; wrote `PLAN_2026-10-29.md` (Phase 0 steps 0.1–0.7, phases 1–4 kept,
  gates now relative, no invented mm targets). FINAL_PLAN marked superseded.
- User asked for more FOVs (now 40–70° step 1°) and an arena-fill criterion (don't waste pixels outside the arena); plan 0.2 updated.
- Researched image size + FOV criterion (user asked to think + search before running):
  - Today 640x480 @ 55° fills only **48%** of the image with arena (half the pixels wasted).
  - Accuracy depends on focal length in px f = (rows/2)/tan(fovy/2): at 49° f=527 px (5.6 mm/px, ball 10.6 px),
    at 55° f=461 (6.4 mm/px). Same rows → same accuracy; extra columns of a 4:3 image only see outside the arena.
  - 480x480 square: same accuracy as 640x480, 25% fewer pixels to render/process, fill 84% at 49°. Real square depth modes
    exist (Azure Kinect WFOV 512x512 / 1024x1024). 720x720: 1.5x pixels/m, ~2.25x cost.
  - FOV rule proposal: geometry fixes it — smallest FOV that keeps arena + 5 cm safety margin (≈1° mounting tilt at 3 m)
    in view at wall-top height → 48.3° → **49°**; the 40–70° grid sweep verifies it and gives the error-vs-FOV plot.
  - New finding: real stereo depth noise = Z²·subpixel/(f·B) (Intel). D435-like at 2.97 m ≈ 33 mm RMS vs our model 7.2 mm;
    k should depend on f (FOV + resolution). Depth noise leaks into x,y by tan(angle) ≈ 0.41 at the arena edge.
    Option for later: ball is planar → back-project with the known floor plane (RGB only, like RoboCup SSL-Vision) → no
    depth noise in x,y. Decide at the sensor step.
- User approved 480x480 + 49°. Wrote `experiments/step_0e_fov_grid.py` (2,4xx grid points, FOV 40–70°, paired noise) and started it.
- 2026-10-30: grid test run 1 → visibility OK (smallest passing FOV 46°, matches analytic 46.1°; 49°: 0 cut, margin 18 px, fill 83%),
  but errors jumped between FOVs (1 mm ↔ 15 mm). Cause: the Panda wrist LED (geom link6_12, material "green", rgba 0 1 0)
  has the ball's colour → detector merged it into the ball at some FOVs. Fix: make_static_panda.py recolours it off-white;
  panda_static.xml regenerated; spot check 49–52° now 0.4–5.5 mm. Grid test re-run started (run 1 results invalid for errors).
- Grid test run 2 (clean): smooth. 49°: 0 cut, margin 18 px, fill 83.3%, ideal median 1.23 mm, noise median 2.79 / p95 7.19 mm,
  corners p95 11.3 mm. Noise median grows 2.63 mm (46°) → 3.02 (55°) → 3.75 (70°). 10 grid points hidden by the arm.
  (FOV 67 shows 1 "cut" point: arm-shadow edge pixel effect, not the image edge.) Locked in `camera_settings_locked.yaml`.
- Step 0.3 part 1 done: generator/checker/viewer switched to 480x480 / 49° (`outputs/test_f49_sq480_with_arm/` quick,
  `outputs/raw_60hz_h3_f49_sq480_with_arm/` full); files now store cam_w/cam_rows (dataset.py; old files assumed 640x480).
  Old 55° quick test moved to outputs/_broken/. Running the 10-launch quick test (3 workers). Full run NOT approved yet.
- Quick test 480x480/49°: 10/10 pass, detection median 1.11 mm (was 1.64 at 640x480/55°), p99 2.90 mm; 1.9 min for 10 launches with 3 workers (15.6 CPU-s/launch) → full 500 ≈ 80–95 min.
- Full 500 run started with 3 workers (user approved).
- Mid-run check (75/500 done, ~80 min left): 8/10 pass. "ball hits arm" fail = expected (first 75 are the slow 0.1–0.8 m/s).
  REAL BUG: "speed never increases" fails in 6/75 (mu 1.13–1.38): at the end of a slide, when mu*dt > speed, friction
  overshoots and reverses v → endless ±1.1–1.6 mm/s jiggle for the rest of the 10 s (v_stop = 1 mm/s never reached).
  Positions unaffected (µm), but true velocity ≈1.4 mm/s instead of 0. Proposed fix in arena_sim.step: stop exactly when
  mu*dt >= speed (same rule as estimators/model.py). Waiting for user: fix + restart run, or not.
- User chose: stop, fix, restart. Run stopped at 89 files (moved to outputs/_broken/raw_f49_sq480_friction_jiggle_89files).
  Fix in arena_sim.step: apply friction only if speed > max(v_stop, mu*dt), else stop exactly. Physics-only check over all
  500 launches: 0 speed increases, 0 not-stopped (177 launches have mu > 1), 89 arm bounces in the dataset.
  Checker gained "balls that should stop are at rest" → quick test 11/11 pass. Full run RESTARTED (3 workers).
- Friction range explained to user: mu = sliding deceleration 0.2–1.5 m/s² (≈ coefficient 0.02–0.15 = mu/g); E 0.5–0.95.
- Full run done (90.5 min): 500/500, 11/11 checks. 934 wall + 89 arm bounces (86 launches) = exactly the physics-only
  prediction. 443/443 should-stop balls end at 0. Camera sees 99.6% of frames; 1,148 missed, all within 0.29 m of the base.
  Detection error median 1.27 / p99 4.35 mm. Error vs distance from centre: median 0.6 mm (<0.4 m) → 2.2 mm (>1.1 m,
  steeper view angle); no dependence on speed (renders have no motion blur).
  Partial occlusion by the arm: 993 frames (0.33%) have error > 10 mm (max 72 mm), all within 0.33 m of the base —
  visible-pixel centroid shifts when the arm hides part of the ball (only 6 from depth read on the arm). Realistic; kept.

### 2026-10-30 — step 0.4
- User decisions: 1 L from XML (yes), 2 new camera/data in step_0d (yes), 3 EKF knows the arm box (yes), 4 keep depth.
- `estimators/model.py`: `propagate(..., L, obs)` + `_box_bounce`: arm box (ball-centre box, cx -0.041, cy 0, hx 0.143,
  hy 0.125) with restitution E. **Swept segment test** (slab method): first version (smallest-penetration rule, like the
  sim) failed with 10 ms forecast sub-steps — wrong face near corners and corner tunnelling (errors up to 2.3 m).
  Started-inside (noisy state) → push out by smallest penetration, flip inward velocity.
- `estimators/filters.py`: EKF19 `L` now REQUIRED (no 0.97 default), `obs` arg; ekf19b extra variance fires on any hit
  (velocity component changes sign = wall or arm); cvkf drops L/obs. Old scripts step_f/step_g (old arena) now fail loudly — legacy.
- `arena_sim.py`: `Arena.estimator_geometry()` → dict(L=1.22, obs=(...)) — single source, read from arena.xml.
- NEW `experiments/step_0f_model_check.py`: true state + true mu/E, forecast 0.5 s, vs simulator (13,234 windows, 371 with
  arm bounce). With arm box: median 0.99 / p95 2.68 mm; arm-bounce windows median 1.30 / p95 3.13 mm. Without arm box:
  arm-bounce windows median 487 / p95 1990 mm. One leftover: launch 303 hits the box exactly at a corner (1 mm) — sim rule
  says side face, swept rule says top face → up to 1.4 m in 5 windows. Accepted (exact corner = ambiguous).
- `experiments/step_0d_baselines_with_arm.py`: RAW = raw_60hz_h3_f49_sq480_with_arm, Pinhole(3, 49, 480, 480), passes
  geometry, args `[stride] [conds|all_conds] [workers=3]`, output `outputs/baselines_h3_f49_sq480_with_arm_stride{N}.json`.
- Quick test stride 10 (50 launches, 6 conditions): 2.6 min. h0.5 RMS (mm), condition ideal / all:
  fd 590/688, cvkf 633/712, ekf19-none 84/100, ekf19b-none 83/98, ekf19_oracle 11/27. Log `outputs/baselines_quick_stride10.log`.
  Same pattern as before: stale guard fixes dropout_silent; hybrid gate hurts under noise for ekf19 (not ekf19b).
  now_vel RMS ≈0.15 m/s even for oracle-ideal = a few frames at bounce instants (top 1% = 99% of the sum of squares);
  median 0.007 m/s. Metric property, not a bug — consider reporting median/p95 too.
- User said go → full baseline run started (all 500, 6 conditions, 3 workers).
- Full baseline run done (500 launches, 6 conditions, 23 min). h0.5 RMS mm, ideal / noise / all:
  fd 591/608/686, cvkf 629/630/711, ekf19-none 75/79/90, ekf19b-none 74/78/89, ekf19_oracle-none 20/32/33.
  dropout_silent: plain EKFs ~185, ekf19b-none+stale 75 (stale guard works). Hybrid gate hurts under noise (ekf19 113 vs 79).
  Oracle 3–4 m/s bin (30 mm) > 4–5 m/s (15): launch 303 (3.1 m/s, exact-corner arm hit) sits in that bin.
- User asked for now_pos + all horizons: shown per condition from the stride1 JSON (no new run). now_pos ideal: EKFs 3.6 mm,
  fd 4.7, cvkf 8.1; "all": ekf19b 7.2, oracle 5.9, fd 34, cvkf 41. h1.0 ideal: ekf19b 216, oracle 54, fd 1454.
  Step 0.5 still waiting for OK.

### 2026-10-30 — deep realism check of the baselines (user asked)
- Scripts: `experiments/step_0g_error_breakdown.py`, `experiments/step_0g_depth_noise_test.py` (run from ball_estimation/).
- Physics sanity: fd h0.5 median 99 mm ≈ ½·0.85·0.5² = 106 mm (ignores friction) ✓; ekf19b h0.5 median 49 mm ≈ ½·|Δμ|·t²
  with median |Δμ|≈0.33 → 41 mm ✓; oracle h0.5 median 3.4 mm (model check from true state: 1 mm) ✓.
- now_pos: ideal → EKF median 1.3 = measurement median 1.32 (error is a fixed view-angle bias, can't be filtered);
  noise → measurement 2.86 → EKF 2.1 (filter helps) ✓.
- RMS dominated by few launches: now_pos RMS 65% from 5 launches (235, 222, 247, 317, 188) = ball rolls into the camera's
  blind spot under the arm (235: hidden 1.38–10 s, stops there), EKF predicts blind → 90–134 mm. Realistic, matters for grasping.
  Oracle h0.5 RMS 19.5 → 14.3 without launch 303 (exact-corner arm hit, sim-vs-model rule mismatch = sim artefact).
- Depth noise is OPTIMISTIC: ours σZ≈7 mm; D435-like stereo ≈33 mm (k=3.74e-3). With 33 mm: measurement median 7.0 mm;
  h0.5 RMS fd 762, ekf19b 103, oracle 76 (vs 608/78/32) → oracle advantage shrinks from 2.4x to 1.35x. EKF R (5 mm) not
  retuned for it.
- Other sim gaps (all optimistic): no motion blur, exact known timestamps/fixed delay, perfect colour segmentation/lighting,
  simple physics (constant sliding friction, no spin, same E walls+arm, tangential speed kept at bounce). "all" uses
  flagged dropout, not silent.
- Proposed decisions (waiting for user): (A) add realistic-depth-noise condition (and set EKF R per condition);
  (B) report median + p95 next to RMS; (C) launch 303: keep + note, or exclude.
- User: 'go to the next phase' → did step 0.5 (Phase 0 not finished: 0.6, 0.7 remain before Phase 1).

### 2026-10-30 — step 0.6 (user: yes to A, B, C)
- A: `dataset.py` new conditions `noise_real` and `all_real` (k_depth = K_REAL = 3.74e-3, ~33 mm at 3 m, D435-like).
  EKF noise (sigma, q) tuned per condition on the VAL launches: `experiments/step_0i_tune_ekf_noise.py`, criterion h0.5 RMS.
  Two families tuned separately (fair): "ekf" (tuned with ekf19b-none+stale, used by all ekf19*) and "cvkf" (needs larger q —
  smoke test with EKF settings gave cvkf now_pos 29 mm). Grid sigma 2–30 mm × q 0.03–20; flags optimum on the grid edge.
  First grid (q 0.1–2, ekf only): tuning changes little for old conditions (≤1–2 mm, default 5 mm/0.5 near-optimal), a lot
  for noise_real (94.5 → 72.8 mm, sigma 12 mm, q 0.1 at grid edge) → widened grid. Old log `outputs/ekf_tuning_val_grid3q.log`.
- B: `step_0d_baselines_with_arm.py` rewritten: per-frame errors → RMS / median / p95 for every metric, reported on ALL
  and on TEST (split_500.json), uses tuned noise per family. Smoke test OK.
- C: launch 303 kept (in test), to be noted in the thesis.
- Previous untuned results kept as `outputs/baselines_h3_f49_sq480_with_arm_stride1_v1_untuned.json` / `.log`.
- 12:48 user: stop everything (charging). Run killed; restart command at the top of this file.
- 15:21 user: restart → tuning + baselines restarted (chained, background).
- 0.6 DONE. Tuned EKF (val): ideal/dropout σ5 q2, noise/all σ8 q0.5, delay σ5 q5, noise_real/all_real σ12 q0.1 (94.5→72.8 mm
  on val). cvkf tuned to q 5–20 (several on grid edge, gain tiny) → behaves ≈ fd.
  h0.5 RMS (median) mm, ALL 500: ideal ekf19b 73 (47), oracle 23 (5), fd 591 (99); noise ekf19b 77 (52), oracle 25 (12);
  all ekf19b 89 (60), oracle 28 (13); noise_real ekf19b 87 (60), oracle 28 (14); all_real ekf19b 99 (69), oracle 32 (15), fd 847.
  → with tuned noise the oracle gap stays ~3x even with realistic depth noise (the 1.35x seen untuned was a tuning artefact).
  Hybrid gate no longer hurts much once sigma is tuned (ideal 75 vs 73).
  TEST vs ALL: medians/p95 match; RMS higher on test (oracle h0.5 40 vs 23, now_pos 6.8 vs 3.8) because test holds launch 303
  (corner hit) and a blind-spot launch → report median/p95 with RMS.
- 0.7: README rewritten, step_0c_generate_with_gui.py exits (legacy), committed (user: 'yes do 0.7 and commit'). Phase 0 complete.
- Commit a711784 (author Haidar Saad, chosen by user; git identity not configured on this laptop). Not pushed.

### 2026-10-31 — Phase 1.1, paper 03
- Read paper 03 + supplement via pdftotext; old guide accurate (incl. 29/30 = 96.7% not 97.7%). Wrote
  `docs/phase1_paper03_takeaways.md` (summary, mapping, proposed 1.2 = μ in the state + max-likelihood learning).
- Gap check (`experiments/step_1a_mu_vs_e_gap.py`): h0.5 ideal RMS/median: nominal 73/47, true μ only 50/6, true E only 55/43,
  both 23/5 (all_real 99/69, 64/20, 81/64, 36/16) → μ = typical error, E = bounce tail. μ first, then E.
- User asked for a full detailed PDF of paper 03 → `papers/explained/03_achterhold_explained_full.pdf` (21 pages, .tex beside it):
  every section + supplement (S.1–S.4: horizons, 26/30 without launch info, initial/learned values translated, learned C,
  full Jacobian incl. impact timing), derivations checked, critique (new slips found: Table S.2 header, ε missing in S.3/S.4,
  likelihood notation), corrected old guide (S.4 DOES treat bounce linearisation), project section with gap check + ekf-μ equations.
  Old short guide kept. Step 1.2 decisions still open (scipy vs PyTorch, μ first, likelihood criterion) / or read 18, 01, 05 first.

### 2026-11-01 — research audit (user: "deep thinking as a researcher about any bug ... anything that prevents publishing")
- Re-read all core code + ran 4 data checks. Full report: **`docs/research_audit_2026-11-01.md`** (no code changed).
- A (rejection risks): **A1 inverse crime** — sim physics (arena_sim) = estimator model exactly (MuJoCo contacts off;
  constant μ/E per launch) → oracle per-launch median 9.5 mm at h0.5; gray-box will look perfect, gray-vs-learned decided
  by generator → need a "mismatch" dataset (slide→roll friction, speed-dependent E, tangential loss, wall≠arm E, tilt).
  **A2** test set already looked at / used in decisions (step_1a, ALL-500) → decisions on train/val only + fresh larger
  final test set. **A3** no CIs: TEST h0.5 ekf19b 77 mm CI [58, 96], oracle 41 [17, 66]; top-5 launches = 54 % / 87 % of
  squared error; 69 % of frames from launches ≥2.5 m/s → per-launch metrics + bootstrap + paired tests.
- B: B1 ray–floor-plane intersection would remove depth noise (reviewer will ask); B2 "ideal" has outward radial bias
  +0.6/+1.6/+3.0 mm (0.5–0.9/0.9–1.3/1.3–1.8 m) and occlusion outliers up to 72 mm; B3 baselines use generator midpoints
  (μ 0.85, E 0.725, sigma_E 0.13 = std of U) → add train-fitted ekf19b; cvkf q on grid edge 5/8; oracle not tuned itself;
  B4 silent dropout + StaleGuard circular (61.5 = 61.5); B5 EKF P ignores bounce (Liu baseline; fix in our filters).
- C: thesis statements (μ is m/s² deceleration, sensor idealisations incl. no motion blur, scope, clamp vs mirror,
  reproducibility: publish raw data, novelty framing vs paper 03). D: branch never pushed + outputs/ not backed up.
- Checked OK: no double physics, sim arm bounces consistent, camera convention, metric timing, seeds, split, no
  stale false positives.
- **Waiting for user:** which fixes to do and in which order (proposed: D push/backup → A3 stats.py → B1+B3 → A1 mismatch
  data → A2 fresh test set → step 1.2). Step 1.2 decisions (scipy / μ first / likelihood) still open too.

### 2026-11-01 (later) — user: "i am ok with everything, and from u treat every detail as a research"
- All audit fixes approved. Memory `feedback-research-rigor.md` saved.
- **D DONE:** public repo → NOT used for chapter 2 (user chose private). Created **private repo
  github.com/haidar996/ball-estimation-private** (remote name `backup`), pushed `ball-estimation` (b1cc2cc) + `main`.
  Outputs backup = private release `outputs-2026-11-01` (21 MB zip: dataset, tuning, baselines, fov_grid, split, camera yaml),
  local copy `~/backups/ball_estimation_outputs_2026-11-01.zip` (sha256 5d0a09e2…). **Push to `backup`, never `origin`.**
- Next in this turn: A3 stats.py, B1 plane measurement, B2 depth ray correction, B3 (fit c,E on train, oracle + cvkf own
  tuning), B4 silent_noisy condition, A2 final-test seed, then re-run tuning + baselines (val+test), then A1 design (needs user).
- **Code changes 2026-11-01 (audit fixes, all approved):**
  - B2 `sensing/camera.py back_project`: surface→centre along the ray (zc = z_s + r·cosθ); outward bias +3.0 → +1.6 mm at
    1.3–1.8 m. Rest = pure 0.08 % scale from the pixel centroid (same in plane method), < 1/3 px → documented, left in.
  - B1 `back_project_plane` + SensorParams `xy_method` ("depth"/"plane"), `plane_dz`; conditions `plane`, `all_plane`
    (same jitter draws as noise_real/all_real → paired). Ideal plane: median 0.9 mm, max 28 mm (depth: 67 mm, arm occlusion).
  - B4 loss_mode `silent_noisy` (lost frame = last detection with fresh noise) + condition `dropout_silent_noisy`.
  - A3 `estimators/stats.py`: per-launch RMS, launch-mean/median + bootstrap CI, paired (bootstrap diff, Wilcoxon, win %).
  - B3 `ekf19b_fit` in make(); `experiments/step_0k_fit_constants_train.py` (c × E grid on TRAIN, ideal; all_real check);
    step_0i rewritten: families ekf / ekf_fit / oracle / cvkf (own wider grid), criterion launch-mean h0.5 on val.
  - step_0d rewritten: VAL + TEST(dev) only, per-family noise, stats tables, saves `outputs/baselines_v3_errors.pkl.gz`.
  - `experiments/step_0j_paired_tests.py` (paired comparisons); step_1a now train+val only with CIs.
  - A2 `final_test_plan.json`: final test seed 20261101 (500 launches), render only when methods frozen.
  - Archived Phase-0: `outputs/ekf_tuning_val_v2_phase0.json/.log`, `baselines_full_stride1_v2_phase0.log`.
- **RUNNING (started 19:16):** `bash outputs/run_audit_chain.sh` (fit → tune → baselines → paired → step_1a), ~1.5–2 h.
  Logs: outputs/fit_constants_train.log, ekf_tuning_val.log, baselines_v3_val_test.log, paired_tests_v3.log,
  mu_vs_e_gap_trainval.log. If the laptop dies: rerun the same command (each step overwrites its own outputs).
- Next: A1 mismatch-physics design (literature) → present to user before coding.
- **A1 design written:** `docs/A1_mismatch_physics_design.md` (slide→roll with spin, speed-dependent rolling resistance,
  launch slip ratio, e(v_n) falling with speed, wall ≠ arm e₀, Coulomb wall friction μ_w (Mathavan 2010: 0.14), re-sliding
  after bounces, exact mirror bounce, floor tilt 0–0.15°; paired launches = same p0/v0 (seed 42), extra params seed 43).
  **Waiting for user on 3 ⚠ decisions:** (1) rolling resistance a₀ realistic 0.1–0.8 m/s² (A, recommended) vs stretched
  0.2–1.5 (B); (2) arm e₀ 0.40–0.90; (3) tilt range; plus oracle on mismatch data = hindsight best-constant (recommended).
- **User decided A1 (all recommended):** rolling resistance 0.1–0.8 m/s², arm e₀ 0.40–0.90, tilt 0–0.15°, oracle on
  mismatch = hindsight best-fit constant (c, E) per launch (to implement when evaluating on mismatch data).
- **A1 implemented:** `arena_sim.py` MismatchPhysics + MismatchArena (step_mismatch, _impact, exact bounces);
  `launches.py` MismatchRanges + with_mismatch (seed 43, p0/v0 identical to seed-42 dataset — verified 0.0 diff);
  `dataset.record_raw` handles launch.phys (stores phys_* keys; mu/e = NaN so the matched oracle fails loudly);
  `step_0c_generate_500_with_arm.py --physics mismatch` (default workers now 3, was 8!).
  `experiments/step_0l_check_mismatch_physics.py`: 12/12 closed-form checks PASS (5/7 rule, roll time, a0(1+kv),
  5/7 g sinα tilt, e(v_n), tangential, wall friction removes 2.5·μ_w(1+e) of top spin, re-roll speed −(5e−2(1−f))/7).
  Key physics fact: rolling ball hitting a wall with e = 0.8 comes back at 0.29–0.47 of its speed (constant model: 0.8).
- Commit 8e98f22 pushed to `backup` (private). TODO after the chain: run step_0l full (500 truth-only, energy/position checks, matched vs mismatch stats), then render
  mismatch (`python3 experiments/step_0c_generate_500_with_arm.py --physics mismatch`, ~90 min), then step_0c checks.
- 19:30 user: "if it takes too much time we can do it after some hours and instead do some work now". Told: the chain runs
  in background (still on step_0k fit at 19:28, ~1.5–2 h left + 90 min mismatch render); light work can run in parallel.
  Offered: design ekf-μ (recommended) / read paper 18 / write thesis Methods+limitations / stop run. User wants to clarify
  first — waiting for their question.
- 19:30 user: stop the run now, rerun later, do other work now → **chain STOPPED** (was in step_0k, nothing finished, no
  result file overwritten: ekf_tuning_val.json is still the Phase-0 one). **To rerun later:** from ball_estimation/:
  `bash outputs/run_audit_chain.sh` (~2 h), then `python3 experiments/step_0l_check_mismatch_physics.py 3`, then
  `python3 experiments/step_0c_generate_500_with_arm.py --physics mismatch` (~90 min).
- User: "i have finished paper 03 reading, time to implement — what is your plan". Wrote **`docs/phase1_step12_ekf_mu_plan.md`**
  (state (x,y,vx,vy,ℓ=log μ); bounce Jacobian −E (B5) + finite-difference unit test; ML on train per condition with a
  batch numpy filter (all 350 launches in lockstep, 600 equal frame times) + scipy L-BFGS-B; variants ekf-μ hand /
  learned / learned-h05; evaluation on matched + mismatch, CIs, μ recovery, NIS/NEES, runtime). No torch/numba installed.
  **Waiting for 4 decisions:** (1) log μ vs a²+ε vs raw; (2) scipy+batch numpy vs PyTorch; (3) likelihood + h0.5 ablation;
  (4) learn E with the noise or keep train-fitted E. Audit chain still paused (rerun commands above).
- User asked for a detailed explanation of the step-1.2 plan → **`docs/phase1_step12_explained.md`** (why μ matters
  ½Δμt², how μ is learned from position via correlation, log μ, bounce Jacobian, the 7 numbers, likelihood with numeric
  example, batch trick, L-BFGS-B, comparisons, evaluation incl. NIS/NEES). Still waiting for the 4 decisions.
- **User decided step 1.2 (all recommended):** (1) log μ in the state; (2) scipy L-BFGS-B + batch numpy filter;
  (3) likelihood main + h0.5-trained ablation; (4) E learned with the other 6 numbers. Implementation started.
- **1.2a DONE:** `estimators/ekf_mu.py` (EKFMu: state x,y,vx,vy,log μ; propagate_jac = model.propagate op-by-op + Jacobian
  rows incl. wall/arm mirror −E and push-out; predicts at every frame tick; Joseph update; accumulates NLL).
  `experiments/step_1b_test_ekf_mu.py`: T1 mean identical 5000/5000; T2 Jacobian vs central FD max err ≤ 7e-9 for free,
  stop, wall bounce, arm bounce, push-out (all PASS). Diagnostic (14 train launches, ideal): |μ̂−μ| median at 0.5 s:
  q=2 → 0.23, q=0.1 → 0.09, q=0.01 → 0.03 (prior 0.29) → μ only observable when q small ⇒ learning matters.
- Next: 1.2b batch filter `estimators/ekf_mu_batch.py` + equality test vs single filter + evaluate().
- **1.2b DONE:** `estimators/ekf_mu_batch.py` (all launches in lockstep; masks for init/dropout/stale/stop/bounces;
  NLL + optional h-errors exactly like metrics.evaluate). `experiments/step_1b_test_ekf_mu_batch.py`: 6 conditions × 2
  random parameter sets × 30 launches: counts identical, NLL rel. diff 1.6e-13, h0.5 diff 3e-13 m → PASS.
  **350 train launches: 2.2 s per NLL evaluation** (3.4 s with h0.5 forecasts).
  Finding (document): forecast sub-step count ceil(dt/0.01) depends on the last bit of dt (0.5 vs 0.4999999 → 50/51
  steps) → up to 5 mm difference near bounces; batch now uses the same float expression as metrics.evaluate.
- Next: 1.2c learning script `experiments/step_1c_learn_ekf_mu.py` (L-BFGS-B, central-difference gradients in a pool of 3).
- **1.2c script written:** `experiments/step_1c_learn_ekf_mu.py nll|h05 [conds] [workers] [--quick]` (L-BFGS-B, central
  differences in pool of 3, 2 starts, saves after each condition, scores on val). Quick smoke test (ideal, 35 launches,
  3 iterations) OK: NLL −18.9 → −26.9, val launch-mean h0.5 38 mm; "low process noise" start better; σ hit lower bound
  0.5 mm → widened to 0.1 mm. Expected full run ~15–30 min per condition per criterion (heavy → overnight with chain).
- **1.2d/e written + smoke-tested:** make("ekfmu_<variant>+stale", **params); EKF19/EKFMu record NIS;
  `experiments/step_1d_tune_ekf_mu_grid.py` (val grid q × q_mu × s_l, σ from ekf_fit tuning, μ0/E = train fit; q tuned
  too, else strawman); `experiments/step_1e_eval_ekf_mu.py` (ekf19b, ekf19b_fit, ekfmu_grid/nll/h05, oracle; val + test(dev);
  stats + paired vs ekf19b_fit; μ recovery at 0.1–2 s; NIS (exp. 2), NEES (exp. 4); ms/update).
  Smoke (5 val launches, ideal, stand-in settings): h0.5 ekf19b 59 → ekfmu 32 mm, oracle 8; NIS of the h0.5-tuned
  filters 0.02–0.04 (≪ 2: σ 5 mm vs true ~1 mm → far too cautious; watch: NLL learning should fix); 0.15 ms/update.
- **ONE COMMAND for all heavy work (~11 h, 3 workers):** `bash experiments/run_overnight.sh` (from ball_estimation/):
  fit → tune → baselines → paired → μ-vs-E → ekf-μ grid → learn nll → learn h05 → eval → mismatch check → mismatch render.
  Progress: `outputs/overnight_progress.log`; finished steps skipped on restart (markers in outputs/.overnight_done/).
  NOT started yet — waiting for user to say when (laptop must stay on + plugged in).
- User: do NOT run now; asked how many hours (old paused run vs new). Answer: the ~11 h pipeline INCLUDES the old paused
  run. Old part ≈ 3.5 h (fit 15 min, tuning 75 min, baselines+paired+μE 10 min, mismatch check 5 min, render 90 min);
  new ekf-μ part ≈ 7.5 h typical (grid 10 min, learn nll ~2.5–3 h, learn h05 ~4 h, eval 20 min); worst case if the
  optimiser uses all 100 iterations: nll ~7 h, h05 ~10 h. Offered: run in two parts, or trim the h05 ablation.
- User asked for a full detailed PDF of Part 1 (math + processes) before running → **`papers/explained/part1_run_explained.pdf`**
  (17 pages, .tex beside it): world + matched physics; camera (f = 526.6 px, 5.6 mm/px), detection, ray correction
  derivation, plane method; sensor model (σZ = kZ², jitter, Markov loss p = rq/(1−r), loss modes, delay), 11 conditions;
  estimators (fd, model, EKF predict/update, Q derivation, gain example, χ² gate 9.21, hybrid, ekf19b σE = 0.13, fit,
  oracle, StaleGuard); metrics; statistics (per-launch RMS, bootstrap, paired, Wilcoxon worked example, multiple
  comparisons); Steps 1–7 each with purpose/procedure/math/outputs; mismatch physics derivations (7/2 μg slip decay,
  5/7 rule, 5/7 tilt, impact + wall friction, re-roll −(5e−2(1−f))/7); restart instructions; glossary.
  Run still NOT started (user: start later).
- **23:08 user: "run the part1 test" → Part 1 STARTED:** `bash experiments/run_overnight.sh part1` (script reordered: mismatch
  check + render now right after step 5; `part1` arg exits after them; Part 2 later = same command without arg, finished
  steps skipped). Expected ~3.5 h → ~02:40. Progress: outputs/overnight_progress.log. If interrupted: rerun same command.
- 00:52 progress: step 1 done (18.9 min, outputs/fit_constants_train.json); step 2 tuning at 675/825 (~95 min total); Part 1 ETA ~03:00.
- 01:05 user: do NOT start the mismatch render automatically after steps 3–6 → created skip marker
  `outputs/.overnight_done/render_mismatch_data` (running script not edited: bash reads scripts while running).
  The progress log will say "skip render mismatch data (done)" — it was NOT rendered. **To render later:**
  `rm outputs/.overnight_done/render_mismatch_data` then `bash experiments/run_overnight.sh part1` (or run step_0c
  `--physics mismatch` directly).
- **01:29 PART 1 DONE** (render skipped as asked). Full results: **`docs/part1_results_2026-11-02.md`**. Key: fit c,E ≈
  midpoints (no gain, p ≥ 0.3); gate hurts; plane −10 mm vs depth (p 4e-11); silent_noisy +36.5 mm; μ-first confirmed on
  train+val (true μ −21 mm vs true E −11 mm); mismatch physics 12/12 + 500-launch checks PASS, similar difficulty.
  **PROBLEM: tuning optima on grid edges** (ekf σ = 2 mm lower edge in 6 conds; oracle σ = 30 mm / q = 0.03 edges everywhere;
  cvkf q = 1000) → proposed: widen grids, re-tune (~2.5 h), re-run baselines + paired (~20 min). Waiting for user.
- 2026-11-02: user: re-run of tuning later. Asked for the Part-1 PDF rebuilt in **Arabic** with expanded explanation of
  the new methods → **`papers/explained/part1_run_explained_ar.pdf`** (20 pages, XeLaTeX + polyglossia + Amiri; .tex beside).
  Expanded "new method" sections #1–#8: ray correction (with figure + cosθ derivation), plane intersection (+ error
  analysis, camera-height sensitivity), silent_noisy (B4), per-launch stats (numeric example), bootstrap (worked
  example), paired test (Var(A−B) derivation) + Wilcoxon, fair fitted baseline + coordinate descent, grid-edge concept,
  inverse crime + mismatch physics; plus a section with the actual Part-1 results and the grid-edge problem.
  Arabic typesetting rules learned: numbers in math mode, ranges as "A إلى B", dates in \LR{}, quotes « », hyperref
  before polyglossia, no tcolorbox env named "code".
- **11:56 user: run re-tuning with wider grids → RUNNING** (`bash outputs/run_retune.sh`: step_0i → 0d → 0j → 1a, ~3 h,
  ETA ~15:00; progress outputs/retune_progress.log). New grids: ekf/ekf_fit σ 0.5–30 mm (9) × q 0.001–20 (9);
  oracle σ 1–120 mm (7) × q {0, 1e-5 … 0.1} (7; q = 0 = physical limit, not flagged as edge); cvkf σ 1–20 mm (7) ×
  q 2–20000 (8); 267 settings. Narrow-grid results kept as outputs/*_narrowgrid.* (tuning, baselines, paired, μE).
  If interrupted: rerun `bash outputs/run_retune.sh` (whole chain again).
- 14:22 check: tuning 750/825 jobs (143 min); then baselines (~17 min) + paired + muE → ETA ~15:05. User asked to be told when ready.
- **14:58 RETUNE DONE.** ekf/ekf_fit: no edges; 21/22 optima identical to narrow grid → baselines + μE unchanged.
  oracle flat in q (≤0.3 mm), σ 30–60 mm; cvkf edges but flat and irrelevant (~450 mm). Paired conclusions unchanged.
  Written to docs/part1_results_2026-11-02.md ("Re-tune with wider grids"). Next (only with user OK): render mismatch
  data (rm outputs/.overnight_done/render_mismatch_data; bash experiments/run_overnight.sh part1) or Part 2 (ekf-μ).

- **2026-11-03 02:15 user: run the render → RUNNING** (bash experiments/run_overnight.sh part1; render marker removed; ETA ~03:45). Output outputs/raw_60hz_h3_f49_sq480_with_arm_mismatch/, log outputs/raw_mismatch_render.log. First files OK (phys_* keys present, mu/e NaN, same p0 as matched). If interrupted: rerun the same command (resumes per launch).
- **03:47 RENDER DONE** (92 min): 500/500 files, 26 MB, no tmp left. Checks: p0/v0 identical to matched (paired), no |pos|>1.22, ~21% fewer bounces than matched, median moving time 2.5 vs 2.3 s, missing detections 0.6% of frames. Next (needs user OK): hindsight best-constant oracle + baselines on mismatch data (--data option), then Part 2 (bash experiments/run_overnight.sh).
- 2026-11-06 user asked hours: nothing running. Done so far ~7 h compute (Part 1 2.4 h, retune 3 h, render 1.5 h). Remaining: mismatch baselines + hindsight oracle ~1–1.5 h (+ ~1 h coding), Part 2 ekf-mu ~7.5 h typical (worst ~17 h). Total ~9 h typical.
- 2026-11-06 user plan: TODAY finish A (hindsight oracle + baselines on mismatch data); TONIGHT ekf-mu grid + learn nll — each run only when user says "run". Proposed A design (waiting yes): (1) oracle = per-launch grid mu 0.1–1.5 x E 0.3–0.95 (~150 pairs) minimising h0.5 with ekf19b_oracle; (2) pick pair on ideal, reuse in all conds; (3) keep matched-val sigma/q; (4) all 11 conds val+test(dev), paired mismatch-vs-matched, hindsight gap, mu-first check (best mu only / best E only). ~1 h code + ~40 min run. Explained in simple words on request.
- 2026-11-06 new chat: user "let us continue". Nothing running. Re-stated status + design A; still waiting for yes/changes before coding A.
- 2026-11-06 user asked which runs remain: listed A (mismatch, not coded, ~40 min) + Part 2 four steps (grid 10 min, nll ~2.5–3 h, h05 ~4 h, eval 20 min; ~7.5 h typical, worst ~17 h). 7 overnight markers done.
- **2026-11-07 user: "run those, one after the other" (A + run 1) → RUNNING** `bash experiments/run_A_then_grid.sh` (A ~30–40 min, then ekf-mu grid ~10 min).
  NEW `experiments/step_0m_mismatch_baselines.py` (design A approved by "run"): hindsight grid mu 0.05–1.50 (30) × E 0.30–1.00 (+0.725, 16)
  = 480 pairs per launch on ideal, twice (oracle noise → ekf19b_oracle; ekf noise → mu-first filter of step_1a); ties → closest to
  nominal; then 13 baselines × 11 conds val+test (MATCHED val-tuned noise + train fit; oracle names = hindsight pair) + mu-first
  (nominal / best mu only / best E only / best both) on ideal + all_real; paired mismatch-vs-matched, hindsight gap, fit, gate.
  Smoke test (--quick 3) OK in 80 s. Logs outputs/mismatch_baselines.log, outputs/ekf_mu_grid_val.log; progress outputs/overnight_progress.log.
  Markers shared with run_overnight.sh. If interrupted: rerun the same command. Runs 2–4 (learn nll, learn h05, eval) NOT started.
- **01:52 A + GRID DONE.** Results → `docs/stepA_mismatch_results_2026-11-07.md`. Key: on mismatch, ekf19b 45→67 mm (val ideal),
  hindsight-best constants 4→~36–39 mm (oracle gap was mostly inverse crime); fd/cvkf improve; μ-first holds but weaker
  (val μ-only −17 vs E-only −10 mm, p 0.002; test diff n.s.); fit −3 mm; gate hurts. Hindsight grid: 15/147 at E=0.30 edge.
  ekf-μ grid (val, matched): ideal 26.6 mm, all_real 47.7 — BUT ALL 11 optima on grid edges (q low edge 0.003, s_l 1.0 high,
  dsn s_l 0.2 low; q_mu flat). Proposed: widen q to 1e-4 and s_l 0.05–3 (~50 min) before learn/eval. Waiting for user.
  Not committed yet.
- **2026-11-07 user: "ok both modify then run" → widened both grids + bug fix, FULL PART 2 RUNNING:** `bash experiments/run_overnight.sh`
  (Part 1 markers skip) = ekf-mu grid (~1 h) → A mismatch (~1.2 h) → learn nll (~2.5–3 h) → learn h05 (~4 h) → eval (~20 min); ~9 h typical.
  Changes: step_1d grid q 1e-4…0.5 (8) × q_mu (3) × s_l 0.05…3 (7), per-axis edge + profiles printed; step_0m μ 0.05–2.0 × E 0.05–1.0
  (+0.725), E = 1.0 not an edge (physical limit), reuses first-run cells (verified identical to fresh); A added to run_overnight.sh
  after the grid; run_A_then_grid.sh deleted. **BUG FIXED step_1c:** start values read ekf_tuning_val_v2_phase0.json (pre-audit, no
  plane/all_plane/dropout_silent_noisy → would have crashed) → now ekf_tuning_val.json; smoke test plane OK.
  First-run outputs kept as outputs/*_narrowgrid.* (ekf_mu_grid_val, mismatch_*). If interrupted: rerun the same command.
- 2026-11-07 ~06:0x user: "do 1 and 2" = only ekf-mu grid + step A now. Created SKIP markers (NOT run!) outputs/.overnight_done/
  ekf-mu_learn_nll, ekf-mu_learn_h05, ekf-mu_evaluation (files contain "SKIPPED"). Chain ends after A (~08:10).
  **To run learning later:** `rm outputs/.overnight_done/ekf-mu_learn_* outputs/.overnight_done/ekf-mu_evaluation` then
  `bash experiments/run_overnight.sh`.
- **08:07 GRID + A DONE** (learn/eval skipped as asked). Appended to docs/stepA_mismatch_results_2026-11-07.md. Step A: wider grid
  changes ≤ 0.8 mm, hindsight bound ~35.5 mm (val ideal), oracle-pair edges 16 → 0, all findings stand. ekf-μ grid val h0.5:
  ideal 26.6, noise 31.3, all_real 47.7, dsn 94.7 mm; remaining edges flat (≤ 0.1 mm) except dsn s_l → 0 (= μ estimation off).
  Next (needs user): learn nll / h05 / eval (rm the 3 SKIPPED markers, bash experiments/run_overnight.sh, ~7 h). Not committed.
- 2026-11-07 user asked to split the ~7 h learn/eval into 3.5 h + 3.5 h. Proposed: `run_overnight.sh --hours 3.5` budget; step_1c checks before each condition (elapsed + mean condition time ≤ budget) else exits code 3 (no marker), rerun resumes. Session 1 ≈ nll all + 1–2 h05 conds; session 2 ≈ rest h05 + eval. Waiting for yes; then rm 3 SKIPPED markers.
- **2026-11-07 21:44 user "yes" → SESSION 1 RUNNING:** `bash experiments/run_overnight.sh --hours 3.5` (learn nll → h05 …, ends by ~01:15).
  New: run_overnight.sh `--hours H` (RUN_DEADLINE env; step starts only if EST fits; eval EST=25 min); step_1c checks before each
  condition (1.1 × mean minutes of learned conds, default nll 15 / h05 22) → exit 3 = clean budget stop (no marker); learn logs
  now appended (>>). Tested: 3-min budget → clean stop, no marker/json; 20-min budget quick → continues. All SKIPPED markers removed.
  **SESSION 2 later: same command** `bash experiments/run_overnight.sh --hours 3.5` (resumes at the next condition).
- **21:46 user: "stop it now, rerun when I tell you" → STOPPED** (session 1 killed 2 min in, during nll ideal; no condition finished,
  no json, no learn markers → nothing lost but those 2 min). **When user says run:** `bash experiments/run_overnight.sh --hours 3.5`
  (learn nll → h05 → eval). Note: `pkill -f run_overnight.sh` also kills the calling shell — kill by PID instead.
- **2026-11-08 04:56 user "run it now" → SESSION 1 RUNNING** `bash experiments/run_overnight.sh --hours 3.5` (ends by ~08:26). Session 2 later = same command.
- 06:19 status: nll ideal DONE (71.5 min!, val h0.5 31.1 mm vs grid 26.6; both starts ABNORMAL_TERMINATION_IN_LNSRCH), nll noise
  running. ~70 min/condition vs 15 estimated → as is: nll ~13 h, h05 ~15–25 h (old 7 h estimate WRONG). Session 1 will stop
  after noise (~07:20–07:30; budget check). Cause: objective converged by ~20–40 gradient evals (change < 0.01), then 60–100 evals
  wasted in failing line searches (noisy central-difference gradients). Proposed fix (waiting user): early stop (no gain > 1e-3
  over 10 evals) → ~25–30 min/cond; options: 1 start only, drop/trim h05 ablation.
- **2026-11-08 06:23 user: stop, fix 1+2, rerun → done.** step_1c: early stop (best gain < TOL over PATIENCE=10 gradient evals;
  TOL nll 1e-3, h05 1e-5 m; returns best point seen) + N_STARTS=1 ("from ekf19b tuning"). First nll/ideal (2 starts, 71 min,
  val h0.5 31.1 mm) kept as outputs/ekf_mu_learned_nll_v1_2starts.json/.log; ideal is RE-LEARNED with the new procedure (consistency).
  Smoke test (PATIENCE 3, quick data) OK. **SESSION 1 (again) RUNNING** ~06:3x `bash experiments/run_overnight.sh --hours 3.5`.
  Expected ~10–15 min/condition → nll ~2–2.5 h, h05 maybe ~3 h, eval 20 min → ~2 sessions.
- **2026-11-08 user: "stop everything now" → ALL STOPPED** (session killed during nll ideal; see next line for what was saved).
  Nothing running. To resume when user says: `bash experiments/run_overnight.sh --hours 3.5` (finished conditions are skipped).
- 2026-11-08 user wants many parts instead of 2 sessions. Proposed: 11 sections = 1 condition each (learn nll → learn h05 → eval, ~30–45 min), report after each; needs step_1e to MERGE into ekf_mu_eval.json/pkl (now overwrites) + experiments/run_section.sh <cond>. Suggested order: ideal, noise, all_real, dropout_silent_noisy first. Waiting for yes.
- **2026-11-08 user: yes, make the change and wait → DONE, NOTHING RUNNING.** Part 2 now runs per SECTION (= 1 condition):
  `bash experiments/run_section.sh <cond>` = learn nll → learn h05 → eval (~30–45 min), marker outputs/.sections_done/<cond>,
  refuses if another step_1c/1e is running, checks the condition name. step_1e now MERGES into ekf_mu_eval.json/_errors.pkl.gz
  (saved after each condition; records `missing` estimators + time). Tested: merge (2 conds in scratch) OK, bad name rejected.
  Suggested order: ideal, noise, all_real, dropout_silent_noisy, then the rest. WAIT for the user to name the section.
  (run_overnight.sh Part 2 still works but is no longer the plan.)
- 2026-11-08 06:31 user: run ideal → SECTION ideal RUNNING (bash experiments/run_section.sh ideal).
- 06:32 user: when ideal finishes as expected → immediately run section noise, then report.
- 06:59 SECTION ideal DONE (28 min) → results in docs/part2_sections_results.md (h05 best: test 30 mm vs fit 55, grid 39 with NIS 39 on test). Section noise STARTED 06:59 as instructed.
- 07:02 user: after noise run all_real then dropout_silent_noisy → queued (background chain waits for noise marker; stops if a section fails).
- 07:29 SECTION noise DONE (h05 34 / nll 35 / grid 36 vs fit 59 mm test; nll NIS 1.96). all_real STARTED 07:29 (chain), then dsn. Outliers: grid diverges on test 461/480 (fast, high μ, arm hits); launch 235 bad for all filters — TODO.
- 08:13 SECTIONS all_real + dropout_silent_noisy DONE (4/11 done: ideal, noise, all_real, dsn). all_real: h05 51 / grid 55 / nll 58 vs fit 76 (test). dsn: NO gain; nll COLLAPSES (σ 0.1 mm, q 78, bounds; 312 mm) = ML fooled by undetectable stale frames. Remaining 7: dropout_flagged, dropout_silent, delay, all, noise_real, plane, all_plane. Nothing running. TODO: launch 235 (bad for all filters), grid divergence 461/480.
- 08:55 user: run remaining sections one by one, each ONLY after user allows. Started dropout_flagged now; then WAIT for permission. Order: dropout_flagged, dropout_silent, delay, all, noise_real, plane, all_plane.
- 09:35 SECTION dropout_flagged DONE (40 min): test h05 30 / grid 32 / nll 37 vs fit 55; no bounds. 5/11 done. WAITING for user permission before dropout_silent.
- 09:36 user: run dropout_silent → RUNNING. After it: report and WAIT for permission.
- 10:09 user: stop everything → STOPPED during dropout_silent learn h05 (nll saved; h05 partial lost; no eval, no marker). Nothing running. To finish it: bash experiments/run_section.sh dropout_silent (nll skipped). Stop by PID with pgrep -f "[s]tep_1c" (bracket avoids killing own shell). Wait for user.
- 2026-11-08 12:53 new chat, user: "continue". Checked: nothing running, charging 48 %. 5/11 sections done; dropout_silent nll saved, h05 lost at eval 31 when stopped (restarts from scratch). NOT started: rule = each section only after explicit permission → asked. Flagged: (1) step_1c/1e read only the matched data, so ekf-mu has NO mismatch-physics result although the plan requires it (A1) → needs code + run; (2) results + code since 741c74f uncommitted, learned outputs not in the 2026-11-01 backup → offered commit + push to `backup`.
- 2026-11-08 ~13:0x user: commit + push, then run dropout_silent. Commit 6d4e027 pushed to `backup` (verified same hash; xml_meshes left out, not ours; outputs/ still NOT backed up since 2026-11-01). SECTION dropout_silent RUNNING (bash experiments/run_section.sh dropout_silent; nll skipped, h05 from zero, ~30 min). After it: report and WAIT for permission before delay. If interrupted: rerun same command.
- 13:22 SECTION dropout_silent DONE, 6/11 (marker set). Result = BIT-IDENTICAL to dropout_flagged (params, objectives, all 4500 per-launch error arrays diff 0) = audit B4 (silent_hold + StaleGuard), not a new bug → merge with flagged in the thesis, do not count as a separate condition/test. Written to docs/part2_sections_results.md (Section 6). Nothing running. Results doc + this file not committed yet. WAITING for user permission before delay. Remaining: delay, all, noise_real, plane, all_plane. Mismatch evaluation of ekf-mu still not coded (timing decision open).
- 13:2x user: run delay → SECTION delay RUNNING (bash experiments/run_section.sh delay; ~15–40 min). After it: report and WAIT for permission before all. If interrupted: rerun same command.
- 13:45 delay still running (h05 eval ~13). Finding: delay NLL learning is bit-identical to ideal (objective -25.140794562392315, same params): the likelihood uses capture-time innovations and delay only shifts arrival (exact known fixed delay), so NLL parameters carry no delay information; h05 differs (seen in its log). Expected, not a bug; note in the Section 7 write-up + known-delay limitation.
- 14:05 SECTION delay DONE (37 min), 7/11. Test h0.5: fit 62, grid 41, nll 43, h05 33; val 54/30/35/30 mm. nll = bit-identical to ideal (expected). Written to docs/part2_sections_results.md (Section 7). Battery switched to DISCHARGING at 81 % around 14:05 (told user: check charger). Nothing running. WAITING for user permission before all. Remaining: all, noise_real, plane, all_plane. Docs + this file not committed since 6d4e027. Mismatch eval of ekf-mu still not coded.
- 14:06 user: run all → SECTION all RUNNING (bash experiments/run_section.sh all; ~15–40 min). After it: report and WAIT for permission before noise_real. If interrupted: rerun same command.
- 14:29 all still in learn nll (23 min, eval 65, slower than earlier conds; q 1.8e-4 heading to low bound 1e-4 as in all_real, mu0 0.40). Battery 61 % discharging (-20 % in 23 min ≈ 70 min left) → told user to plug in.
- 14:57 SECTION all DONE (51 min), 8/11. Test h0.5: fit 67, grid 41, nll 45, h05 39; val 61/36/42/34 mm. NEES test grid 676, nll 396, h05 1.5 (TODO: cause). Written to docs/part2_sections_results.md (Section 8). Battery 35 % DISCHARGING (~0.9 %/min) → do NOT start noise_real until plugged in. Nothing running. WAITING for user permission before noise_real. Remaining: noise_real, plane, all_plane. Docs + this file not committed since 6d4e027. Mismatch eval of ekf-mu still not coded.
- 22:18 user: run noise_real → SECTION noise_real RUNNING (bash experiments/run_section.sh noise_real; ~15–50 min). Battery at start: 80 Discharging . After it: report and WAIT for permission before plane. If interrupted: rerun same command.
- 22:49 SECTION noise_real DONE (29 min), 9/11. Test h0.5: fit 68, grid 48, nll 62, h05 45; val 61/40/55/38 mm. NLL gain nearly gone (-5.8, p 0.055). Checked (read-only, scratchpad): all_real params give train NLL -17.014 vs learned -17.069 but val h0.5 43.8 vs 55.3 → NLL flat, poor proxy for forecasting; early stop NOT yet ruled out (gain 0.00098 vs TOL 0.001, E still moving). Proposed: continuation test from learned point with tighter TOL (scratch only, no overwrite, needs user OK). Written to docs/part2_sections_results.md (Section 9). Battery 43 Discharging . Nothing running. WAITING for user before plane. Remaining: plane, all_plane. Docs + this file not committed since 6d4e027. Mismatch eval of ekf-mu still not coded.
- 22:58 user: run plane (chose this over the continuation test, which stays open) → SECTION plane RUNNING (bash experiments/run_section.sh plane; ~15–50 min). Battery at start: 40 Discharging . After it: report and WAIT for permission before all_plane. If interrupted: rerun same command.
- 23:17 plane: nll DONE + saved (14.3 min, val h0.5 35.2 mm, q 1.4e-4 near low bound), h05 at eval 9. Battery 15 Discharging  (about -1.3 %/min) → will not finish unplugged; told user to plug in NOW. Safety copy of ekf_mu_learned_*.json + ekf_mu_eval.json/_errors.pkl.gz in outputs/_safety_copy_2026-11-08/. If power dies: bash experiments/run_section.sh plane (nll skipped); check those files load first.
- 23:18 user: stop everything → ALL STOPPED during plane learn h05 (nll saved; h05 partial lost; no eval, no marker). Battery 6 Discharging . Nothing running, result files load OK. 9/11 done. To finish plane: rerun its section command (nll skipped; ~25 min). Then all_plane. Stopping tip: pgrep -f matches the calling shell if its command text contains the pattern anywhere (it killed its own shell this time) → list PIDs first, then kill exact PIDs in a separate call. Open: NLL early-stop continuation test, NEES 396–676 on all, grid outliers, launch 235, mismatch eval of ekf-mu not coded. Docs + this file not committed since 6d4e027. WAIT for user.
- 2026-11-09 00:15 user: continue → SECTION plane RESUMED (nll skipped, h05 from zero, ~25 min). Battery 19 % charging. After it: report and WAIT for permission before all_plane. If interrupted: rerun same section command.
- 2026-11-09 00:36 SECTION plane DONE, 10/11. Test h0.5: fit 58, grid 33, nll 37, h05 33; val 51/30/35/29 mm. Paired plane − noise_real: h05 test −12.7, val −8.5; grid −14.4/−10.2; fit −10.1/−10.2 (matches Part 1). h05 and grid tie here. NEES calm (test 4.4/14.8/0.9). Early-stop concern stronger: nll in plane + all_real + noise_real all start q 0.1, only noise_real stops near it (19 grads) → continuation test for noise_real recommended (scratch only, needs user OK). Written to docs/part2_sections_results.md (Section 10). Battery 36 Charging . Nothing running. WAITING for user before all_plane (last section). Docs + this file not committed since 6d4e027. Mismatch eval of ekf-mu not coded.
- 2026-11-09 00:47 user: run all_plane → LAST SECTION RUNNING (~15–50 min). Battery at start: 46 Charging . After it: report 11/11 and WAIT (continuation test on noise_real + commit both still need user OK). If interrupted: rerun same section command.
- 2026-11-09 01:31 SECTION all_plane DONE (44 min) → **ALL 11/11 SECTIONS DONE**, matched physics only. Test h0.5: fit 65, grid 38, nll 40, h05 37; val 59/34/36/33 mm. Overall: ekf-μ −20 to −29 mm vs fit in 9/10 distinct conditions (dsn: no gain); h05 ≈ grid (≤ 3 mm typically; grid outliers on test); h05 ≥ nll everywhere; h05 covariances too cautious. Paired tables + reading in docs/part2_sections_results.md (Section 11 + Overall). Battery 74 Charging . Nothing running. WAITING for user. Next options: (1) commit + push docs (nothing committed since 6d4e027); backup outputs/ as release (last 2026-11-01); (2) noise_real continuation test, scratch only; (3) design + code ekf-μ evaluation on mismatch physics (A1, biggest gap); (4) diagnostics NEES / grid outliers 461,480 / launch 235.
- 2026-11-09 02:01 user: check. Nothing running, 11/11 done. Still waiting for go on commit/backup + noise_real continuation test.
- 2026-11-09 02:03 user: go → commit 3777d27 pushed to backup; backup zip built at ~/backups/ball_estimation_outputs_2026-11-09.zip (103.6 MB), private release upload IN PROGRESS, see next line. noise_real NLL continuation test RUNNING (2 starts, scratchpad only, log + json continue_nll_noise_real.* in session scratchpad; no project file changed).
- 2026-11-09 02:15 private release outputs-2026-11-09 created (sha256 verified by re-download). Continuation test still running.
- 2026-11-09 02:28 CONTINUATION TEST DONE (val only, nothing overwritten): noise_real NLL was stopped too early. Published −17.069 / q 0.093 / val h0.5 55.3 → A −17.173 / q 0.00021 / 44.7; B −17.174 / q 0.00022 / 47.7 (grid 40.0, h05 37.9). 100 × TOL left though last-10 gain was 0.00098 → stop rule unsafe. "NLL gain nearly vanishes" CORRECTED in docs Section 9. Whole NLL column PROVISIONAL. Others suspicious: noise, all_plane (q ≈ 13 × their same-noise partners all, plane), dropout_flagged. h05 stop rule untested. Script experiments/step_1f_continue_nll_check.py; results outputs/continue_nll_noise_real.*. Proposed (needs user OK): tighter NLL stop, re-learn NLL for 9 conditions (delay = ideal, silent = flagged reuse) + eval, ~3 h; also test h05 stop on 1 condition. Mismatch design after that. Not committed since 3777d27. Nothing running. WAIT for user.
- 2026-11-09 02:33 user: check. Nothing running. Waiting for decision: h05 stop-rule check, then NLL rerun (~3 h).
- 2026-11-09 02:34 user: go → h05 CONTINUATION TEST RUNNING (experiments/step_1g_continue_h05_check.py, noise_real, 2 starts, stop 1e-6 m or 40 grads, val only, ~30–45 min; writes only outputs/continue_h05_noise_real.*). If interrupted: delete a partial outputs/continue_h05_noise_real.json, then rerun. After it: report + WAIT. NLL rerun not started. Not committed since 3777d27.
- 2026-11-09 02:48 h05 CONTINUATION TEST DONE (noise_real, val only): 14 min total. Published train 44.645 / val 37.9 → A 44.643 / 37.9; B (from all_real) 44.686 / 37.8. h05 stable, no sign early stop hurt it (one condition; optimiser barely moved on a non-smooth objective, so not proof of global optimum). Written to docs Section 9. Proposed NLL rerun (needs user OK + decisions): step_1c TOL nll 1e-4; move current results to ekf_mu_learned_nll_v1.json + ekf_mu_eval_v1.* first; FIRST section noise_real as acceptance test (must reach ~-17.17 from the original start); then ideal, noise, all_real, dsn, dropout_flagged, all, plane, all_plane; delay + silent reuse; eval all 11. ~3 h. Nothing running. Not committed since 3777d27. WAIT for user.
- 2026-11-09 02:55 user: check. Nothing running. Waiting for go on NLL rerun: noise_real first as acceptance test, then 8 other fits (9 total).
- 2026-11-09 03:26 user: go → NLL RERUN started. step_1c TOL nll 1e-4 (h05 unchanged). Old results MOVED to outputs/ekf_mu_learned_nll_tol1e-3.*, ekf_mu_eval_tol1e-3.*, ekf_mu_eval_tol1e-3_errors.pkl.gz, .sections_done_tol1e-3 (11 conds, checksums OK). ekf_mu_learned_h05.json kept. Current ekf_mu_eval* start empty and fill per section. Acceptance test RUNNING: section noise_real (must reach train NLL ≈ -17.17 from the original start q 0.1). If it fails: stop + tell user. Then 8 others only with permission: ideal, noise, all_real, dropout_silent_noisy, dropout_flagged, all, plane, all_plane; delay + dropout_silent re-evaluated reusing params. Battery 100 Full . Not committed since 3777d27.
- 2026-11-09 03:57 ACCEPTANCE TEST PASSED, noise_real NLL rerun: train −17.1724 from the original start (86 grads, 28.7 min), q 0.0003. val h0.5 47 (old 55), test 51 (old 62); fit 61/68, grid 40/48, h05 38/45. Other estimators bit-identical to archive. NIS 2.00 but NEES test 132 (old 7) → low q may drive the NEES blow-ups (untested). Written at the top of docs/part2_sections_results.md. Nothing running. WAIT for permission per section. Remaining NLL reruns (8): noise, all_plane, dropout_flagged, ideal, all_real, dropout_silent_noisy, all, plane; each up to ~30 min + 2 min eval, ~4 h total. Then eval delay + dropout_silent. Command: bash experiments/run_section.sh <cond>. Not committed since 3777d27 (TOL change, tests, docs).
- 2026-11-09 04:09 user: run noise → NLL RERUN section noise RUNNING (TOL 1e-4; ~up to 30 min + eval). Battery 100 Full . After it: report and WAIT. If interrupted: rerun same section command.
- 2026-11-09 04:20 section noise rerun DONE: identical to old fit (−21.80748, q 0.0024; line search stuck). BUT noise train NLL at the old all params = −21.86183, better by 0.054, with val h0.5 35.9 vs 31.0 → TOL 1e-4 NOT sufficient, "acceptance passed" held only for noise_real; better likelihood forecasts worse (early stop may flatter NLL). NLL RERUN PAUSED, all_plane NOT started. Proposed (needs user OK): pilot on noise, scratch only: restart L-BFGS on stall + second low-q start, best train NLL, accept if ≤ −21.86; ~30 min. If it works, code into step_1c and rerun 9 fits, maybe 6–8 h. Alternative: report NLL as limited by the optimiser. Current state: ekf_mu_eval* hold noise_real + noise only; 11-cond archive = *_tol1e-3.*. Docs top section updated. Nothing running. Not committed since 3777d27. WAIT.
- 2026-11-09 04:31 user: check. Nothing running. Waiting for decision on noise pilot (restart on stall + low-q start).
- 2026-11-09 04:32 user: go → NLL PILOT RUNNING (experiments/step_1h_nll_restart_pilot.py, noise, 2 starts, restart on stall, cap 80 grads/start, up to ~50 min; writes only outputs/pilot_nll_noise.*; train + val only). Pass: train NLL ≤ −21.86 (fixed before running). After it: report + WAIT. If interrupted: delete partial pilot json, rerun.
- 2026-11-09 05:19 PILOT DONE, PASSED on noise: start A (original + restarts) −21.80783, val 31.0; start B (low-q + restarts) −21.87164, val 32.8 (published −21.80748 / 31.0). q at lower bound 1e-4 in B. Both starts and restarts are needed. Better NLL forecasts slightly worse on val (+1.8 mm, unpaired). Hypothesis: matched physics → ML wants q → 0 (inverse crime), untested. Docs updated. Decision needed: (1) code N_STARTS 2 + restart on stall (NLL only) in step_1c + rerun 9 fits + 11 evals, ~7.5 h, section by section; (2) q lower bound: recommended lower to 1e-6 so it is not an active constraint (grid-edge rule); (3) alternative: do mismatch evaluation first, since the q → 0 behaviour may differ there. Nothing running. ekf_mu_eval* hold only noise_real + noise under the 1e-4 rule; 11-cond archive = *_tol1e-3.*. Not committed since 3777d27.
- 2026-11-09 22:10 user: go and commit → b3ec46c (tests + TOL 1e-4 + docs) and 29ce137 (step_1c: NLL 2 starts + restart on stall, cap 80/start, q lower bound 1e-6; h05 path unchanged, smoke test identical) pushed to backup. noise_real + noise 1e-4 results archived as outputs/*_tol1e-4.* + .sections_done_tol1e-4 (checksums OK). Current ekf_mu_learned_nll.json / ekf_mu_eval* start EMPTY. SECTION noise RUNNING with the new rule (~50 min). Expect train NLL ≤ −21.87 (pilot). After it: report and WAIT. Then decide: other 8 fits (~7 h) or mismatch design first. If interrupted: rerun same section command.
- 2026-11-09 22:22 user: stop → noise section (new rule) STOPPED during learn nll, round 2 of start 1; nothing saved for noise (no json written), all archives intact. Battery 42 Discharging . Nothing running. To resume: bash experiments/run_section.sh noise (~50 min, plugged in). Then other 8 fits or mismatch design. WAIT.
- 2026-11-10 11:42 user: run noise → SECTION noise RUNNING again (new NLL rule, from zero, ~50 min). Battery 93 Discharging . After it: report and WAIT. If interrupted: rerun same section command.
- 2026-11-10 12:36 SECTION noise DONE with final NLL rule (53 min): train −21.87180 (start B, q 8.1e-5, cap 80 hit), val h0.5 33.1 (old 31.0; paired +2.2, p 0.006), test 36.2 (old 35.5, n.s.). NIS 2.00, NEES test 662. Pattern: NEES rises as q falls (7 → 662). Better likelihood forecasts slightly worse → inverse-crime hypothesis (ML → q → 0 on matched physics). Docs updated. 1/9 fits done under the final rule (noise). Nothing running. Decision needed: remaining 8 fits (~50 min each, ~7 h, section by section) and/or mismatch design now (no compute; I recommend writing it now). Remaining fits: noise_real, all_plane, dropout_flagged, ideal, all_real, dropout_silent_noisy, all, plane; then eval delay + dropout_silent. Not committed since 29ce137 (docs + RESUME).
- 2026-11-10 user: ok → committed d61d9e1 (noise final-rule docs) + pushed to backup. Wrote docs/A1_ekfmu_mismatch_eval_design.md (no compute): arm T = matched-learned params applied to mismatch (9 conds, eval only, NLL part waits for matched reruns); arm R = re-learn everything on mismatch incl. baselines (fairness), subset ideal/noise/all_real/dsn, ~7.3 h; pre-registered predictions (H-q: mismatch NLL q ≥ 1e-3 in ideal/noise/all_real; NEES < 20; NLL–h05 gap narrower; NLL degrades more in transfer; ekf-μ gain stays positive, bounded by ~28 mm); Holm on main claims; PHYSICS env switch + _mismatch output suffix, matched outputs untouched. Matched NLL q (old rule): ideal 0.016, noise 0.0024, all_real 1e-4, dsn 78; final noise 8.1e-5. Nothing running. WAITING for 5 decisions (arms, R subset, baseline re-tune, order, predictions). noise_real matched rerun NOT started (needs explicit "run noise_real").
- 2026-11-10 user: ok → all 5 design decisions accepted (recorded in docs/A1_ekfmu_mismatch_eval_design.md, predictions frozen). CODED (uncommitted): physics_cfg.py (PHYSICS=matched|mismatch, output suffix _mismatch), switch in step_0i/0k/1c/1d/1e, step_1e ARM=transfer|retrain + hindsight row + corr(mu_hat, a0) + gap closed, --quick in 0i/0k/1e, run_section.sh mismatch modes (baselines / retrain section / ARM=transfer). Smoke 1 (matched identical, 300 arrays) PASS; smoke 2 (T baselines + hindsight = Step A, 250 arrays) PASS; smoke 3 (R chain, scratchpad smoke_R.sh, temp *_mismatch.json auto-removed) RUNNING. Quick files outputs/*_quick* are smoke outputs (delete before real runs). Next after smoke: commit, then ask go for R pilot = `PHYSICS=mismatch bash experiments/run_section.sh baselines` (~1 h) then `PHYSICS=mismatch bash experiments/run_section.sh ideal` (~100 min).
- 2026-11-10 smoke 3 (arm R chain) PASS; temp + quick files removed. Amendment 4: mismatch q_mu grid widened to 1.0 (smoke hit edge 1e-2, still falling). run_section.sh refuses 7 bad invocations. Timing from smoke: step_0k mismatch ~50 min (1.4 min / 10 launches), step_0i ~20 min → baselines section ~1 h 10 min; ideal R section ~105 min → pilot ~2 h 55 min. Committed + pushed (see git log). Nothing running. Battery 93 % DISCHARGING at last check. WAITING for user go: `PHYSICS=mismatch bash experiments/run_section.sh baselines` then `... ideal`.
- 2026-11-10 user: run baselines → RUNNING `PHYSICS=mismatch bash experiments/run_section.sh baselines` (step_0k fit c,E on mismatch train ~50 min → step_0i re-tune ekf/ekf_fit noise on mismatch val, 4 conds ~20 min). Started at 79 % battery DISCHARGING. Each step writes only at its end: if interrupted, rerun the same command (restarts that step). Logs outputs/fit_constants_train_mismatch.log, ekf_tuning_val_mismatch.log. After it: report and WAIT for go on `PHYSICS=mismatch bash experiments/run_section.sh ideal`.
- 2026-11-10 15:34 user: stop everything → STOPPED (battery 5 %). Baselines section: step_0k DONE (36 min) → outputs/fit_constants_train_mismatch.json: c 0.5, E 0.5, train h0.5 51.5 mm, interior (not on grid edge). step_0i (noise re-tune) KILLED at 150/300 jobs (nothing saved; no marker). To resume: rerun ONLY step_0i: `PHYSICS=mismatch python3 -u experiments/step_0i_tune_ekf_noise.py 3 >> outputs/ekf_tuning_val_mismatch.log 2>&1 && touch outputs/.sections_done_mismatch/baselines` (~45 min; NOT the whole section, which would redo the 36-min fit). Plug in first. Then wait for go on ideal.
- 2026-11-10 20:37 user: continue → RUNNING step_0i alone (mismatch noise re-tune, ~45 min, fit kept). Battery 70 % DISCHARGING. On success the marker .sections_done_mismatch/baselines is set. If interrupted: same command as in the 15:34 entry. Then WAIT for go on ideal.
- 2026-11-10 21:23 mismatch noise re-tune DONE (46 min) → outputs/ekf_tuning_val_mismatch.json. BUT 4/8 optima ON GRID EDGE: ekf ideal q = 20 (top); ekf_fit sigma = 30 mm (top) in noise, all_real, dsn. Grid-edge rule → baselines not final. I REMOVED the marker .sections_done_mismatch/baselines so the ideal section cannot start on them. Proposal (needs user OK): amendment 5 = widen mismatch grid (sigma up to 120 mm, q up to 100) and rerun step_0i (~75 min), file kept as *_edge.json. val h0.5 (mm) mismatch vs matched: ekf ideal 64.9/45.3, noise 71.7/50.2, all_real 89.9/67.6, dsn 100.0/92.9; ekf_fit 51.6/46.9, 57.1/51.9, 70.2/70.8, 82.3/94.5. Battery 13 % DISCHARGING. Nothing running.
- 2026-11-11 16:25 user: continue the project → amendment 5 applied (step_0i mismatch grid: sigma + {50, 80, 120} mm, q + {100}; run-1 files renamed *_mismatch_edge.*; design doc updated). RUNNING step_0i run 2 (~70 min, 120 cells × 2 families × 4 conds), battery 93 % DISCHARGING at start. On success: marker .sections_done_mismatch/baselines set. If interrupted: rerun the same step_0i command (fit kept). Then: check edges again; if clean, report and WAIT for go on `PHYSICS=mismatch bash experiments/run_section.sh ideal` (~105 min).
- 2026-11-11 16:28 user: stop it now → step_0i run 2 STOPPED 2 min in (nothing saved; no marker; run-1 edge file + fit intact). Nothing running. To resume: `PHYSICS=mismatch python3 -u experiments/step_0i_tune_ekf_noise.py 3 >> outputs/ekf_tuning_val_mismatch.log 2>&1 && touch outputs/.sections_done_mismatch/baselines` (~70 min, plugged in). WAIT for user.
- 2026-11-11 user: rethink + new plan → docs/plan_2026-11-11_paper_plan.md (F1–F11 evidence table, claims C1–C5, work plan P0–P10 ≈ 25 h compute, open predictions, 5 decisions). Rethink changes: (a) amendment 6 in A1 design: prediction 5 "bounded by 28 mm" was WRONG (ekf-μ's μ drifts; hindsight constant bound is a reference, gap closed may exceed 1; new 5b, 5c); (b) propose CORE SET ideal, noise, all_real, all_plane, dsn for all remaining heavy steps, other 5 conds matched-only appendix without NLL; (c) NLL reruns only for core set (4 fits); (d) new cheap step: covariance calibration α on val for h05 (NIS→2); (e) learned comparison (C5, KalmanNet-style, torch) is the biggest open risk, scheduled after mismatch arm; (f) dsn = detection problem, stop optimising estimators there. Nothing running. Battery 86 % DISCHARGING. WAITING for the 5 decisions in plan §6; then P0 = step_0i run 2 (70 min) on go.
- 2026-11-12 00:10 user: go → all 5 plan decisions accepted (recorded in plan doc). NOTE: core set adds all_plane to arm R, so step_0i on mismatch must also tune all_plane: run 2 is started on the 4-cond default; all_plane will be added by a separate step_0i call with the cond argument (~18 min) BEFORE its arm-R section (step_0i overwrites the json: merge needed → handle then). P0 RUNNING: step_0i run 2 (widened grid, ~70 min, started 00:11, battery 17 % CHARGING). On success marker .sections_done_mismatch/baselines is set. If interrupted: same step_0i command. Then: check edges → report → WAIT for go on P1 (`PHYSICS=mismatch bash experiments/run_section.sh ideal`, ~105 min).
- 2026-11-12 00:25 user: AUTONOMOUS MODE: chain runs back to back without asking; every 15 min check charging + battery; if NOT charging stop the running job (exact PIDs) and wait, restart it when charging again. Gate failure: apply written fallback once, then stop and report. Still: docs + RESUME + commit (backup only) after every section. Queue: P0 (running) → P1 arm-R ideal → P2 NLL reruns ideal, all_real, all_plane, dsn (`bash experiments/run_section.sh <c>` after moving old nll entry? NO: step_1c skips learned conds → use `--redo` via a direct step_1c call, see P2 note when reached) → P3 calibration (code first) → P4 arm R noise, all_real, all_plane, dsn (all_plane needs step_0i mismatch tuning first, merge json) → P5 arm T. Cron job every 15 min created in this session (session-only; a new chat must re-create it).
- 2026-11-12 01:18 P0 DONE (68 min): only ekf_fit dsn still on edge (σ 120 mm, 82.3→81.8 mm, flat → accepted as saturation per fallback). Marker baselines set. Starting P1: `PHYSICS=mismatch bash experiments/run_section.sh ideal` (~105 min; grid → nll → h05 → eval retrain). Battery 69 % charging.
- 01:33 P1 ideal: grid step done (10 min): q 0.5, q_mu 1.0, s_l 0.5 → 52.3 mm, ON EDGE q (top, still falling 57.2→52.3) and q_mu (top, 54.3→52.3). Fallback (widen once) TO DO after the section ends: add q {2, 5} and q_mu {10} for mismatch in step_1d, rerun `PHYSICS=mismatch python3 -u experiments/step_1d_tune_ekf_mu_grid.py ideal` (~15 min) then re-eval `PHYSICS=mismatch ARM=retrain python3 -u experiments/step_1e_eval_ekf_mu.py ideal 3` (~15 min). Direction already supports prediction 5c (q_mu ≫ matched 1e-3) and H-q-like behaviour for q. nll learning running since 01:29.
- 2026-11-12 11:25 (session was idle 02:27 → 11:18). P1 ideal DONE 02:27: ekf-μ gains ≤ 3 mm over ekf19b_fit on mismatch ideal (test h05 −3.1 p 3e-5; val n.s.; grid/nll ≈ 0); hindsight still 18 mm better; nll q 0.065 (H-q supported), NEES 20; h05 q 4.5. Prediction 5 effectively refuted on ideal. Written to docs/A1_ekfmu_mismatch_results.md. Grid edge fallback prepared: step_1d mismatch grid widened (q +{2,5}, q_mu +{10}); run-1 grid file renamed *_run1.json. Battery 91 % DISCHARGING → nothing started. NEXT when charging: (1) `PHYSICS=mismatch python3 -u experiments/step_1d_tune_ekf_mu_grid.py ideal >> outputs/ekf_mu_grid_val_mismatch.log 2>&1` (~20 min) then `PHYSICS=mismatch ARM=retrain python3 -u experiments/step_1e_eval_ekf_mu.py ideal 3 >> outputs/ekf_mu_eval_mismatch_retrain.log 2>&1` (~2 min); (2) P2 NLL reruns (matched, core set): `python3 -u experiments/step_1c_learn_ekf_mu.py nll ideal 3 --redo >> outputs/ekf_mu_learned_nll.log 2>&1` (ekf_mu_learned_nll.json currently holds only noise, so --redo not even needed) then step_1e ideal (matched), same for all_real, all_plane, dropout_silent_noisy.
- 2026-11-12 16:40 user: go continue (battery 90 % DISCHARGING, explicit override) → RUNNING ideal grid rerun (wider grid, ~20 min) + retrain re-eval (~2 min). Then P2 NLL reruns (matched core set) if still allowed; 15-min checks continue (stop if the user's override is not repeated and battery < 40 %).
- 16:56 ideal grid rerun DONE (q 5, q_mu 10, still edge but flattening → accepted, widened twice). Re-eval: grid −2.9 mm vs fit (test), = h05. Docs updated. Battery 72 % DISCHARGING: P2 (NLL rerun ideal, ~50 min) would end near 20 % → HOLDING until charging (or user says go). Next command: `python3 -u experiments/step_1c_learn_ekf_mu.py nll ideal 3 >> outputs/ekf_mu_learned_nll.log 2>&1 && python3 -u experiments/step_1e_eval_ekf_mu.py ideal 3 >> outputs/ekf_mu_eval.log 2>&1` (matched, PHYSICS unset).
- 2026-11-12 17:50 user: go go continue (override, discharging) → P2 RUNNING: matched NLL rerun ideal (~50 min) + eval. Will continue to all_real, all_plane, dsn while the user's override holds; stop below 25 % unless charging.
- 18:05 P2 ideal nll STOPPED at 38 % (drain 1.5 %/min, would hit 25 % before the next check; nothing saved: step_1c saves per condition at the end). Resume when charging with the same command (RESUME 16:56 entry). Note: start A reached -25.14080 in round 2 (= old value), so ideal's NLL optimum may already be the old one; start B pending.
- 18:19 user: go; CHARGING 51 % → P2 ideal nll rerun RESTARTED (~50 min) + eval. Then all_real, all_plane, dsn (same commands with the condition name).
- 18:34 charger OFF at 42 % → P2 ideal nll STOPPED again per rule (nothing saved; 14 min lost). Resume with the same command when charging.
- 18:34 user: ignore charger, run NLL down to 25 % → NEW RULE for P2: stop only if battery < 25 % and not charging. P2 ideal nll RESTARTED.
- 18:36 user: do not stop until 15 % and not charging → stop rule now battery < 15 % while not charging.
- 19:22 P2 ideal DONE: NLL −25.14106 = old point (confirmed), test 38 mm. Starting P2 all_real.
- 20:09 P2 all_real DONE: NLL −17.18408 (start B, q 1e-5), test 57 mm (old 58). Starting P2 all_plane (83 % discharging; rule: stop < 15 %). Then dsn. USAGE LIMIT reached in this chat at 20:10: a new chat must re-create the 15-min cron (see feedback-autonomous-run-mode memory).
- 23:22 laptop SLEPT ~20:30 → 23:18 (journal); all_plane nll resumed by itself (45 grads so far, log alive). Battery 61 % discharging, rule < 15 %. Chain continues: all_plane → dsn.
- 23:55 user: do not stop it even at 15 % → NO battery stop any more for this chain (all_plane eval → dsn). Monitoring only.
- 2026-11-13 00:05 P2 all_plane DONE: NLL −23.163 (cap 80), test 39 mm (old 40) → confirmed. dsn nll running (start A −16.121 AT BOUND, start B running). Battery 13 % CHARGING.
- 2026-11-13 00:22 P2 COMPLETE: dsn NLL −16.807 (start B, s_l at bound), test 138 mm (old 312), still +42 vs fit. Core-set NLL column now final. NEXT P3: write experiments/step_1i_calibrate.py (post-hoc covariance inflation α for ekf-μ: α from val NEES, report test NEES/NIS before/after; core set, matched now, mismatch when available), then P4 (arm R noise/all_real/all_plane/dsn; all_plane needs step_0i mismatch tuning first).
- 00:40 P3 script experiments/step_1i_calibrate.py written + validated (before-values = step_1e). noise result: scalar P inflation CANNOT calibrate h05 (NIS R-dominated: alpha_nis 3e-4, NIS 0.23→0.32) and val-fitted alpha does not transfer to test NEES (grid 330→20 with alpha_nis but nll 662→813; nees-alpha gives test 106–309). RUNNING chain (scratchpad chain_p3_p4.sh): calib matched core → calib mismatch ideal → step_0i mismatch all_plane (merged) → arm R noise (~2 h total). Battery 29 % charging.
- 00:51 user: stop if battery < 30 % while not charging (new rule; replaces 'no stop').
- 2026-11-13 04:22 (gap: usage limit 02:00 → 04:20; chain had finished 02:02). P4 noise DONE: grid −1.8, h05 −1.1 (n.s.), nll +25 mm vs fit; nll calibrated (NIS 1.9). Battery 100 % Full (AC). Starting P4 all_real. P3 calibration write-up still pending (outputs/ekf_mu_calibration*.json).
- 04:30 P3 written up in docs/part2_sections_results.md (negative result: scalar α cannot calibrate; C2 fix withdrawn). P4 all_real RUNNING since 04:22 (~100 min). Then all_plane, dsn, then P5 arm T.
- 05:33 P4 all_real DONE: grid −2.5, h05 −2.2 (n.s.), nll +9 vs fit. Battery 47 % DISCHARGING → starting all_plane anyway (rule: stop < 30 %; stages save separately: grid json → nll json → h05 json → eval).
- 05:35 user: stop now, rerun at 06:05 when charging → all_plane STOPPED (grid step, nothing saved). One-shot cron at 06:05: `PHYSICS=mismatch bash experiments/run_section.sh all_plane` (start only if charging; else wait for charger).
- 06:03 charging (39 %) → all_plane arm R RESTARTED (~100 min). The 06:05 one-shot will find it running (run_section.sh refuses a second run).
- 07:23 P4 all_plane DONE (grid −2.5, h05 −0.2 n.s., nll +4.3). Starting P4 dsn (last arm-R section). Then P5 arm T (eval only, 10 conds × ~2 min).
- 07:25 P5 chain prepared: scratchpad chain_p5.sh (10 transfer evals, ~25 min total); start it after dsn arm R finishes.
- 08:42 P4 dsn DONE → ARM R COMPLETE (5/5). Summary table in docs/A1_ekfmu_mismatch_results.md. Starting P5 (arm T, 10 transfer evals, ~25 min; scratchpad chain_p5.sh). Battery 50 % discharging (rule < 30 %).
- 09:02 P5 DONE (16 min) → P0–P5 COMPLETE. Transfer catastrophic: matched-learned ekf-μ 30–58 mm worse than matched-tuned fixed μ on mismatch, every condition; nll transfers less badly than h05 (prediction 4 refuted). Written in docs/A1_ekfmu_mismatch_results.md. Battery 27 % DISCHARGING → nothing started. NEXT (light, no heavy compute): P6 launch-235 look + notes; then rewrite plan claims (C1/C3), then P7 learned comparison (needs torch install + code; heavy training only when charging).
- 12:45 P6 done (launch 235 = exact two-wall corner hit at 0.34 s; notes written). Charging 54 %. Starting P7 prep: pip install torch (CPU) in background (user approved at plan decision 2).
