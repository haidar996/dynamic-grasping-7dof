# Part 1 results (run 2026-11-01 23:07 → 2026-11-02 01:29; render skipped on user request)

Logs/JSON in `outputs/`: fit_constants_train, ekf_tuning_val, baselines_v3_val_test (+ errors .pkl.gz), paired_tests_v3,
mu_vs_e_gap_trainval, check_mismatch_physics. TEST = test (dev), 75 launches (73 with an h0.5 frame).

## Step 1: fit c, E on train (ideal; all_real check)
- ideal: best c 0.75, E 0.725 → launch-mean h0.5 51.2 mm; nominal (0.85, 0.725) 51.4 mm.
- all_real: best = nominal (0.85, 0.725), 75.3 mm.
- The surface is flat near the midpoints. The generator midpoints happen to be near-optimal, so audit B3 turns out to be
  small in practice (now measured, not assumed).

## Step 2: noise tuning on val (launch-mean h0.5)
- ekf: σ 2 mm (ideal-type conditions), 12 mm (noise_real/all_real), 3 mm (plane), 20 mm (silent_noisy).
- **Grid-edge problem:**
  - ekf σ = 2 mm (lower edge) in 6 conditions;
  - oracle σ = 30 mm (upper edge) and q = 0.03 (lower edge) in all conditions: the oracle wants to trust its perfect model
    and average the camera heavily (an inverse-crime symptom);
  - cvkf q at 1000 (upper edge) in several conditions.
  → The grids must be widened and the tuning re-run before these numbers are final.
- The per-launch criterion chooses σ = 2 mm (Phase 0 pooled criterion: 5 mm), closer to the real ~1 mm noise.

## Step 3: baselines, TEST (dev), launch-mean h0.5 [95 % CI] mm
| condition | fd | cvkf | ekf19b+stale | ekf19b-hybrid | ekf19b_fit+stale | oracle+stale |
|---|---|---|---|---|---|---|
| ideal | 417 | 416 | 54 [47,62] | 63 | 55 | 7 [4,14] |
| noise | 456 | 436 | 59 [52,67] | 70 | 59 | 8 |
| dropout_flagged | 412 | 412 | 55 | 64 | 55 | 11 |
| dropout_silent | 454 | 470 | 55 | 117 | 55 | 11 |
| delay | 466 | 466 | 62 | 70 | 62 | 8 |
| all | 499 | 480 | 67 | 79 | 67 | 10 |
| noise_real | 660 | 461 | 67 [60,75] | 67 | 68 | 18 |
| all_real | 710 | 507 | 76 [67,84] | 76 | 76 | 20 [14,30] |
| plane | 442 | 432 | 57 [50,65] | 61 | 58 | 8 |
| all_plane | 487 | 478 | 65 [57,74] | 69 | 65 | 9 |
| dropout_silent_noisy | 490 | 473 | 95 [80,112] | 108 | 96 | 42 |

## Step 4: paired tests, TEST (dev), h0.5 (A − B, mean [95 % CI], Wilcoxon p, % launches A better)
- ekf19b_fit vs ekf19b: +0.0 … +1.1 mm, p ≥ 0.3 in every condition → **no difference**.
- oracle vs ekf19b: −45 … −58 mm, p ~1e-12, 95–100 % → the gap is large and certain.
- gate (hybrid vs none): hurts: +8 to +11 mm (noise p 3e-3, all p 7e-4), **+62 mm in dropout_silent** (p 3e-8); ~0 for
  noise_real/all_real.
- **B1 plane vs depth: −9.8 mm [−11.7, −7.6] vs noise_real (p 4e-11, 95 % of launches); −10.4 mm all_plane vs
  all_real.** The plane measurement is clearly better: the reviewer objection was right.
- **B4: dropout_silent_noisy vs noise: +36.5 mm [+22.9, +52.7]** (p 5e-11). Undetectable stale frames cost a lot; the
  oracle suffers too (+34 mm).
- Camera noise cost: noise vs ideal +4.5 mm; noise_real vs noise +8.2 mm.

## Step 5: μ vs E on train + val (425 launches), launch-mean h0.5 [CI] / median, paired vs nominal
- ideal: nominal 50 [47,53] / 49; true μ only 29 / 18 (−21 mm, 87 % better); true E only 40 / 38 (−11, 60 %);
  both 12 / 8.
- all_real: nominal 74; true μ only 45 (−29, 88 %); true E only 63 (−11, 56 %); both 28.
- → **"μ first" confirmed without test data** (A2 clean).

## Step 6: mismatch physics check
- 12/12 closed-form checks PASS; 0 energy-gaining impacts, 0 steps inside the arm or beyond the walls.
- matched vs mismatch: bounces per launch 2.0 vs 1.6; arm-hit launches 17.2 % vs 14.0 %; median moving time 2.3 vs 2.6 s;
  still moving at 10 s 0.8 % vs 0.2 % → similar difficulty (the slide phase and re-sliding after bounces offset the lower
  rolling resistance).

## Step 7: render: skipped (user request); rerun instructions in RESUME.md.
