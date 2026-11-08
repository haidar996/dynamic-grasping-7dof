# Step A — baselines on mismatch physics + hindsight constants (2026-11-07)

Script `experiments/step_0m_mismatch_baselines.py`, run by `experiments/run_A_then_grid.sh` (00:04 → 01:33, 89 min;
stage 1 = 70 min, much more than the 15 min estimated: fast launches cost far more than the slow ones used for timing).
Outputs: `outputs/mismatch_hindsight.json` (per-launch grids + picks + true phys params), `mismatch_baselines.json`,
`mismatch_errors.pkl.gz`, `mismatch_paired.json`, log `outputs/mismatch_baselines.log`.

Setup: 75 val + 75 test (dev) launches of the mismatch data (same p0, v0, noise seeds as matched → paired).
All filters use the MATCHED val-tuned noise and train-fitted (c, E); nothing re-tuned on mismatch data.
"Oracle" rows = ekf19b_oracle with the per-launch **hindsight best constant pair** (grid μ 0.05–1.50 × E 0.30–1.00,
480 pairs, picked on ideal by that launch's h0.5, reused in all conditions) = upper bound for constant-parameter filters.

## h0.5 launch-mean (mm), matched → mismatch

| cond (val) | fd | cvkf | ekf19b+stale | ekf19b_fit | oracle / hindsight |
|---|---|---|---|---|---|
| ideal | 435 → 291 | 434 → 294 | 45 → 67 | 47 → 63 | 4 → 39 |
| noise | 466 → 335 | 448 → 306 | 50 → 72 | 52 → 68 | 5 → 53 |
| all_real | 742 → 627 | 542 → 367 | 68 → 91 | 71 → 87 | 11 → 60 |
| dropout_silent_noisy | 612 → 396 | 514 → 341 | 93 → 103 | 94 → 99 | 45 → 58 |

Test (dev) shows the same pattern (ideal: ekf19b 54 → 67, fit 55 → 65, oracle 7 → 38; all_real 76 → 91, 76 → 87, 23 → 54).

## Findings

1. **Model-based filters get worse, model-free get better.** ekf19b +21 mm on val ideal (paired, p 9e-6), +24 mm all_real.
   fd/cvkf −140 mm (balls slow down more gently, fewer bounces: ~21 % fewer). Inverse-crime effect is real but moderate
   for the nominal EKF (45 → 67 mm, still 4× better than cvkf).
2. **The oracle advantage was mostly an inverse crime.** Matched: true constants cut h0.5 by 41 mm vs ekf19b (4 vs 45 mm).
   Mismatch: even the best constants per launch, chosen in hindsight, leave 36–39 mm (ideal). No constant (μ, E) can
   describe rolling resistance a0(1+k|v|), tilt and spin after bounces. → Gap the learned / adaptive methods can close is
   bounded below by ~35 mm with this model structure; going lower needs a richer model (speed-dependent decel, tilt).
3. **Oracle noise is over-confident on mismatch.** The oracle family's noise (σ 30 mm, q 0.1) trusts the model; with the
   wrong model, now_pos under noise is +12.7 mm worse than ekf19b (val). With ekf-family noise the same best pair gives
   36 mm (ideal val) vs 39 mm → report the hindsight bound as ~35 mm, not the oracle row.
4. **μ-first still holds, but weaker.** Paired vs nominal (val ideal): best μ only −17.1 mm, best E only −9.9 mm,
   μ-only better than E-only by 7.2 mm (p 1.6e-3). On all_real −18.2 vs −11.4 (diff p 0.04). On test (dev): −15.1 vs −12.7,
   difference not significant (p 0.2; all_real p 0.39). Matched data had −21 vs −11. Both μ and E matter; μ ≥ E.
   Sanity: best μ correlates with the true rolling-resistance a0 (r = 0.74), best E with true e_wall (r = 0.44).
5. **Train-fitted constants help a little:** ekf19b_fit vs ekf19b −3.4 mm (val ideal, p 2e-6), −4.0 mm all_real.
6. **Gate (hybrid) still hurts under noise:** +25 mm (val noise, p 5e-9). Same as matched.

## Grid-edge notes (hindsight grid)
- oracle pair on an edge in 16/147 launches: 15 at E = 0.30 (lower edge), 1 at E = 1.0. ekf pair 10/147, μ-only 5 at
  μ = 1.5, E-only 5 at 0.30 and 7 at 1.0. Effective E below 0.30 is plausible (wall friction + spin re-roll lose energy).
  Effect on the bound is likely small (bound is a minimum; a wider grid can only lower it). Optional: widen E to 0.05–1.2.
- 29–30 ties = launches without a bounce in the 0.5 s windows (E irrelevant) → E set to nominal.

## Then: ekf-μ grid tuning on val (Part 2, run 1) — 01:33 → 01:52
`outputs/ekf_mu_grid_val.json`, log `outputs/ekf_mu_grid_val.log` (matched data, val, 54 settings per condition).
val h0.5: ideal 26.6 mm (ekf19b 45, oracle 4), noise 31.6, all_real 47.7 (ekf19b 68), dropout_silent_noisy 98.5 (93).
**ALL 11 optima on a grid edge.** Profiles (min over the other parameters):
- q_mu: flat (≤ 0.2 mm) → edge harmless.
- q at lower edge 0.003 and still falling in noise, all, noise_real, all_real, plane, all_plane (e.g. all 37.9 → 36.5).
- s_l at upper edge 1.0 and still falling in ideal, dropouts, delay, noise_real, all_real (all_real 49.1 → 47.7);
  dropout_silent_noisy at lower edge 0.2 (107.9 → 98.5).
→ Proposed: widen q down to 1e-4 (and 0), s_l to {0.05, 0.1, …, 2, 3}; ~50 min. Must be done before ekf-μ evaluation.

## Re-run with wider grids (2026-11-07, 05:55 → 08:07)
First-run outputs kept as `outputs/*_narrowgrid.*`.

**Step A** (μ 0.05–2.0 × E 0.05–1.0, 840 pairs; 480 cells reused, verified identical to fresh): all numbers move ≤ 0.8 mm.
oracle pair edges 16 → 0 (9 picks changed); ekf pair 2 edges, μ-only 4 at μ = 2.0 (line with E fixed at 0.725), E-only 2 at 0.05.
Hindsight bound (ekf noise, best μ & E) val ideal 36.0 → 35.5 mm, all_real 55.1; test 35.5 / 50.4. ekf19b unchanged.
μ-first unchanged: val μ-only vs E-only −7.7 mm (p 0.002) ideal, −7.2 (p 0.04) all_real; test −2.5 / −2.4 (n.s.).
All findings 1–6 above stand.

**ekf-μ grid** (q 1e-4…0.5 × q_mu 3 × s_l 0.05…3, 168 settings, ~5.3 min per condition), val h0.5 launch-mean:
ideal 26.6, noise 31.3, dropout_flagged 26.9, dropout_silent 26.9, delay 29.8, all 36.0, noise_real 40.0, all_real 47.7,
plane 29.8, all_plane 34.4, dropout_silent_noisy 94.7 mm (first run: changes ≤ 0.5 mm except dsn 98.5 → 94.7).
Remaining edges, all harmless:
- q_mu (5 conds): flat, ≤ 0.1 mm across 1e-6…1e-2.
- q at 1e-4 (all, noise_real): flat, ≤ 0.1 mm vs 3e-4 (= q → 0 limit).
- ideal s_l = 3: flat (1, 2, 3 all 26.6).
- dropout_silent_noisy s_l = 0.05, still falling (97.6 → 94.7): s_l → 0 means "μ fixed at μ0" = no μ estimation. Under silent
  stale frames + noise the μ state does not help (ekf19b_fit val dsn 94 mm on matched). Report as: ekf-μ ≈ fixed-μ filter in dsn.
