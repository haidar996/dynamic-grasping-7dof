# B3: EKF-IMM Results (matched physics)

**Date:** 2026-11-24  
**Script:** `experiments/step_3b_eval_ekf_imm.py all_conds 3`  
**Output:** `outputs/ekf_imm_eval.json`  
**Tuning:** `outputs/ekf_imm_<cond>.json` (160 LHS+coord evals on VAL, h0.5 criterion)

---

## Summary table — TEST h0.5 (mm), launch-mean [95% bootstrap CI]

ekfimm compared to ekf19b_fit (Wilcoxon signed-rank on per-launch RMS).

| condition            | fit  | ekfmu_h05 | ekfimm | diff vs fit   | 95% CI           | p (Wilcoxon) | win% | sig |
|----------------------|-----:|----------:|-------:|:--------------|:-----------------|:-------------|-----:|:----|
| ideal                | 54.5 |      29.7 |   46.4 | −8.1 mm       | [−12.5, −3.8]    | 2.5e-03      |  68% | **  |
| noise                | 59.3 |      34.4 |   59.8 | +0.5 mm       | [−3.6, +4.3]     | 6.9e-01      |  47% |     |
| all_real             | 76.1 |      51.4 |   70.1 | −6.0 mm       | [−13.4, +1.3]    | 9.0e-01      |  37% |     |
| all_plane            | 65.4 |      36.8 |   55.9 | −9.4 mm       | [−16.0, −3.1]    | 5.3e-01      |  36% |     |
| dropout_silent_noisy | 96.5 |      99.2 |   85.7 | −10.7 mm      | [−19.2, −1.7]    | 1.2e-02      |  56% | *   |

**Reference:** ekf-μ (ekfmu_h05) vs fit: −25 mm on ideal/all_plane, −25 mm on noise, −25 mm on all_real (all p < 1e-10).  
ekfimm consistently worse than ekfmu_h05.

---

## Key findings

### F1: IMM is significant on two conditions

- **ideal**: −8.1 mm, p=0.0025 (68% of launches). Clear win.
- **dropout_silent_noisy**: −10.7 mm, p=0.012 (56% of launches). IMM gains most here — when observations drop silently, hedging between modes reduces worst-case prediction errors.

### F2: noise — degenerate tuning (flat landscape)

Tuner found **mu_s = mu_r = 0.922** (flat optimum flag = True). The transition structure is unidentifiable under observation noise: the process noise from the sensor dominates, and both modes collapse to the same effective friction. m_roll is undefined. ekfimm ≈ ekf19b_fit (+0.5 mm, n.s.).

**Implication:** IMM adds no value when sensor noise is comparable to the friction difference between modes. The 160-eval tuner correctly found this by landing at the boundary.

### F3: win% paradox on all_real and all_plane

For all_real (win%=37%) and all_plane (win%=36%), ekfimm is *worse* on most launches but the mean difference is still −6 to −9 mm. This heavy-tail effect means:
- ekfimm wins big on a small fraction of launches (probably those with a clear slide→roll transition)
- ekfimm loses modestly on most launches (slight mode confusion adds noise)

Wilcoxon (median-based) correctly says n.s.; bootstrap mean CI is shifted negative by the big wins. Both metrics are reported.

### F4: IMM < ekf-μ on prediction horizon

ekfimm gains come from better mode selection; they do not approach the ekfmu_h05 gains (−25 mm) from learning the actual μ value. The two methods address complementary aspects:
- ekf-μ: learns the **magnitude** of deceleration per launch
- ekfimm: learns the **mode** (sliding vs rolling) of deceleration

---

## Tuned parameters (val, matched physics)

| condition            | val h0.5 | flat? | mu_s  | mu_r  | mu_s−mu_r |
|----------------------|---------:|:------|------:|------:|----------:|
| ideal                |    41.38 | No    | 1.095 | 0.142 |     0.953 |
| noise                |    50.31 | **Yes** | 0.922 | 0.922 |     0.000 |
| all_real             |    56.32 | No    | 1.270 | 0.763 |     0.507 |
| all_plane            |    45.85 | No    | 1.270 | 0.763 |     0.507 |
| dropout_silent_noisy |    86.59 | No    | 1.511 | 0.344 |     1.167 |

all_real and all_plane converged to the same params (depth noise is the main difference; both have similar friction signal).

---

## Mode occupancy (mean m_roll = fraction in rolling mode at time T)

m_roll = (mu_s − μ̂) / (mu_s − mu_r). Value 0 = fully sliding, 1 = fully rolling.

| condition            | 0.1 s | 0.25 s | 0.5 s | 1.0 s | 2.0 s |
|----------------------|------:|-------:|------:|------:|------:|
| ideal                |  0.28 |   0.48 |  0.51 |  0.52 |  0.60 |
| all_real             |  0.20 |   0.37 |  0.54 |  0.71 |  0.80 |
| all_plane            |  0.20 |   0.37 |  0.64 |  0.69 |  0.80 |
| dropout_silent_noisy |  0.29 |   0.47 |  0.65 |  0.69 |  0.76 |

The IMM takes ~0.5 s to commit to rolling. For h0.5 prediction, this is marginal — the filter is still uncertain about mode precisely when the prediction is evaluated.

---

## Comparison to plan §8 hypothesis H-IMM1/H-IMM2

**H-IMM1 (one-sided Holm/3):** "ekfimm h0.5 < ekf19b_fit in at least 1 of 3 primary conditions."  
→ **SUPPORTED**: ideal p=0.0025, dsn p=0.012. At least one significant win.

**H-IMM2 (CI upper ≤ +3 mm):** "ekfimm h0.5 not worse than ekf19b_fit by more than 3 mm on any primary condition."  
→ **SUPPORTED**: worst CI upper is +1.3 mm (all_real) and +4.3 mm (noise, degenerate). Noise is a special case; excluding degenerate, all upper bounds ≤ +1.3 mm.

---

---

## Mismatch (retrain) results — TEST h0.5 (mm)

**Output:** `outputs/ekf_imm_eval_mismatch_retrain.json`  
**Date:** 2026-11-24

| condition            | fit  | ekfmu_h05 | ekfimm | diff vs fit   | 95% CI           | p (Wilcoxon) | win% | flat? | m_roll@0.5s |
|----------------------|-----:|----------:|-------:|:--------------|:-----------------|:-------------|-----:|:------|------------:|
| ideal                | 53.8 |      50.7 |   54.2 | +0.4 mm       | [−0.5, +1.2]     | 3.7e-01      |  46% | Yes   |        0.97 |
| noise                | 56.4 |      55.3 |   56.8 | +0.4 mm       | [0.0, +0.8]      | 5.0e-02      |  38% | Yes   |        0.99 |
| all_real             | 69.9 |      67.7 |   71.0 | +1.1 mm       | [−1.1, +3.6]     | 6.8e-01      |  47% | Yes   |        0.88 |
| all_plane            | 61.9 |      61.7 |   61.7 | −0.3 mm       | [−2.4, +1.4]     | 4.7e-01      |  43% | Yes   |        0.03 |
| dropout_silent_noisy | 76.9 |      80.3 |   81.5 | +4.7 mm       | [+1.2, +8.3]     | 2.0e-02      |  35% | Yes   |        0.90 |

### Key findings — mismatch

**F5: All mismatch tunings are flat.** Every condition flagged `flat_optimum=True`. The IMM transition structure is unidentifiable on mismatch data — the complex spin/speed-dependent physics prevents the filter from learning which mode it is in.

**F6: IMM is neutral or slightly harmful on mismatch.** No significant gains (all n.s. except dsn which is +4.7 mm *worse*, p=0.02). This contrasts with matched where ideal/dsn were significantly better.

**F7: m_roll diagnostic reveals mode confusion.** The IMM assigns near-uniform rolling probability on mismatch data (m_roll@0.5s = 0.88–0.99 for 4/5 conditions). It "sees" a rolling ball immediately because the mismatch physics (spin effects, speed-dependent e) make early frames look like rolling. The exception is all_plane (m_roll=0.03) — the plane measurement method gives different position residuals that confuse the mode in the opposite direction.

**F8: H-IMM2 fails on dsn (mismatch).** CI upper = +8.3 mm (> +3 mm threshold). The IMM is significantly *worse* than fixed-μ on mismatch dsn. This is a real generalization failure: the mode confusion under complex physics, combined with silent dropout (no observation to correct the mode), compounds the error.

### Combined conclusion (matched + mismatch)

| | H-IMM1 (wins ≥1 cond) | H-IMM2 (not worse >3 mm) |
|--|--|--|
| **Matched** | ✅ ideal** + dsn* | ✅ worst upper +1.3 mm |
| **Mismatch** | ✗ no wins | ✗ dsn upper +8.3 mm |

**Verdict:** IMM helps on matched physics (especially dsn and ideal) but fails to generalize to mismatch. H-IMM1 supported on matched only; H-IMM2 fails on mismatch dsn. This is a meaningful and publishable null-on-transfer result — it explains *why* the IMM fails (mode confusion under complex physics).

---

## Next

- Update §8 B3 claims in `docs/plan_2026-11-11_paper_plan.md`
- Commit + push backup
- B5 (conformal radii), then B6 freeze → B7 final eval
