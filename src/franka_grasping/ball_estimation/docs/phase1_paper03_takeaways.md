# Phase 1.1 — Paper 03 (Achterhold et al., L4DC 2023): what we take

Source read: `papers/03_achterhold_graybox_ekf_2023.pdf` via `pdftotext` (2026-10-31), incl. supplementary.
Full detailed guide (2026-10-31, 21 pages, every section + supplement + derivations): `papers/explained/03_achterhold_explained_full.pdf`.
Older short guide: `papers/explained/03_achterhold_explained.pdf` (accurate, except it says bounce linearisation is not discussed — supplement S.4 does).

## The paper in short
- Table-tennis ball, 3D flight. EKF with a physics model: drag + Magnus + gravity (Euler step at 180 Hz),
  table impact = learned linear map C (6×6) on (v, ω), impact time inside a step solved analytically (Eq. 5).
- State z = (p, v, ω, a_d, a_m) ∈ R¹¹. **Physics constants are put in the state** (k_d = a_d² + ε, k_m = a_m² + ε),
  constant in the dynamics, so the filter can adjust them per trajectory.
- **Learned from data (Ψ):** C, process noise σ_q (11, diagonal), measurement noise σ_r (3), initial covariances
  σ_p, σ_v, σ_ω, σ_ad, σ_am, initial means of a_d, a_m, and a small NN: launcher settings → initial spin.
  All variances through softplus [x]₊ = log(1+eˣ) + 1e-6 (positive without constraints).
- **Training criterion:** maximise the marginal log-likelihood of the measurements given the filter,
  Σ_n log N(m_n; predicted mean, predicted covariance (+R)), on random chunks of N = 50 measurements, conditioned on the
  first two (they initialise p and v). Adam, lr 5e-3, batch 64. Missing measurements → several predict steps.
- Results: beats two black-box models (RSSM, TVAE) at all horizons; launch info for spin cuts error a lot; median 1 s
  error 10.3 cm on unseen launcher positions; real robot returned 29/30 balls (paper prints 97.7 %, actually 96.7 %).
- Weak points: no hand-tuned EKF baseline (only black boxes); metric = max of last 5 errors; learned C has an
  unphysical entry (spin about z amplified); bounce linearisation not discussed.

## Mapping to our project (state on 2026-10-31)
| Paper | Ours |
|---|---|
| drag k_d, Magnus k_m in the state | friction μ (and later restitution E) in the state |
| table impact, learned C | walls + arm box, restitution E (paper 18 is the main source for E) |
| spin from launcher via NN | not needed (no spin; planar ball) |
| 4 RGB cameras, 180 Hz | one depth camera, 60 Hz, 8 sensor conditions |
| 108 train trajectories | 350 train launches (`split_500.json`) |

**Why it matters for us:** ekf19b uses one fixed μ = 0.85 and E = 0.725 for every launch; the oracle (true μ, E)
is ~3× better at 0.5 s in every condition (73 vs 23 mm ideal, 99 vs 32 mm all_real). Most of that gap is the
unknown per-launch μ (forecast error ≈ ½·Δμ·t²). Paper 03's idea — constants in the state + learn the filter's
noise and initial beliefs by maximum likelihood on training data — attacks exactly this gap.

## Proposed step 1.2 (needs user decisions before code)
1. **ekf-μ:** state (x, y, vx, vy, μ); μ constant in the model (small random walk q_μ); Jacobian gets ∂v/∂μ = −h·v/|v|;
   E stays fixed (E in the state = step 1.3 with paper 18).
2. **Learn by maximum likelihood on the 350 train launches:** q_pos/q_vel, q_μ, σ_r, initial mean + variance of μ,
   initial velocity variance (~6–8 numbers). Optimiser: scipy (no PyTorch needed for so few numbers) — or PyTorch.
3. **Compare** on TEST: ekf19b+stale (tuned) vs ekf-μ (hand-tuned) vs ekf-μ (learned) vs oracle, 8 conditions,
   RMS / median / p95; also how well μ itself is recovered over time.

## Check: friction or restitution? (`experiments/step_1a_mu_vs_e_gap.py`, all 500, tuned noise, ekf19b-none+stale)
| h0.5 RMS / median (mm) | ideal | all_real |
|---|---|---|
| nominal μ & E (= ekf19b) | 73 / 47 | 99 / 69 |
| true μ only | 50 / 6 | 64 / 20 |
| true E only | 55 / 43 | 81 / 64 |
| true μ & E (= oracle) | 23 / 5 | 36 / 16 |

→ μ explains almost all of the **typical** error (median 47 → 6 mm); E matters for the **tail** (launches with
bounces: RMS 50 → 23 needs E too). Order: μ in the state first (1.2, paper 03), then E (1.3, paper 18).
