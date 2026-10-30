# Paper selection (updated 2026-10-12)

## Scope (corrected twice)
Planar 2D motion only: no flying, no spin. Walls, friction, fast and accelerating balls.

Ball speed is NOT capped. The 0-0.7 m/s in `benchmark_grasping_methods.py` belongs to
chapter 1 and is not a limit for this chapter. The ball can move fast with varying
velocities and accelerations (friction, bounces, curved paths), so fast-ball, bounce and
maneuvering papers are in scope. Chapter focus: estimator comparison, learned estimators
required, papers whose methods can be implemented. Delay modelled from paper 11 (~40 ms).

| # | Paper | Role |
|---|---|---|
| 01 | Jankowski, air hockey piecewise-linear KF (2025) | mode-aware KF baseline (floating / wall / contact) |
| 02 | Adzemovic, deep Gaussian / NODE filters (2024) | learned motion model + Bayes fusion, missing data |
| 03 | Achterhold, gray-box EKF (2023) | bounce, spin, predict-only occlusion, learned Q |
| 04 | Revach, KalmanNet (2022) | learned Kalman gain |
| 05 | Payne, uncertainty-aware Salted KF (2022) | covariance through a bounce |
| 06 | Kloss, differentiable filters (2021) | learned Q/R, heteroscedastic noise |
| 07 | Bi, transformer ball predictor (2026) | sequence predictor, sim-to-real swap |
| 08 | Chen, FruitNinja (2026) | estimator covariance -> risk-aware intercept |
| 09 | Boyalakuntla, KARL (2025) | EKF under tracking loss |
| 10 | Aljalbout, reality-gap survey (2025) | sensor-model framing |
| 11 | Liu, Robot Air Hockey Challenge retrospective (2024) | tracking loss > noise; ~40 ms delay |
| 12 | Chatterjee & Govindu, depth noise (2015) | sigma_Z ~ Z^2 |
| 13 | Wu, KalmanNet++ (2025) | learned gain with regime-conditioned hypernetwork (extension of 04) |
| 15 | Xiao, differentiable factor graph (2025) | smoother/batch estimator family, learned dynamics |
| 16 | Zhao, IMMNet (2026) | learned IMM: KalmanNet per mode + transformer mode classifier (CV/CA/turn) |
| 17 | Nguyen, OIPP (2026) | learned acceleration / early-stage impact prediction |
| 18 | Singh, physics-guided 3D tracking (2025) | RGB-D fast ball, depth NaN fill, restitution-based bounce |

Replaced: Singh 2025a (image-plane KF trackers) -> superseded by 18 and 01
(in `papers/replaced/`).
Alternates added: Piga ROFT (delay sync with optical flow), Wang EV-Catcher (event camera,
13 m/s), Islam, Akinola.
Not fetched (paywalled): Xiao & Chen delayed/irregular visual feedback.

## 2D update
- Added 19: Liu et al. 2021, EKF with friction term + wall-collision model + gating/reset on
  unexpected collisions + Bayesian-optimisation system identification (air hockey, 2D).
- Moved 14 (Kreiser, event camera, 3D table tennis) to `alternates/`; keep only its idea of
  adding a velocity measurement to the filter.
- Flight/spin content of 03, 07, 15, 17 is out of scope; use them only for: learned Q/R and
  occlusion handling (03), sequence-predictor architecture (07), smoother family + yaw
  invariance (15), learned acceleration / early prediction (17).
- Alternates: Song & Boularias (planar sliding friction identification).
