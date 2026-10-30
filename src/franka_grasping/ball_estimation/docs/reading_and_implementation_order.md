# Reading, understanding and implementation order

Scope: planar 2D ball, fast, varying velocity/acceleration, friction + wall bounces +
occlusion, depth camera, CPU, simulation. Learned estimators required.

## A. Reading + understanding order
| # | Paper | Read it for |
|---|---|---|
| 1 | 10 Aljalbout, reality gap | why simulated depth is too clean; what to add |
| 2 | 11 Liu, air hockey retrospective | tracking loss > noise; ~40 ms delay; MuJoCo benchmark |
| 3 | 12 Chatterjee, depth noise | sigma_Z grows as Z^2 |
| 4 | 19 Liu 2021, air hockey EKF | simplest complete EKF: friction + collision + gating/reset + system ID |
| 5 | 01 Jankowski | modes (floating / wall) learned from data, uncertainty rollout |
| 6 | 18 Singh 2025 | heuristic baseline to beat; depth NaN fill; restitution bounce |
| 7 | 03 Achterhold (+ explained guide) | learned Q, predict-only occlusion, bounce (ignore spin) |
| 8 | 05 Payne | covariance through a bounce (fixes the crude "skip update" of 19) |
| 9 | 08 FruitNinja | estimator covariance -> risk-aware intercept score |
| 10 | 09 KARL | EKF under tracking loss (evidence) |
| 11 | 06 Kloss | differentiable EKF, learned Q/R, heteroscedastic noise |
| 12 | 04 Revach (KalmanNet) | learned Kalman gain |
| 13 | 13 Wu (KalmanNet++) | extension of 04 for regime changes |
| 14 | 02 Adzemovic | learned motion model + Bayes fusion, missing data |
| 15 | 16 Zhao (IMMNet) | needs 04: learned IMM, regime classifier (CV / CA / bounce) |
| 16 | 15 Xiao | factor-graph smoother, yaw invariance (ignore spin) |
| 17 | 07 Bi | sequence-predictor architecture (ignore table-tennis data) |
| 18 | 17 Nguyen (OIPP) | learned acceleration, early prediction (low priority) |

## B. Implementation order (README steps)
0. Data generator in simulation (needed by every learned model): many launches with
   different speeds/accelerations/walls, ground truth saved.
1. Scene: overhead camera -> friction -> walls.
2. Sensor (10, 12, 11, 18): depth render, sigma_Z ~ Z^2, dropout/tracking loss, ~40 ms
   delay, colour detector -> 3D point, frame-rate limit for fast balls.
3. Baselines (19, 01, 11, 18): finite difference -> CV-KF -> friction EKF with collision
   (19) -> mode-aware KF with learned parameters (01). Metrics incl. error during loss.
4. Gray-box + bounce (03, 05, 16): friction/restitution in state, predict-only occlusion,
   learned Q, saltation covariance at walls, regime switching CV/CA/bounce.
5. Planner link (08, 09): wrapper scripts feed estimate via get_state; pass a predictor
   (not straight-line extrapolation); uncertainty term in the CROFT objective;
   catch-success benchmark. Do this on the baselines BEFORE step 6.
6. Learned (06 -> 04 -> 13 -> 02 -> 16 -> 15 -> 07 -> 17), all plugged into the same
   benchmark harness. Needs torch (ask before installing).
