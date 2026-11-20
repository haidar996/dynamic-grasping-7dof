# Literature solutions for the ball-forecasting weaknesses (sim-only), 2026-11-16

VERIFICATION STATUS (read first). Every entry below was located through web search results
(title, authors, venue, arXiv id seen in the result). I did NOT download or pdftotext any PDF in
this pass. So: "V-meta" = existence, authors and venue seen in search results; the "what it does"
text comes from abstracts/snippets, NOT from reading the paper body. Numeric results quoted are
only those that appeared in snippets. Before citing any of these in the thesis, read the PDF
(curl + pdftotext) and re-check the claim. Items marked [UNVERIFIED] came from a snippet
with incomplete bibliographic data.

UPDATE 2026-11-16 (pdftotext verification of 16 PDFs): tags [PDF-VERIFIED] mark claims confirmed in the paper text; corrections
are in place. Summary: all arXiv ids resolve to the cited papers; fixed Achterhold authors; Tobin is visual-only randomisation
(PARTLY); Split-KalmanNet is SLAM and no calibrated-P claim (PARTLY); Zhong restitution is constant (our e(v) idea is not theirs).
Stankeviciute and Sarkka-Nummenmaa 2009 could not be fetched (not on arXiv).

## (a) Annotated bibliography

### Ball / object prediction with friction and bounces
1. Achterhold, Tobuschat, Ma, Buechler, Muehlebach, Stueckler, "Black-Box vs. Gray-Box: A Case Study on Learning Table
   Tennis Ball Trajectory Prediction with Spin and Impacts", L4DC 2023, arXiv:2305.15189 (V-meta).
   Gray-box: EKF + physical model, parameters (dynamics and EKF) learned from offline data, NN for
   initial state. Reported better than two black-box baselines; real robot returned 29/30.
   [PDF-VERIFIED] authors corrected (was wrongly "Otto"); "parameters of the dynamics model, of an EKF, and of a neural model [for initial condition] trained from offline data"; "superior prediction performance over two black-box approaches"; return rate 29/30 (97.7 %). Note: ball FLIGHT with drag/spin/impacts, not rolling.
   Maps to: our gray-box story and "learn EKF parameters offline". Closest published template
   for ekf-mu tuned by forecast loss; shows learned-params + EKF is publishable. Cost: low
   (we already do it). Sim-only: yes (they validate on real, we cannot; justify by controlled
   mismatch families, see 14-17).
2. Ajay, Wu, Fazeli, Bauza, Kaelbling, Tenenbaum, Rodriguez, "Augmenting Physical Simulators with
   Stochastic Neural Networks: Case Study of Planar Pushing and Bouncing", IROS 2018,
   arXiv:1808.03246 (V-meta). Analytical simulator + NN learning the residual; includes bouncing
   ball; hybrid beat pure-physics and pure-NN (per search snippet). [PDF-VERIFIED] "Our model outperforms both purely analytical and purely learned simulators" (benchmarks: toy bouncing ball + MIT Push; the bouncing ball is a toy).
   Maps to: unmodelled slide->roll, speed-dependent restitution. Residual on the EKF predict step
   (x+ = f_phys(x) + g_theta(x)) is the standard fix. Cost: med. Sim-only: yes.
3. Zhong, Dey, Chakraborty, "Extending Lagrangian and Hamiltonian Neural Networks with
   Differentiable Contact Models", NeurIPS 2021, arXiv:2102.06794 (V-meta). Learns restitution
   and friction coefficients inside a differentiable contact model.
   [PDF-VERIFIED title/authors/NeurIPS-era; "learning coefficients of restitution and friction"; restitution is a constant e_P per contact, NOT speed-dependent: the e(v) idea is ours, not theirs.]
   Maps to: identifiable restitution(speed) and friction parameters. Cost: high (full model),
   but the idea (learn e(v) as a small parametric function) is low. Sim-only: yes.
4. Lee, Oh, Moon, "Catching Robot: Predicting the Trajectory of a Rolling Ball Using Transformer",
   IEEE Access 2024 (V-meta from DOAJ snippet; authors as seen). Transformer predicts rolling-ball
   trajectory without Kalman filter. Maps to: black-box forecaster baseline for the comparison
   table. Cost: med. Sim-only: yes.
5. Springer chapter DOI 10.1007/978-3-030-27544-0_32 (authors not captured) [UNVERIFIED]:
   compares DES, AR, quadratic prediction, RLS for a soccer ball rolling on ground with LuGre
   friction in simulation (Simulink). Maps to: precedent that sim-only rolling-ball friction
   prediction studies are published; baseline list (RLS). Cost: low.
6. "Catch the Ball: Accurate High-Speed Motions", arXiv:2003.07489 (2020) (V-meta): Kalman filter
   with aerodynamic drag in predict step, 85% real catch. Weak relevance (flying ball); cite as
   gray-box EKF precedent only.

### Robust / adaptive / multiple-model filtering
7. Sarkka and Nummenmaa, "Recursive Noise Adaptive Kalman Filtering by Variational Bayesian
   Approximations", IEEE TAC 2009 (V-meta); Sarkka and Hartikainen, "Variational Bayesian
   Adaptation of Noise Covariances in Non-Linear Kalman Filtering", arXiv:1302.0681 (V-meta).
   VB-AKF estimates R (and process noise in extensions) online.
   [NOT opened: Sarkka papers not on arXiv, UNAVAILABLE; arXiv:1302.0681 not downloaded.]
   Maps to: covariance inconsistency (NEES >> dim): adapt Q online so P matches error. Cost: med.
   Sim-only: yes. Expected to fix calibration, NOT bias from wrong physics.
8. Sage-Husa / innovation-based adaptive estimation (Mohamed and Schwarz 1999; Yang and Xu,
   J. Navigation 2003) (V-meta via secondary snippet; details unverified). Windowed or
   fading-memory estimate of Q/R from innovations. Maps to NEES inflation. Cost: low
   (10-20 lines). Known flaw: indefinite covariances, needs clipping.
9. Murphy, "Switching Kalman Filters", 1998 tech report (V-meta); Brady, Luo, Wang, Elvira, Li, "Differentiable
   Interacting Multiple Model Particle Filtering", arXiv:2410.00620 (PDF-VERIFIED: SMC parameter learning for jump models; "DIMMPF is the best performing algorithm in all experiments"; it is a particle filter, not an EKF-IMM); "Filtering Jump Markov
   Systems with Partially Known Dynamics: A Model-Based Deep Learning Approach",
   arXiv:2511.09569 (PDF-VERIFIED: Stamatelis and Alexandropoulos, JMFNet; "outperforms classical model-based filters (e.g. IMM and particle filters) as well as model-free baselines"; unknown noise statistics, not a rolling ball); parameter tuning of IMM without ground truth,
   arXiv:2307.06618 (PDF-VERIFIED: Brandenburger, Hoffmann, Charlish; IMM params optimised "solely using measurements, thus without ground-truth"). Maps to: slide->roll regime switch. IMM with 2 modes (sliding
   deceleration mu_s, rolling deceleration mu_r) plus mixing. Cost: med. Sim-only: yes.
   Caveat: Mehrfard et al. arXiv:2411.16930 report KalmanNet LOSES to an IMM on radar data
   (PDF-VERIFIED: "KalmanNet is outperformed by the IMM filter"; Mehrfard, Duraisamy, Haag, Geiss, Mercedes-Benz, automotive RadarScenes, not ball tracking), supporting an IMM baseline in our table.
10. Li, Shi, Yu, Shi, "Variational Robust Kalman Filters: A Unified Framework", arXiv:2512.15419
    (PDF-VERIFIED; "can recover conventional KF, robust KF, and adaptive KF by tuning parameters"): claims one framework recovering KF, robust KF, adaptive KF. Maps to
    combining adaptive Q with outlier robustness. Cost: med.

### Learned filters
11. Revach, Shlezinger, Ni, Escoriza, van Sloun, Eldar, "KalmanNet", IEEE TSP 70, 2022,
    arXiv:2107.10043 (V-meta). Our baseline. Snippet claims gains under model mismatch vs EKF.
12. Choi, Park, Shlezinger, Eldar, Lee, "Split-KalmanNet: A Robust Model-Based Deep Learning Approach for SLAM", arXiv:2210.09636 (V-meta). Two RNNs
    learn prior and innovation covariances separately; [PDF-VERIFIED] "two RNNs independently learn the covariance matrices for a prior state estimate and the innovation"; compensates "state and measurement model mismatch independently". PARTLY: validated on SLAM; it learns matrices used for the GAIN, it is not shown to give calibrated posterior P (no NEES claim found), so "yields an explicit P" is our inference. Maps to: our "noisier velocity" and non-consistent covariance, because it yields an
    explicit P. Cost: med (our knet code exists). Sim-only: yes.
13. Dahan, Revach, Dunik, Shlezinger, "Bayesian KalmanNet: Quantifying Uncertainty in Deep
    Learning Augmented Kalman Filter", arXiv:2309.03058 (V-meta); and KalmanNet uncertainty from
    internal features, arXiv:2110.04738 (V-meta). Ensemble/MC-dropout covariance;
    [PDF-VERIFIED mechanism: J KalmanNet realisations (MC-dropout) averaged "to recover both the state and its predicted error"; the phrase "varying knowledge accuracy" was not found by text search.] Maps to NEES. Cost: med
    (ensemble of N knets, cheap at our scale).
14. Revach, Shlezinger, Locher, Ni, van Sloun, Eldar, "Unsupervised Learned Kalman Filtering", arXiv:2110.09005
    (PDF-VERIFIED; "adapts in an unsupervised online manner to dynamics that differ from the offline trained model without ground-truth"; EUSIPCO venue not checked). Trains without ground-truth states; online adaptation to changed SSM.
    Maps to: held-out physics shift: adapt per-episode on the observed 60 Hz measurements, which
    is legitimate at test time (no truth needed). Cost: med.
15. Haarnoja, Ajay, Levine, Abbeel, "Backprop KF", NeurIPS 2016, arXiv:1605.07148 (V-meta); and
    "How to Train Your Differentiable Filter", Autonomous Robots 2021 (Kloss et al., V-meta).
    Learn heteroscedastic R and train through filter; survey notes process noise rarely learned.
    Maps to: justification/guide for forecast-loss and NLL training of our EKF; suggests state-
    dependent Q(speed) rather than constant Q. Cost: low-med.

### System identification and simulation-only evaluation
16. Ramos, Possas, Fox, "BayesSim", RSS 2019, arXiv:1906.01728 (V-meta). Likelihood-free posterior
    over simulator parameters from trajectories. Maps to: a posterior over (mu, restitution,
    tilt) per episode instead of a point mu; gives calibrated forecast spread. Cost: high.
17. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the
    Real World", IROS 2017, arXiv:1703.06907 (PDF-VERIFIED title/authors; PARTLY: randomises RENDERING (textures), detector accurate to 1.5 cm; no physics randomisation. Use only as the general sim-to-real randomisation rationale; physics-parameter randomisation is a different, later line of work).
    Rationale for training over randomized physics. Maps to: training knet / ekf-mu tuning over a distribution of unmodelled-physics
    families, then testing on held-out families. Cost: low (we have the generators).
18. Held-out-physics evaluation precedents (V-meta, snippets only): ACWM-Phys, arXiv:2605.08567
    (separate ID/OOD splits of physical parameters); Robust imitation learning under dynamics
    variations, arXiv:2206.09314 (PDF-VERIFIED: Chae et al., "Robust Imitation Learning against Variations in Environment Dynamics"; tests gravity/mass perturbations 50%-150% (1-D up to 230%), trained on e.g. 50%/150%; it is imitation learning/RL, not state estimation); MetaPhysiCa,
    arXiv:2303.03181 (OOD ODE parameters); real-robot learned-dynamics dataset, ICRA 2020
    (argues iid test error is a poor transfer indicator). Maps to inverse-crime / A2 audit
    threats: report results on held-out physics families, never on the family used to tune.
19. Ozmen, Hespanha, Byl, "Learning Transferable Friction Models and LuGre Identification via PINNs",
    arXiv:2504.12441 (V-meta). Notes MuJoCo/PyBullet use simple friction models. Maps to:
    cross-simulator discrepancy argument and speed-dependent (Stribeck/LuGre) friction as the
    right richer model for slide->roll. Cost: med.
20. Differentiable contact gradients in MuJoCo/MJX, arXiv:2506.14186 (V-meta, snippet states MJX
    gradients can be wrong in contact cases) and MJX sysid, arXiv:2508.04696 (V-meta). Maps to:
    differentiable-sim sysid for mu / restitution. Caution: gradients unreliable through bounces;
    use gradient-free (CMA-ES) fitting instead. Cost: med.
21. Conformal forecasting: Stankeviciute, Alaa, van der Schaar, NeurIPS 2021 (V-meta only; not on arXiv, UNAVAILABLE for pdftotext); Gibbs and
    Candes, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021, arXiv:2106.00170 (PDF-VERIFIED: "provably achieves the desired coverage frequency over long-time intervals irrespective of the true data generating process");
    Cleaveland, Lee, Pappas, Lindemann, arXiv:2304.01075 (V-meta, multi-step trajectory
    conformal regions). Maps to: covariance inconsistency. Replace nominal P by split-conformal
    per-horizon radius calibrated on a held-out sim split: guaranteed coverage regardless of
    filter model. Cost: low. Strongly sim-only.
22. NEES/NIS consistency: Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking
    and Navigation", Wiley 2001 (standard reference; not opened); arXiv:2108.10712 "Time
    Dependence in Kalman Filter Tuning" and arXiv:1807.08855 (both V-meta): NEES/NIS used as tuning
    fitness. Maps to: report time-averaged NEES with chi-square bounds, tune Q against NEES+NLL.
    Cost: low.

(Not found: no paper specifically on IMM for sliding vs rolling ball tracking; no paper
cross-validating a ball-forecast filter across MuJoCo/PyBullet. Both are gaps we could claim.)

## (b) Weakness -> candidate solutions -> evidence -> cost

| Weakness | Candidate | Evidence (source) | Cost | Sim-only |
|---|---|---|---|---|
| Slide->roll transition unmodelled, retrain gain only ~3 mm | 2-mode IMM (sliding mu_s, rolling mu_r) | [9] jump-Markov DL, diff-IMM | med | yes |
| Same | Stateful regime flag in EKF state (speed threshold = v_slide->roll, physical rule v_c = 5/7 v_0 style) | billiards physics, snippet | low | yes |
| Speed-dependent restitution | parametric e(v) learned in forecast-loss tuning; or NN residual on bounce map | [2],[3] | low / med | yes |
| Table tilt | augment state with constant gravity-along-slope acceleration (2 extra states) | state augmentation, as ekf-mu does | low | yes |
| ekf-mu transfer 30-58 mm worse | train/tune over randomized physics families, test on held-out family | [17],[18] | low | yes |
| Same | shrink mu to prior (MAP) / per-episode unsupervised adaptation | [14],[7] | low-med | yes |
| NEES >> dim | adaptive Q (Sage-Husa, VB-AKF) | [7],[8] | low-med | yes |
| NEES >> dim | split-conformal calibration of forecast radius per horizon | [21] | low | yes |
| NEES >> dim (knet) | Split-KalmanNet / Bayesian KalmanNet | [12],[13] | med | yes |
| knet noisy velocity, no gain over EKF | physics residual on predict + keep analytic gain; or multi-model knet | [2], Han et al. Multi-Model KalmanNet (unverified) | med | yes |
| Oracle 2-3x better than any estimate | per-episode likelihood-free posterior over (mu, e, tilt) | [16] | high | yes |
| "Sim-only / inverse crime" | evaluate on a second simulator (PyBullet) or independent integrator with different contact model; held-out families; CIs by bootstrap over episodes | [18],[19] | med | yes |

## (c) Top 5 recommendations (ranked)

1. Evaluate on held-out physics families with training over randomized families (domain
   randomization) and bootstrap CIs over episodes [17][18]. Cost low. Reason: it directly
   answers the sim-only/inverse-crime and A2 reviewer attacks, and it likely changes the
   transfer verdict (30-58 mm worse) because mu tuned on a single family overfits. Do this first;
   all later results are only credible under it.
2. Split-conformal (plus ACI for drift) per-horizon forecast radii on top of any filter [21].
   Cost low. Reason: NEES >> dim is a model-structure problem; conformal gives valid coverage
   without fixing it, and is reportable as a calibration metric alongside NEES/NIS [22].
3. Two-regime IMM (sliding/rolling) as the principled fix for the slide->roll mismatch [9]. Cost
   med. Reason: it is the largest named unmodelled effect; literature shows IMM can beat
   learned filters (Mehrfard 2411.16930) and no published rolling-ball IMM was found (novelty).
4. Physics-residual or parametric-restitution term inside the EKF predict/bounce map, trained with the
   existing multi-step forecast loss [2][3][15]. Cost med (parametric e(v): low). Reason: targets
   speed-dependent restitution, which mu augmentation cannot capture, while keeping the EKF
   structure and covariance.
5. Adaptive Q (Sage-Husa first, VB-AKF if time) combined with Bayesian/Split KalmanNet as the
   learned-filter variant [7][8][12][13]. Cost low-med / med. Reason: cheap NEES improvement for
   the analytic filter; Split/Bayesian KalmanNet gives KalmanNet a defensible covariance.
   Optional cross-simulator check (PyBullet) rides on item 1 as a strong inverse-crime defence.
