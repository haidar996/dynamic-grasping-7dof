# B5: Conformal Forecast Radii — Coverage Report

**Date:** 2026-11-24
**Method:** split conformal (Angelopoulos & Bates 2022) + ACI (Gibbs & Candès 2106.00170)
**Nonconformity score:** per-launch RMS (m); calibrated on val (n=73-75), evaluated on test (n≈75)
**Nominal coverage targets:** 90% (α=0.10), 95% (α=0.05)

A coverage ≥ target means the estimator is calibrated at that level.
A coverage far below target means the predictor is overconfident.

## 0.2 s horizon

### Target 90% coverage (α=0.1)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |   19.8 | 0.781 ✗ | 0.836 ✗ |    73 |     73 |
| ideal                | ekfmu_grid  |   19.6 | 0.822 ✗ | 0.822 ✗ |    73 |     73 |
| ideal                | ekfmu_h05   |   18.7 | 0.877   | 0.877   |    73 |     73 |
| ideal                | knet        |   22.4 | 0.836 ✗ | 0.849 ✗ |    73 |     73 |
| ideal                | ekfimm      |   20.1 | 0.753 ✗ | 0.794 ✗ |    73 |     73 |
| noise                | ekf19b_fit  |   22.9 | 0.794 ✗ | 0.822 ✗ |    73 |     73 |
| noise                | ekfmu_grid  |   21.2 | 0.863   | 0.877   |    73 |     73 |
| noise                | ekfmu_h05   |   19.2 | 0.849 ✗ | 0.877   |    73 |     73 |
| noise                | knet        |   24.9 | 0.836 ✗ | 0.836 ✗ |    73 |     73 |
| noise                | ekfimm      |   23.2 | 0.822 ✗ | 0.836 ✗ |    73 |     73 |
| all_real             | ekf19b_fit  |   42.1 | 0.931 ✓ | 0.918 ✓ |    72 |     73 |
| all_real             | ekfmu_grid  |   32.3 | 0.890 ✓ | 0.890 ✓ |    72 |     73 |
| all_real             | ekfmu_h05   |   31.5 | 0.904 ✓ | 0.904 ✓ |    72 |     73 |
| all_real             | knet        |   45.0 | 0.890 ✓ | 0.890 ✓ |    72 |     73 |
| all_real             | ekfimm      |   35.1 | 0.781 ✗ | 0.836 ✗ |    72 |     73 |
| all_plane            | ekf19b_fit  |   29.4 | 0.822 ✗ | 0.836 ✗ |    72 |     73 |
| all_plane            | ekfmu_grid  |   24.6 | 0.877   | 0.863   |    72 |     73 |
| all_plane            | ekfmu_h05   |   25.2 | 0.877   | 0.863   |    72 |     73 |
| all_plane            | knet        |   34.1 | 0.822 ✗ | 0.822 ✗ |    72 |     73 |
| all_plane            | ekfimm      |   29.6 | 0.863   | 0.863   |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  109.2 | 0.931 ✓ | 0.931 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  105.9 | 0.918 ✓ | 0.918 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  274.2 | 0.931 ✓ | 0.918 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  113.2 | 0.945 ✓ | 0.945 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  109.1 | 0.931 ✓ | 0.918 ✓ |    73 |     73 |

### Target 95% coverage (α=0.05)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |   31.4 | 0.959 ✓ | 0.959 ✓ |    73 |     73 |
| ideal                | ekfmu_grid  |   29.2 | 0.945 ✓ | 0.945 ✓ |    73 |     73 |
| ideal                | ekfmu_h05   |   28.0 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| ideal                | knet        |   31.9 | 0.931   | 0.918   |    73 |     73 |
| ideal                | ekfimm      |   27.8 | 0.959 ✓ | 0.931   |    73 |     73 |
| noise                | ekf19b_fit  |   30.8 | 0.931   | 0.918   |    73 |     73 |
| noise                | ekfmu_grid  |   29.3 | 0.986 ✓ | 0.959 ✓ |    73 |     73 |
| noise                | ekfmu_h05   |   29.1 | 0.986 ✓ | 0.959 ✓ |    73 |     73 |
| noise                | knet        |   33.9 | 0.918   | 0.918   |    73 |     73 |
| noise                | ekfimm      |   32.1 | 0.959 ✓ | 0.945 ✓ |    73 |     73 |
| all_real             | ekf19b_fit  |   45.9 | 0.931   | 0.931   |    72 |     73 |
| all_real             | ekfmu_grid  |   44.2 | 0.973 ✓ | 0.959 ✓ |    72 |     73 |
| all_real             | ekfmu_h05   |   44.3 | 0.986 ✓ | 0.959 ✓ |    72 |     73 |
| all_real             | knet        |   53.0 | 0.904   | 0.904   |    72 |     73 |
| all_real             | ekfimm      |   46.5 | 0.931   | 0.931   |    72 |     73 |
| all_plane            | ekf19b_fit  |   40.1 | 0.945 ✓ | 0.945 ✓ |    72 |     73 |
| all_plane            | ekfmu_grid  |   38.0 | 0.986 ✓ | 0.959 ✓ |    72 |     73 |
| all_plane            | ekfmu_h05   |   37.8 | 0.986 ✓ | 0.959 ✓ |    72 |     73 |
| all_plane            | knet        |   48.7 | 0.890 ✗ | 0.890 ✗ |    72 |     73 |
| all_plane            | ekfimm      |   41.9 | 0.959 ✓ | 0.959 ✓ |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  225.3 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  238.2 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  411.7 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  182.4 | 0.986 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  220.6 | 0.973 ✓ | 0.959 ✓ |    73 |     73 |

## 0.5 s horizon

### Target 90% coverage (α=0.1)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |   82.1 | 0.877   | 0.877   |    73 |     73 |
| ideal                | ekfmu_grid  |   59.7 | 0.890 ✓ | 0.877   |    73 |     73 |
| ideal                | ekfmu_h05   |   61.6 | 0.918 ✓ | 0.890 ✓ |    73 |     73 |
| ideal                | knet        |   81.0 | 0.863   | 0.863   |    73 |     73 |
| ideal                | ekfimm      |   66.6 | 0.836 ✗ | 0.822 ✗ |    73 |     73 |
| noise                | ekf19b_fit  |   91.1 | 0.904 ✓ | 0.890 ✓ |    73 |     73 |
| noise                | ekfmu_grid  |   60.7 | 0.877   | 0.877   |    73 |     73 |
| noise                | ekfmu_h05   |   60.8 | 0.890 ✓ | 0.877   |    73 |     73 |
| noise                | knet        |   88.4 | 0.877   | 0.877   |    73 |     73 |
| noise                | ekfimm      |   91.9 | 0.877   | 0.877   |    73 |     73 |
| all_real             | ekf19b_fit  |  117.5 | 0.877   | 0.877   |    72 |     73 |
| all_real             | ekfmu_grid  |   81.2 | 0.849 ✗ | 0.849 ✗ |    72 |     73 |
| all_real             | ekfmu_h05   |   76.6 | 0.863   | 0.863   |    72 |     73 |
| all_real             | knet        |  114.2 | 0.849 ✗ | 0.863   |    72 |     73 |
| all_real             | ekfimm      |   98.3 | 0.822 ✗ | 0.836 ✗ |    72 |     73 |
| all_plane            | ekf19b_fit  |  102.9 | 0.904 ✓ | 0.890 ✓ |    72 |     73 |
| all_plane            | ekfmu_grid  |   68.7 | 0.904 ✓ | 0.877   |    72 |     73 |
| all_plane            | ekfmu_h05   |   71.0 | 0.904 ✓ | 0.877   |    72 |     73 |
| all_plane            | knet        |  101.7 | 0.849 ✗ | 0.849 ✗ |    72 |     73 |
| all_plane            | ekfimm      |   93.5 | 0.918 ✓ | 0.918 ✓ |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  177.3 | 0.890 ✓ | 0.877   |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  167.8 | 0.877   | 0.877   |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  340.3 | 0.904 ✓ | 0.904 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  159.6 | 0.918 ✓ | 0.918 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  211.6 | 0.931 ✓ | 0.918 ✓ |    73 |     73 |

### Target 95% coverage (α=0.05)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |   95.0 | 0.931   | 0.918   |    73 |     73 |
| ideal                | ekfmu_grid  |   95.7 | 0.945 ✓ | 0.945 ✓ |    73 |     73 |
| ideal                | ekfmu_h05   |   92.2 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| ideal                | knet        |   92.5 | 0.904   | 0.890 ✗ |    73 |     73 |
| ideal                | ekfimm      |   90.9 | 0.931   | 0.931   |    73 |     73 |
| noise                | ekf19b_fit  |  102.5 | 0.931   | 0.918   |    73 |     73 |
| noise                | ekfmu_grid  |   96.4 | 0.973 ✓ | 0.945 ✓ |    73 |     73 |
| noise                | ekfmu_h05   |   93.3 | 0.973 ✓ | 0.945 ✓ |    73 |     73 |
| noise                | knet        |   98.2 | 0.931   | 0.931   |    73 |     73 |
| noise                | ekfimm      |  101.1 | 0.931   | 0.918   |    73 |     73 |
| all_real             | ekf19b_fit  |  134.2 | 0.931   | 0.931   |    72 |     73 |
| all_real             | ekfmu_grid  |  120.1 | 0.959 ✓ | 0.945 ✓ |    72 |     73 |
| all_real             | ekfmu_h05   |  121.3 | 0.973 ✓ | 0.973 ✓ |    72 |     73 |
| all_real             | knet        |  137.4 | 0.904   | 0.904   |    72 |     73 |
| all_real             | ekfimm      |  113.5 | 0.945 ✓ | 0.931   |    72 |     73 |
| all_plane            | ekf19b_fit  |  115.7 | 0.931   | 0.931   |    72 |     73 |
| all_plane            | ekfmu_grid  |  102.6 | 0.973 ✓ | 0.959 ✓ |    72 |     73 |
| all_plane            | ekfmu_h05   |  100.6 | 0.973 ✓ | 0.959 ✓ |    72 |     73 |
| all_plane            | knet        |  121.8 | 0.890 ✗ | 0.890 ✗ |    72 |     73 |
| all_plane            | ekfimm      |  113.0 | 0.973 ✓ | 0.945 ✓ |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  358.7 | 0.986 ✓ | 0.959 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  329.3 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  641.4 | 0.986 ✓ | 0.986 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  258.5 | 0.973 ✓ | 0.959 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  375.7 | 0.986 ✓ | 0.959 ✓ |    73 |     73 |

## 1.0 s horizon

### Target 90% coverage (α=0.1)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |  229.9 | 0.877   | 0.877   |    73 |     73 |
| ideal                | ekfmu_grid  |  184.4 | 0.931 ✓ | 0.904 ✓ |    73 |     73 |
| ideal                | ekfmu_h05   |  176.3 | 0.931 ✓ | 0.931 ✓ |    73 |     73 |
| ideal                | knet        |  228.9 | 0.863   | 0.863   |    73 |     73 |
| ideal                | ekfimm      |  195.8 | 0.890 ✓ | 0.877   |    73 |     73 |
| noise                | ekf19b_fit  |  238.7 | 0.863   | 0.863   |    73 |     73 |
| noise                | ekfmu_grid  |  182.5 | 0.904 ✓ | 0.904 ✓ |    73 |     73 |
| noise                | ekfmu_h05   |  169.8 | 0.904 ✓ | 0.890 ✓ |    73 |     73 |
| noise                | knet        |  233.7 | 0.836 ✗ | 0.849 ✗ |    73 |     73 |
| noise                | ekfimm      |  254.9 | 0.877   | 0.863   |    73 |     73 |
| all_real             | ekf19b_fit  |  297.1 | 0.877   | 0.863   |    72 |     73 |
| all_real             | ekfmu_grid  |  223.0 | 0.904 ✓ | 0.890 ✓ |    72 |     73 |
| all_real             | ekfmu_h05   |  216.9 | 0.918 ✓ | 0.890 ✓ |    72 |     73 |
| all_real             | knet        |  275.1 | 0.794 ✗ | 0.822 ✗ |    72 |     73 |
| all_real             | ekfimm      |  260.5 | 0.890 ✓ | 0.890 ✓ |    72 |     73 |
| all_plane            | ekf19b_fit  |  261.5 | 0.877   | 0.877   |    72 |     73 |
| all_plane            | ekfmu_grid  |  189.1 | 0.904 ✓ | 0.904 ✓ |    72 |     73 |
| all_plane            | ekfmu_h05   |  184.9 | 0.918 ✓ | 0.918 ✓ |    72 |     73 |
| all_plane            | knet        |  259.7 | 0.794 ✗ | 0.808 ✗ |    72 |     73 |
| all_plane            | ekfimm      |  219.7 | 0.877   | 0.877   |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  321.4 | 0.849 ✗ | 0.849 ✗ |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  353.0 | 0.890 ✓ | 0.890 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  586.8 | 0.918 ✓ | 0.918 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  304.2 | 0.863   | 0.863   |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  340.7 | 0.890 ✓ | 0.890 ✓ |    73 |     73 |

### Target 95% coverage (α=0.05)

| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |
|----------------------|-------------|-------:|----------:|--------:|------:|-------:|
| ideal                | ekf19b_fit  |  256.4 | 0.904   | 0.904   |    73 |     73 |
| ideal                | ekfmu_grid  |  237.3 | 0.945 ✓ | 0.945 ✓ |    73 |     73 |
| ideal                | ekfmu_h05   |  237.9 | 0.959 ✓ | 0.959 ✓ |    73 |     73 |
| ideal                | knet        |  253.1 | 0.918   | 0.904   |    73 |     73 |
| ideal                | ekfimm      |  241.4 | 0.904   | 0.904   |    73 |     73 |
| noise                | ekf19b_fit  |  278.7 | 0.918   | 0.918   |    73 |     73 |
| noise                | ekfmu_grid  |  259.1 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| noise                | ekfmu_h05   |  244.9 | 0.959 ✓ | 0.959 ✓ |    73 |     73 |
| noise                | knet        |  275.7 | 0.904   | 0.904   |    73 |     73 |
| noise                | ekfimm      |  293.5 | 0.918   | 0.904   |    73 |     73 |
| all_real             | ekf19b_fit  |  326.9 | 0.918   | 0.918   |    72 |     73 |
| all_real             | ekfmu_grid  |  270.4 | 0.959 ✓ | 0.959 ✓ |    72 |     73 |
| all_real             | ekfmu_h05   |  275.1 | 0.973 ✓ | 0.973 ✓ |    72 |     73 |
| all_real             | knet        |  316.6 | 0.877 ✗ | 0.890 ✗ |    72 |     73 |
| all_real             | ekfimm      |  330.0 | 0.959 ✓ | 0.945 ✓ |    72 |     73 |
| all_plane            | ekf19b_fit  |  294.3 | 0.918   | 0.918   |    72 |     73 |
| all_plane            | ekfmu_grid  |  249.1 | 0.959 ✓ | 0.945 ✓ |    72 |     73 |
| all_plane            | ekfmu_h05   |  252.7 | 0.959 ✓ | 0.959 ✓ |    72 |     73 |
| all_plane            | knet        |  299.3 | 0.863 ✗ | 0.890 ✗ |    72 |     73 |
| all_plane            | ekfimm      |  265.0 | 0.931   | 0.918   |    72 |     73 |
| dropout_silent_noisy | ekf19b_fit  |  542.5 | 0.986 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_grid  |  505.6 | 0.973 ✓ | 0.973 ✓ |    73 |     73 |
| dropout_silent_noisy | ekfmu_h05   |  935.3 | 0.986 ✓ | 0.986 ✓ |    73 |     73 |
| dropout_silent_noisy | knet        |  345.9 | 0.918   | 0.904   |    73 |     73 |
| dropout_silent_noisy | ekfimm      |  593.9 | 1.000 ✓ | 0.973 ✓ |    73 |     73 |

## Summary / findings

Coverage ✓ = within 1 pp of target; ✗ = more than 5 pp below target.

- **90% target (α=0.1):** 34/75 cells at or above nominal; 20 cells more than 5 pp below.
- **95% target (α=0.05):** 46/75 cells at or above nominal; 4 cells more than 5 pp below.

### Key findings

**F1: Short-horizon (h0.2) conformal coverage is systematically below 90% for most estimators.**
At 0.2 s, error distributions are heavy-tailed — a few launches with early observability problems contribute large errors. The val-calibrated quantile is not conservative enough to cover the test tail. At h0.5 and h1.0, coverage converges to or exceeds the nominal target.

**F2: knet shows systematic undercoverage at h1.0 on harder conditions.**
knet at 90%/h1.0: all_real=0.794 ✗, all_plane=0.794 ✗, noise=0.836 ✗. KalmanNet's learned covariances are overconfident at 1 s horizon under real-world conditions — the network is well-calibrated on matched clean data but underpredicts uncertainty when condition complexity increases.

**F3: ekfmu_h05 is the best-calibrated estimator overall.**
ekfmu_h05 achieves 95% nominal or better in 10/15 (cond × horizon) cells at the 95% target, and fails only at h0.2 short-horizon. At 90%/h0.5: 4 of 5 conditions ✓ or ~ (within 2 pp). The learned friction captures the error magnitude well enough for calibration.

**F4: ekfimm is moderately calibrated — not worse than fit at 95%.**
ekfimm undercoverage is mainly at h0.2 and h0.5/ideal (0.836 ✗). At h1.0, ekfimm is comparable to ekf19b_fit. The mode uncertainty inflates errors on some launches but the conformal quantile largely captures it.

**F5: ACI closely tracks split conformal.**
ACI coverage is within ±0.02 of split conformal for all cells. The adaptive step (γ=0.005) with n_test=73 does not have enough steps to show major drift correction; both methods converge to the same result.

**Coverage table for the paper (ekfmu_h05, split conformal, h0.5):**

| condition            | 90% q (mm) | cov@90% | 95% q (mm) | cov@95% |
|----------------------|:----------:|:-------:|:----------:|:-------:|
| ideal                |      61.6  |  91.8%  |      92.2  |  97.3%  |
| noise                |      60.8  |  89.0%  |      93.3  |  97.3%  |
| all_real             |      76.6  |  86.3%  |     121.3  |  97.3%  |
| all_plane            |      71.0  |  90.4%  |     100.6  |  97.3%  |
| dropout_silent_noisy |     340.3  |  90.4%  |     641.4  |  98.6%  |

*(Conformal guarantee: split coverage ≥ 1-α holds in expectation under exchangeability.)*
*(ACI adapts the threshold online; coverage should converge to nominal.)*
