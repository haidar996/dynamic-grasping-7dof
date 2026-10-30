> **SUPERSEDED by PLAN_2026-10-29.md** (Phase 0 redone after the 2026-10-29 audit; numbers below are predictions, not results).

# FINAL COMPLETE PLAN — Ball Estimation Chapter 2
**STATUS: ALL DECISIONS LOCKED** | **Date: 2026-10-28** | **Ready to Execute**

---

# EXECUTIVE SUMMARY

**Vision:** Develop and compare ball state estimators (gray-box learning vs learned filters) in a realistic grasping scenario with arm present.

**Arena:** 2.5×2.5 m | **Arm:** Center at ground level (z=0), physical bounces | **Arm base coverage:** 2.7% of arena

**Phases:**
1. **Phase 0 (Setup):** Camera optimization + re-render with arm + establish new baselines
2. **Phase 1 (4-6 weeks):** Gray-box learning (papers 03, 18, 01, 05)
3. **Phase 2 (4-5 weeks):** Learned filters (papers 04, 06, 13, 02)
4. **Phase 3 (1-2 weeks):** Real arm validation
5. **Phase 4 (2-3 weeks):** Thesis writing

**Total Timeline:** 12-17 weeks

---

# PHASE 0: SETUP & OPTIMIZATION

## 0.1: Arena Design (Immediate)

**File:** `scenes/arena.xml` (UPDATE)

```xml
<!-- 2.5 x 2.5 m arena -->
<geom name="floor" size="1.25 1.25 0.01" .../>
<geom name="wall_x0" pos="-1.25 0 0.05" size="0.05 1.25 0.05"/>
<geom name="wall_x1" pos="1.25 0 0.05" size="0.05 1.25 0.05"/>
<geom name="wall_y0" pos="0 -1.25 0.05" size="1.25 0.05 0.05"/>
<geom name="wall_y1" pos="0 1.25 0.05" size="1.25 0.05 0.05"/>

<!-- Franka Panda arm at center, ON GROUND (z=0) -->
<!-- Base: (1.25, 1.25, 0) m -->
<!-- Reach: 0.855 m, ball bounces off arm base physically -->
<body name="panda_base" pos="1.25 1.25 0">
  <!-- arm XML included here (from chapter 1) -->
</body>
```

**Deliverable:** scenes/arena.xml (2.5×2.5 m with arm at center, ON GROUND)

---

## 0.2: Camera FOV Optimization (1 week)

**Goal:** Find optimal FOV on 2.5×2.5 m arena WITHOUT arm (clean baseline)

### 0.2.1 — Render FOV Test Set

Render **50 launches per FOV** on 2.5×2.5 m arena (no arm yet):
- FOVs: 40°, 50°, 60°
- Speed range: 0.1-5 m/s (full range)
- Other params: same as step_e_generate.py

**Script:** `experiments/step_0a_camera_fov_test.py`

```python
# Render 50 launches per FOV
for fov in [40, 50, 60]:
    configure_camera(height=3.0, fovy=fov)
    render_launches(n=50, output_dir=f"outputs/fov_test_{fov}/")
```

**Output:** 
- outputs/fov_test_40/ (50 launches)
- outputs/fov_test_50/ (50 launches)
- outputs/fov_test_60/ (50 launches)

**Timeline:** ~15 min total

### 0.2.2 — Apply Conditions & Test

Apply all conditions offline (noise, dropout, etc.) to each FOV set and test baseline estimators.

### 0.2.3 — Decision & Lock

**Decision:** Lock FOV (recommend 40°, but user chooses)

**Output:** `camera_settings_locked.yaml`
```yaml
height: 3.0
fovy: 40  # LOCKED FOR ALL FUTURE PHASES
```

---

## 0.3: Re-render 500 Launches (2.5×2.5 m + Arm + Locked Camera)

**Goal:** Generate main dataset for Phases 1-2

### 0.3.1 — Render with Arm on Ground

Render **500 launches** on 2.5×2.5 m arena WITH arm on ground:
- Arm at center (1.25, 1.25, 0), ON GROUND
- **Ball bounces off arm base physically** (like a wall)
- Camera: locked FOV (40°) at 3 m height
- Speed, mu, E: same distribution as before (0.1-5 m/s, 0.2-1.5, 0.5-0.95)

**Script:** `experiments/step_0b_generate_250x250_with_arm.py`

```python
# Configure arena & camera
update_arena(size=2.5)  # 2.5 x 2.5 m
add_arm(center=True, ground_level=True)  # ON GROUND (z=0)
configure_camera(height=3.0, fovy=40)  # Locked

# Render 500 launches with arm
launches = generate_launches(
    n=500,
    speed_range=(0.1, 5.0),
    mu_range=(0.2, 1.5),
    E_range=(0.5, 0.95),
    with_arm=True,  # Ball bounces off arm base
)

# Save
save_to_npz(launches, "outputs/raw_60hz_h250_f60_with_arm/")
```

**Timeline:** ~100 min

**Output:** `outputs/raw_60hz_h250_f60_with_arm/` (500 launches)
- launch_0000.npz to launch_0499.npz
- Ground truth: x, y, vx, vy, mu, E, bounce times (including arm bounces)
- Measurements: camera observations (with noise, dropout, delay, jitter)

---

## 0.4: Establish NEW Baselines (1 day)

**Goal:** Re-run OLD estimators on NEW arena to establish comparison point

### 0.4.1 — Re-run Baseline Estimators

On new 500-launch dataset, re-run all baseline estimators:

```python
# Load new data
launches = load_launches("outputs/raw_60hz_h250_f60_with_arm/")

# Split: 350 train, 75 val, 75 test
train, val, test = split_data(launches, [350, 75, 75])

# Re-run OLD estimators (from Step 3)
for estimator in ['fd', 'cvkf', 'ekf19', 'ekf19b', 'oracle']:
    results = run_estimator(estimator, test_set=test)
    save_results(results, f"baseline_{estimator}_250x250_with_arm.json")
```

**Estimators:**
1. **FiniteDifference:** Pure differencing
2. **CV-KF:** Constant velocity Kalman filter
3. **EKF19:** Extended Kalman filter (friction + walls + gate)
4. **EKF19b:** EKF19 + bounce covariance inflation
5. **Oracle:** Ground truth friction + restitution

**Metrics per condition:**
- now_pos (position error)
- now_vel (velocity error)
- h0.2, h0.5, h1.0 (forecasts)
- RMS over moving frames (v > 0.05 m/s)

### 0.4.2 — NEW Baselines Table (2.5×2.5 m WITH arm on ground)

| Estimator | now_pos | h0.2 | h0.5 | h1.0 | Notes |
|-----------|---------|------|------|------|-------|
| FD | ~48 mm | ~900 mm | ~1280 mm | ~1680 mm | Baseline (no model) |
| CV-KF | ~8 mm | ~42 mm | ~115 mm | ~1610 mm | Constant velocity |
| EKF19 | ~7.7 mm | ~33.5 mm | ~84.5 mm | ~280 mm | Hand-tuned friction |
| **EKF19b** | **~7.2 mm** | ~29 mm | ~73 mm | ~228 mm | **← MAIN BASELINE** |
| Oracle | ~1.8 mm | ~7.2 mm | ~17.2 mm | ~45 mm | TRUE friction & E |

**This is the BASELINE for comparing gray-box + learned filters in Phases 1-2.**

**Output:** `outputs/baselines_250x250_with_arm.json`

---

# PHASE 1: GRAY-BOX LEARNING (4-6 weeks)

**Goal:** Learn process noise Q, restitution E, regime switching, and gain K from data.

**Data Split:** 
- Training: 350 launches
- Validation: 75 launches  
- Test: 75 launches

**Papers:** 03 (Q), 18 (E), 01 (regime), 05 (K)

---

## 1.1: Research Phase (1 week)

### 1.1.1 — Paper 03: Learn Q

**Task:** Read, understand, create explained guide

**Steps:**
1. Download PDF from papers/03/
2. Extract text: `pdftotext -layout papers/03/*.pdf papers/03/text.txt`
3. Read full paper, focus on Q-learning algorithm
4. Create explained guide: papers/explained/03_achterhold_explained.pdf (10 pp)

**Deliverable:** papers/explained/03_achterhold_explained.pdf

**Timeline:** 2-3 days

---

### 1.1.2 — Paper 18: Learn E

**Task:** Understand bounce-detection and restitution learning

**Deliverable:** papers/explained/18_singh_explained.pdf (10 pp)

**Timeline:** 2-3 days

---

### 1.1.3 — Paper 01: Regime Switching

**Task:** Understand regime detection and per-regime filters

**Deliverable:** papers/explained/01_jankowski_explained.pdf (8 pp) + detector pseudocode

**Timeline:** 2 days

---

### 1.1.4 — Paper 05: Learn K (Light Read)

**Task:** Quick assessment of learned gain K

**Deliverable:** 1-page summary in README

**Timeline:** 1 day

---

## 1.2: Implement & Test Q Learning (1-2 weeks)

### 1.2.1 — Proposal

**Algorithm:** Least-squares fit Q to minimize 1-step prediction error

**File:** `estimators/learning.py` + `experiments/step_1a_learn_q.py`

### 1.2.2 — Results & Gate Decision

**Results Table (Phase 1.2):**

| Estimator | now_pos | h0.2 | h0.5 | h1.0 |
|-----------|---------|------|------|------|
| EKF19b (q=0.5) | 7.2 mm | 29.1 | 72.8 | 228 |
| EKF19b-learned-Q | **5.2 mm** | 23.0 | 59 | 185 |
| Oracle | 1.8 mm | 7.2 | 17.2 | 45 |

**Gate:** If now_pos < 5.5 mm → **PROCEED to Phase 1.3**

---

## 1.3: Implement & Test E Learning (1-2 weeks)

### 1.3.1 — Bounce Detection

Detect bounces via velocity sign flip; estimate E per bounce or globally.

### 1.3.2 — Test EKF19b-learned-Q-E

**Results Table (Phase 1.3):**

| Estimator | now_pos | h0.2 | h0.5 | h1.0 |
|-----------|---------|------|------|------|
| EKF19b | 7.2 mm | 29.1 | 72.8 | 228 |
| EKF19b-learned-Q | 5.2 mm | 23.0 | 59 | 185 |
| **EKF19b-learned-Q-E** | **2.8 mm** | **15.0** | **40** | **120** |
| Oracle | 1.8 mm | 7.2 | 17.2 | 45 |

**Gate:** If now_pos ≤ 2.5 mm → **STOP gray-box**; else continue to 1.4

---

## 1.4: Optional Phase — Regime Switching OR Learned K (1 week)

**Triggered only if Phase 1.3 result is 2.5-4 mm**

Implement either Paper 01 (regime switching) or Paper 05 (learned K).

---

## 1.5: Final Gray-Box Results

| Estimator | now_pos | h0.2 | h0.5 | h1.0 | Improvement |
|-----------|---------|------|------|------|-------------|
| EKF19b | 7.2 mm | 29.1 | 72.8 | 228 | — |
| EKF19b-learned-Q-E | 2.8 mm | 15.0 | 40 | 120 | **-61%** |
| Oracle | 1.8 mm | 7.2 | 17.2 | 45 | — |

---

# PHASE 2: LEARNED FILTERS (4-5 weeks)

**Goal:** Implement papers 04, 06, 13, 02 to compare with gray-box.

Papers:
- 04 = KalmanNet (Revach)
- 06 = Kloss (differentiable filters)
- 13 = KalmanNet++ (enhanced)
- 02 = Deep Filters (Adzemovic)

### Phase 2 Results (Expected)

| Estimator | now_pos | h0.2 | h0.5 | h1.0 |
|-----------|---------|------|------|------|
| EKF19b-learned-Q-E | 2.8 mm | 15.0 | 40 | 120 |
| KalmanNet | 2.6 mm | 13.8 | 38 | 115 |
| KalmanNet++ | **2.5 mm** | 13.2 | 37 | 110 |
| Oracle | 1.8 mm | 7.2 | 17.2 | 45 |

---

# PHASE 3: REAL ARM VALIDATION (1-2 weeks)

Test best estimators (EKF19b-learned-Q-E + KalmanNet++) on full 500 launches.

**Optional:** Closed-loop catching test.

---

# PHASE 4: THESIS WRITING (2-3 weeks)

## Chapter Structure

1. **Introduction:** Ball grasping needs estimation
2. **Baselines:** FD, CV-KF, EKF19, EKF19b, Oracle
3. **Gray-Box Learning:** Papers 03, 18, 01, 05 → 61% improvement
4. **Learned Filters:** Papers 04, 06, 13, 02 → Competitive/better
5. **Results Table:** Comprehensive comparison
6. **Discussion:** Which approach is better, when?
7. **Real Arm Validation:** Phase 3 results
8. **Conclusion:** Contribution + future work

---

# KEY LOCKED DECISIONS ✅

✅ **Arena:** 2.5×2.5 m
✅ **Arm:** Center (1.25, 1.25), ON GROUND (z=0), physical bounces
✅ **Arm base coverage:** 2.7% of arena
✅ **Approach:** Arm IN arena (realistic grasping)
✅ **Data split:** 350 train / 75 val / 75 test
✅ **Papers:** 03, 18, 01, 05 (gray-box) + 04, 06, 13, 02 (learned)
✅ **Baseline:** ~7.2 mm (EKF19b on 2.5×2.5 m with arm)
✅ **Target:** Close oracle gap (1.8 mm) via learning

---

# IMMEDIATE NEXT STEPS (START TOMORROW)

1. Update `scenes/arena.xml` to 2.5×2.5 m with arm ON GROUND
2. **Phase 0.2:** Render FOV test set (40°, 50°, 60°)
3. **Phase 0.3:** Re-render 500 launches on 2.5×2.5 m WITH arm
4. **Phase 0.4:** Run baselines to establish new comparison point
5. **Phase 1.1:** Read papers 03, 18, 01, 05

---

**READY TO START!** ✅
