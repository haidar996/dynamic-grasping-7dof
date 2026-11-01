# A1 — "mismatch" physics dataset: design proposal (2026-11-01, waiting for user decisions)

## Goal
The current simulator (`arena_sim.py`) uses exactly the estimators' motion model: constant deceleration μ, mirror bounce
with one E, tangential speed kept. That is an "inverse crime": any method that estimates that model's constants looks
almost perfect. We need a second dataset whose physics the estimators do **not** contain. A conclusion counts only if it
holds on both datasets.

**Pairing rule:** the mismatch launches reuse the **same start positions and velocities** as the current 500 launches
(seed 42 stream). The extra physics parameters come from a separate random stream (seed 43). So the matched and mismatch
data differ only in the physics, and they are compared launch by launch.

## Physics of the mismatch simulator (standard rolling-ball model, as in billiards simulators)

Simulated at 2 ms like now. The ball stays planar: two slide joints, and rendering is unchanged (a uniform green
sphere, so rotation is invisible anyway). The spin ω is integrated in Python, like the friction is today.

1. **Slide → roll on the floor.**
   - The contact-point slip is u = v − r(ω × ẑ).
   - While sliding (|u| > 0), floor friction μ_s·g opposes u. It slows the centre and spins the ball up, and the slip
     decays at a rate of (7/2)·μ_s·g.
   - For a solid ball (I = 2/5·m·r²) launched with no spin, rolling starts at **5/7 of the launch speed**.
   - While rolling (u = 0), only rolling resistance acts, which is much weaker.
   - Constant-μ model sees: a large deceleration first, then a small one. This is the main mismatch for a launched
     ball.
2. **Rolling resistance that depends on speed:** a_roll(v) = a₀·(1 + k·v), with a₀ drawn per launch. The literature
   reports a deceleration that is about constant for spheres on deformable surfaces, with weak speed dependence, so k
   is small (0–0.1 s/m).
3. **Launch spin:** the initial slip is a fraction s ∈ [0, 1] of v₀. s = 0 means already rolling (a ball released by
   hand or a ramp); s = 1 means pure sliding (struck or launched).
4. **Wall and arm bounces:**
   - **Normal direction:** e(v_n) = e₀ − β·|v_n|, clipped to [0.3, 0.98]. Restitution falls with impact speed, which
     is the common experimental trend (though not universal).
   - Walls and the arm get **different e₀**.
   - **Tangential direction:** a Coulomb friction impulse at the contact with coefficient μ_w (about 0.14 for a ball
     on a cushion, Mathavan et al. 2010). It is capped at the value that stops the contact slip. This changes the
     tangential speed and the vertical-axis spin ω_z.
   - **After a bounce the ball slides again**, because its rolling spin now points the wrong way. It decelerates hard
     for a short time and then rolls again. The current model cannot represent this at all.
   - The bounce is resolved exactly at the crossing inside the 2 ms step (mirror, not clamp: fixes audit item C4 for
     this dataset).
5. **Floor tilt:** a constant acceleration g·sin α in a random direction, with α ∈ [0°, 0.15°]. Real floors are rarely
   level to better than about 0.1°. 0.1° gives 0.017 m/s², which is comparable to rolling resistance on a hard floor.

## Proposed parameter ranges (uniform per launch). The ones marked ⚠ need your decision

| Parameter | Range | Source / reason |
|---|---|---|
| μ_s, floor sliding friction | 0.15 – 0.40 | billiard ball on cloth ≈ 0.2; rubber on floor higher |
| a₀, rolling resistance ⚠ | 0.10 – 0.80 m/s² | c_rr ≈ 0.01–0.08: carpet 0.013–0.047 measured (PocketLab, wheelchair studies) up to turf; hard wood ≈ 0.005 would let balls roll for > 10 s |
| k, speed dependence of rolling | 0 – 0.1 s/m | weak; constant deceleration reported for deformable surfaces |
| s, launch slip ratio | 0 – 1 | covers rolled to struck launches |
| e₀ walls | 0.55 – 0.95 | same span as now |
| e₀ arm ⚠ | 0.40 – 0.90 | robot base: different material from the walls |
| β, e drop with speed | 0 – 0.04 s/m | at most −0.2 at 5 m/s |
| μ_w, wall/arm friction | 0.10 – 0.30 | Mathavan 2010: 0.14 |
| α, floor tilt ⚠ | 0° – 0.15°, random direction | typical floor flatness |

⚠ **a₀ is the biggest choice.** Realistic rolling resistance (0.1–0.8 m/s²) is lower than today's μ range (0.2–1.5 m/s²).
Balls will roll longer and bounce more, so the mismatch data is also *harder*. That mixes "model mismatch" with "harder
data".
- **Option A (recommended):** use the realistic range, and report the matched data at matching speeds so the effect can
  be separated.
- **Option B:** stretch a₀ to 0.2–1.5 so the stopping distances match today's data (less realistic: a very rough surface).

## The oracle on mismatch data ⚠
There is no "true μ, E" on mismatch data. Proposal:
- **(i) Hindsight oracle (recommended):** the EKF with the constant (c, E) that best fits *this launch's* true
  trajectory. This is the best any constant-parameter model can do, i.e. the honest upper bound for ekf-μ and ekf-μE.
- **(ii) Full-model oracle:** the EKF that knows the full mismatch physics. This is more work, and it is only needed if
  we later build a filter with slide→roll in it.

## Cost
- Code: `arena_sim.py` gets a physics flag (the current physics stays the default, so Phase 0 results are untouched).
  `launches.py` samples the extra parameters from stream 43. A checker script compares the physics against the formulas
  (5/7 rule, e(v), tilt drift).
- Rendering: 500 launches, about 90 min with 3 workers. Output in `outputs/raw_60hz_h3_f49_sq480_with_arm_mismatch/`.
  The same split (`split_500.json`) applies, because the launches are paired.
- The final test set (`final_test_plan.json`) will be rendered in both physics.

## Sources
- Sliding → rolling at 5/7·v₀ for a solid sphere (angular momentum about the contact point): standard mechanics.
- Rolling resistance on carpet ≈ 0.013–0.047, wood ≈ 0.005: PocketLab rolling-resistance labs; rugby wheelchair
  coast-down study (carpet 0.014–0.021).
- Brearley & de Mestre, "Rolling of a rigid ball on a horizontal deformable surface", ANZIAM J. 46 (2004) 249–264
  (billiard ball on carpet, lawn bowl on grass).
- Mathavan, Jackson, Parkin, "A theoretical analysis of billiard ball dynamics under cushion impacts", Proc. IMechE C 224
  (2010) 1863–1873: e ≈ 0.98, ball–cushion friction 0.14, spin changes rebound speed and angle.
- Restitution decreasing with impact speed: experimental reviews summarised in Kuninaka & Hayakawa (arXiv
  cond-mat/0011375); non-monotonic exceptions (arXiv cond-mat/0209490).
