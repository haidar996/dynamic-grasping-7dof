# Step 1.2 (ekf-μ) explained step by step (2026-11-01)

Companion to `docs/phase1_step12_ekf_mu_plan.md`, written for the user before the 4 decisions.

## 1. What problem are we solving?
Today's filter (ekf19b) assumes **the same friction μ = 0.85 m/s² for every launch**. In reality, each launch has its own
μ somewhere between 0.2 and 1.5.

If the filter's μ is wrong by Δμ, its forecast of a rolling ball is wrong by about **½·Δμ·t²**. Example: true μ = 1.2,
filter assumes 0.85, so Δμ = 0.35:
- forecast 0.5 s ahead: ½ · 0.35 · 0.5² = **44 mm** wrong. That is the size of our typical error (median 47 mm).
- with the true μ, the typical error drops to 6 mm (our gap check).

So the job is: **let the filter find out the μ of each launch by itself, from the camera images.**

## 2. What the filter does now
The filter keeps 4 numbers, the state (x, y, vx, vy), plus a table P that says how unsure it is about each number and how
the uncertainties are linked. Every frame (60 per second) it does two things:
1. **Predict:** move the state forward with the physics (friction, walls, arm). Uncertainty grows.
2. **Correct:** compare the predicted position with the camera's position. The difference is called the "innovation".
   Move the state part of the way towards the measurement. Uncertainty shrinks.

How far it moves is decided by the **Kalman gain**: if the filter is unsure and the camera is precise, it trusts the
camera more, and the other way round.

## 3. Putting μ into the state: how can the filter learn μ if the camera only sees position?
The state becomes 5 numbers: (x, y, vx, vy, **log μ**).

The camera never measures μ. But μ changes where the ball goes:
- higher μ → the ball slows more → it is **behind** the predicted position;
- lower μ → it is **ahead**.

The filter tracks this link in P (the "correlation" between μ and position). It builds that link from the derivative
"if μ were a bit larger, how much would the position change?" (∂p/∂μ). When the ball keeps arriving behind the
prediction, the gain turns that into "μ must be larger" and raises μ̂. This is exactly how paper 03 learns drag and spin,
which it never measures either.

**How fast?** The effect of a μ error grows like ½·Δμ·t². For Δμ = 0.1 it is 2 mm after 0.2 s and 12.5 mm after 0.5 s. With
5 mm camera noise (and many frames to average), μ̂ becomes good after a few tenths of a second. Measuring this
convergence time is one of our results.

**When can it NOT learn μ?** When the ball is at rest (no motion, no friction effect), and right after the start.

**Why log μ and not μ?**
- μ must be positive. With log μ, any number the filter produces gives μ = e^(log μ) > 0. With plain μ, a noisy update
  could give μ = −0.1, i.e. friction that speeds the ball up.
- Uncertainty becomes relative. "log μ = log 0.85 ± 0.5" means μ is roughly between 0.52 and 1.40 (×/÷ 1.65). That covers
  our range well, and ±20 % means the same for small and large μ.
- The paper uses a² + ε instead. That also stays positive, but a and −a give the same value, and at a = 0 the
  derivative is zero (the filter "gets stuck"). log μ has neither problem.

## 4. Bounce handling (audit item B5)
At a wall the ball's position is mirrored and its normal velocity becomes −E·v.

**What our current filter does wrong:** it mirrors the state but **not the uncertainty table P**. Example: before the
bounce, P says "if the ball is 1 cm further right than I think, it is also moving faster to the right". After the bounce
the truth is reversed: being further right before means being further **left** after, by E·1 cm. The current filter keeps
the old, now wrong-signed link, so its corrections after a bounce go partly in the wrong direction.

**The fix:** at a bounce, the derivative of the new position with respect to the old one is −E (the same holds for the
velocity), so P gets multiplied by that. For our model this is exact, and it already includes the "impact time inside a
step" effect that paper 03 derives in its supplement S.4.

**Unit test:** a computer check that compares my derivative formulas with numbers obtained by nudging each input a tiny
bit and re-running the physics. It includes steps with a wall bounce and an arm bounce. If a formula is wrong, the test
fails, so this kind of bug cannot slip into the thesis.

## 5. Learning: which 7 numbers, and what each one means
| Number | Meaning | Today |
|---|---|---|
| σ | camera noise: how far a measurement typically is from the truth | tuned by grid on val |
| q | motion noise: how much the real motion differs from the model (random pushes, model errors) | tuned by grid on val |
| q_μ | how much μ may change during one launch. Matched data: μ is truly constant, so we expect ≈ 0. Realistic data (slide→roll): μ really changes, so we expect it larger. That comparison is itself an interesting result | — (new) |
| μ̄ | starting guess for μ before any frame | 0.85 (range midpoint) |
| s_ℓ | how unsure that starting guess is | — (new) |
| σ_v0 | how unsure the starting velocity is (the first frame gives only a position) | fixed 3 m/s |
| E | bounce restitution (still one fixed number in 1.2) | 0.725 (range midpoint) |

These numbers cannot simply be set by hand. Only σ follows from the camera. q and the rest describe how wrong the model
is, and that has to come from data.

## 6. The learning criterion: "likelihood", explained
At each frame the filter does not only predict *where* the next measurement will be, it also says *how sure* it is: a
spread S (in mm). The likelihood asks: **how probable was the measurement we really got, under the filter's
prediction?** In numbers, for each frame the cost is

  cost = log S + (prediction error)² / S

(summed over all frames of all 350 train launches; smaller is better).

Example with a real prediction error of 10 mm:

| Filter says its spread is | log S part | error² / S part | Total cost |
|---|---|---|---|
| 2 mm (over-confident) | −12.4 | +25.0 | **12.6** (bad) |
| 10 mm (honest) | −9.2 | +1.0 | **−8.2** (best) |
| 50 mm (too vague) | −6.0 | +0.04 | **−5.95** |

So the likelihood rewards a filter that is **accurate AND honest about its uncertainty**. An over-confident filter is
punished hard; a vague one is punished by the log S part. The learning changes the 7 numbers until the total cost is
lowest. This is paper 03's equation 7.

**Separately for each sensor condition:** the camera noise differs per condition (ideal, noisy, D435-like, …), so the best
numbers differ too. The baselines are tuned per condition as well, so the comparison is fair.

**Train set only:** learning uses the 350 train launches. All reported results come from launches the learning never saw.

## 7. Speed: why "all launches at once"
- One likelihood evaluation = 350 launches × 600 frames × ~3 sub-steps ≈ 630,000 filter steps. In plain Python that
  is ≈ 30 s.
- The optimiser needs a few hundred evaluations per condition, so ≈ 3 hours per condition, ≈ 1.5 days for 11 conditions.
  Too slow.
- **The trick:** every launch has exactly the same 600 frame times. So instead of 350 separate runs, I run **one run where
  every number is an array of 350 values** (numpy does the 350 at once in C). Dropout and bounces are handled with
  masks ("this launch has no frame now", "this launch bounces now"). Expected: ~1–2 s per evaluation, ~10 min per
  condition.
- **Equality test:** the batch version must give exactly the same numbers as the one-launch version. Otherwise speed
  would come at the price of a hidden bug.

## 8. The optimiser (L-BFGS-B)
It is like walking downhill in fog. It measures the slope of the cost with respect to each of the 7 numbers by nudging
each one a little (7 nudges + 1 = 8 evaluations), then steps downhill, and it remembers past slopes to step smarter.
After ~30–60 steps it reaches the bottom.

PyTorch could compute the slopes automatically ("autograd"). But it is not installed, it is an extra dependency, and on a
CPU it is slower per step for such a small filter. For 7 numbers, scipy is enough.

## 9. The comparisons: each one answers one question
| Comparison | Question it answers |
|---|---|
| ekf19b_fit vs ekf-μ hand-tuned | Does putting μ into the state help by itself? |
| ekf-μ hand-tuned vs ekf-μ learned | Does paper 03's learning add anything beyond the state? |
| ekf-μ learned (likelihood) vs ekf-μ learned (forecast error) | Is likelihood the right training goal, or should we train directly on what we care about (the 0.5 s forecast)? |
| ekf-μ learned vs oracle (true μ, E) | How much of the gap is closed, and how much remains for step 1.3 (E)? |

ekf19b_fit is the baseline with the constant μ and E fitted on the train set (audit B3), not the range midpoints, so any
improvement comes from the idea, not from extra tuning.

## 10. How the results are measured
- **Data:** validation launches + old test launches (development). The final test set is rendered and used once at the
  very end.
- **Both physics:** matched (sim = model) and realistic (slide→roll, spin, …). A method only "wins" if it wins on both.
- **All 11 sensor conditions:** ideal, noise, dropouts, delay, D435-like, plane measurement, …
- **Errors per launch** (position now, velocity now, forecasts 0.2 / 0.5 / 1.0 s), averaged over launches with 95 %
  confidence intervals, and **paired tests** (same launches, same noise) to say whether a difference is real or luck.
- **μ convergence:** a plot of μ̂ over time vs the true μ. How many tenths of a second until it is right? Slower for slow
  balls? Disturbed by bounces?
- **Honest uncertainty (NIS/NEES):** if the filter says "±10 mm", the real errors should be about 10 mm.
  - NEES uses the true state.
  - NIS uses the innovations: it should average to 2 for a 2-D measurement.
  - Far above means over-confident, far below means too cautious.
  A likelihood-trained filter should pass. Reviewers like this check.
- **Run time per update:** the camera gives a frame every 16.7 ms, so one update must be far faster than that for the
  real robot.

## 11. The 4 decisions
1. **How to store μ:** log μ (recommended: always positive, relative uncertainty, never stuck) / a² + ε (exactly like the
   paper) / plain μ clipped at 0 (simplest, but clipping breaks the filter's maths).
2. **Optimiser:** scipy + all-launches numpy filter (recommended: no new install, fast enough) / install PyTorch
   (automatic slopes, extra dependency, slower here).
3. **Training goal:** likelihood as the main method + forecast-error training as a comparison (recommended: tests the
   paper's choice) / likelihood only (less work, one question unanswered).
4. **E:** learned together with the other 6 numbers (recommended: one consistent fit) / kept at the value fitted on train
   in step 0k.
