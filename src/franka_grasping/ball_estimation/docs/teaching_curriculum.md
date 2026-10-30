# Teaching curriculum (chapter 2, ball estimation) - taught from zero, one small idea per message

CURRENT LESSON: F3 layer 1 given (docs/lessons/F3_L1.html: dropout-flagged condition—oracle 31mm vs 27mm noise (+14%), ekf19b 234mm vs 209mm (+12%), small degradation because filter knows to skip bad measurement (predict-only); marked difference will be dropout-silent where filter doesn't know); skipped: A3 L4-6, A4 L6, A5 L4-6, C2 L4/L6, C3 L4/L6, C4 L3/L4, C5 L3-6, B3/B4, D3 L6, D7 L2; next: F4+ remaining conditions, then summary
GOAL: mastery. Layers per lesson: 1 intuition, 2 definition + notation, 3 math/derivation + numeric check, 4 where it lives in our code/data, 5 assumptions and limits, 6 exercise. One small piece per message.

## Part A - The problem and the world
- A1 What the project is: a robot arm and a moving ball; why we need to know where the ball is and will be
- A2 What a simulator is (MuJoCo); why simulation only
- A3 The arena: floor, walls, ball, camera (sizes)
- A4 Ball physics: friction (mu), wall bounce (E), planar motion
- A5 Launches: random speed / direction / mu / E; why 500 launches; the speed grid

## Part B - The camera
- B1 Pixels, field of view, focal length
- B2 Depth image; back-projection (pixel + depth -> x, y)
- B3 Finding the ball in the image (colour mask, centroid)
- B4 Camera height and field of view (3 m / 60 deg provisional)

## Part C - Imperfections of a real sensor
- C1 Depth noise and centroid jitter
- C2 Quantisation
- C3 Frame rate, capture time, delay, arrival time
- C4 Dropout: flagged vs silent hold, the two-state Markov chain
- C5 The dataset: raw 60 Hz files, six conditions, seeds, why offline

## Part D - Estimation from zero
- D1 State, estimate, forecast
- D2 Finite difference (fd)
- D3 Kalman filter in 1D: predict, correct, gain
- D4 Uncertainty P, model noise Q, camera noise R
- D5 The friction model, the Jacobian, the EKF
- D6 Walls in the model; uncertainty at a bounce (ekf19b)
- D7 The gate, chi-square, policies none/skip/reset/hybrid
- D8 The oracle; the stale-repeat guard

## Part E - Measuring
- E1 now_pos / now_vel, forecasts h0.2/0.5/1.0, moving frames, RMS
- E2 Why RMS hides things; pooling; sampling noise

## Part F - What we found (tables, one condition at a time)
- F1 ideal, F2 noise, F3 dropout flagged, F4 dropout silent, F5 delay, F6 all together
- F7 gate diagnosis and the bounce fix, F8 sigma/q grid

## Part G - Papers and what each gave (10, 11, 12, 19, and the unread ones)
## Part H - What is still open and what comes next
