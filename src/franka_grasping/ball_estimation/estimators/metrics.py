"""Evaluate an estimator on one launch record (from dataset.apply_condition)."""
import numpy as np

HORIZONS = (0.2, 0.5, 1.0)


def evaluate(est, rec, horizons=HORIZONS, every=6, warmup=0.1, v_min=0.05):
    """Errors of the estimator at the arrival time of every `every`-th frame.
    now_pos (m), now_vel (m/s): state at the arrival time; h<h>: position error (m) h seconds later."""
    est.reset()
    tp, pp, vp = rec["t_phys"], rec["pos"], rec["vel"]
    out = {"now_pos": [], "now_vel": [], **{f"h{h}": [] for h in horizons}}
    first = None
    for k in range(len(rec["t_capture"])):
        est.update(rec["t_capture"][k], rec["xy"][k], rec["valid"][k])
        if not est.ready:
            continue
        if first is None:
            first = rec["t_capture"][k]
        ta = rec["t_arrival"][k]
        if rec["t_capture"][k] - first < warmup or k % every:
            continue
        tvx, tvy = np.interp(ta, tp, vp[:, 0]), np.interp(ta, tp, vp[:, 1])
        if np.hypot(tvx, tvy) < v_min:            # ball already at rest: nothing to estimate
            continue
        x, y, vx, vy = est.forecast(ta - est.t)
        tx, ty = np.interp(ta, tp, pp[:, 0]), np.interp(ta, tp, pp[:, 1])
        out["now_pos"].append(np.hypot(x - tx, y - ty))
        out["now_vel"].append(np.hypot(vx - tvx, vy - tvy))
        for h in horizons:
            if ta + h > tp[-1]:
                continue
            fx, fy, _, _ = est.forecast(ta + h - est.t)
            out[f"h{h}"].append(np.hypot(fx - np.interp(ta + h, tp, pp[:, 0]),
                                         fy - np.interp(ta + h, tp, pp[:, 1])))
    return {k: np.array(v) for k, v in out.items()}
