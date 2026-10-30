#!/usr/bin/env python3
"""Step 0.4 check: does the estimators' motion model (estimators/model.py) follow the simulator?
No camera, no noise: from the TRUE state every 0.1 s, forecast 0.5 s ahead with the TRUE mu and E,
compare to the simulated path.  Done with the arm box (new) and without it (old model), split by
whether the 0.5 s window contains an arm bounce.
Usage: python3 experiments/step_0f_model_check.py"""
import sys
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena, ARM
from estimators.model import forecast

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
H, EVERY = 0.5, 0.1


def main():
    geo = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
    err = {"arm": [], "no_arm": []}
    arm_win = []
    for f in sorted(RAW.glob("launch_????.npz")):
        r = np.load(f)
        t, p, v, mu, e = r["t_phys"].astype(float), r["pos"].astype(float), r["vel"].astype(float), float(r["mu"]), float(r["e"])
        arm_t = r["bounces"][r["bounces"][:, 3] == ARM, 0]
        for t0 in np.arange(0.0, t[-1] - H, EVERY):
            i = int(np.searchsorted(t, t0))
            if np.hypot(*v[i]) < 0.05:
                continue
            j = int(np.searchsorted(t, t[i] + H))
            s = (*p[i], *v[i])
            for key, obs in (("arm", geo["obs"]), ("no_arm", None)):
                fx, fy, _, _ = forecast(s, t[j] - t[i], mu, e, geo["L"], obs)
                err[key].append(np.hypot(fx - p[j, 0], fy - p[j, 1]))
            arm_win.append(np.any((arm_t > t[i]) & (arm_t <= t[j])))
    arm_win = np.array(arm_win)
    print(f"{len(arm_win)} forecast windows of {H} s from the true state ({arm_win.sum()} contain an arm bounce)")
    print(f"{'model':8s} {'windows':22s} {'median mm':>10s} {'p95 mm':>8s} {'max mm':>8s}")
    for key in ("arm", "no_arm"):
        e_ = 1000 * np.array(err[key])
        for name, m in (("all", np.ones_like(arm_win)), ("with arm bounce", arm_win), ("no arm bounce", ~arm_win)):
            print(f"{key:8s} {name:22s} {np.median(e_[m]):10.2f} {np.percentile(e_[m], 95):8.2f} {e_[m].max():8.1f}")


if __name__ == "__main__":
    main()
