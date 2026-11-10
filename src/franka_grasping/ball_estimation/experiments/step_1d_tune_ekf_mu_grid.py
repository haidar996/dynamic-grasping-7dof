#!/usr/bin/env python3
"""Step 1.2d: grid-tuned ekf-mu (ablation: "mu in the state" without paper 03's learning).
Same tuning budget and criterion as the baselines (step_0i): grid on the VAL launches, launch-mean h0.5 RMS.
  fixed : sigma from the ekf_fit family (val-tuned, outputs/ekf_tuning_val.json), mu0 = c and E fitted on train (step_0k)
  grid  : q (white acceleration) x q_mu (random walk of log mu) x s_l (prior std of log mu); sigma_v0 = 3 m/s as ekf19b
q is tuned too: with the baselines' q (0.1-5) the velocity noise explains the slowing and mu is hardly observable
(step_1b diagnostic), so keeping it would make this ablation a strawman.
Output: outputs/ekf_mu_grid_val.json
PHYSICS=mismatch (physics_cfg.py, A1 arm R): mismatch val launches, sigma / c / E from the mismatch re-tune and fit
(ekf_tuning_val_mismatch.json, fit_constants_train_mismatch.json) -> outputs/ekf_mu_grid_val_mismatch.json
Usage: [PHYSICS=mismatch] python3 experiments/step_1d_tune_ekf_mu_grid.py [conds|all_conds]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, itertools, time
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators import ekf_mu_batch as eb
from estimators.stats import per_launch

import physics_cfg as phys
RAW = phys.RAW
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
TUNED = json.loads(phys.out_path("ekf_tuning_val.json").read_text())["best"]["ekf_fit"]
FIT = json.loads(phys.out_path("fit_constants_train.json").read_text())
cam = Pinhole(3.0, 49.0, 480, 480)
# 2026-11-07: widened after the first run had every optimum on an edge (q at 0.003, s_l at 0.2 / 1.0; old results in
# outputs/ekf_mu_grid_val_narrowgrid.json).  q = 2.0 dropped (worst everywhere: 42-116 mm).
Q = [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5]
Q_MU = [1e-6, 1e-4, 1e-2]
S_L = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
if phys.MISMATCH:   # 2026-11-10, before any real result: the arm-R smoke test (quick noise tuning) hit the q_mu edge 1e-2 and was
    Q_MU = Q_MU + [0.1, 1.0]   # still falling; on mismatch mu really drifts (slide -> roll).  1.0 = step_1c's upper bound.


def main():
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all_conds" else list(CONDITIONS)
    out = {}
    for cond in conds:
        t0 = time.time()
        data = eb.prepare([apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
                           for i in SPLIT["val"]])
        grid = {}
        for q, qm, sl in itertools.product(Q, Q_MU, S_L):
            p = dict(sigma=TUNED[cond]["sigma"], q=q, q_mu=qm, mu0=FIT["c"], s_l=sl, sigma_v0=3.0, E=FIT["E"], sigma_E=0.13)
            _, _, errs = eb.run(data, p, GEO["L"], GEO["obs"], horizon=0.5)
            grid[(q, qm, sl)] = 1000 * float(np.nanmean(per_launch([{"h": e} for e in errs], "h")))
        k = min(grid, key=grid.get)
        axes = (("q", Q), ("q_mu", Q_MU), ("s_l", S_L))
        edge_axes = [n for a, (n, vals) in enumerate(axes) if k[a] in (vals[0], vals[-1])]
        edge = bool(edge_axes)
        # profile along each axis (min over the other two): shows whether an edge optimum is flat or still falling
        prof = {n: [min(v for kk, v in grid.items() if kk[a] == x) for x in vals] for a, (n, vals) in enumerate(axes)}
        out[cond] = dict(params=dict(sigma=TUNED[cond]["sigma"], q=k[0], q_mu=k[1], mu0=FIT["c"], s_l=k[2], sigma_v0=3.0,
                                     E=FIT["E"], sigma_E=0.13),
                         val_h05_launch_mean_mm=grid[k], on_grid_edge=edge, edge_axes=edge_axes, profiles=prof,
                         grid={f"{a}|{b}|{c}": v for (a, b, c), v in grid.items()})
        print(f"{cond:22s} q {k[0]:<6} q_mu {k[1]:<7} s_l {k[2]:<4} -> val launch-mean h0.5 {grid[k]:6.1f} mm"
              + (f"  ON GRID EDGE {edge_axes}" if edge else "") + f"  ({time.time() - t0:.0f} s)", flush=True)
        for n, vals in axes:
            print(f"      profile {n:5s}: " + "  ".join(f"{x:g}:{v:.1f}" for x, v in zip(vals, prof[n])), flush=True)
    path = phys.out_path("ekf_mu_grid_val.json")
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **out}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
