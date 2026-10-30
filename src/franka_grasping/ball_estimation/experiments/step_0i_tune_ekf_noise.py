#!/usr/bin/env python3
"""Step 0.6a: tune the EKF measurement noise sigma and process noise q per sensor condition, on the VALIDATION
launches only (split_500.json), criterion = h0.5 RMS.  Two families, each tuned on its own (fair baselines):
  ekf : tuned with ekf19b-none+stale, used by every ekf19* estimator (incl. oracle) in that condition
  cvkf: the constant-velocity KF has no friction/bounce model, so it needs its own (larger) q
(used by step_0d_baselines_with_arm.py).
Output: outputs/ekf_tuning_val.json
Usage: python3 experiments/step_0i_tune_ekf_noise.py [workers=3]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, itertools, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
cam = Pinhole(3.0, 49.0, 480, 480)
SIG = [0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03]
Q = [0.03, 0.1, 0.5, 2.0, 5.0, 20.0]
FAMILIES = {"ekf": "ekf19b-none+stale", "cvkf": "cvkf"}
KEY = "h0.5"


def work(args):
    i, cond = args
    rec = apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
    out = {}
    for fam, name in FAMILIES.items():
        for s, q in itertools.product(SIG, Q):
            r = evaluate(make(name, rec, sigma=s, q=q, **GEO), rec)
            out[(fam, s, q)] = {k: (float(np.sum(r[k] ** 2)), len(r[k])) for k in ("now_pos", KEY)}
    return cond, out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    conds = list(CONDITIONS)
    jobs = [(i, c) for c in conds for i in SPLIT["val"]]
    print(f"tuning {FAMILIES} on {len(SPLIT['val'])} val launches x {len(conds)} conditions, "
          f"grid sigma {SIG} x q {Q}, criterion {KEY} RMS", flush=True)
    acc, t0 = {}, time.time()
    with Pool(workers) as p:
        for cond, out in p.imap_unordered(work, jobs, chunksize=4):
            for sq, d in out.items():
                for k, (ss, n) in d.items():
                    a = acc.setdefault((cond, sq, k), [0.0, 0]); a[0] += ss; a[1] += n
    rms = lambda c, sq, k: 1000 * np.sqrt(acc[(c, sq, k)][0] / acc[(c, sq, k)][1])
    best = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        for c in conds:
            grid = {(s, q): rms(c, (fam, s, q), KEY) for s, q in itertools.product(SIG, Q)}
            sq = min(grid, key=grid.get)
            edge = sq[0] in (SIG[0], SIG[-1]) or sq[1] in (Q[0], Q[-1])
            best[fam][c] = dict(sigma=sq[0], q=sq[1], h05_rms_mm=grid[sq], now_pos_rms_mm=rms(c, (fam, *sq), "now_pos"),
                                default_h05_rms_mm=grid[(0.005, 0.5)], on_grid_edge=edge)
            print(f"\n[{fam}] {c}:  sigma(mm) \\ q  " + "  ".join(f"{q:>7}" for q in Q))
            for s in SIG:
                print(f"   {1000 * s:12.0f}     " + "  ".join(f"{grid[(s, q)]:7.1f}" + ("*" if (s, q) == sq else " ") for q in Q))
            print(f"   -> sigma {1000 * sq[0]:.0f} mm, q {sq[1]}: h0.5 {grid[sq]:.1f} mm "
                  f"(default 5 mm / 0.5: {grid[(0.005, 0.5)]:.1f})" + ("   ON GRID EDGE" if edge else ""))
    out = root / "outputs" / "ekf_tuning_val.json"
    out.write_text(json.dumps(dict(families=FAMILIES, criterion=f"{KEY} RMS on val", grid_sigma=SIG, grid_q=Q,
                                   best=best), indent=1))
    print(f"\nsaved {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
