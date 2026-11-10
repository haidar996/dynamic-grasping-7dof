#!/usr/bin/env python3
"""Step 0.6a / 0.8b: tune the measurement noise sigma and process noise q per sensor condition, on the VALIDATION
launches only (split_500.json).  Criterion: launch-mean h0.5 RMS (estimators/stats.py; per-frame pooled RMS was used
in Phase 0, see outputs/ekf_tuning_val_v2_phase0.json).  Each family is tuned on its own (same tuning budget):
  ekf    : ekf19b-none+stale, c/E = generator midpoints 0.85 / 0.725; used by every ekf19* / ekf19b* estimator
  ekf_fit: ekf19b_fit-none+stale, c/E fitted on train (outputs/fit_constants_train.json, step_0k)
  oracle : ekf19b_oracle-none+stale (true c/E); used by every *_oracle estimator, so the upper bound is tuned too
  cvkf   : constant-velocity KF, own grid (Phase 0 ran into the q edge in 5 of 8 conditions)
Output: outputs/ekf_tuning_val.json (+ the printed grids in the log)
PHYSICS=mismatch (physics_cfg.py, A1 arm R, 2026-11-10): mismatch val launches, ekf_fit uses the mismatch train fit
(outputs/fit_constants_train_mismatch.json); families ekf and ekf_fit only (oracle needs the true mu, E, which do not
exist on mismatch data; cvkf is not reported by step_1e); conditions from argv (default: the arm-R subset).
Output: outputs/ekf_tuning_val_mismatch.json
Usage: [PHYSICS=mismatch] python3 experiments/step_0i_tune_ekf_noise.py [workers=3] [conds, comma-separated]"""
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
from estimators.stats import per_launch

import physics_cfg as phys
RAW = phys.RAW
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
FIT = json.loads(phys.out_path("fit_constants_train.json").read_text())
cam = Pinhole(3.0, 49.0, 480, 480)
# grids widened 2026-11-02: the first run (outputs/ekf_tuning_val_narrowgrid.json) had optima on the grid edges
# (ekf sigma 2 mm / q 0.03 lowest; oracle sigma 30 mm highest and q 0.03 lowest; cvkf q 1000 highest)
SIG = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03]
Q = [0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 2.0, 5.0, 20.0]
SIG_OR = [0.001, 0.002, 0.005, 0.012, 0.03, 0.06, 0.12]
Q_OR = [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.1]      # q = 0 is the physical lower limit (perfect model), not a grid edge
SIG_CV = [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02]
Q_CV = [2.0, 5.0, 20.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
FAMILIES = {   # name, sigma grid, q grid, extra kw
    "ekf":     ("ekf19b-none+stale", SIG, Q, {}),
    "ekf_fit": ("ekf19b_fit-none+stale", SIG, Q, dict(c=FIT["c"], E=FIT["E"])),
    "oracle":  ("ekf19b_oracle-none+stale", SIG_OR, Q_OR, {}),
    "cvkf":    ("cvkf", SIG_CV, Q_CV, {}),
}
if phys.MISMATCH:
    FAMILIES = {f: FAMILIES[f] for f in ("ekf", "ekf_fit")}
R_SUBSET = ["ideal", "noise", "all_real", "dropout_silent_noisy"]
KEY = "h0.5"


def work(args):
    i, cond = args
    rec = apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
    out = {}
    for fam, (name, sig, qs, kw) in FAMILIES.items():
        for s, q in itertools.product(sig, qs):
            r = evaluate(make(name, rec, sigma=s, q=q, **kw, **GEO), rec)
            out[(fam, s, q)] = {k: per_launch([r], k)[0] for k in ("now_pos", KEY)}     # per-launch RMS (NaN: no frame)
    return cond, i, out


def main():
    quick = "--quick" in sys.argv                        # smoke test: 5 val launches, *_quick output
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(argv[0]) if argv else 3
    conds = (argv[1].split(",") if len(argv) > 1 else R_SUBSET if phys.MISMATCH else list(CONDITIONS))
    val = SPLIT["val"][::15] if quick else SPLIT["val"]
    assert all(c in CONDITIONS for c in conds), conds
    jobs = [(i, c) for c in conds for i in val]
    print(f"tuning {list(FAMILIES)} ({phys.PHYSICS} physics) on {len(val)} val launches x {len(conds)} conditions, "
          f"criterion launch-mean {KEY} RMS; ekf_fit uses c {FIT['c']}, E {FIT['E']}", flush=True)
    acc, t0 = {}, time.time()
    with Pool(workers) as p:
        for n, (cond, i, out) in enumerate(p.imap_unordered(work, jobs, chunksize=2), 1):
            for key, d in out.items():
                acc.setdefault((cond, key), {})[i] = d
            if n % 75 == 0:
                print(f"  {n}/{len(jobs)} jobs, {(time.time() - t0) / 60:.0f} min", flush=True)

    def score(cond, key, k):
        return 1000 * float(np.nanmean([d[k] for d in acc[(cond, key)].values()]))

    best = {fam: {} for fam in FAMILIES}
    for fam, (name, sig, qs, kw) in FAMILIES.items():
        for c in conds:
            grid = {(s, q): score(c, (fam, s, q), KEY) for s, q in itertools.product(sig, qs)}
            sq = min(grid, key=grid.get)
            edge = sq[0] in (sig[0], sig[-1]) or sq[1] == qs[-1] or (sq[1] == qs[0] and qs[0] > 0)
            best[fam][c] = dict(sigma=sq[0], q=sq[1], h05_launch_mean_mm=grid[sq],
                                now_pos_launch_mean_mm=score(c, (fam, *sq), "now_pos"), on_grid_edge=edge, **kw)
            print(f"\n[{fam}] {c}:  sigma(mm) \\ q  " + "  ".join(f"{q:>7}" for q in qs))
            for s in sig:
                print(f"   {1000 * s:12.0f}     " + "  ".join(f"{grid[(s, q)]:7.1f}" + ("*" if (s, q) == sq else " ") for q in qs))
            print(f"   -> sigma {1000 * sq[0]:.0f} mm, q {sq[1]}: launch-mean h0.5 {grid[sq]:.1f} mm"
                  + ("   ON GRID EDGE" if edge else ""), flush=True)
    out = phys.out_path("ekf_tuning_val.json", phys.SUFFIX + ("_quick" if quick else ""))
    out.write_text(json.dumps(dict(physics=phys.PHYSICS, fit_c_E=[FIT["c"], FIT["E"]], conditions=conds,
                                   families={f: v[0] for f, v in FAMILIES.items()}, criterion=f"launch-mean {KEY} RMS on val",
                                   grids={f: dict(sigma=v[1], q=v[2]) for f, v in FAMILIES.items()}, best=best), indent=1))
    print(f"\nsaved {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
