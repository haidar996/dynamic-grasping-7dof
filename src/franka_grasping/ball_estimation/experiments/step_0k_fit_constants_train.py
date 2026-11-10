#!/usr/bin/env python3
"""Step 0.8a (research audit B3, 2026-11-01): fit the fixed friction c and restitution E of ekf19b on the TRAIN launches.

Why: ekf19b uses c = 0.85, E = 0.725, the midpoints of the ranges the launches were drawn from (knowledge of the data
generator).  The learned filters of Phase 1 are fitted on the train launches, so the fair fixed-constant baseline must be
fitted on the same data with the same criterion.  Physical constants do not depend on the sensor, so they are fitted
once on the clean condition ("ideal") and the fit is repeated on "all_real" only as a check.
Criterion: launch-mean h0.5 RMS (estimators/stats.py).  sigma, q: the Phase-0 val tuning of the "ekf" family
(outputs/ekf_tuning_val_v2_phase0.json); a single coordinate-descent pass (sigma, q are re-tuned on val afterwards
with the fitted c, E in step_0i).
Output: outputs/fit_constants_train.json
PHYSICS=mismatch (physics_cfg.py, A1 arm R, 2026-11-10): mismatch train launches; sigma, q = the current matched val tuning
of the "ekf" family (outputs/ekf_tuning_val.json, the best noise available before re-tuning on mismatch val in step_0i);
wider grid (Step A: per-launch best constants on mismatch spread far beyond the generator ranges); "ideal" only (the
fit is taken from "ideal" in both physics; the all_real check is not repeated).  Output: outputs/fit_constants_train_mismatch.json
Usage: [PHYSICS=mismatch] python3 experiments/step_0k_fit_constants_train.py [workers=3]"""
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
from estimators.stats import per_launch, _boot

import physics_cfg as phys
RAW = phys.RAW
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
NOISE = json.loads((root / "outputs" / ("ekf_tuning_val.json" if phys.MISMATCH else "ekf_tuning_val_v2_phase0.json"))
                   .read_text())["best"]["ekf"]
cam = Pinhole(3.0, 49.0, 480, 480)
C = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
E = [0.625, 0.675, 0.725, 0.775, 0.825]
CONDS = ["ideal", "all_real"]
if phys.MISMATCH:
    C = [round(0.2 + 0.1 * k, 2) for k in range(15)]          # 0.2 .. 1.6
    E = [round(0.30 + 0.05 * k, 2) for k in range(14)]        # 0.30 .. 0.95
    CONDS = ["ideal"]
NAME, KEY = "ekf19b_fit-none+stale", "h0.5"


def work(args):
    i, cond = args
    rec = apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
    nz = dict(sigma=NOISE[cond]["sigma"], q=NOISE[cond]["q"])
    return cond, i, {(c, e): evaluate(make(NAME, rec, c=c, E=e, **nz, **GEO), rec)[KEY] for c, e in itertools.product(C, E)}


def main():
    quick = "--quick" in sys.argv                        # smoke test: 10 train launches, *_quick output
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(argv[0]) if argv else 3
    train = SPLIT["train"][::35] if quick else SPLIT["train"]
    jobs = [(i, c) for c in CONDS for i in train]
    print(f"fit c x E = {C} x {E} on {len(train)} train launches ({phys.PHYSICS} physics), conditions {CONDS}, criterion launch-mean {KEY} RMS",
          flush=True)
    res, t0 = {c: {} for c in CONDS}, time.time()
    with Pool(workers) as p:
        for cond, i, out in p.imap_unordered(work, jobs, chunksize=4):
            res[cond][i] = out
    best = {}
    for cond in CONDS:
        ids = sorted(res[cond])
        grid = {}
        for ce in itertools.product(C, E):
            r = per_launch([{KEY: res[cond][i][ce]} for i in ids], KEY)
            grid[ce] = float(np.nanmean(r))
        ce = min(grid, key=grid.get)
        r_best = per_launch([{KEY: res[cond][i][ce]} for i in ids], KEY); r_best = r_best[~np.isnan(r_best)]
        edge = ce[0] in (C[0], C[-1]) or ce[1] in (E[0], E[-1])
        best[cond] = dict(c=ce[0], E=ce[1], launch_mean_h05_mm=1000 * grid[ce],
                          ci_mm=[1000 * v for v in _boot(r_best, np.mean)],
                          nominal_launch_mean_h05_mm=1000 * grid.get((0.85, 0.725), float("nan")), on_grid_edge=edge)
        print(f"\n[{cond}] launch-mean h0.5 RMS (mm), c (rows) x E (cols)\n        " + "".join(f"{e:>8}" for e in E))
        for c in C:
            print(f"  {c:5.2f} " + "".join(f"{1000 * grid[(c, e)]:7.1f}" + ("*" if (c, e) == ce else " ") for e in E))
        print(f"  -> c {ce[0]}, E {ce[1]}: {1000 * grid[ce]:.1f} mm (nominal 0.85 / 0.725: {1000 * grid.get((0.85, 0.725), float('nan')):.1f})"
              + ("   ON GRID EDGE" if edge else ""))
    out = phys.out_path("fit_constants_train.json", phys.SUFFIX + ("_quick" if quick else ""))
    out.write_text(json.dumps(dict(grid_c=C, grid_E=E, criterion=f"launch-mean {KEY} RMS on train", used="ideal", physics=phys.PHYSICS,
                                   c=best["ideal"]["c"], E=best["ideal"]["E"], per_condition=best), indent=1))
    print(f"\nsaved {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
