#!/usr/bin/env python3
"""Phase 1.1 check: how much of the ekf19b-vs-oracle gap is friction mu vs restitution E (true mu only / true E only).
Research audit A2 (2026-11-01): run on TRAIN + VAL only (the first version used all 500, including the test launches,
and the "mu first" decision was taken from it), with launch-level statistics and paired tests vs the nominal filter.
Noise (sigma, q) = "ekf" family tuned on val (outputs/ekf_tuning_val.json).
Usage: python3 experiments/step_1a_mu_vs_e_gap.py [workers=3]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import EKF19, StaleGuard
from estimators.metrics import evaluate
from estimators.stats import summary, paired

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
SPLIT = json.loads((root / "split_500.json").read_text())
T = json.loads((root / "outputs" / "ekf_tuning_val.json").read_text())["best"]["ekf"]
cam = Pinhole(3.0, 49.0, 480, 480)
CONDS = ("ideal", "all_real")


def task(a):
    i, cond = a
    rec = apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
    kw = dict(sigma=T[cond]["sigma"], q=T[cond]["q"], policy="none", sigma_E=0.13, **GEO)
    V = {"nominal mu & E": (0.85, 0.725), "true mu only": (rec["mu"], 0.725),
         "true E only": (0.85, rec["e"]), "true mu & E": (rec["mu"], rec["e"])}
    return {k: evaluate(StaleGuard(EKF19(c=c, E=e, **kw)), rec) for k, (c, e) in V.items()}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ids = sorted(SPLIT["train"] + SPLIT["val"])
    out = {}
    for cond in CONDS:
        with Pool(workers) as p:
            res = p.map(task, [(i, cond) for i in ids])
        print(f"\n[{cond}] train+val, {len(ids)} launches; per-launch RMS (mm): launch-mean [95% CI], launch-median;"
              f" paired vs nominal: mean diff [95% CI], % launches better")
        for k in res[0]:
            items = [r[k] for r in res]
            row = []
            for h in ("h0.5", "h1.0"):
                s = summary(items, h)
                pr = paired(items, [r["nominal mu & E"] for r in res], h)
                row.append(f"{h} {1000 * s['mean']:5.0f} [{1000 * s['mean_ci'][0]:3.0f},{1000 * s['mean_ci'][1]:3.0f}]"
                           f" med {1000 * s['median']:4.0f}" +
                           (f" | vs nominal {1000 * pr['mean_diff']:+5.0f} [{1000 * pr['mean_diff_ci'][0]:+4.0f},"
                            f"{1000 * pr['mean_diff_ci'][1]:+4.0f}] {100 * pr['a_better_frac']:3.0f}%" if k != "nominal mu & E" else ""))
                out.setdefault(cond, {}).setdefault(k, {})[h] = dict(summary=s, vs_nominal=pr)
            print(f"  {k:16s} " + "  ||  ".join(row))
    (root / "outputs" / "mu_vs_e_gap_trainval.json").write_text(json.dumps(out, indent=1))
    print("\nsaved outputs/mu_vs_e_gap_trainval.json")


if __name__ == "__main__":
    main()
