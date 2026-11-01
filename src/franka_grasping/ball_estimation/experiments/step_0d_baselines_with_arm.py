#!/usr/bin/env python3
"""
Phase 0.6 / 0.8: baseline estimators on the 500-launch dataset (2.5×2.5 m, static arm, camera 3 m / 49° / 480×480).
Usage: python3 step_0d_baselines_with_arm.py [conditions|all_conds] [workers]
Runs on the VAL and TEST launches of split_500.json (research audit 2026-11-01: headline numbers never on ALL 500, which
mixes in the train launches; the current TEST set has been looked at in Phase 0, so it is reported as "test (dev)" and a
fresh final test set is rendered once the methods are frozen, see final_test_plan.json).
The EKFs get the walls (L) and the arm box from scenes/arena.xml via Arena.estimator_geometry(), and (sigma, q) tuned per
condition and per family on val (outputs/ekf_tuning_val.json, step_0i).  ekf19b_fit uses c, E fitted on train (step_0k).
Statistics (estimators/stats.py): per-launch RMS -> launch-mean and launch-median with 95 % bootstrap intervals;
pooled per-frame RMS kept for continuity.  All per-frame errors are saved, so tables and paired tests (step_0j) can be
recomputed without re-running.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, time, json, gzip, pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import mujoco
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
from estimators.stats import summary

# dataset of Phase 0.3; camera = camera_settings_locked.yaml (checked against every file below)
RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R = 3.0, 49.0, 480, 480
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
SPLIT = json.loads((root / "split_500.json").read_text())
TUNED = json.loads((root / "outputs" / "ekf_tuning_val.json").read_text())["best"]

ESTIMATORS = ["fd", "fd+stale", "cvkf", "ekf19-none", "ekf19-hybrid",
              "ekf19b-none", "ekf19b-none+stale", "ekf19b-hybrid", "ekf19b_fit-none+stale",
              "ekf19_oracle-none", "ekf19_oracle-hybrid", "ekf19b_oracle-none",
              "ekf19b_oracle-none+stale"]

KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]

# Camera: must match what was used to render the dataset
cam = Pinhole(CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R)


def family(name):
    if name.startswith("fd"):
        return None
    if name.startswith("cvkf"):
        return "cvkf"
    if "_oracle" in name:
        return "oracle"
    if name.startswith("ekf19b_fit"):
        return "ekf_fit"
    return "ekf"


def noise_kw(name, cond):
    """Tuned (sigma, q) of the estimator's family (+ fitted c, E for ekf_fit); fd has none."""
    fam = family(name)
    if fam is None:
        return {}
    t = TUNED[fam][cond]
    return {k: t[k] for k in ("sigma", "q", "c", "E") if k in t}


def task(args):
    """Evaluate one launch on one condition: per-frame errors of every estimator."""
    i, cond = args
    raw = np.load(RAW / f"launch_{i:04d}.npz")
    rec = apply_condition(cam, raw, CONDITIONS[cond], 1000 + i)
    return i, {name: evaluate(make(name, rec, **noise_kw(name, cond), **GEO), rec) for name in ESTIMATORS}


def table(res, idx, title):
    print(f"\n  [{title}: {len(idx)} launches]  per-launch RMS -> launch-mean [95% CI] | launch-median | pooled-frame RMS"
          f"   (mm; now_vel m/s)")
    print(f"  {'Estimator':26s} " + " ".join(f"{k:>30s}" for k in ("now_pos", "now_vel", "h0.5", "h1.0")))
    out = {}
    for name in ESTIMATORS:
        items = [res[i][name] for i in idx]
        out[name] = {k: summary(items, k) for k in KEYS}

        def cell(k):
            s, f = out[name][k], (1 if k == "now_vel" else 1000)
            if "mean" not in s:
                return "-"
            d = 2 if k == "now_vel" else 0
            return (f"{s['mean'] * f:.{d}f} [{s['mean_ci'][0] * f:.{d}f},{s['mean_ci'][1] * f:.{d}f}]"
                    f" {s['median'] * f:.{d}f} {s['pooled_rms'] * f:.{d}f}")
        print(f"  {name:26s} " + " ".join(f"{cell(k):>30s}" for k in ("now_pos", "now_vel", "h0.5", "h1.0")))
    return out


def main():
    print("Phase 0.8: baselines (audit fixes), 2.5×2.5 m arena with static arm, camera 3 m / 49° / 480×480\n")
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all_conds" else list(CONDITIONS)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    val, test = SPLIT["val"], SPLIT["test"]
    idx = sorted(val + test)
    print(f"Dataset: {RAW.name}/  ({len(val)} val + {len(test)} test (dev) launches)")
    print(f"EKF geometry: L = {GEO['L']:.3f} m, arm box (cx, cy, hx, hy) = {tuple(round(v, 3) for v in GEO['obs'])}")
    print(f"Conditions: {', '.join(conds)}")

    summary_out, errors = {}, {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            res = dict(pool.map(task, [(i, cond) for i in idx], chunksize=1))
            errors[cond] = res
            tun = " | ".join(f"{f} sigma {TUNED[f][cond]['sigma'] * 1000:g} mm q {TUNED[f][cond]['q']:g}" for f in TUNED)
            print(f"\n{'=' * 150}\nCondition: {cond} | tuned on val: {tun} | {time.time() - t0:.0f} s\n{'=' * 150}")
            summary_out[cond] = dict(noise={f: TUNED[f][cond] for f in TUNED},
                                     test=table(res, test, "TEST (dev)"), val=table(res, val, "VAL (tuning set)"))
            print(flush=True)

    out = root / "outputs" / "baselines_v3_val_test.json"
    out.write_text(json.dumps(summary_out, indent=1))
    with gzip.open(root / "outputs" / "baselines_v3_errors.pkl.gz", "wb") as f:
        pickle.dump(dict(split=dict(val=val, test=test), estimators=ESTIMATORS, errors=errors), f)
    print(f"\nSaved: {out.name}, baselines_v3_errors.pkl.gz")


if __name__ == "__main__":
    main()
