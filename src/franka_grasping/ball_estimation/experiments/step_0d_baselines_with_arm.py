#!/usr/bin/env python3
"""
Phase 0.6: Baseline estimators on the 500-launch dataset (2.5×2.5 m, static arm, camera 3 m / 49° / 480×480)
Usage: python3 step_0d_baselines_with_arm.py [stride] [conditions|all_conds] [workers]
  stride=1 → all 500 launches (full run)
  stride=10 → 50 launches (quick test)
The EKFs get the walls (L) and the arm box from scenes/arena.xml via Arena.estimator_geometry(), and the
measurement / process noise (sigma, q) tuned per condition on the validation launches (outputs/ekf_tuning_val.json,
from step_0i_tune_ekf_noise.py).  Results are reported on all launches and on the test launches of split_500.json:
RMS, median and p95 of every metric (moving frames only).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, time, json
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

# dataset of Phase 0.3; camera = camera_settings_locked.yaml (checked against every file below)
RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R = 3.0, 49.0, 480, 480
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
TEST = set(json.loads((root / "split_500.json").read_text())["test"])
TUNED = json.loads((root / "outputs" / "ekf_tuning_val.json").read_text())["best"]

# Estimators to compare (V2 set: includes ekf19b with stale guard)
ESTIMATORS = ["fd", "fd+stale", "cvkf", "ekf19-none", "ekf19-hybrid",
              "ekf19b-none", "ekf19b-none+stale", "ekf19b-hybrid",
              "ekf19_oracle-none", "ekf19_oracle-hybrid", "ekf19b_oracle-none",
              "ekf19b_oracle-none+stale"]

KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]
BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.01)]

# Camera: must match what was used to render the dataset
cam = Pinhole(CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R)


def noise_kw(name, cond):
    """Tuned (sigma, q) of the estimator's family; fd has none."""
    if name.startswith("fd"):
        return {}
    t = TUNED["cvkf" if name.startswith("cvkf") else "ekf"][cond]
    return dict(sigma=t["sigma"], q=t["q"])


def task(args):
    """Evaluate one launch on one condition: per-frame errors of every estimator."""
    path, cond, seed = args
    raw = np.load(path)
    rec = apply_condition(cam, raw, CONDITIONS[cond], seed)
    out = {name: evaluate(make(name, rec, **noise_kw(name, cond), **GEO), rec) for name in ESTIMATORS}
    return float(np.linalg.norm(raw["v0"])), out


def stats(items, key):
    """RMS, median, p95 over the frames of several launches."""
    e = np.concatenate([o[key] for o in items]) if items else np.array([])
    if not len(e):
        return dict(rms=float("nan"), median=float("nan"), p95=float("nan"))
    return dict(rms=float(np.sqrt(np.mean(e ** 2))), median=float(np.median(e)), p95=float(np.percentile(e, 95)))


def table(res, idx, title):
    print(f"\n  [{title}: {len(idx)} launches]   mm (now_vel: m/s), each cell RMS / median / p95")
    print(f"  {'Estimator':24s} " + " ".join(f"{k:>20s}" for k in KEYS) + " | h0.5 RMS by speed (0-1 .. 4-5 m/s)")
    out = {}
    for name in ESTIMATORS:
        items = [res[i][1][name] for i in idx]
        row = {k: stats(items, k) for k in KEYS}
        bins = [stats([res[i][1][name] for i in idx if lo <= res[i][0] < hi], "h0.5")["rms"] for lo, hi in BINS]
        out[name] = dict(row, h05_rms_by_speed=bins)
        cell = lambda k, s=1000: "/".join(f"{row[k][m] * (1 if k == 'now_vel' else s):.{2 if k == 'now_vel' else 0}f}"
                                          for m in ("rms", "median", "p95"))
        print(f"  {name:24s} " + " ".join(f"{cell(k):>20s}" for k in KEYS) + " | " +
              " ".join(f"{b * 1000:5.0f}" for b in bins))
    return out


def main():
    print("Phase 0.6: baselines, 2.5×2.5 m arena with static arm, camera 3 m / 49° / 480×480\n")
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all_conds" else list(CONDITIONS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    files = sorted(RAW.glob("launch_????.npz"))
    if len(files) != 500:
        sys.exit(f"expected 500 launch files in {RAW}, found {len(files)}")
    idx = list(range(0, len(files), stride))
    test = [i for i in idx if i in TEST]

    print(f"Dataset: {RAW.name}/  ({len(idx)} launches, stride {stride}; {len(test)} of them in the test split)")
    print(f"Camera: height {CAMERA_HEIGHT} m, FOV {CAMERA_FOV}°, {IMG_W}x{IMG_R}")
    print(f"EKF geometry: L = {GEO['L']:.3f} m, arm box (cx, cy, hx, hy) = {tuple(round(v, 3) for v in GEO['obs'])}")
    print(f"Conditions: {', '.join(conds)}")

    summary = {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            res = dict(zip(idx, pool.map(task, [(files[i], cond, 1000 + i) for i in idx], chunksize=1)))
            te, tc = TUNED["ekf"][cond], TUNED["cvkf"][cond]
            print(f"\n{'=' * 150}\nCondition: {cond} | tuned on val: EKFs sigma {te['sigma'] * 1000:.0f} mm q {te['q']}, "
                  f"cvkf sigma {tc['sigma'] * 1000:.0f} mm q {tc['q']} | {time.time() - t0:.0f} s\n{'=' * 150}")
            summary[cond] = dict(noise_ekf=te, noise_cvkf=tc, all=table(res, idx, "ALL"), test=table(res, test, "TEST"))

    out = root / "outputs" / f"baselines_h3_f49_sq480_with_arm_stride{stride}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
