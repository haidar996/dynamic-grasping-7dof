"""Step F: baseline estimators on the dataset (every launch x condition x estimator).
Usage: python3 step_f_baselines.py [stride] [conditions, comma separated]   (stride 1 = all 500 launches)"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")                # one BLAS thread per worker (4x faster here)
import sys, time, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate

RAW = root / "outputs" / "raw_60hz_h3_f60"
ESTIMATORS = ["fd", "cvkf", "ekf19-none", "ekf19-skip", "ekf19-reset", "ekf19-hybrid", "ekf19_oracle-hybrid"]
if os.environ.get("ESTIMATORS_V2"):              # 2026-10-17: + oracle without gate, ekf19b, stale-repeat guard
    ESTIMATORS = ["fd", "fd+stale", "cvkf", "ekf19-none", "ekf19-hybrid", "ekf19b-none", "ekf19b-none+stale",
                  "ekf19b-hybrid", "ekf19_oracle-none", "ekf19_oracle-hybrid", "ekf19b_oracle-none",
                  "ekf19b_oracle-none+stale"]
TAG = "baselines_v2" if os.environ.get("ESTIMATORS_V2") else "baselines"
KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]
BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.01)]
cam = Pinhole(3.0, 60.0)                        # the camera setting of the recording


def task(args):
    path, cond, seed = args
    raw = np.load(path)
    rec = apply_condition(cam, raw, CONDITIONS[cond], seed)
    out = {}
    for name in ESTIMATORS:
        res = evaluate(make(name, rec), rec)
        out[name] = {k: (float(np.sum(v ** 2)), int(len(v))) for k, v in res.items()}   # pooled RMS later
    return float(np.linalg.norm(raw["v0"])), out


def rms(items, key):
    ss = sum(o[key][0] for o in items); n = sum(o[key][1] for o in items)
    return np.sqrt(ss / n) if n else float("nan")


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONDITIONS)
    files = sorted(RAW.glob("launch_????.npz")); idx = list(range(0, len(files), stride))
    summary = {}
    with Pool(4) as pool:
        for cond in conds:
            t0 = time.time()
            res = pool.map(task, [(files[i], cond, 1000 + i) for i in idx], chunksize=1)
            print(f"\n=== {cond}: {len(idx)} launches, {time.time()-t0:.0f} s   (RMS over frames where the ball moves)")
            print(f"{'estimator':22s} now_pos(mm) now_vel(m/s) h0.2(mm) h0.5(mm) h1.0(mm) | h0.5 (mm) by launch speed 0-1,1-2,2-3,3-4,4-5")
            summary[cond] = {}
            for name in ESTIMATORS:
                items = [r[1][name] for r in res]
                row = {k: rms(items, k) for k in KEYS}
                bins = [rms([r[1][name] for r in res if lo <= r[0] < hi], "h0.5") for lo, hi in BINS]
                summary[cond][name] = dict(row, h05_by_speed=bins)
                print(f"{name:22s} {row['now_pos']*1000:9.1f} {row['now_vel']:11.2f} {row['h0.2']*1000:9.1f} "
                      f"{row['h0.5']*1000:9.1f} {row['h1.0']*1000:9.1f} | " + " ".join(f"{b*1000:6.0f}" for b in bins))
    out = root / "outputs" / f"{TAG}_stride{stride}.json"
    out.write_text(json.dumps(summary, indent=1)); print("\nsaved", out)


if __name__ == "__main__":
    main()
