"""Step 0.4 realism check: our depth noise (k=8.2e-4, ~7 mm at 3 m) vs D435-like stereo (k=3.74e-3, ~33 mm). Run from ball_estimation/."""
import os, sys
os.environ["OMP_NUM_THREADS"]="1"; sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
from pathlib import Path
from multiprocessing import Pool
from arena_sim import Arena
from sensing.camera import Pinhole
from sensing.sensor import SensorParams
from dataset import apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
GEO=Arena(mujoco.MjModel.from_xml_path('scenes/arena.xml')).estimator_geometry()
cam=Pinhole(3.0,49.0,480,480)
files=sorted(Path('outputs/raw_60hz_h3_f49_sq480_with_arm').glob('launch_????.npz'))
CONDS={"ours k=8.2e-4 (7 mm)":SensorParams(k_depth=8.2e-4, jitter_px=0.3),
       "D435-like k=3.74e-3 (33 mm)":SensorParams(k_depth=3.74e-3, jitter_px=0.3)}
NAMES=["fd","ekf19b-none+stale","ekf19_oracle-none"]
def task(a):
    i,c=a; raw=np.load(files[i]); rec=apply_condition(cam,raw,CONDS[c],1000+i)
    m=rec["valid"]&(np.hypot(*raw["true_v"].T)>0.05)
    out={"meas":np.hypot(*(rec["xy"][m]-raw["true_xy"][m]).T)}
    for n in NAMES: out[n]=evaluate(make(n,rec,**GEO),rec)
    return out
if __name__=="__main__":
    for c in CONDS:
        with Pool(3) as p: res=p.map(task,[(i,c) for i in range(500)])
        meas=1000*np.concatenate([r["meas"] for r in res])
        print(f"\n{c}: measurement error median {np.median(meas):.1f} p95 {np.percentile(meas,95):.1f} mm")
        for n in NAMES:
            row=[]
            for k in ("now_pos","h0.2","h0.5","h1.0"):
                e=1000*np.concatenate([r[n][k] for r in res]); row.append(f"{k} {np.sqrt(np.mean(e**2)):6.1f} (med {np.median(e):5.1f})")
            print(f"  {n:20s} "+" | ".join(row))
