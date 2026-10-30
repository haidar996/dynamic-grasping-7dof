"""Step 0.4 realism check: median / p95 / RMS and the launches that dominate RMS (ideal + noise). Run from ball_estimation/."""
import os, sys
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
from pathlib import Path
from multiprocessing import Pool
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
GEO=Arena(mujoco.MjModel.from_xml_path('scenes/arena.xml')).estimator_geometry()
cam=Pinhole(3.0,49.0,480,480)
files=sorted(Path('outputs/raw_60hz_h3_f49_sq480_with_arm').glob('launch_????.npz'))
NAMES=["fd","ekf19b-none+stale","ekf19_oracle-none"]
def task(a):
    i,cond=a
    raw=np.load(files[i]); rec=apply_condition(cam,raw,CONDITIONS[cond],1000+i)
    # raw measurement error on moving frames
    v=np.hypot(*rec["true_v"].T) if "true_v" in rec else None
    m=rec["valid"] & (np.hypot(*raw["true_v"].T)>0.05)
    meas=np.hypot(*(rec["xy"][m]-raw["true_xy"][m]).T)
    out={"meas":meas}
    for n in NAMES:
        out[n]=evaluate(make(n,rec,**GEO),rec)
    return i,out
if __name__=="__main__":
    for cond in ("ideal","noise"):
        with Pool(3) as p: res=dict(p.map(task,[(i,cond) for i in range(500)]))
        meas=np.concatenate([r["meas"] for r in res.values()])*1000
        print(f"\n== {cond}: raw measurement error (moving, valid) median {np.median(meas):.2f} p95 {np.percentile(meas,95):.2f} RMS {np.sqrt(np.mean(meas**2)):.2f} mm  (frames>10mm: {np.mean(meas>10)*100:.2f}%)")
        for n in NAMES:
            for k in ("now_pos","h0.2","h0.5","h1.0"):
                e=np.concatenate([res[i][n][k] for i in range(500)])*1000
                ss=np.array([np.sum(res[i][n][k]**2) for i in range(500)])
                top=np.argsort(ss)[::-1][:5]
                e_no303=np.concatenate([res[i][n][k] for i in range(500) if i!=303])*1000
                print(f"  {n:20s} {k:7s} median {np.median(e):7.1f} p95 {np.percentile(e,95):7.1f} RMS {np.sqrt(np.mean(e**2)):7.1f} | RMS w/o 303 {np.sqrt(np.mean(e_no303**2)):7.1f} | top5 launches share {100*ss[top].sum()/ss.sum():4.0f}% {list(top)}")
