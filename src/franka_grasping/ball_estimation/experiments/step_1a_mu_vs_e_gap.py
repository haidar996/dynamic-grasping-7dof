"""Phase 1.1 check: how much of the ekf19b-vs-oracle gap is friction mu vs restitution E (true mu only / true E only). Run from ball_estimation/."""
import os, sys, json
os.environ["OMP_NUM_THREADS"]="1"; sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
from multiprocessing import Pool
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import EKF19, StaleGuard
from estimators.metrics import evaluate
GEO=Arena(mujoco.MjModel.from_xml_path('scenes/arena.xml')).estimator_geometry()
cam=Pinhole(3.0,49.0,480,480)
T=json.load(open('outputs/ekf_tuning_val.json'))["best"]["ekf"]
def task(a):
    i,cond=a; raw=np.load(f'outputs/raw_60hz_h3_f49_sq480_with_arm/launch_{i:04d}.npz'); rec=apply_condition(cam,raw,CONDITIONS[cond],1000+i)
    kw=dict(sigma=T[cond]["sigma"], q=T[cond]["q"], policy="none", sigma_E=0.13, **GEO)
    V={"nominal mu & E":(0.85,0.725),"true mu only":(rec["mu"],0.725),"true E only":(0.85,rec["e"]),"true mu & E":(rec["mu"],rec["e"])}
    return {k:evaluate(StaleGuard(EKF19(c=c,E=e,**kw)),rec) for k,(c,e) in V.items()}
if __name__=="__main__":
    for cond in ("ideal","all_real"):
        with Pool(3) as p: res=p.map(task,[(i,cond) for i in range(500)])
        print(cond)
        for k in res[0]:
            row=[]
            for h in ("h0.5","h1.0"):
                e=1000*np.concatenate([r[k][h] for r in res]); row.append(f"{h} RMS {np.sqrt(np.mean(e**2)):5.0f} med {np.median(e):4.0f}")
            print(f"  {k:16s} "+" | ".join(row))
