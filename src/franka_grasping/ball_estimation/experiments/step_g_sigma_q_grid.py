import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, json, itertools, numpy as np
from pathlib import Path
root = Path("/home/haidar/dynamic-grasping-7dof/src/franka_grasping/ball_estimation"); sys.path.insert(0, str(root))
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
from multiprocessing import Pool
cam = Pinhole(3.0, 60.0)
allf = sorted((root/"outputs/raw_60hz_h3_f60").glob("launch_*.npz"))
SIG = [0.002, 0.003, 0.004, 0.005]; Q = [0.05, 0.1, 0.2, 0.5]; CONDS = ["noise", "all"]
KEYS = ["now_pos", "now_vel", "h0.5", "h1.0"]
def work(args):
    i, cond = args
    rec = apply_condition(cam, np.load(allf[i]), CONDITIONS[cond], i); out = {}
    for s, q in itertools.product(SIG, Q):
        r = evaluate(make("ekf19b-none", rec, sigma=s, q=q), rec)
        out[f"{s}_{q}"] = {k: (float(np.sum(v**2)), len(v)) for k, v in r.items() if k in KEYS}
    return cond, out
def run(idx, tag):
    jobs = [(i, c) for i in idx for c in CONDS]; acc = {}
    with Pool(4) as p:
        for j, (cond, out) in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            for key, d in out.items():
                for k, (ss, n) in d.items():
                    a = acc.setdefault((cond, key, k), [0.0, 0]); a[0] += ss; a[1] += n
            if j % 20 == 19: print(tag, "progress", j + 1, "/", len(jobs), flush=True)
    return acc
def show(acc, tag):
    for cond in CONDS:
        print(f"\n[{tag}] {cond}:  sigma mm, q | now_pos mm | now_vel mm/s | h0.5 mm | h1.0 mm")
        for s, q in itertools.product(SIG, Q):
            f = lambda k: 1e3*np.sqrt(acc[(cond, f"{s}_{q}", k)][0]/acc[(cond, f"{s}_{q}", k)][1])
            print(f"  {1e3*s:3.0f}  {q:5.2f} | {f('now_pos'):6.2f} | {f('now_vel'):7.0f} | {f('h0.5'):6.1f} | {f('h1.0'):6.1f}")
if __name__ == "__main__":
    A = run(range(0, 500, 10), "tune"); show(A, "TUNE launches 0,10,..")
    B = run(range(5, 500, 10), "check"); show(B, "CHECK launches 5,15,..")
