"""Render the 500 launches once (50 speeds x K=10) at the provisional camera (3 m, 60 deg)."""
import sys, time; from pathlib import Path
import numpy as np, mujoco
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from arena_sim import Arena
from launches import sample_launches
from sensing.camera import DepthCamera
from dataset import record_raw

K, T, OUT = 10, 10.0, root / "outputs" / "raw_60hz_h3_f60"
model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
arena, cam = Arena(model), DepthCamera(model, 3.0, 60.0)
launches = sample_launches(np.random.default_rng(0), arena, K)
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); done = 0
for i, l in enumerate(launches):
    f = OUT / f"launch_{i:04d}.npz"
    if f.exists(): continue
    tmp = OUT / f"launch_{i:04d}.tmp.npz"
    np.savez_compressed(tmp, **record_raw(model, arena, cam, l, T)); tmp.rename(f)
    done += 1
    if done % 10 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(launches)}  {el/60:.1f} min, ~{el/done*(len(launches)-i-1)/60:.0f} min left", flush=True)
print("finished", flush=True)
cam.close()
