#!/usr/bin/env python3
"""Replay saved launches in the MuJoCo viewer (positions straight from the .npz files).
Usage: step_0c_view_with_arm.py [dir] [--speed 1.0]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import ARM

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default=str(root / "outputs" / "test_f49_sq480_with_arm"))
ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
a = ap.parse_args()

model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
model.geom("arm_base").rgba[:] = [1, 0, 0, 0.35]       # show the bounce box
data = mujoco.MjData(model)
files = sorted(Path(a.dir).glob("launch_????.npz"))
dt = model.opt.timestep
every = 8                                               # draw every 8 physics steps (~60 Hz)

with mujoco.viewer.launch_passive(model, data) as v:
    v.opt.geomgroup[4] = 1
    v.cam.lookat[:] = [0, 0, 0.3]; v.cam.distance = 4.0; v.cam.elevation = -55; v.cam.azimuth = 90
    while v.is_running():
        for f in files:
            r = np.load(f)
            pos, vel, b = r["pos"], r["vel"], r["bounces"]
            arm_hits = int((b[:, 3] == ARM).sum()) if len(b) else 0
            info = (f"{f.stem}  speed {np.linalg.norm(r['v0']):.1f} m/s  mu {float(r['mu']):.2f}  "
                    f"E {float(r['e']):.2f}  bounces {len(b)} (arm {arm_hits})")
            print(info, flush=True)
            moving = np.nonzero(np.linalg.norm(vel, axis=1) > 0)[0]
            end = min(len(pos), (moving[-1] if len(moving) else 0) + int(1.0 / dt))   # stop 1 s after the ball halts
            t0 = time.time()
            for k in range(0, end, every):
                if not v.is_running():
                    sys.exit()
                with v.lock():
                    data.qpos[:2] = pos[k]; data.time = k * dt
                    mujoco.mj_forward(model, data)
                v.sync()
                lag = k * dt / a.speed - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            time.sleep(0.5)
