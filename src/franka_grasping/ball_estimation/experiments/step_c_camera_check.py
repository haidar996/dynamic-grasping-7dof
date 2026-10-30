"""Step C check: ideal-camera error over many ball positions, for several heights / FOVs."""
import sys; from pathlib import Path
import numpy as np, mujoco
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from sensing.camera import DepthCamera

model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
data = mujoco.MjData(model)
rng = np.random.default_rng(0)
pts = rng.uniform(-0.97, 0.97, size=(200, 2))          # ball positions inside the arena

cam = DepthCamera(model)                                # one renderer for all settings
print("height  fovy | ball radius px | visible | mean err mm | max err mm")
for H in (2.0, 3.0, 4.0):
    for fovy in (45.0, 60.0, 75.0):
        cam.configure(H, fovy)
        errs = []
        for p in pts:
            data.qpos[:2] = p; mujoco.mj_forward(model, data)
            m = cam.measure(data)
            if m is not None:
                errs.append(np.linalg.norm(m - p) * 1000)
        e = np.array(errs)
        print(f"{H:5.1f} {fovy:5.0f} | {cam.radius * cam.f / (H - cam.radius):14.1f} | "
              f"{len(e) / len(pts):7.2f} | {e.mean():11.2f} | {e.max():10.2f}")
cam.close()
