"""Step A check: launch the ball, plot its path and speed, verify bounces."""
import sys
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena

model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
data = mujoco.MjData(model)

MU, E = 0.5, 0.8                    # friction (m/s^2), wall restitution (placeholders)
arena = Arena(model)
arena.reset(data, p0=[-0.5, -0.5], v0=[2.4, 1.6])

T = 8.0
n = int(T / model.opt.timestep)
t = np.zeros(n); pos = np.zeros((n, 2)); vel = np.zeros((n, 2))
bounces = []                        # (time, axis, v_before, v_after)
for i in range(n):
    v_before = data.qvel[:2].copy()
    arena.step(data, MU, E)
    # a bounce flips the sign of one velocity component
    for ax in (0, 1):
        if v_before[ax] * data.qvel[ax] < 0:
            bounces.append((data.time, ax, v_before[ax], data.qvel[ax]))
    t[i] = data.time
    pos[i] = data.qpos[:2]
    vel[i] = data.qvel[:2]

speed = np.linalg.norm(vel, axis=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(pos[:, 0], pos[:, 1]); ax[0].set_aspect("equal")
ax[0].set_xlim(-1.1, 1.1); ax[0].set_ylim(-1.1, 1.1)
ax[0].set_title("ball path (top view)"); ax[0].grid()
ax[1].plot(t, speed); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("speed (m/s)")
ax[1].set_title("speed: friction slope + wall bounces"); ax[1].grid()
out = root / "experiments" / "step_a_launch.png"
fig.savefig(out, dpi=120)
print("saved", out)
print("max |x|,|y| =", np.abs(pos).max(axis=0), " (limit", arena.Lx - arena.radius, ")")
print("max speed =", speed.max(), " final speed =", speed[-1])
print("bounces (t, axis, v_before, v_after, ratio):")
for tb, axis, vb, va in bounces:
    print(f"  t={tb:5.2f} axis={'xy'[axis]} {vb:+.3f} -> {va:+.3f}  ratio={-va/vb:.3f}")
