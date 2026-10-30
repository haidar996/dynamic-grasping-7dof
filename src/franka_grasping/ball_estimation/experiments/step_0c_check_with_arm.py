#!/usr/bin/env python3
"""Realism checks for launches rendered with the arm.  Usage: step_0c_check_with_arm.py [dir]"""
import sys
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena, ARM
from sensing.camera import Pinhole

d = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "outputs" / "test_f49_sq480_with_arm"
files = sorted(d.glob("launch_????.npz"))
model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
arena = Arena(model)
lim = arena.Lx - arena.radius
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


print(f"{len(files)} launches in {d}\n")
check("arm has no joints (static)", model.nq == 2, f"nq = {model.nq}")

R = [np.load(f) for f in files]
cam = Pinhole(float(R[0]["cam_height"]), float(R[0]["cam_fovy"]), int(R[0]["cam_w"]), int(R[0]["cam_rows"]))
print(f"camera: {cam.H} m, {cam.fovy} deg, {cam.W}x{cam.R}")
speed0 = np.array([np.linalg.norm(r["v0"]) for r in R])
first_v = np.array([np.linalg.norm(r["vel"][0]) for r in R])
check("start speed kept after 1 step", np.all(first_v > 0.95 * speed0 - 0.01),
      f"worst ratio {np.min(first_v / speed0):.3f}")

allpos = np.concatenate([r["pos"] for r in R])
check("ball stays inside walls", np.all(np.abs(allpos) <= lim + 1e-6), f"max |x|,|y| = {np.abs(allpos).max():.3f} (limit {lim:.3f})")
inside = np.all(np.abs(allpos - arena.obs_c) < arena.obs_h - 1e-6, axis=1)
check("ball never inside arm base", not inside.any(), f"{inside.sum()} samples inside")

sp = [np.linalg.norm(r["vel"], axis=1) for r in R]
check("speed never increases", all(np.all(np.diff(s) < 1e-6) for s in sp), "friction + bounces only remove energy")
should_stop = [np.linalg.norm(r["v0"]) / r["mu"] < 0.8 * float(r["T"]) for r in R]     # straight-line stop time well inside T
stopped = [s[-1] == 0 for s, k in zip(sp, should_stop) if k]
check("balls that should stop are at rest at the end", all(stopped), f"{sum(stopped)}/{len(stopped)} at exactly 0 m/s")

nb = np.array([len(r["bounces"]) for r in R])
na = np.array([(r["bounces"][:, 3] == ARM).sum() if len(r["bounces"]) else 0 for r in R])
def reaches_wall(r):
    s = np.linalg.norm(r["v0"]); u = r["v0"] / s
    to_wall = min((lim * np.sign(u[k]) - r["p0"][k]) / u[k] for k in (0, 1) if abs(u[k]) > 1e-9)
    return s * s / (2 * r["mu"]) > to_wall + 0.05            # stop distance clearly beyond the wall
must = np.array([reaches_wall(r) for r in R])
check("launches that reach a wall bounce off it", np.all(nb[must] > 0) and must.any(),
      f"{must.sum()} must bounce; wall bounces per launch: {list(nb - na)}")
check("ball hits the arm in some launch", na.sum() > 0, f"arm bounces per launch: {list(na)}")

err, miss_d, n_frames, n_seen = [], [], 0, 0
for r in R:
    uvz, txy = r["uvz"], r["true_xy"]
    ok = ~np.isnan(uvz[:, 0])
    n_frames += len(ok); n_seen += ok.sum()
    err += [np.linalg.norm(cam.back_project(*u) - t) for u, t in zip(uvz[ok], txy[ok])]
    miss_d += list(np.linalg.norm(txy[~ok] - arena.obs_c, axis=1))
err, miss_d = np.array(err) * 1000, np.array(miss_d)
check("camera sees the ball most of the time", n_seen / n_frames > 0.9, f"{100 * n_seen / n_frames:.1f}% of frames")
check("misses only near the arm", len(miss_d) == 0 or miss_d.max() < 0.6,
      f"{len(miss_d)} missed frames, max distance from base {miss_d.max() if len(miss_d) else 0:.2f} m")
check("detection error small", np.median(err) < 5, f"median {np.median(err):.2f} mm, p99 {np.percentile(err, 99):.2f} mm")

fig, ax = plt.subplots(figsize=(7, 7))
for r, s in zip(R, speed0):
    ax.plot(r["pos"][:, 0], r["pos"][:, 1], lw=0.7, label=f"{s:.1f} m/s")
    ok = np.isnan(r["uvz"][:, 0])
    ax.plot(r["true_xy"][ok, 0], r["true_xy"][ok, 1], "kx", ms=3)
c, h = arena.obs_c, arena.obs_h - arena.radius
ax.add_patch(plt.Rectangle(c - h, *(2 * h), color="r", alpha=0.4))
ax.add_patch(plt.Rectangle((-lim - arena.radius,) * 2, 2 * arena.Lx, 2 * arena.Ly, fill=False))
ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right"); ax.set_title("paths (x = camera missed ball, red = arm base)")
fig.savefig(d / "check.png", dpi=120)
print(f"\nplot: {d / 'check.png'}")
print(f"\n{sum(results)}/{len(results)} checks passed")
