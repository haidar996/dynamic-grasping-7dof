"""Raw per-launch recordings (rendered once) + offline application of sensor conditions."""
from pathlib import Path
import numpy as np
import mujoco
from sensing.sensor import Sensor, SensorParams

# sensor conditions for the ablation.  k_depth: 8.2e-4 -> sigma_Z ~7 mm at 3 m (optimistic, good ToF-class);
# 3.74e-3 -> ~33 mm = D435-like stereo, RMS = Z^2 * subpixel / (f * B) (Intel), added 2026-10-30 (step 0.6)
K_REAL = 3.74e-3
CONDITIONS = {
    "ideal":           SensorParams(),
    "noise":           SensorParams(k_depth=8.2e-4, jitter_px=0.3),
    "dropout_flagged": SensorParams(loss_rate=0.05, mean_gap=6, loss_mode="flagged"),
    "dropout_silent":  SensorParams(loss_rate=0.05, mean_gap=6, loss_mode="silent_hold"),
    "delay":           SensorParams(delay=0.04),
    "all":             SensorParams(k_depth=8.2e-4, jitter_px=0.3, loss_rate=0.05,
                                    mean_gap=6, loss_mode="flagged", delay=0.04),
    "noise_real":      SensorParams(k_depth=K_REAL, jitter_px=0.3),
    "all_real":        SensorParams(k_depth=K_REAL, jitter_px=0.3, loss_rate=0.05,
                                    mean_gap=6, loss_mode="flagged", delay=0.04),
}


def record_raw(model, arena, camera, launch, T=10.0, fps=60.0):
    """Truth at the physics rate + ideal detections (u, v, z) at `fps`."""
    data = mujoco.MjData(model)
    arena.reset(data, launch.p0, launch.v0)
    n = int(T / model.opt.timestep)
    t = np.zeros(n); pos = np.zeros((n, 2)); vel = np.zeros((n, 2)); bounces = []
    fr = {k: [] for k in ("t", "uvz", "true_xy", "true_v")}
    next_t = 0.0
    for i in range(n):
        hits = arena.step(data, launch.mu, launch.e)
        t[i], pos[i], vel[i] = data.time, data.qpos[:2], data.qvel[:2]
        bounces += [(data.time, *h) for h in hits]
        if data.time >= next_t:
            mujoco.mj_forward(model, data)
            parts = camera.measure_parts(data)
            fr["t"].append(data.time)
            fr["uvz"].append(np.full(3, np.nan) if parts is None else np.array(parts))
            fr["true_xy"].append(data.qpos[:2].copy())
            fr["true_v"].append(data.qvel[:2].copy())
            next_t += 1.0 / fps
    return dict(t_phys=t.astype(np.float32), pos=pos.astype(np.float32),
                vel=vel.astype(np.float32), bounces=np.array(bounces).reshape(-1, 4),
                frame_t=np.array(fr["t"]), uvz=np.array(fr["uvz"]),
                true_xy=np.array(fr["true_xy"]), true_v=np.array(fr["true_v"]),
                p0=launch.p0, v0=launch.v0, mu=launch.mu, e=launch.e,
                fps=fps, T=T, cam_height=camera.H, cam_fovy=camera.fovy, cam_w=camera.W, cam_rows=camera.R)


def generate_raw(model, arena, camera, launches, out_dir, T=10.0, fps=60.0):
    """One file per launch; skips files that exist, so it can be stopped and resumed."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i, launch in enumerate(launches):
        f = out_dir / f"launch_{i:04d}.npz"
        if f.exists():
            continue
        tmp = out_dir / f"launch_{i:04d}.tmp.npz"
        np.savez_compressed(tmp, **record_raw(model, arena, camera, launch, T, fps))
        tmp.rename(f)                                  # a half-written file is never visible


def apply_condition(camera, raw, params, seed, decimate=1):
    """Offline sensor: same output as sensing.sensor.record_launch."""
    size = (int(raw["cam_w"]), int(raw["cam_rows"])) if "cam_w" in raw else (640, 480)   # old files: 640x480
    assert (float(raw["cam_height"]), float(raw["cam_fovy"]), *size) == (camera.H, camera.fovy, camera.W, camera.R), \
        "camera setting differs from the one used to record this launch"
    p = SensorParams(**{**params.__dict__, "seed": seed, "fps": float(raw["fps"]) / decimate})
    sensor = Sensor(camera, p)
    xy, valid, lost = [], [], []
    for row in raw["uvz"][::decimate]:
        parts = None if np.isnan(row[0]) else tuple(row)
        a, v, l = sensor.apply(lambda: parts)
        xy.append(np.full(2, np.nan) if a is None else a); valid.append(v); lost.append(l)
    t = raw["frame_t"][::decimate]
    return dict(t_capture=t, t_arrival=t + p.delay, xy=np.array(xy), valid=np.array(valid),
                was_lost=np.array(lost), true_xy=raw["true_xy"][::decimate],
                true_v=raw["true_v"][::decimate], mu=float(raw["mu"]), e=float(raw["e"]),
                bounces=raw["bounces"], t_phys=raw["t_phys"], pos=raw["pos"], vel=raw["vel"],
                v0=raw["v0"])


def load_condition(camera, raw_dir, params, base_seed=0, decimate=1):
    """All launches under one condition. Seed = base_seed + launch index, so the loss gaps
    are identical across conditions that share loss parameters (paired comparison)."""
    files = sorted(Path(raw_dir).glob("launch_????.npz"))
    return [apply_condition(camera, np.load(f), params, base_seed + i, decimate)
            for i, f in enumerate(files)]
