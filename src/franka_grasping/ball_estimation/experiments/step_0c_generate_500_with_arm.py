#!/usr/bin/env python3
"""Phase 0.3: render the 500 launches on the 2.5x2.5 m arena with the static arm (Q_HOME) at the
centre, overhead camera 3 m / 49 deg / 480x480.  Resumable, parallel.

  python3 experiments/step_0c_generate_500_with_arm.py --quick 10   # 10 launches -> outputs/test_f49_sq480_with_arm/
  python3 experiments/step_0c_generate_500_with_arm.py              # all 500 -> outputs/raw_60hz_h3_f49_sq480_with_arm/
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

ARENA_XML = root / "scenes" / "arena.xml"
CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R = 3.0, 49.0, 480, 480     # camera_settings_locked.yaml
DURATION, SEED = 10.0, 42

_w = {}


def _init():
    import mujoco
    from arena_sim import Arena
    from sensing.camera import DepthCamera
    m = mujoco.MjModel.from_xml_path(str(ARENA_XML))
    _w.update(model=m, arena=Arena(m), cam=DepthCamera(m, CAMERA_HEIGHT, CAMERA_FOV, width=IMG_W, rows=IMG_R))


def _render(job):
    from dataset import record_raw
    i, launch, out_dir = job
    out, tmp = out_dir / f"launch_{i:04d}.npz", out_dir / f"launch_{i:04d}.tmp.npz"
    np.savez_compressed(tmp, **record_raw(_w["model"], _w["arena"], _w["cam"], launch, DURATION))
    tmp.rename(out)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0, help="render only N launches spread over the speeds")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import mujoco
    from arena_sim import Arena
    from launches import sample_launches
    arena = Arena(mujoco.MjModel.from_xml_path(str(ARENA_XML)))
    launches = sample_launches(np.random.default_rng(SEED), arena, k=10)     # 50 speeds x 10 = 500

    if a.quick:
        out_dir = root / "outputs" / "test_f49_sq480_with_arm"
        idx = np.linspace(0, len(launches) - 1, a.quick).astype(int)
    else:
        out_dir = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
        idx = np.arange(len(launches))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.tmp.npz"):
        f.unlink()
    jobs = [(int(i), launches[i], out_dir) for i in idx if not (out_dir / f"launch_{i:04d}.npz").exists()]
    print(f"{len(idx)} launches, {len(jobs)} to render -> {out_dir}  ({a.workers} workers)", flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for n, _ in enumerate(pool.imap_unordered(_render, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el / 60:.1f} min, ~{el / n * (len(jobs) - n) / 60:.0f} min left", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
