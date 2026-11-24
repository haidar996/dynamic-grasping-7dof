#!/usr/bin/env python3
"""B6: render the final held-out test set (seed 20261101, as specified in final_test_plan.json).
Outputs to outputs/final_test_matched/ and outputs/final_test_mismatch/.
Never run on dev-test data; never re-run once B7 has started.

Usage:
  python3 experiments/step_5a_render_final_test.py                    # matched
  python3 experiments/step_5a_render_final_test.py --physics mismatch
  python3 experiments/step_5a_render_final_test.py --quick 10         # smoke test
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

ARENA_XML       = root / "scenes" / "arena.xml"
CAMERA_HEIGHT, CAMERA_FOV, IMG_W, IMG_R = 3.0, 49.0, 480, 480  # camera_settings_locked.yaml
DURATION        = 10.0
SEED_LAUNCHES   = 20261101   # final_test_plan.json
SEED_MISMATCH   = 20261102   # mismatch physics parameters for final test

_w = {}


def _init(physics="matched"):
    import mujoco
    from arena_sim import Arena, MismatchArena
    from sensing.camera import DepthCamera
    m = mujoco.MjModel.from_xml_path(str(ARENA_XML))
    arena = MismatchArena(m) if physics == "mismatch" else Arena(m)
    _w.update(model=m, arena=arena, cam=DepthCamera(m, CAMERA_HEIGHT, CAMERA_FOV, width=IMG_W, rows=IMG_R))


def _render(job):
    from dataset import record_raw
    i, launch, out_dir = job
    out, tmp = out_dir / f"launch_{i:04d}.npz", out_dir / f"launch_{i:04d}.tmp.npz"
    np.savez_compressed(tmp, **record_raw(_w["model"], _w["arena"], _w["cam"], launch, DURATION))
    tmp.rename(out)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--physics", choices=("matched", "mismatch"), default="matched")
    a = ap.parse_args()

    import mujoco
    from arena_sim import Arena
    from launches import sample_launches, with_mismatch
    arena = Arena(mujoco.MjModel.from_xml_path(str(ARENA_XML)))
    launches = sample_launches(np.random.default_rng(SEED_LAUNCHES), arena, k=10)  # 500 launches

    suffix = ""
    if a.physics == "mismatch":
        launches = with_mismatch(launches, np.random.default_rng(SEED_MISMATCH))
        suffix = "_mismatch"

    if a.quick:
        out_dir = root / "outputs" / f"final_test_quick{suffix}"
        idx = np.linspace(0, len(launches) - 1, a.quick).astype(int)
    else:
        out_dir = root / "outputs" / f"final_test{suffix}"
        idx = np.arange(len(launches))

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.tmp.npz"):
        f.unlink()
    jobs = [(int(i), launches[i], out_dir) for i in idx if not (out_dir / f"launch_{i:04d}.npz").exists()]
    print(f"[final test {a.physics}] {len(idx)} launches, {len(jobs)} to render -> {out_dir}  ({a.workers} workers)", flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.physics,)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_render, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f} min, ~{el/n*(len(jobs)-n)/60:.0f} min left", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
