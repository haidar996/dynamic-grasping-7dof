#!/usr/bin/env python3
"""
Phase 0.3: Render launches WITH GUI visualization
Shows: 2.5×2.5 m arena, arm at center, ball launching, camera at FOV 55°
"""
import sys
sys.exit("LEGACY (old FOV 55° / 640x480 setup, own physics loop, writes into a dataset folder) — do not run. "
         "Use experiments/step_0c_view_with_arm.py to watch launches.")

import sys, time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from arena_sim import Arena
from launches import sample_launches
from sensing.camera import DepthCamera
from dataset import record_raw

# LOCKED SETTINGS
ARENA_XML = root / "scenes" / "arena.xml"
CAMERA_HEIGHT = 3.0
CAMERA_FOV = 55.0
N_LAUNCHES = 500
DURATION = 10.0
OUTPUT_DIR = root / "outputs" / "raw_60hz_h250_f55_with_arm"

print("╔════════════════════════════════════════════════════════════════╗")
print("║   Phase 0.3: GUI Visualization + Launch Rendering            ║")
print("╚════════════════════════════════════════════════════════════════╝")
print()
print("Arena: 2.5×2.5 m WITH Franka Panda arm (center)")
print("Camera: FOV 55° (LOCKED), height 3.0 m")
print()
print("📺 GUI showing live simulation...")
print("   You can watch the arena and ball launches")
print("   Press CTRL+C to stop the GUI and continue rendering")
print()

# Load arena
model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
data = mujoco.MjData(model)
arena = Arena(model)
cam = DepthCamera(model, CAMERA_HEIGHT, CAMERA_FOV)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample launches
rng = np.random.default_rng(42)
launches = sample_launches(rng, arena, N_LAUNCHES)

print()
print("Starting visualization and rendering...")
print()

# Open viewer in background
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0_render = time.time()
    done = 0

    try:
        for i, launch in enumerate(launches):
            out_file = OUTPUT_DIR / f"launch_{i:04d}.npz"
            if out_file.exists():
                continue

            tmp_file = OUTPUT_DIR / f"launch_{i:04d}.tmp.npz"
            try:
                # Reset and record launch
                arena.reset(data, launch.p0, launch.v0)
                mujoco.mj_forward(model, data)
                viewer.sync()

                # Record the launch
                np.savez_compressed(tmp_file, **record_raw(model, arena, cam, launch, DURATION))
                tmp_file.rename(out_file)
                done += 1

                if done % 10 == 0:
                    elapsed = time.time() - t0_render
                    print(f"  {done}/{N_LAUNCHES} launches  ({elapsed/60:.1f} min)")

            except Exception as e:
                print(f"  Error on launch {i}: {e}")
                if tmp_file.exists():
                    tmp_file.unlink()

    except KeyboardInterrupt:
        print()
        print("✓ GUI stopped by user")
        elapsed = time.time() - t0_render
        print(f"  Rendered: {done} launches in {elapsed/60:.1f} minutes")
        print()
        print("To continue rendering remaining launches without GUI:")
        print("  Run: python3 step_0c_generate_500_with_arm.py")
        print()

cam.close()
