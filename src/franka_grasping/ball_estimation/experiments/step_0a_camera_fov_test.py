#!/usr/bin/env python3
"""
Phase 0.2: Camera FOV Optimization
Render 50 launches per FOV (40°, 50°, 60°) on 2.5×2.5 m arena WITHOUT arm.
Find optimal camera field of view for ball state observation.
"""
import sys, time
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from arena_sim import Arena
from launches import sample_launches
from sensing.camera import DepthCamera
from dataset import record_raw

# Configuration
ARENA_XML = root / "scenes" / "arena_no_arm.xml"
CAMERA_HEIGHT = 3.0
FOVS = [50, 55, 58, 60, 62, 65, 70, 75, 80]  # Full range: narrow to wide
LAUNCHES_PER_FOV = 30  # 30 launches per FOV for comprehensive testing
DURATION = 10.0

print(f"✓ Phase 0.2: Camera FOV Optimization")
print(f"  Arena: 2.5×2.5 m (NO ARM)")
print(f"  Camera height: {CAMERA_HEIGHT} m")
print(f"  FOVs to test: {FOVS}")
print(f"  Launches per FOV: {LAUNCHES_PER_FOV}")
print(f"  Total launches: {len(FOVS) * LAUNCHES_PER_FOV}")
print()

# Load arena
model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
arena = Arena(model)

# Test each FOV
for fov in FOVS:
    print(f"\n{'='*60}")
    print(f"Testing FOV: {fov}°")
    print(f"{'='*60}")

    # Create output directory
    out_dir = root / "outputs" / f"fov_test_{fov}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create camera
    cam = DepthCamera(model, CAMERA_HEIGHT, float(fov))

    # Sample launches
    rng = np.random.default_rng(fov)  # seed by FOV for reproducibility
    launches = sample_launches(rng, arena, LAUNCHES_PER_FOV)

    # Render launches
    # Check how many already exist
    existing = len(list(out_dir.glob("launch_*.npz")))
    done = 0

    if existing >= LAUNCHES_PER_FOV:
        print(f"  ✓ FOV {fov}° already has {existing} launches (skipping)")
    else:
        t0 = time.time()
        for i, launch in enumerate(launches):
            out_file = out_dir / f"launch_{i:04d}.npz"
            if out_file.exists():
                continue

            # Stop if we already have enough launches
            if existing + done >= LAUNCHES_PER_FOV:
                break

            tmp_file = out_dir / f"launch_{i:04d}.tmp.npz"
            try:
                np.savez_compressed(tmp_file, **record_raw(model, arena, cam, launch, DURATION))
                tmp_file.rename(out_file)
                done += 1
            except Exception as e:
                print(f"  Error on launch {i}: {e}")
                if tmp_file.exists():
                    tmp_file.unlink()

            if done % 10 == 0 and done > 0:
                elapsed = time.time() - t0
                remaining = elapsed / done * (LAUNCHES_PER_FOV - existing - done)
                print(f"  FOV {fov}°: {existing + done}/{LAUNCHES_PER_FOV} launches  "
                      f"({elapsed/60:.1f} min, ~{remaining/60:.0f} min left)", flush=True)

    cam.close()
    print(f"  ✓ Finished FOV {fov}°: {done} launches saved to {out_dir}")

print(f"\n{'='*60}")
print(f"✓ Phase 0.2 COMPLETE: Camera FOV test data generated")
print(f"  Outputs:")
print(f"    - {root / 'outputs' / 'fov_test_40'}")
print(f"    - {root / 'outputs' / 'fov_test_50'}")
print(f"    - {root / 'outputs' / 'fov_test_60'}")
print(f"  Next: Analyze images to select best FOV")
print(f"{'='*60}")
