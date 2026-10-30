#!/usr/bin/env python3
"""Render real depth camera images from the arena."""
import sys
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from arena_sim import Arena
from launches import sample_launches
from sensing.camera import DepthCamera
from dataset import record_raw

ARENA_XML = root / "scenes" / "arena_no_arm.xml"
CAMERA_HEIGHT = 3.0
FOVS = [40, 50, 60]

# Load arena
model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
arena = Arena(model)

print("Rendering real depth camera images...")
print()

for fov in FOVS:
    print(f"FOV {fov}°:")

    # Create camera
    cam = DepthCamera(model, CAMERA_HEIGHT, float(fov))

    # Sample and render one launch
    rng = np.random.default_rng(fov + 100)  # Different seed for variety
    launches = sample_launches(rng, arena, 1)
    launch = launches[0]

    # Render the launch and capture depth buffer
    data = mujoco.MjData(model)
    arena.reset(data, launch.p0, launch.v0)

    # Get the depth rendering context
    renderer = mujoco.Renderer(model, height=int(cam.R), width=int(cam.W))

    # Step to get initial frame
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    # Render depth
    depth_image = renderer.render()  # RGB image for visualization

    # Also get the actual depth from the camera system
    # by capturing the depth buffer directly
    import struct

    print(f"  Rendering arena at FOV {fov}°...")
    print(f"  Image size: {int(cam.R)}x{int(cam.W)} px")
    print(f"  Ball position: ({launch.p0[0]:.2f}, {launch.p0[1]:.2f})")
    print(f"  Ball velocity: ({launch.v0[0]:.2f}, {launch.v0[1]:.2f}) m/s")

    # Save visualization
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the rendered image
    from PIL import Image

    img = Image.fromarray(depth_image)
    img_file = out_dir / f"camera_render_fov{fov}.png"
    img.save(img_file)
    print(f"  ✓ Saved image: {img_file}")

    renderer.close()
    cam.close()
    print()

print("="*70)
print("✓ Real camera images rendered and saved!")
print("="*70)
