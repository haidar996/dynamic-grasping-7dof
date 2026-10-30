#!/usr/bin/env python3
"""
Visualize the arena in MuJoCo viewer.
Shows: 2.5x2.5 m arena, Franka Panda arm at center, ball, walls.
"""

import mujoco
import mujoco.viewer

# Load the arena scene
import os
import sys

# Get absolute paths
project_root = "/home/haidar/dynamic-grasping-7dof"
scenes_path = f"{project_root}/src/franka_grasping/ball_estimation/scenes"
arena_path = f"{scenes_path}/arena.xml"

# Change working directory so relative includes work
os.chdir(scenes_path)

try:
    model = mujoco.MjModel.from_xml_path(arena_path)
    data = mujoco.MjData(model)

    print(f"✓ Loaded arena: {arena_path}")
    print(f"✓ Arena ready for visualization")
    print(f"\nOpening MuJoCo viewer...")
    print(f"  - Arena: 2.5×2.5 m (x,y ∈ [-1.25, 1.25])")
    print(f"  - Arm: Franka Panda at center (0, 0, 0)")
    print(f"  - Ball: Green sphere at center")
    print(f"  - Camera: Overhead view")
    print(f"\nControls:")
    print(f"  - Right-click + drag: Rotate view")
    print(f"  - Scroll: Zoom")
    print(f"  - Space: Pause/resume")
    print(f"  - Ctrl+click + drag: Pan")
    print(f"\nClose the viewer window to exit.")

    # Open viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

except FileNotFoundError as e:
    print(f"✗ Error: Could not find {arena_path}")
    print(f"  Make sure you're running from the experiments/ directory")
    print(f"  Error: {e}")
except Exception as e:
    print(f"✗ Error loading arena: {e}")
    import traceback
    traceback.print_exc()
