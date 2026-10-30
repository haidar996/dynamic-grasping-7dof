#!/usr/bin/env python3
"""Quick check: visualize arena with ball (no rendering, just view)"""
import sys
from pathlib import Path
import mujoco
import mujoco.viewer

root = Path(__file__).resolve().parents[1]

ARENA_XML = root / "scenes" / "arena.xml"

print("Loading arena...")
model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
data = mujoco.MjData(model)

print(f"✓ Model loaded")
print(f"✓ Bodies: {model.nbody}")
print(f"✓ Geoms: {model.ngeom}")

# Check for ball
try:
    ball_body = model.body("ball")
    print(f"✓ Ball found: body_id={ball_body.id}, pos={data.xpos[ball_body.id]}")
except:
    print("✗ Ball NOT found in model!")

print()
print("Opening viewer... (should show arena with ball at center)")
print("Close window to exit")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

print("✓ Done")
