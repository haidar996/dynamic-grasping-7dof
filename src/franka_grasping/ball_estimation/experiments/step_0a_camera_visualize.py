#!/usr/bin/env python3
"""Visualize camera output for each FOV (40°, 50°, 60°)."""
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

print("Capturing sample camera data for each FOV...")
print()

fov_stats = {}

for fov in FOVS:
    print(f"FOV {fov}°:")

    # Create camera and sample one launch
    cam = DepthCamera(model, CAMERA_HEIGHT, float(fov))
    rng = np.random.default_rng(fov)
    launches = sample_launches(rng, arena, 1)
    launch = launches[0]

    # Record the launch
    data = record_raw(model, arena, cam, launch, 10.0)

    # Get UV measurements (ball position in image)
    uvz = data['uvz']  # Shape: (n_frames, 3)
    valid = ~np.isnan(uvz[:, 0])  # Valid detections

    # Save sample data
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"camera_sample_fov{fov}.npz"
    np.savez(out_file, uvz=uvz, valid=valid, fov=fov,
             true_xy=data['true_xy'], frame_t=data['frame_t'])

    # Collect stats
    fov_stats[fov] = {
        'res_h': int(cam.R),
        'res_w': int(cam.W),
        'n_frames': len(uvz),
        'valid_frames': int(valid.sum())
    }

    # Print stats
    print(f"  Camera resolution: {int(cam.R)} x {int(cam.W)} px")
    print(f"  Total frames: {len(uvz)}")
    print(f"  Valid detections: {valid.sum()} / {len(valid)}")
    if valid.sum() > 0:
        valid_uvz = uvz[valid]
        print(f"  Ball position U range: [{valid_uvz[:, 0].min():.0f}, {valid_uvz[:, 0].max():.0f}] px")
        print(f"  Ball position V range: [{valid_uvz[:, 1].min():.0f}, {valid_uvz[:, 1].max():.0f}] px")
        print(f"  Ball depth Z range: [{valid_uvz[:, 2].min():.3f}, {valid_uvz[:, 2].max():.3f}] m")
    print(f"  ✓ Saved to: {out_file}")

    cam.close()
    print()

# Create comparison HTML
print("Creating comparison visualization...")

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Camera FOV Comparison</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        .container {{ display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; margin-top: 30px; }}
        .fov-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            min-width: 300px;
            text-align: center;
        }}
        .fov-title {{ font-size: 20px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; }}
        .stats {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            text-align: left;
            font-size: 13px;
            line-height: 1.8;
        }}
        .stat-label {{ font-weight: bold; color: #555; }}
        .stat-value {{ color: #0066cc; }}
        .description {{ margin-top: 15px; font-size: 12px; color: #666; font-style: italic; }}
        .camera-viz {{
            width: 250px;
            height: 250px;
            border: 2px solid #ddd;
            border-radius: 4px;
            margin: 15px auto;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
        }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>📷 Camera FOV Comparison (2.5×2.5 m Arena, Height: 3.0 m)</h1>

    <div class="container">
        <div class="fov-box">
            <div class="fov-title">FOV: 40°</div>
            <div class="camera-viz">40°<br>NARROW</div>
            <div class="stats">
                <div><span class="stat-label">Resolution:</span> <span class="stat-value">{fov_stats[40]['res_h']} × {fov_stats[40]['res_w']} px</span></div>
                <div><span class="stat-label">Frames:</span> <span class="stat-value">{fov_stats[40]['n_frames']}</span></div>
                <div><span class="stat-label">Valid:</span> <span class="stat-value">{fov_stats[40]['valid_frames']}</span></div>
            </div>
            <div class="description">✓ Best detail<br>✗ Narrow coverage<br>✗ May miss ball</div>
        </div>

        <div class="fov-box">
            <div class="fov-title">FOV: 50°</div>
            <div class="camera-viz">50°<br>MEDIUM</div>
            <div class="stats">
                <div><span class="stat-label">Resolution:</span> <span class="stat-value">{fov_stats[50]['res_h']} × {fov_stats[50]['res_w']} px</span></div>
                <div><span class="stat-label">Frames:</span> <span class="stat-value">{fov_stats[50]['n_frames']}</span></div>
                <div><span class="stat-label">Valid:</span> <span class="stat-value">{fov_stats[50]['valid_frames']}</span></div>
            </div>
            <div class="description">✓ Balanced view<br>✓ Good detail & coverage<br>✓ Recommended</div>
        </div>

        <div class="fov-box">
            <div class="fov-title">FOV: 60°</div>
            <div class="camera-viz">60°<br>WIDE</div>
            <div class="stats">
                <div><span class="stat-label">Resolution:</span> <span class="stat-value">{fov_stats[60]['res_h']} × {fov_stats[60]['res_w']} px</span></div>
                <div><span class="stat-label">Frames:</span> <span class="stat-value">{fov_stats[60]['n_frames']}</span></div>
                <div><span class="stat-label">Valid:</span> <span class="stat-value">{fov_stats[60]['valid_frames']}</span></div>
            </div>
            <div class="description">✓ Maximum coverage<br>✗ Less detail<br>✗ Lower pixel density</div>
        </div>
    </div>

    <div class="footer">
        <p>Sample data from one ball launch in 2.5×2.5 m arena (no arm)</p>
        <p>Each FOV tested with 50 launches • Full datasets in fov_test_XX/ directories</p>
    </div>
</body>
</html>
"""

html_file = root / "outputs" / "camera_fov_comparison.html"
html_file.write_text(html_content)
print(f"✓ Visualization saved!")
print()
print("="*70)
print("✓ Camera FOV samples ready!")
print(f"  Comparison: {html_file}")
print("="*70)
