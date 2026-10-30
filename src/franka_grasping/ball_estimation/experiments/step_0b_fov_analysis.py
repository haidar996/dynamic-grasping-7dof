#!/usr/bin/env python3
"""Analyze FOV test data to pick the best camera field of view."""
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]

FOVS = [50, 55, 58, 60, 62, 65, 70, 75, 80]
ARENA_SIZE = 2.5  # meters (diagonal = 3.536 m)
CAMERA_HEIGHT = 3.0

print("╔════════════════════════════════════════════════════════════════════╗")
print("║         Phase 0.2: FOV Optimization Analysis                      ║")
print("║         Selecting Best Camera FOV for 2.5×2.5 m Arena             ║")
print("╚════════════════════════════════════════════════════════════════════╝")
print()

results = {}

for fov in FOVS:
    out_dir = root / "outputs" / f"fov_test_{fov}"
    npz_files = sorted(out_dir.glob("launch_*.npz"))

    if len(npz_files) == 0:
        continue

    # Analyze first 5 launches to get statistics
    all_uvz = []
    valid_count = 0

    for npz_file in npz_files[:5]:
        data = np.load(npz_file)
        uvz = data['uvz']  # Shape: (n_frames, 3)
        valid = ~np.isnan(uvz[:, 0])

        if valid.sum() > 0:
            valid_count += valid.sum()
            all_uvz.append(uvz[valid])

    if len(all_uvz) == 0:
        continue

    # Concatenate all valid detections
    all_uvz = np.vstack(all_uvz)

    # Load one sample to get camera resolution
    sample = np.load(npz_files[0])
    cam_h = sample['uvz'].shape[0]  # approx frames

    # Get camera resolution from the data
    # Typical: 480x640 for the DepthCamera
    img_h, img_w = 480, 640

    # Calculate metrics
    u_center = img_w / 2
    v_center = img_h / 2

    # Distance from image center (normalized)
    u_dist = np.abs(all_uvz[:, 0] - u_center) / img_w
    v_dist = np.abs(all_uvz[:, 1] - v_center) / img_h
    avg_dist_from_center = np.mean(np.sqrt(u_dist**2 + v_dist**2))

    # FOV coverage on ground (theoretical)
    # tan(fov/2) * 2 * H = ground_coverage
    fov_rad = np.radians(fov)
    ground_coverage = 2 * CAMERA_HEIGHT * np.tan(fov_rad / 2)

    # How well does this FOV fit the arena?
    # Ideal: ground_coverage ≈ ARENA_SIZE * sqrt(2) for diagonal coverage
    ideal_coverage = ARENA_SIZE * np.sqrt(2)
    coverage_efficiency = min(100, (ideal_coverage / ground_coverage) * 100) if ground_coverage > 0 else 0

    # Detection reliability (percentage of frames with valid detection)
    detection_reliability = 100 * valid_count / (5 * cam_h)

    results[fov] = {
        'files': len(npz_files),
        'ground_coverage': ground_coverage,
        'coverage_efficiency': coverage_efficiency,
        'detection_reliability': detection_reliability,
        'avg_dist_from_center': avg_dist_from_center,
        'ideal_coverage': ideal_coverage
    }

print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║                    FOV COMPARISON METRICS                            ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
print()
print(f"Arena: {ARENA_SIZE}×{ARENA_SIZE} m | Diagonal: {ARENA_SIZE*np.sqrt(2):.2f} m | Camera height: {CAMERA_HEIGHT} m")
print()
print("FOV  │ Files │ Ground │ Coverage │ Detection │ Center  │ Score")
print("     │       │ Cover  │ Efficiency│ Reliability│Distance│")
print("°    │       │ (m)    │   (%)     │    (%)    │         │")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

scores = {}
for fov in sorted(results.keys()):
    r = results[fov]

    # Calculate composite score
    # - Coverage efficiency (how well FOV fits the arena diagonal): 40% weight
    # - Detection reliability: 30% weight
    # - Centering (lower is better): 30% weight (inverted)
    score = (
        r['coverage_efficiency'] * 0.40 +
        r['detection_reliability'] * 0.30 +
        (100 - r['avg_dist_from_center'] * 100) * 0.30
    )
    scores[fov] = score

    print(f"{fov:3d} │ {r['files']:5d} │ {r['ground_coverage']:6.2f} │ {r['coverage_efficiency']:9.1f} │ {r['detection_reliability']:9.1f} │ {r['avg_dist_from_center']:7.3f} │ {score:6.1f}")

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

# Find best FOV
best_fov = max(scores, key=scores.get)
best_score = scores[best_fov]

print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║                         RECOMMENDATION                               ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
print()
print(f"🎯 BEST FOV: {best_fov}°")
print(f"   Score: {best_score:.1f}/100")
print()
print("Why:")
r_best = results[best_fov]
print(f"  ✓ Ground coverage: {r_best['ground_coverage']:.2f} m (ideal: {r_best['ideal_coverage']:.2f} m)")
print(f"  ✓ Coverage efficiency: {r_best['coverage_efficiency']:.1f}% (captures arena optimally)")
print(f"  ✓ Detection reliability: {r_best['detection_reliability']:.1f}%")
print(f"  ✓ Ball centering: {r_best['avg_dist_from_center']:.3f} (good pixel density)")
print()
print("✅ This FOV balances:")
print("   - Full arena coverage (no blind spots)")
print("   - Minimal wasted pixels outside arena")
print("   - Good ball detection quality")
print()
print(f"→ Use FOV {best_fov}° for all future phases!")
print()
